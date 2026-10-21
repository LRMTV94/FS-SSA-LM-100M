# --------------------------------------------------------------------
#    FS2-SSA v2.1.0 IN RECURRENT FORM: RELOAD THE MODEL AND TEST IT
# --------------------------------------------------------------------
#
#  The model of the v2.1.0 results: spiking residual highway, K=3 for
#  every FS neuron, data-dependent decay.
#
#  Self-contained: it defines the model as it was trained (the classes
#  of v2.1.0/model.py; SignedFSNeuron keeps its gain as alpha1, the name
#  in the checkpoints), the helpers the classes need and the validation
#  data, so it depends on nothing in the session and runs in a fresh
#  runtime too. Validation tokens come from the cache the training
#  script writes on Drive (searched under DATA); without it, from the
#  first documents of FineWeb-Edu.
#
#  Weights from Hugging Face (SOURCE = "hf", HF_FILE in HF_REPO) or from
#  Drive (SOURCE = "drive", CONFIG's file in OUT). The checkpoint is
#  loaded twice: once in the parallel form used in training and once
#  with every attention block switched to its recurrent form. Same
#  weights, nothing retrained. Per head, with gamma_t the decay the gate
#  computes at token t, the parallel forward computes
#
#      out_t = scale / Z_t * sum_{j<=t} D_tj (q_t . k_j) v_j
#      D_tj  = prod_{j<l<=t} gamma_l,      Z_t = sum_{j<=t} D_tj
#
#  and the recurrent one carries a state of fixed size instead:
#
#      S_t = gamma_t * S_{t-1} + k_t^T v_t      (d_head x d_head per head)
#      Z_t = gamma_t * Z_{t-1} + 1
#      out_t = scale * (q_t S_t) / Z_t
#
#  Four tests: same logits, same validation loss, same generated text
#  at the same seed with the generation time of both, and the logits
#  again in float64.
# =======================================================================

import copy
import glob
import math
import os
import time
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE = "hf"                            # "hf": the published weights, "drive": CONFIG's checkpoint in OUT
HF_REPO = "Matt-94/FS-SSA-LM-100M"
HF_FILE = "v2.1.0/model.pt"              # the K=3 checkpoint in HF_REPO, next to the model.pt of v2.0
DATA = "/content/drive/MyDrive/fsssa"
OUT = "/content/drive/MyDrive/fsssa"     # where v2.1.0/model.py writes its checkpoints

SEED = 1234
N_VAL = 10            # validation batches for the loss comparison
NEW_TOKENS = 256      # generated tokens for the timing test

BLOCK      = 1024
D_MODEL    = 576
N_HEADS    = 9
N_LAYER    = 16
MLP_RATIO  = 4
DROPOUT    = 0.


WIDTH          = 1.1                   # surrogate half-width
READOUT_SCALE  = 1.0                   # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT   = 0.75                 # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT  = 1.0                  # mlp_scale = this x measured std of the MLP pre-activation
QK_SCALE  = QK_SIGMA_MULT * 1.000      # x the std model.py measures at init; they only set the
MLP_SCALE = MLP_SIGMA_MULT * 0.271     # initial thresholds, which the checkpoint overwrites
MICRO_BATCH = 16                       # sequences per validation batch, as in the v2.0 check: same seed, same windows

GEN_TOKENS      = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 200

W_MIN = 8.0
W_MAX = BLOCK
GAMMA_MIN, GAMMA_MAX = 0.50, 0.9999


# =====================================================================
#                   HELPERS AND VALIDATION DATA
# =====================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB = tokenizer.n_vocab
decode = lambda ids: tokenizer.decode([int(i) for i in ids])
_record = lambda module, spike_count: None          # spike counting is not needed here

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print(f"Drive not mounted ({e})")


def stream_tokens(n):

    ''' The first n tokens of FineWeb-Edu, documents closed by EOT as in model.py '''

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = []
    for entry in ds:
        out += tokenizer.encode_ordinary(entry["text"]) + [tokenizer.eot_token]
        if len(out) >= n:
            return torch.tensor(out[:n])


cache = sorted(glob.glob(f"{DATA}/**/fineweb*_val_gpt2.bin", recursive=True))
if cache:
    val_data = torch.from_numpy(np.fromfile(cache[0], dtype=np.int32).astype(np.int64))
    print(f"validation tokens: {cache[0]}, {len(val_data):,}")
else:
    val_data = stream_tokens(2_000_000)
    print("validation cache not found: first 2M tokens of FineWeb-Edu, from the training portion")


def get_batch(split, block_size, generator=None):

    ''' Validation windows drawn as in model.py, so a seed gives the same batches '''

    ix = torch.randint(len(val_data) - block_size - 1, (MICRO_BATCH,), generator=generator)
    x = torch.stack([val_data[i:i + block_size] for i in ix])
    y = torch.stack([val_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# =====================================================================
#                    FS NEURONS + SPIKE ACCOUNTING
# =====================================================================


def gamma_ladder(n_heads, w_min, w_max):

    r = (w_max / w_min) ** (1.0 / max(1, n_heads - 1))
    return [1.0 - 1.0 / (w_min * r ** h) for h in range(n_heads)]

gamma_window = lambda g: float("inf") if g >= 1.0 else 1.0 / (1.0 - g)

class TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.clamp(1.0 - x.abs() / ctx.width, min=0.0), None

spike = TriangularSpike.apply


class FSNeuron(nn.Module):

    ''' T, h scaled by threshold_scale; d by readout_scale '''

    def __init__(self, K, width, threshold_scale, readout_scale):
        super().__init__()
        self.K = K
        self.width = width
        self.threshold_scale = threshold_scale
        self.readout_scale = readout_scale
        self.spike_sum = 0.0
        self.spike_n = 0

        geom = 2.0 ** -(torch.arange(K, dtype=torch.float32))
        self.register_buffer("T", threshold_scale * geom.clone())
        self.register_buffer("h", threshold_scale * geom.clone())
        self.register_buffer("d", readout_scale * geom.clone())

    def forward(self, x):
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - self.T[i], self.width)
            out = out + s * self.d[i]
            v = v - s * self.h[i]
            cnt = cnt + s
        _record(self, cnt)
        return out


class LearnableFSNeuron(nn.Module):

    ''' FS neuron with learnable thresholds '''

    def __init__(self, K, surrogate_width, threshold_scale, readout_scale, per_channel, n_channels):
        super().__init__()

        self.K = K
        self.width = surrogate_width
        self.per_channel = per_channel
        self.threshold_scale = threshold_scale
        self.readout_scale = readout_scale
        self.spike_sum = 0.0
        self.spike_n = 0

        geom = 2.0 ** -(torch.arange(K, dtype=torch.float32))
        raw_thr = self._invert(threshold_scale * geom)                          # T, h
        raw_out = self._invert(readout_scale * geom)                            # d

        if per_channel:
            raw_thr = raw_thr[:, None].repeat(1, n_channels)
            raw_out = raw_out[:, None].repeat(1, n_channels)

        self.raw_T = nn.Parameter(raw_thr.clone())
        self.raw_h = nn.Parameter(raw_thr.clone())
        self.raw_d = nn.Parameter(raw_out.clone())

    @staticmethod
    def _invert(values):
        increments = torch.cat([values[:-1] - values[1:], values[-1:]])
        return torch.log(torch.expm1(increments.clamp(min=1e-6)))

    @staticmethod
    def _ladder(raw):
        inc = F.softplus(raw)
        return inc.flip(0).cumsum(0).flip(0)

    def thresholds(self):
        return (self._ladder(self.raw_T), self._ladder(self.raw_d), self._ladder(self.raw_h))

    def forward(self, x):

        T, d, h = self.thresholds()
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - T[i], self.width)
            out = out + s * d[i]
            v = v - s * h[i]
            cnt = cnt + s
        _record(self, cnt)

        return out

    def ladder_stats(self):

        '''Learned thresholds, to check if they drifted from init '''

        T, _, _ = self.thresholds()
        if self.per_channel:
            return {"T_mean": T.mean(dim=1).tolist(), "T_std": T.std(dim=1).tolist()}
        return {"T": T.tolist()}


def make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):

    """Single place where the FS variant is chosen. No bypasses."""

    if learnable:
        return LearnableFSNeuron(K, width, threshold_scale, readout_scale, per_channel, n_channels)
    return FSNeuron(K, width, threshold_scale, readout_scale)


class SignedFSNeuron(nn.Module):

    '''ON/OFF pair: out = FS(x) - FS(-x) '''

    def __init__(self, K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):
        super().__init__()
        self.K = K
        self.width = width
        self.fs_on = make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels)
        self.fs_off = make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels)
        self.alpha1 = nn.Parameter(0.1 + 0.9 * torch.rand(n_channels))

    def forward(self, x):
        return self.alpha1 * (self.fs_on(x) - self.fs_off(-x))


def make_activation(kind, K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels):
    if kind == "gelu":
        return nn.GELU()
    if kind == "fs":
        return make_fs(K, width, threshold_scale, readout_scale, learnable, per_channel, n_channels)
    raise ValueError(f"unknown activation: {kind}")


# =====================================================================
#                              MODEL
# =====================================================================

class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout, block):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = h(q), h(k), h(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(dim=-1)
        att = self.attn_drop(att)
        return self.resid_drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, C)))


class CausalSpikingSelfAttention(nn.Module):

    ''' Softmax-free causal attention with Spiking Highway Pre-Activation.

    W_qkv and W_proj receive signed spike codes, turning dense floating-point
    matrix multiplications into sparse synaptic additions (ACs). Rows are
    normalised by a fixed per-head decay ladder instead of a softmax. '''

    def __init__(self, d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min=W_MIN, w_max=W_MAX):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.signed = signed
        self.learnable = learnable
        self.recurrent = False

        # Highway read: the RMS-normed residual is coded into signed spikes before W_qkv
        self.norm_in = RMSNorm(d_model)
        self.fs_in = SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)

        # Projections fed by spikes: all-AC matrix products
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.norm_v = RMSNorm(d_model)

        # Q and K signed (ON/OFF pairs) when `signed`, V always non-negative
        def enc():
            if signed:
                return SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)
            return make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.fs_q = enc()
        self.fs_k = enc()
        self.fs_v = make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        # Read of the attention output before W_proj
        self.norm_o = RMSNorm(d_model)
        self.fs_o = SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block, block)))
        self.register_buffer("n_keys", torch.arange(1, block + 1, dtype=torch.float32))

        self.use_decay = use_decay
        self.gammas = None

        if use_decay:
            if gamma is None:
                self.gammas = gamma_ladder(n_heads, w_min, w_max if w_max is not None else block / 2)
            elif isinstance(gamma, (int, float)):
                self.gammas = [float(gamma)] * n_heads
            else:
                self.gammas = [float(g) for g in gamma]

            assert len(self.gammas) == n_heads, f"Needed {n_heads} gamma, have {len(self.gammas)}"
            assert all(0.0 < g <= 1.0 for g in self.gammas), "gamma out of interval (0, 1]"

            g = torch.tensor(self.gammas, dtype=torch.float32)[:, None, None]
            idx = torch.arange(block, dtype=torch.float32)

            dist = (idx[:, None] - idx[None, :]).clamp(min=0)
            D = torch.pow(g, dist[None]) * self.mask
            self.register_buffer("decay", D)
            self.register_buffer("decay_sum", D.sum(-1, keepdim=True))

    def decay_stats(self):
        if not self.use_decay:
            return None
        return {"gamma": [round(g, 5) for g in self.gammas], "window": [round(gamma_window(g), 1) for g in self.gammas], "learned": False}

    def gamma_per_head(self):
        if self.use_decay:
            return self.decay[:, 1, 0]
        return torch.ones(self.n_heads, device=self.mask.device, dtype=self.mask.dtype)

    def encode(self, x):
        ''' Position-wise part: highway read, W_qkv on spikes, RMSNorm and FS coding of Q, K, V '''
        s_in = self.fs_in(self.norm_in(x))
        q, k, v = self.qkv(s_in).chunk(3, dim=-1)
        return self.fs_q(self.norm_q(q)), self.fs_k(self.norm_k(k)), self.fs_v(self.norm_v(v))

    def mix_parallel(self, q, k, v):
        ''' Training form: T x T scores, weighted by the decay and normalised by its row sums '''
        T = q.shape[2]
        att = (q @ k.transpose(-2, -1)) * self.attn_scale

        if self.use_decay:
            att = att * self.decay[None, :, :T, :T]
            att = att / self.decay_sum[None, :, :T]
        else:
            att = att * self.mask[:T, :T]
            att = att / self.n_keys[:T][None, None, :, None]

        return att @ v

    def mix_recurrent(self, q, k, v):
        ''' Same result as mix_parallel, one token at a time: S_t = g S_(t-1) + k_t^T v_t, Z_t = g Z_(t-1) + 1 '''
        B, H, T, Dh = q.shape
        g = self.gamma_per_head().to(q.dtype)[None, :, None, None]
        S = q.new_zeros(B, H, Dh, Dh)
        Z = q.new_zeros(B, H, 1, 1)
        out = []

        for t in range(T):
            S = g * S + k[:, :, t, :, None] * v[:, :, t, None, :]
            Z = g * Z + 1.0
            out.append((q[:, :, t, None, :] @ S) * self.attn_scale / Z)

        return torch.cat(out, dim=2)

    def step(self, x_t, state):
        ''' One token of the recurrent form: the state (S, Z) per head replaces the T x T matrix '''
        B, C = x_t.shape
        H, Dh = self.n_heads, self.d_head

        s_in = self.fs_in(self.norm_in(x_t))
        q, k, v = self.qkv(s_in).chunk(3, dim=-1)

        q = self.fs_q(self.norm_q(q)).reshape(B, H, Dh)
        k = self.fs_k(self.norm_k(k)).reshape(B, H, Dh)
        v = self.fs_v(self.norm_v(v)).reshape(B, H, Dh)

        g = self.gamma_per_head().to(x_t.dtype)[None, :, None, None]

        if state is None:
            S = x_t.new_zeros(B, H, Dh, Dh)
            Z = x_t.new_zeros(B, H, 1, 1)
        else:
            S, Z = state

        S = g * S + k[..., :, None] * v[..., None, :]
        Z = g * Z + 1.0
        o = (q[..., None, :] @ S) * self.attn_scale / Z

        # attention output coded into spikes before W_proj
        o_flat = o.reshape(B, C)
        o_spikes = self.fs_o(self.norm_o(o_flat))

        return self.resid_drop(self.proj(o_spikes)), (S, Z)

    def forward(self, x):
        B, T, C = x.shape
        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = (h(t) for t in self.encode(x))

        o = self.mix_recurrent(q, k, v) if self.recurrent else self.mix_parallel(q, k, v)
        o_flat = o.transpose(1, 2).reshape(B, T, C)

        # attention output coded into spikes before W_proj
        o_spikes = self.fs_o(self.norm_o(o_flat))

        return self.resid_drop(self.proj(o_spikes))


# =====================================================================
#              DATA-DEPENDENT DECAY, ONE GATE PER HEAD
# =====================================================================
#
#  gamma_t,h = exp(-softplus(W s_t + b + base)_h), with s_t the spike
#  input of W_qkv, so the gate is an all-AC product (576 x 9 weights).
#  base is the fixed ladder, kept as a buffer; W and b start at zero, so
#  at init gamma_t is the ladder for every token, and weight decay pulls
#  the gate back to the ladder instead of towards a 2-token window.
#
#  Parallel form: D_ij = prod_{j<l<=i} gamma_l = exp(c_i - c_j), with c
#  the cumulative sum of log gamma. Recurrent form: the same S and Z as
#  with the fixed ladder, decayed with the gamma of each token.
# ======================================================================


class DataDecay(nn.Module):

    ''' log gamma per token and head, from the spike input of W_qkv; at init it is the fixed ladder '''

    def __init__(self, d_model, n_heads, gammas):
        super().__init__()
        g = torch.tensor(gammas, dtype=torch.float32).clamp(max=1.0 - 1e-6)
        # softplus(base) = -log(gamma): the ladder, a buffer, so neither Adam nor weight decay moves it
        self.register_buffer("base", torch.log(torch.expm1(-torch.log(g))))
        # plain parameters, not nn.Linear: FSGPT._init would overwrite the zeros
        self.weight = nn.Parameter(torch.zeros(n_heads, d_model))
        self.bias = nn.Parameter(torch.zeros(n_heads))

    def forward(self, s):
        return -F.softplus(F.linear(s, self.weight, self.bias) + self.base)


class DataDecaySpikingSelfAttention(CausalSpikingSelfAttention):

    ''' CausalSpikingSelfAttention with the decay of every head computed per token. The rest is
    inherited from the fixed-ladder attention: highway read, Q/K/V codes, and the read of the output through norm_o. '''

    def __init__(self, d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min=W_MIN, w_max=W_MAX):
        super().__init__(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min, w_max)
        assert use_decay, "the gate starts from the decay ladder: use_decay must be on"
        self.gate = DataDecay(d_model, n_heads, self.gammas)

        # the T x T ladder is now computed from the data at every forward
        del self.decay
        del self.decay_sum

    def gamma_per_head(self):
    
        ''' Part of the decay that does not depend on the token: the ladder plus the learned bias '''
        
        return torch.exp(-F.softplus(self.gate.base + self.gate.bias))

    def encode(self, x):
    
        ''' Highway read, W_qkv on spikes, FS codes of Q, K, V, and log gamma from the same spikes '''
        
        s_in = self.fs_in(self.norm_in(x))
        q, k, v = self.qkv(s_in).chunk(3, dim=-1)
        return self.fs_q(self.norm_q(q)), self.fs_k(self.norm_k(k)), self.fs_v(self.norm_v(v)), self.gate(s_in)

    def mix_parallel(self, q, k, v, log_g):
    
        ''' Training form: scores weighted by D_ij = exp(c_i - c_j) and normalised by the row sums of D '''
        
        T = q.shape[2]
        c = log_g.cumsum(dim=-1)
        D = torch.exp((c[..., :, None] - c[..., None, :]).masked_fill(self.mask[:T, :T] == 0, float("-inf")))
        att = (q @ k.transpose(-2, -1)) * self.attn_scale * D
        return (att / D.sum(-1, keepdim=True)) @ v

    def mix_recurrent(self, q, k, v, log_g):
    
        ''' Same result as mix_parallel, one token at a time: S_t = g_t S_(t-1) + k_t^T v_t, Z_t = g_t Z_(t-1) + 1 '''
        
        B, H, T, Dh = q.shape
        S = q.new_zeros(B, H, Dh, Dh)
        Z = q.new_zeros(B, H, 1, 1)
        out = []

        for t in range(T):
            g = log_g[:, :, t, None, None].exp()
            S = g * S + k[:, :, t, :, None] * v[:, :, t, None, :]
            Z = g * Z + 1.0
            out.append((q[:, :, t, None, :] @ S) * self.attn_scale / Z)

        return torch.cat(out, dim=2)

    def step(self, x_t, state):
    
        ''' One token of the recurrent form, decayed with the gamma of this token '''
        
        B, C = x_t.shape
        H, Dh = self.n_heads, self.d_head

        s_in = self.fs_in(self.norm_in(x_t))
        q, k, v = self.qkv(s_in).chunk(3, dim=-1)

        q = self.fs_q(self.norm_q(q)).reshape(B, H, Dh)
        k = self.fs_k(self.norm_k(k)).reshape(B, H, Dh)
        v = self.fs_v(self.norm_v(v)).reshape(B, H, Dh)
        g = self.gate(s_in).exp()[..., None, None]

        if state is None:
            S = x_t.new_zeros(B, H, Dh, Dh)
            Z = x_t.new_zeros(B, H, 1, 1)
        else:
            S, Z = state

        S = g * S + k[..., :, None] * v[..., None, :]
        Z = g * Z + 1.0
        o = (q[..., None, :] @ S) * self.attn_scale / Z

        o_spikes = self.fs_o(self.norm_o(o.reshape(B, C)))
        
        return self.resid_drop(self.proj(o_spikes)), (S, Z)

    def forward(self, x):
        B, T, C = x.shape
        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v, log_g = self.encode(x)
        q, k, v = h(q), h(k), h(v)
        log_g = log_g.transpose(1, 2)

        o = self.mix_recurrent(q, k, v, log_g) if self.recurrent else self.mix_parallel(q, k, v, log_g)
        o_flat = o.transpose(1, 2).reshape(B, T, C)

        o_spikes = self.fs_o(self.norm_o(o_flat))
        
        return self.resid_drop(self.proj(o_spikes))


class SpikingMLP(nn.Module):

    ''' Spiking MLP where both W_up and W_down operate on discrete spike inputs '''

    def __init__(self, d_model, hidden, K, width, in_scale, scale, readout_scale, learnable, dropout):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fs_in = SignedFSNeuron(K, width, in_scale, readout_scale, learnable, True, d_model)
        self.fc1 = nn.Linear(d_model, hidden)

        self.act = make_activation("fs", K, width, scale, readout_scale, learnable, True, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # highway read, then W_up on spikes
        s_in = self.fs_in(self.norm(x))
        h = self.fc1(s_in)

        # the FS neuron emits spikes, so W_down is also an all-AC product
        h_spikes = self.act(h)
        out = self.fc2(h_spikes)

        return self.drop(out)


class Block(nn.Module):

    def __init__(self, d_model, n_heads, attention, activation, K, width, qk_scale, mlp_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, data_decay):
        super().__init__()

        if attention == "softmax":
            self.norm1 = RMSNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads, dropout, block)
            self.norm2 = RMSNorm(d_model)
            hidden = d_model * MLP_RATIO

            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden),
                make_activation(activation, K, width, mlp_scale, readout_scale, learnable, True, hidden),
                nn.Linear(hidden, d_model),
                nn.Dropout(dropout),
            )

            self.is_spiking_highway = False

        elif attention == "ssa":
            attn_cls = DataDecaySpikingSelfAttention if use_decay and data_decay else CausalSpikingSelfAttention
            self.attn = attn_cls(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma)
            hidden = d_model * MLP_RATIO

            self.mlp = SpikingMLP(d_model, hidden, K, width, qk_scale, mlp_scale, readout_scale, learnable, dropout)
            self.is_spiking_highway = True

        else:
            raise ValueError(attention)

    def forward(self, x):
        if self.is_spiking_highway:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class FSGPT(nn.Module):

    def __init__(self, attention, activation, K, signed, learnable, qk_scale, mlp_scale, block, use_decay, gamma, data_decay):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(block, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_HEADS, attention, activation, K, WIDTH, qk_scale, mlp_scale, READOUT_SCALE, signed, learnable, DROPOUT, block, use_decay, gamma, data_decay)
            for _ in range(N_LAYER)])

        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))

        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block:])
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    # ----------------------------------------------------------------
    #                         RECURRENT FORM
    # ----------------------------------------------------------------

    def set_recurrent(self, on=True):
        for blk in self.blocks:
            if isinstance(blk.attn, CausalSpikingSelfAttention):
                blk.attn.recurrent = on
        return self

    def step(self, idx_t, t, states):
        x = self.drop(self.tok(idx_t) + self.pos.weight[t])
        new_states = []

        for blk, st in zip(self.blocks, states):
            if blk.is_spiking_highway:
                a, st = blk.attn.step(x, st)
                x = x + a
                x = x + blk.mlp(x)
            else:
                a, st = blk.attn.step(blk.norm1(x), st)
                x = x + a
                x = x + blk.mlp(blk.norm2(x))
            new_states.append(st)

        return self.head(self.norm(x)), new_states

    @torch.no_grad()
    def generate_recurrent(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        assert all(isinstance(b.attn, CausalSpikingSelfAttention) for b in self.blocks), "SSA model only"
        assert idx.shape[1] + max_new_tokens <= self.block, "Position bound reached"

        states = [None] * len(self.blocks)
        for t in range(idx.shape[1]):
            logits, states = self.step(idx[:, t], t, states)

        for i in range(max_new_tokens):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)

            if i < max_new_tokens - 1:
                logits, states = self.step(nxt[:, 0], idx.shape[1] - 1, states)

        return idx

    # ----------------------------------------------------------------
    #                          STATISTICS
    # ----------------------------------------------------------------

    def spike_stats(self):
        leaves = lambda m: [x for x in m.modules() if isinstance(x, (FSNeuron, LearnableFSNeuron))]

        def rate(sites):
            tot, cnt = 0.0, 0
            for s in sites:
                ls = leaves(s)
                if not ls or ls[0].spike_n == 0:
                    continue
                tot += sum(x.spike_sum for x in ls)
                cnt += ls[0].spike_n
            return tot / cnt if cnt else float("nan")

        attn, mlp = [], []
        for blk in self.blocks:
            if blk.is_spiking_highway:
                attn += [getattr(blk.attn, nm) for nm in ("fs_in", "fs_q", "fs_k", "fs_v", "fs_o") if getattr(blk.attn, nm, None) is not None]
                mlp += [blk.mlp.fs_in, blk.mlp.act]
            else:
                attn += [getattr(blk.attn, nm) for nm in ("fs_q", "fs_k", "fs_v") if getattr(blk.attn, nm, None) is not None]
                if leaves(blk.mlp[1]):
                    mlp.append(blk.mlp[1])

        return {"attention": rate(attn), "mlp": rate(mlp)}

    def decay_stats(self):
        return [blk.attn.decay_stats() for blk in self.blocks if getattr(blk.attn, "use_decay", False)]


#  (checkpoint filename, then the tuple of v2.1.0/model.py:
#   label, attn, act, K, signed, learnable, block, use_decay, gamma, data_decay)
CONFIG = ("ckpt_fineweb_100m_ssa_K=3_p-_L_g_var_alpha_app_data_decay_s1.pt", "ssa K=3 +/- L g var alpha app data decay", "ssa", "fs", 3, True, True, BLOCK, True, gamma_ladder(N_HEADS, W_MIN, W_MAX), True)


# --------------------------------------------------------------------
#                       THE RECURRENT ATTENTION
# --------------------------------------------------------------------
#
#  The classes above carry the recurrent form already: mix_recurrent for
#  a whole sequence and step for one token, both decayed with the gamma
#  of each token, and FSGPT.generate_recurrent samples one step per token.
#  Switching it on changes the time mixing only, not the weights.
# ====================================================================

def to_recurrent(model):

    ''' A copy of the model with every spiking attention block in recurrent form '''

    rec = copy.deepcopy(model)
    for blk in rec.blocks:
        assert isinstance(blk.attn, CausalSpikingSelfAttention), "only the SSA model has a recurrent form"
    return rec.set_recurrent(True).eval()


# --------------------------------------------------------------------
#                                 LOAD
# --------------------------------------------------------------------

def load(ckpt, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm, dd):

    ''' Weights from Hugging Face or from Drive, read on the CPU: the
    checkpoint also holds the optimizer state, which is not needed here '''

    if SOURCE == "hf":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    else:
        path = f"{OUT}/{ckpt}"

        if not os.path.exists(path):
            print(f"not found -> {ckpt}\ncheckpoints in the folder:")

            for f in sorted(glob.glob(f"{OUT}/ckpt_*.pt")):
                print(f"  {os.path.basename(f)}")
            raise SystemExit

    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model", ck.get("model_state_dict", ck))
    blk = sd["pos.weight"].shape[0] if "pos.weight" in sd else blk_cfg
    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm, dd)

    # constant buffers may or may not be in the file, depending on the version
    own = set(model.state_dict())
    sd = {k: v for k, v in sd.items()
          if k in own or k.rsplit(".", 1)[-1] not in ("decay", "decay_sum", "mask", "n_keys", "dist")}
    model.load_state_dict(sd)

    best = ck.get("best", {})
    print(f"{label}\n{path}\nstep {ck.get('iter')}, best val {best.get('val', float('nan')):.4f}"
          f", context {blk}, {sum(p.numel() for p in model.parameters()):,} params")
    del ck, sd
    return model.to(device).eval()


# --------------------------------------------------------------------
#                                 TESTS
# --------------------------------------------------------------------

@torch.no_grad()
def test_logits(model, rec, x, title="1) logits"):

    ''' Same input through both forms, full sequence '''

    a, _ = model(x)
    b, _ = rec(x)
    d = (a - b).abs()
    same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"\n{title} on {x.shape[0]} x {x.shape[1]} tokens")
    print(f"   max |diff| {d.max().item():.2e}   relative {(d.max() / a.abs().max()).item():.2e}"
          f"   same argmax {100 * same:.3f}%")


@torch.no_grad()
def test_logits_f64(model, x):

    ''' Test 1 again in float64. In float32 the two forms round differently,
    and an FS input that sits within ~1e-7 of its threshold can fire in one
    form and not in the other; the flipped spike then spreads along the
    sequence through the attention. In float64 that margin shrinks by nine
    orders of magnitude. The data-dependent decay is computed by the gate,
    in float64 like the rest; a fixed ladder would be rebuilt in float64
    from the same gammas. Both forms use identical weights: what is left is
    the method. '''

    m64 = copy.deepcopy(model).double()
    for blk in m64.blocks:
        at = blk.attn
        if at.use_decay and hasattr(at, "decay"):
            g = at.decay[:, 1, 0]
            i = torch.arange(at.decay.shape[-1], device=g.device, dtype=g.dtype)
            dist = (i[:, None] - i[None, :]).clamp(min=0)
            at.decay = torch.pow(g[:, None, None], dist[None]) * at.mask.to(g.dtype)
            at.decay_sum = at.decay.sum(-1, keepdim=True)
    test_logits(m64, to_recurrent(m64), x, title="4) logits in float64")
    del m64


@torch.no_grad()
def test_loss(model, rec, n):

    ''' Validation loss of both forms on the same batches '''

    g = torch.Generator().manual_seed(SEED)
    la, lb = [], []
    for _ in range(n):
        x, y = get_batch("val", model.block, g)
        la.append(model(x, y)[1].item())
        lb.append(rec(x, y)[1].item())
    a, b = sum(la) / n, sum(lb) / n
    print(f"\n2) validation loss on {n} batches")
    print(f"   quadratic {a:.5f} (ppl {math.exp(a):.3f})   recurrent {b:.5f} (ppl {math.exp(b):.3f})"
          f"   diff {abs(a - b):.1e}")


@torch.no_grad()
def test_generation(model, rec, prompt, new_tokens):

    ''' Same prompt, same seed: same text? And how long does each take? '''

    idx = torch.tensor([tokenizer.encode_ordinary(prompt)], dtype=torch.long, device=device)
    new_tokens = min(new_tokens, model.block - idx.shape[1])
    sync = torch.cuda.synchronize if idx.is_cuda else (lambda: None)

    torch.manual_seed(SEED)
    sync()
    t0 = time.time()
    a = model.generate(idx, new_tokens, temperature=0.8, top_k=200)
    sync()
    t_quad = time.time() - t0

    torch.manual_seed(SEED)
    sync()
    t0 = time.time()
    b = rec.generate_recurrent(idx, new_tokens, temperature=0.8, top_k=200)
    sync()
    t_rec = time.time() - t0

    print(f"\n3) generation, {new_tokens} tokens from {prompt!r}")
    print(f"   same tokens: {torch.equal(a, b)}")
    print(f"   time: quadratic {t_quad:.2f} s   recurrent {t_rec:.2f} s   ({t_quad / t_rec:.1f}x)")
    print("\n   " + decode(b[0].tolist()).replace("\n", "\n   "))


model = load(*CONFIG)
rec = to_recurrent(model)

x, _ = get_batch("val", model.block, torch.Generator().manual_seed(SEED))
test_logits(model, rec, x[:2])
test_loss(model, rec, N_VAL)
test_generation(model, rec, "The process of photosynthesis", NEW_TOKENS)
test_logits_f64(model, x[:2])
