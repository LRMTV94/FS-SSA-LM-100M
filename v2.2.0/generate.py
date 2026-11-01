# --------------------------------------------------------------------
#       GENERATION, v2.2.0: K=3, DATA DECAY, SCHEDULED SURROGATE
# --------------------------------------------------------------------
#
#  Same prompts, same seed, every arm held in memory at once so the
#  continuations of a prompt print one under the other. That is the whole
#  point: read them in a column, not all of one model and then all of the
#  other. If it runs out of memory, drop one entry from CONFIGS and run
#  it twice.
#
#  Self-contained: it defines the model as it was trained (the classes
#  of v2.2.0/model.py; SignedFSNeuron keeps its gain as alpha1, the name
#  in the checkpoints), the tokenizer and the sampler, so it depends on
#  nothing in the session. The surrogate width only enters the backward
#  pass, so the forward is the one of v2.1.0, and the v2.1 checkpoints
#  load here too.
#
#  Weights from Hugging Face (SOURCE = "hf": HF_FILE in HF_REPO, the K=3
#  model with the scheduled surrogate) or from Drive (SOURCE = "drive":
#  every entry of CONFIGS, from OUT; add the v2.1 runs there to compare
#  the arms). It ends with the log-probability margins of the agreement
#  and binding probes.
# --------------------------------------------------------------------

import copy
import glob
import math
import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE = "hf"                         # "hf": the published weights, "drive": the checkpoints of CONFIGS in OUT
HF_REPO = "Matt-94/FS-SSA-LM-100M"
HF_FILE = "v2.2.0/model.pt"           # the K=3 checkpoint with the scheduled surrogate in HF_REPO
OUT = "/content/drive/MyDrive/fsssa"  # where v2.2.0/model.py writes its checkpoints
SEED = 1234

BLOCK      = 1024
D_MODEL    = 576
N_HEADS    = 9
N_LAYER    = 16
MLP_RATIO  = 4
DROPOUT    = 0.

WIDTH          = 1.5                  # surrogate half-width at the start of training, in units of s; it enters the backward pass only
READOUT_SCALE  = 1.0                  # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT   = 0.75                # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT  = 1.0                 # mlp_scale = this x measured std of the MLP pre-activation
QK_SCALE  = QK_SIGMA_MULT * 1.000     # x the std model.py measures at init; they only set the
MLP_SCALE = MLP_SIGMA_MULT * 0.271   # initial thresholds, which the checkpoint overwrites

W_MIN = 8.0
W_MAX = BLOCK


# =====================================================================
#                       HELPERS AND TOKENIZER
# =====================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB = tokenizer.n_vocab
encode = tokenizer.encode_ordinary
decode = lambda ids: tokenizer.decode([int(i) for i in ids])
_record = lambda module, spike_count: None          # spike counting is not needed here

if SOURCE == "drive":
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"Drive not mounted ({e})")


# =====================================================================
#                    FS NEURONS + SPIKE ACCOUNTING
# =====================================================================


def gamma_ladder(n_heads, w_min, w_max):

    r = (w_max / w_min) ** (1.0 / max(1, n_heads - 1))
    return [1.0 - 1.0 / (w_min * r ** h) for h in range(n_heads)]

gamma_window = lambda g: float("inf") if g >= 1.0 else 1.0 / (1.0 - g)

_WIDTH_SCALE = 1.0      # set by the training loop; at inference it does not matter, the width only enters the backward pass


class TriangularSpike(torch.autograd.Function):

    ''' Heaviside step in the forward pass '''

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        x, width = ctx.saved_tensors
        return grad_out * torch.clamp(1.0 - x.abs() / width, min=0.0) / width, None


spike = TriangularSpike.apply


class FSNeuron(nn.Module):

    ''' Few-spike neuron with fixed ladders: at step k it spikes if the residual reaches T_k,
    subtracts h_k and emits d_k, with T = h = s * 2^-k and d = r * 2^-k. The surrogate has
    half-width width * _WIDTH_SCALE * s '''

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
        w = self.width * _WIDTH_SCALE * self.T[0]
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - self.T[i], w)
            out = out + s * self.d[i]
            v = v - s * self.h[i]
            cnt = cnt + s
        _record(self, cnt)
        return out


class LearnableFSNeuron(nn.Module):

    ''' Few-spike neuron with learnable ladders, one per channel if per_channel. Each ladder is stored
    as raw increments: softplus keeps them positive and the reverse cumulative sum keeps it decreasing.
    At init it equals the fixed FSNeuron. The surrogate has half-width width * _WIDTH_SCALE * T_0, with
    T_0 the learned top threshold of each channel, detached: the width follows the ladder, no gradient
    flows through it. '''

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
        raw_thr = self._invert(threshold_scale * geom)
        raw_out = self._invert(readout_scale * geom)

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
        w = self.width * _WIDTH_SCALE * T[0].detach()
        v = x
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)

        for i in range(self.K):
            s = spike(v - T[i], w)
            out = out + s * d[i]
            v = v - s * h[i]
            cnt = cnt + s
        _record(self, cnt)
        return out

    def ladder_stats(self):
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

        self.norm_in = RMSNorm(d_model)
        self.fs_in = SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.norm_v = RMSNorm(d_model)

        def enc():
            if signed:
                return SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)
            return make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.fs_q = enc()
        self.fs_k = enc()
        self.fs_v = make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

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

        o_flat = o.reshape(B, C)
        o_spikes = self.fs_o(self.norm_o(o_flat))

        return self.resid_drop(self.proj(o_spikes)), (S, Z)

    def forward(self, x):
        B, T, C = x.shape
        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = (h(t) for t in self.encode(x))

        o = self.mix_recurrent(q, k, v) if self.recurrent else self.mix_parallel(q, k, v)
        o_flat = o.transpose(1, 2).reshape(B, T, C)

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

        self.register_buffer("base", torch.log(torch.expm1(-torch.log(g))))
        self.weight = nn.Parameter(torch.zeros(n_heads, d_model))
        self.bias = nn.Parameter(torch.zeros(n_heads))

    def forward(self, s):
        return -F.softplus(F.linear(s, self.weight, self.bias) + self.base)


class DataDecaySpikingSelfAttention(CausalSpikingSelfAttention):

    ''' CausalSpikingSelfAttention with the decay of every head computed per token '''

    def __init__(self, d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min=W_MIN, w_max=W_MAX):

        super().__init__(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min, w_max)
        assert use_decay, "the gate starts from the decay ladder: use_decay must be on"
        self.gate = DataDecay(d_model, n_heads, self.gammas)

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

    ''' Spiking MLP where both W_up and W_down operate on discrete spike inputs.

    fs_in reads the RMS-normed residual at in_scale; the FS neuron is the
    non-linearity, with thresholds at scale on the pre-activation of fc1. '''

    def __init__(self, d_model, hidden, K, width, in_scale, scale, readout_scale, learnable, dropout):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fs_in = SignedFSNeuron(K, width, in_scale, readout_scale, learnable, True, d_model)
        self.fc1 = nn.Linear(d_model, hidden)

        self.act = make_activation("fs", K, width, scale, readout_scale, learnable, True, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        s_in = self.fs_in(self.norm(x))
        h = self.fc1(s_in)

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


# --------------------------------------------------------------------
#                            CONFIGURATIONS
# --------------------------------------------------------------------
#
#  (checkpoint filename on Drive, label, attention, activation, K, signed,
#   learnable, block, use_decay, gamma, data_decay). With SOURCE = "hf"
#  only the spiking entry is loaded, from the published weights.

CONFIGS = [

    ("ckpt_fineweb_100m_ssa_K=3_p-_L_g_var_alpha_app_data_decay_width_sched_s1.pt",
     "ssa K=3 +/- L g var alpha app data decay width sched", "ssa", "fs", 3, True, True, BLOCK, True,
     gamma_ladder(N_HEADS, W_MIN, W_MAX), True),
]

# --------------------------------------------------------------------
#                             PROMPT SETS
# --------------------------------------------------------------------
#
#  Each set carries its own sampling parameters, because they are not
#  measuring the same thing.
#
#  Sampled at 0.8 for prose: judging fluency, so you want the model's
#  actual output distribution, not its mode.
#
#  Greedy (top_k=1) for everything with one correct answer. Sampling a
#  test that has a single right answer only adds noise to it.
#
#  WHAT TO EXPECT. At this scale the arithmetic and code sets will fail
#  in both arms. That is scale, not architecture: GPT-2 small saw 40 GB
#  and still cannot add two digits reliably, and FineWeb-Edu is filtered
#  educational prose with very little code in it. They stay because the
#  FAILURE MODE is the informative part. Prose is forgiving, you can lose
#  the thread and still sound fine. Code and lists are not: an unclosed
#  bracket from forty tokens back is a visible attention failure, and
#  attention is the thing under test.
#
#  The induction set is the one that should actually work. Induction
#  heads emerge early in training and are purely attentional, so this is
#  the sharpest probe in the file. If the spiking arm cannot continue a
#  pattern it has just been shown twice, that is a concrete, nameable
#  result, worth more than "the prose reads better"
#
# ----------------------------------------------------------------------

PROMPT_SETS = [
    ("PROSE  (sampled, temp 0.8)", 0.8, 200, 250, [
        "The process of photosynthesis",
        "In mathematics, a prime number is",
        "When scientists study the ocean floor, they",
    ]),

    ("INDUCTION  (greedy)", 1.0, 1, 24, [
        "The capital of France is Paris. The capital of Italy is Rome. "
        "The capital of Spain is",
        "apple red, banana yellow, grape purple, apple red, banana yellow, grape",
        "Dr. Alvarez studied volcanoes. Dr. Mehta studied glaciers. "
        "Dr. Alvarez studied",
    ]),

    ("STRUCTURE / CODE  (greedy)", 1.0, 1, 120, [
        "def factorial(n):\n    if n == 0:\n        return 1\n    return",
        "Here are the three states of matter:\n1. Solid\n2. Liquid\n3.",
        "import numpy as np\n\ndef mean(values):\n    total = 0\n    for v in values:",
    ]),

    ("ARITHMETIC  (greedy, expected to fail)", 1.0, 1, 16, [
        "2 + 2 =",
        "There are 12 eggs in a carton. Three cartons contain",
        "7 times 8 equals",
    ]),

    ("LONG-RANGE AGREEMENT  (greedy)", 1.0, 1, 40, [
        "The samples that the researchers collected from the lake bed "
        "during the summer expedition",
        "The teacher, along with the students who had arrived early that "
        "morning from the neighbouring village,",
    ]),
]


# --------------------------------------------------------------------
#                               SAMPLER
# --------------------------------------------------------------------

@torch.no_grad()
def sample(model, idx, max_new, temperature, top_k, blk):

    ''' nanoGPT semantics: crop the context to the block size, take the
    last position, scale by temperature, truncate to top_k, draw.
    top_k=1 is greedy, verified identical to argmax step by step. '''

    for _ in range(max_new):
        c = idx if idx.shape[1] <= blk else idx[:, -blk:]
        lg = model(c)[0][:, -1, :] / max(temperature, 1e-8)
        if top_k:
            v, _ = torch.topk(lg, min(top_k, lg.shape[-1]))
            lg = lg.masked_fill(lg < v[:, [-1]], -float("inf"))
        idx = torch.cat([idx, torch.multinomial(lg.softmax(-1), 1)], dim=1)
    return idx


# --------------------------------------------------------------------
#                                 LOAD
# --------------------------------------------------------------------

def load_one(ckpt, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm, dd):

    if SOURCE == "hf":
        if attn != "ssa":
            print(f"\n### {label}: not on Hugging Face, only the spiking model is published")
            return None
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    else:
        path = f"{OUT}/{ckpt}"

        if not os.path.exists(path):
            print(f"\n### {label}: not found -> {ckpt}")
            print("    checkpoints actually in the folder:")

            for f in sorted(glob.glob(f"{OUT}/ckpt_*.pt")):
                print(f"      {os.path.basename(f)}")
            return None

    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model", ck.get("model_state_dict", ck))
    blk = sd["pos.weight"].shape[0] if "pos.weight" in sd else blk_cfg
    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm, dd)

    own = set(model.state_dict())
    sd = {k: v for k, v in sd.items()
          if k in own or k.rsplit(".", 1)[-1] not in ("decay", "decay_sum", "mask", "n_keys", "dist")}
    model.load_state_dict(sd)

    best = ck.get("best", {})
    print("\n" + "#" * 70)
    print(f"#  {label}")
    print(f"#  {path}")
    print(f"#  step {ck.get('iter')}, best val {best.get('val', float('nan')):.4f}"
          f", context {blk}, {sum(p.numel() for p in model.parameters()):,} params")
    print("#" * 70)
    del ck, sd
    return {"label": label, "model": model.to(device).eval(), "block": blk}


# --------------------------------------------------------------------
#                                 RUN
# --------------------------------------------------------------------

arms = [r for r in (load_one(*c) for c in CONFIGS) if r is not None]
assert arms, "no checkpoint loaded"

for label, temp, top_k, max_new, prompts in PROMPT_SETS:
    print("\n" + "=" * 70)
    print(f" {label}")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n--- {prompt!r}")
        for a in arms:
            torch.manual_seed(SEED)          # same seed for every arm
            idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            out = sample(a["model"], idx, max_new, temp, top_k, a["block"])
            txt = decode(out[0].tolist()[len(idx[0]):])
            print(f"\n  [{a['label']}]")
            print("    " + txt.replace("\n", "\n    "))


# --------------------------------------------------------------------
#       LOG-PROBABILITY MARGINS: AGREEMENT AND BINDING PROBES
# --------------------------------------------------------------------
#
#  Scores the models loaded above in `arms`, after the generations.
#
#  Every probe compares two continuations of the same prompt by the sum
#  of the log-probabilities of their tokens, so a word split into
#  several tokens is scored whole. margin = log p(correct) - log p(wrong)
#  in nats, > 0 when the model prefers the correct one; p is the share
#  of the correct one between the two.
#
#  The margin is computed in float64, where the spikes do not flip on
#  rounding, and in float32, as generate.py runs: a different sign means
#  a near tie, decided by rounding rather than by the model.
#
#  Agreement: the published prompts, the teacher one with shorter
#  incisi, and the numbers swapped (plural subject with a singular
#  attractor, singular subject with a plural one), so that a fixed
#  preference for one verb shows up. Binding: Alvarez and Mehta in both
#  orders, asking for both names, so that recency and binding come apart.
# --------------------------------------------------------------------

# (label, prompt, correct continuation, wrong continuation)
AGREEMENT = [
    ("teacher, short", "The teacher, along with the students,", " was", " were"),
    ("teacher, medium", "The teacher, along with the students who had arrived early,", " was", " were"),
    ("teacher, published", "The teacher, along with the students who had arrived early that morning from the neighbouring village,", " was", " were"),
    ("teachers / student", "The teachers, along with the student who had arrived early that morning from the neighbouring village,", " were", " was"),
    ("samples, published", "The samples that the researchers collected from the lake bed during the summer expedition", " were", " was"),
    ("sample / researchers", "The sample that the researchers collected from the lake bed during the summer expedition", " was", " were"),
]

# (label, prompt, correct continuation, wrong continuation, True if the correct object is the more recent one)
BINDING = [
    ("Alvarez first, ask Alvarez", "Dr. Alvarez studied volcanoes. Dr. Mehta studied glaciers. Dr. Alvarez studied", " volcanoes", " glaciers", False),
    ("Alvarez first, ask Mehta", "Dr. Alvarez studied volcanoes. Dr. Mehta studied glaciers. Dr. Mehta studied", " glaciers", " volcanoes", True),
    ("Mehta first, ask Alvarez", "Dr. Mehta studied glaciers. Dr. Alvarez studied volcanoes. Dr. Alvarez studied", " volcanoes", " glaciers", True),
    ("Mehta first, ask Mehta", "Dr. Mehta studied glaciers. Dr. Alvarez studied volcanoes. Dr. Mehta studied", " glaciers", " volcanoes", False),
]


@torch.no_grad()
def continuation_logprob(model, prompt, continuation):

    ''' Sum of the log-probabilities of the tokens of continuation after prompt, by teacher forcing and the top-1 token at the end of the prompt '''

    p = encode(prompt)
    c = encode(continuation)
    idx = torch.tensor([p + c], dtype=torch.long, device=device)
    logp = model(idx)[0][0].log_softmax(-1)

    pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=device)
    tok = torch.tensor(c, dtype=torch.long, device=device)
    return logp[pos, tok].sum().item(), int(logp[len(p) - 1].argmax())


def score(model, prompt, good, bad):

    ''' Margin log p(good) - log p(bad), and the top-1 token after the prompt '''

    lp_good, top = continuation_logprob(model, prompt, good)
    lp_bad, _ = continuation_logprob(model, prompt, bad)
    return lp_good - lp_bad, top


# --------------------------------------------------------------------
#                               SCORE
# --------------------------------------------------------------------

probes = [(lab, pr, g, b) for lab, pr, g, b in AGREEMENT] + [(lab, pr, g, b) for lab, pr, g, b, _ in BINDING]
results = {}

for a in arms:
    m64 = copy.deepcopy(a["model"]).double()
    rows = []

    for lab, pr, g, b in probes:
        margin32, top32 = score(a["model"], pr, g, b)
        margin64, _ = score(m64, pr, g, b)
        rows.append((margin64, margin32, top32))

    results[a["label"]] = rows
    del m64
    if device == "cuda":
        torch.cuda.empty_cache()

# --------------------------------------------------------------------
#                        ONE PROBE, EVERY ARM
# --------------------------------------------------------------------

share = lambda m: 1.0 / (1.0 + math.exp(-m))
width = max(len(a["label"]) for a in arms)

for i, (lab, pr, g, b) in enumerate(probes):
    print("\n" + "=" * 70)
    print(f" {lab}: {g!r} ({len(encode(g))} tok) vs {b!r} ({len(encode(b))} tok)")
    print(f" {pr!r}")
    print("=" * 70)

    for a in arms:
        margin64, margin32, top32 = results[a["label"]][i]
        tie = "   <- near tie: the sign changes with the precision" if (margin64 > 0) != (margin32 > 0) else ""
        print(f"  {a['label']:<{width}}  margin {margin64:+7.3f} nats  p {share(margin64):.3f}  | float32 {margin32:+7.3f}  top-1 {decode([top32])!r}{tie}")

# --------------------------------------------------------------------
#                              SUMMARY
# --------------------------------------------------------------------

print("\n" + "=" * 70)
print(" SUMMARY, float64 margins in nats")
print("=" * 70)
print(" agreement: " + " | ".join(lab for lab, *_ in AGREEMENT))

for a in arms:
    rows = results[a["label"]]
    agreement = rows[:len(AGREEMENT)]
    binding = rows[len(AGREEMENT):]
    recent = [r[0] for r, probe in zip(binding, BINDING) if probe[4]]
    earlier = [r[0] for r, probe in zip(binding, BINDING) if not probe[4]]
    m_recent = sum(recent) / len(recent)
    m_earlier = sum(earlier) / len(earlier)

    if m_recent > 0 and m_earlier > 0:
        verdict = "binding: it follows the name"
    elif m_recent > 0:
        verdict = "recency: it takes the last object"
    elif m_earlier > 0:
        verdict = "primacy: it takes the first object"
    else:
        verdict = "swapped: it takes the object of the other name"

    print(f"\n  [{a['label']}]")
    print("    agreement  " + "  ".join(f"{r[0]:+7.3f}" for r in agreement))
    print(f"    binding    answer is the recent object {m_recent:+7.3f} | answer is the earlier object {m_earlier:+7.3f}  -> {verdict}")
