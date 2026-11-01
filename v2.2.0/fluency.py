# --------------------------------------------------------------------
#        FLUENCY TEST, v2.2.0: MANY PROMPTS, MANY SEEDS, PAIRED
# --------------------------------------------------------------------
#
#  One completion per prompt says more about the seed than about the
#  model. This test samples every prompt with several seeds, the same
#  seeds for every arm, so the arms are compared prompt by prompt and
#  seed by seed, and it scores the continuations instead of reading
#  them:
#    rep-4     share of repeated 4-grams in the continuation: the loops
#    ref loss  mean loss of the continuation under GPT-2 large, given the
#              prompt: fluency as a larger model sees it. Loops lower it
#              too, so read it next to rep-4
#    pmi       how much the prompt makes the continuation more likely
#              for GPT-2 large, in nats per token: near zero when the
#              text has left the topic of the prompt
#
#  Prompts in three groups: the three prose prompts of generate.py,
#  twelve written by hand on other topics (half a few words, half a
#  full sentence), and fifteen beginnings of documents drawn from the
#  validation cache, whose real continuation is scored too, as the
#  human reference.
#
#  Self-contained like generate.py: the classes of v2.2.0/model.py,
#  which load the v2.1 checkpoints too. Each arm of CONFIGS has its own
#  source, "hf" (its file in HF_REPO) or "drive" (its checkpoint in
#  OUT), as SOURCE in generate.py. A 24 GB GPU is enough (an L4). Every
#  sample is written to OUT, with a shuffled copy without labels for a
#  blind reading.
# --------------------------------------------------------------------

import glob
import math
import json
import random
import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm


HF_REPO = "Matt-94/FS-SSA-LM-100M"
OUT = "/content/drive/MyDrive/fsssa"   # checkpoints, the validation cache, and the samples this test writes
SEED = 1234
SEEDS_PER_PROMPT = 5                   # continuations per prompt and arm, the same random draws for every arm
NEW_TOKENS = 250                       # as the prose set of generate.py
TEMPERATURE = 0.8
TOP_K = 200
VAL_PROMPTS = 15                       # beginnings of documents from the validation cache
VAL_PROMPT_TOKENS = 32
LOOP_REP4 = 0.3                        # rep-4 above this counts as a loop; real text stays near zero
REF_MODEL = "gpt2-large"               # same tokenizer as the model, so its token ids go in as they are

BLOCK      = 1024
D_MODEL    = 576
N_HEADS    = 9
N_LAYER    = 16
MLP_RATIO  = 4
DROPOUT    = 0.

WIDTH          = 1.5                 # surrogate half-width at the start of training, in units of s; it enters the backward pass only
READOUT_SCALE  = 1.0                 # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT   = 0.75               # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT  = 1.0                # mlp_scale = this x measured std of the MLP pre-activation
QK_SCALE  = QK_SIGMA_MULT * 1.000    # x the std model.py measures at init; they only set the
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
#  (source, checkpoint filename on Drive, file on Hugging Face, label,
#   attention, activation, K, signed, learnable, block, use_decay, gamma,
#   data_decay). source is "hf" or "drive", one per arm, so the two can
#  come from different places. The first arm is the one the others are
#  compared with. On Drive the file name of the v2.1 K=3 may hold
#  another run: take it from Hugging Face.
# ---------------------------------------------------------------------

CONFIGS = [

    ("hf", "ckpt_fineweb_100m_ssa_K=3_p-_L_g_var_alpha_app_data_decay_width_sched_s1.pt", "v2.2.0/model.pt",
     "v2.2, scheduled surrogate", "ssa", "fs", 3, True, True, BLOCK, True,
     gamma_ladder(N_HEADS, W_MIN, W_MAX), True),

    ("hf", "ckpt_fineweb_100m_ssa_K=3_p-_L_g_var_alpha_app_data_decay_s1.pt", "v2.1.0/model.pt",
     "v2.1, fixed surrogate", "ssa", "fs", 3, True, True, BLOCK, True,
     gamma_ladder(N_HEADS, W_MIN, W_MAX), True),
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

def load_one(source, ckpt, hf_file, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm, dd):

    ''' One arm, from Hugging Face or from Drive, read on the CPU: the
    checkpoint also holds the optimizer state, which is not needed here '''

    if source == "hf":
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo_id=HF_REPO, filename=hf_file)
        except Exception as e:
            print(f"\n### {label}: {hf_file} could not be downloaded ({type(e).__name__}); if it is not on Hugging Face yet, set its source to \"drive\"")
            return None
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
#                               PROMPTS
# --------------------------------------------------------------------
#
#  (group, text). The hand-written ones cover topics the three of
#  generate.py do not: half are a few words, which leave the model free
#  to wander, half a full sentence, which gives it a topic to hold.
# ---------------------------------------------------------------------

PUBLISHED = [
    "The process of photosynthesis",
    "In mathematics, a prime number is",
    "When scientists study the ocean floor, they",
]

HANDWRITTEN = [
    "The human heart pumps blood",
    "During the Industrial Revolution,",
    "Volcanoes form when",
    "Vitamin D is important because",
    "A fraction represents",
    "The first computers were",
    "Plants need water, light and carbon dioxide to grow. In this lesson we will look at how the roots take up water from the soil and",
    "The French Revolution began in 1789, when the people of Paris, angry about the price of bread and the debts of the monarchy,",
    "Earthquakes happen when the plates of the Earth's crust move against each other. Scientists measure their strength with",
    "Regular physical activity has many benefits for children. Doctors recommend at least one hour of exercise every day, because",
    "To solve a linear equation such as 3x + 5 = 20, the first step is to isolate the term with the variable. We subtract",
    "Black holes are regions of space where gravity is so strong that nothing, not even light, can escape. They form when",
]


def validation_prompts():

    ''' Beginnings of documents of the validation cache, with their real continuation: VAL_PROMPT_TOKENS of
    prompt and NEW_TOKENS of continuation from a document long enough for both, drawn with a fixed seed '''

    cache = sorted(glob.glob(f"{OUT}/**/fineweb*_val_gpt2.bin", recursive=True))
    if not cache:
        print(f"validation cache not found in {OUT}: the validation group is skipped")
        return []

    data = np.fromfile(cache[0], dtype=np.int32)
    eot = tokenizer.eot_token
    span = VAL_PROMPT_TOKENS + NEW_TOKENS
    starts = np.flatnonzero(data == eot) + 1
    starts = [int(s) for s in starts if s + span <= len(data) and not (data[s:s + span] == eot).any()]

    rng = np.random.default_rng(SEED)
    picked = sorted(rng.choice(len(starts), size=min(VAL_PROMPTS, len(starts)), replace=False))
    out = []
    for p in picked:
        s = starts[p]
        out.append((data[s:s + VAL_PROMPT_TOKENS].tolist(), data[s + VAL_PROMPT_TOKENS:s + span].tolist()))
    return out


prompts = [("published", encode(t), None) for t in PUBLISHED]
prompts += [("hand-written", encode(t), None) for t in HANDWRITTEN]
prompts += [("validation", p, human) for p, human in validation_prompts()]


# -------------------------------------------------------------------------
#                              GENERATION
# -------------------------------------------------------------------------
#
#  Prompt i draws its continuations with seed SEED + i, the same for every
#  arm: while the distributions of two arms stay close, the same draws
#  pick the same tokens, so the samples start coupled and the comparison
#  is paired. The SEEDS_PER_PROMPT continuations of a prompt are the rows
#  of one batch.
# -------------------------------------------------------------------------

arms = [r for r in (load_one(*c) for c in CONFIGS) if r is not None]
assert arms, "no checkpoint loaded"

samples = []
for a in arms:
    for i, (group, ids, _) in enumerate(tqdm(prompts, desc=a["label"])):
        torch.manual_seed(SEED + i)
        idx = torch.tensor([ids] * SEEDS_PER_PROMPT, dtype=torch.long, device=device)
        out = sample(a["model"], idx, NEW_TOKENS, TEMPERATURE, TOP_K, a["block"])
        for j, row in enumerate(out.tolist()):
            samples.append({"arm": a["label"], "group": group, "prompt": i, "seed": j, "prompt_ids": ids, "ids": row[len(ids):]})

for i, (group, ids, human) in enumerate(prompts):
    if human is not None:
        samples.append({"arm": "human", "group": group, "prompt": i, "seed": 0, "prompt_ids": ids, "ids": human})

for a in arms:
    del a["model"]
if device == "cuda":
    torch.cuda.empty_cache()


# --------------------------------------------------------------------
#                                SCORING
# --------------------------------------------------------------------

ref = GPT2LMHeadModel.from_pretrained(REF_MODEL).to(device).eval()

def rep4(ids):
    ''' Share of the 4-grams of the continuation that already appeared in it '''
    grams = [tuple(ids[k:k + 4]) for k in range(len(ids) - 3)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def ref_loss(context, ids):
    ''' Mean loss of ids after context under the reference model, in nats per token '''
    x = torch.tensor([context + ids], dtype=torch.long, device=device)
    logits = ref(x).logits[0, len(context) - 1:-1].float()
    return F.cross_entropy(logits, torch.tensor(ids, dtype=torch.long, device=device)).item()


for s in tqdm(samples, desc=f"scoring with {REF_MODEL}"):
    s["rep4"] = rep4(s["ids"])
    s["ref_loss"] = ref_loss(s["prompt_ids"], s["ids"])
    s["pmi"] = ref_loss([tokenizer.eot_token], s["ids"]) - s["ref_loss"]


# --------------------------------------------------------------------
#                                REPORT
# --------------------------------------------------------------------

def mean(v):
    return sum(v) / len(v)

# two-sided 95% quantiles of Student's t for 1 to 30 degrees of freedom; beyond, its expansion around 1.96
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df):
    if df <= len(T95):
        return T95[max(df, 1) - 1]
    z = 1.959964
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def stats(rows):
    return (mean([r["rep4"] for r in rows]), mean([r["rep4"] > LOOP_REP4 for r in rows]),
            mean([r["ref_loss"] for r in rows]), mean([r["pmi"] for r in rows]), len(rows))


groups = ["published", "hand-written", "validation"]
labels = [a["label"] for a in arms] + ["human"]
width = max(len(l) for l in labels)

print("\n" + "=" * 92)
print(f" means over prompts and seeds; loops: rep-4 above {LOOP_REP4}")
print("=" * 92)

for group in groups + ["all"]:
    rows_g = [s for s in samples if group in ("all", s["group"])]
    if not rows_g:
        continue
    print(f"\n [{group}]")
    print(f"   {'arm':<{width}}  {'rep-4':>6}  {'loops':>6}  {'ref loss':>8}  {'pmi':>6}  {'n':>4}")
    
    for lab in labels:
        rows = [s for s in rows_g if s["arm"] == lab]

        if rows and not (group == "all" and lab == "human"):
            r4, loops, rl, pmi, n = stats(rows)
            print(f"   {lab:<{width}}  {r4:6.3f}  {100 * loops:5.1f}%  {rl:8.3f}  {pmi:6.3f}  {n:>4}")


def paired(base, other, group):

    ''' Differences base - other, averaged over the seeds of each prompt, then over the prompts, with a 95%
    interval from Student's t over the prompts: the seeds of one prompt are not independent '''

    out = {}
    pids = sorted({s["prompt"] for s in samples if s["arm"] == base and group in ("all", s["group"])})
    if not pids:
        return None
    for key in ("rep4", "ref_loss", "pmi"):
        per_prompt = []
        for p in pids:
            a = {s["seed"]: s[key] for s in samples if s["arm"] == base and s["prompt"] == p}
            b = {s["seed"]: s[key] for s in samples if s["arm"] == other and s["prompt"] == p}
            per_prompt.append(mean([a[j] - b[j] for j in a if j in b]))
        m = mean(per_prompt)
        sd = (sum((d - m) ** 2 for d in per_prompt) / max(1, len(per_prompt) - 1)) ** 0.5
        out[key] = (m, t95(len(per_prompt) - 1) * sd / len(per_prompt) ** 0.5, len(per_prompt))
    return out


base = arms[0]["label"]
for other in [a["label"] for a in arms[1:]]:
    print("\n" + "=" * 92)
    print(f" paired: {base} minus {other}")
    print("   rep-4 and ref loss: below zero, the first arm is better; pmi: above zero, the first arm holds the topic more")
    print("=" * 92)
    for group in groups + ["all"]:
        d = paired(base, other, group)
        if d is None:
            continue
        print(f"   {group:<13} " + "   ".join(f"{k} {m:+.3f} ± {ci:.3f}" for k, (m, ci, n) in d.items()) + f"   ({d['rep4'][2]} prompts)")


# --------------------------------------------------------------------
#                                 SAVE
# --------------------------------------------------------------------
#
#  fluency_samples.jsonl holds every continuation with its scores.
#  fluency_blind.txt shows, for every prompt and seed, the continuations
#  of the arms in a random order under the letters A, B, ...; the order
#  is in fluency_blind_key.json, to open after reading.
# -------------------------------------------------------------------

for s in samples:
    s["prompt_text"] = decode(s["prompt_ids"])
    s["text"] = decode(s["ids"])

with open(f"{OUT}/fluency_samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps({k: v for k, v in s.items() if k not in ("prompt_ids", "ids")}) + "\n")

shuffle = random.Random(SEED)
key = {}
with open(f"{OUT}/fluency_blind.txt", "w") as f:
    for i, (group, ids, _) in enumerate(prompts):
        for j in range(SEEDS_PER_PROMPT):
            rows = [s for s in samples if s["prompt"] == i and s["seed"] == j and s["arm"] != "human"]
            shuffle.shuffle(rows)
            key[f"{i}.{j}"] = {chr(65 + n): r["arm"] for n, r in enumerate(rows)}
            f.write("=" * 70 + f"\n[{i}.{j}] {decode(ids)!r}\n")
            for n, r in enumerate(rows):
                f.write(f"\n  {chr(65 + n)}:  " + r["text"].replace("\n", "\n      ") + "\n")
            f.write("\n")

with open(f"{OUT}/fluency_blind_key.json", "w") as f:
    json.dump(key, f, indent=1)

print(f"\nwritten to {OUT}: fluency_samples.jsonl, fluency_blind.txt, fluency_blind_key.json")
