# --------------------------------------------------------------------
#       ZERO-SHOT BENCHMARKS, v2.4.0: ARC-EASY, PIQA, HELLASWAG
# --------------------------------------------------------------------
#
#  v2.3 and v2.2 next to the dense control on multiple-choice
#  benchmarks, zero shot, through the LM Evaluation Harness of
#  EleutherAI (lm-eval): the prompts, the splits and the two accuracies
#  are those of every model evaluated with it, so the numbers can be
#  set next to theirs.
#    acc       the choice with the highest log-likelihood
#    acc_norm  the same, with the log-likelihood divided by the length
#              of the choice in characters, so that a long answer is
#              not penalised for its length
#
#  The dense control is Softmax + GELU trained by v2.3.0/model.py on
#  the recipe of v2.2 and v2.3: the same 10,000 steps, learning rate and
#  data, not the same batches (3.455, perplexity 31.67).
#
#  Three tasks: ARC-Easy on its test split of 2,376 questions, PIQA and
#  HellaSwag on their validation splits of 1,838 and 10,042 questions,
#  whose labels are public. One request per choice: about 53,000 per
#  arm, three quarters of them HellaSwag.
#
#  Paired, as the fluency test: every arm answers the same questions, so
#  every pair of arms is compared question by question. Next to the
#  accuracies it keeps the margin of each question, the log-likelihood
#  of the right choice minus that of the best wrong one. An accuracy
#  moves only when a margin changes sign; the margin moves on every
#  question, so it sees differences the accuracy does not.
#
#  Self-contained like generate.py: the classes of v2.3.0/model.py,
#  which trained the dense control and load v2.2.0 with the head in
#  floats. Weights from Hugging Face only. One sequence per forward
#  pass: padding several into one batch could change the rounding, and
#  in float32 a rounding can flip a spike, so the scores would depend
#  on how the requests are grouped. Every question of every arm is
#  written to OUT.
#
#  Use this command before launching the script:
#  !pip install -q lm-eval tiktoken
# --------------------------------------------------------------------

import json
import math
import os
import time
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from lm_eval import evaluator
from lm_eval.api.model import LM

HF_REPO = "Matt-94/FS-SSA-LM-100M"

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

        # ----------------------------------------------------------------
        #                 Spiking output head
        # ----------------------------------------------------------------
        self.fs_head = None
        if attention == "ssa":
            rng = torch.get_rng_state()
            self.fs_head = SignedFSNeuron(K, WIDTH, qk_scale, READOUT_SCALE, learnable, True, D_MODEL)
            torch.set_rng_state(rng)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def read_head(self, x):
        ''' Logits from the residual: the final RMSNorm, coded into signed spikes on the SSA model, then W_head '''
        h = self.norm(x)
        if self.fs_head is not None:
            h = self.fs_head(h)
        return self.head(h)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))

        for blk in self.blocks:
            x = blk(x)
        logits = self.read_head(x)

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

        return self.read_head(x), new_states

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

        stats = {"attention": rate(attn), "mlp": rate(mlp)}
        if self.fs_head is not None:
            stats["head"] = rate([self.fs_head])
        return stats

    def decay_stats(self):
        return [blk.attn.decay_stats() for blk in self.blocks if getattr(blk.attn, "use_decay", False)]




# --------------------------------------------------------------------
#                            CONFIGURATIONS
# --------------------------------------------------------------------
#
#  (file in HF_REPO, label, attention, activation, K, signed, learnable,
#   block, use_decay, gamma, data_decay), as in generate.py. Every pair
#  of arms is compared, the earlier of CONFIGS minus the later. Whether
#  the head reads spikes is taken from the checkpoint, not from CONFIGS.
#  The dense control is Dense_Baseline/model.pt in HF_REPO: if it is uploaded
#  under another name, change it here.
# =====================================================================

CONFIGS = [

    ("Dense_Baseline/model.pt", "dense, softmax + GELU", "softmax", "gelu", 2, False, False, BLOCK, False, None, False),

    ("v2.2.0/model.pt", "v2.2, float head", "ssa", "fs", 3, True, True, BLOCK, True, gamma_ladder(N_HEADS, W_MIN, W_MAX), True),

    ("v2.3.0/model.pt", "v2.3, spiking head", "ssa", "fs", 3, True, True, BLOCK, True, gamma_ladder(N_HEADS, W_MIN, W_MAX), True),
]

#  Task names of lm-eval, run in this order. OUT on Drive, when it
#  mounts, keeps the finished tasks through a disconnection.

TASKS = ["arc_easy", "piqa", "hellaswag"]
LIMIT = None
OUT = "."


# --------------------------------------------------------------------
#                                 LOAD
# --------------------------------------------------------------------

def load_one(hf_file, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm, dd):

    ''' One arm, from Hugging Face, read on the CPU: the
    checkpoint also holds the optimizer state, which is not needed here '''

    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=hf_file)
    except Exception as e:
        print(f"\n### {label}: {hf_file} could not be downloaded from {HF_REPO} ({type(e).__name__})")
        return None

    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model", ck.get("model_state_dict", ck))
    blk = sd["pos.weight"].shape[0] if "pos.weight" in sd else blk_cfg
    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm, dd)

    # ----------------------------------------------------------------
    #          The head reads spikes only if the checkpoint
    #         holds the head code: without it, as in v2.2.0
    #         and v2.1.0, the code is taken out of the model
    #          and the head reads floats, as it was trained
    # ----------------------------------------------------------------
    spike_head = any(k.startswith("fs_head.") for k in sd)
    if not spike_head:
        model.fs_head = None

    own = set(model.state_dict())
    sd = {k: v for k, v in sd.items()
          if k in own or k.rsplit(".", 1)[-1] not in ("decay", "decay_sum", "mask", "n_keys", "dist")}
    model.load_state_dict(sd)

    best = ck.get("best", {})
    print("\n" + "#" * 70)
    print(f"#  {label}")
    print(f"#  {path}")
    print(f"#  step {ck.get('iter')}, best val {best.get('val', float('nan')):.4f}"
          f", context {blk}, {sum(p.numel() for p in model.parameters()):,} params"
          f", {'spiking' if spike_head else 'float'} head")
    print("#" * 70)
    del ck, sd
    return {"label": label, "model": model.to(device).eval(), "block": blk}



# --------------------------------------------------------------------
#                            LM-EVAL WRAPPER
# --------------------------------------------------------------------

class FS2SSA(LM):

    ''' One arm as lm-eval sees a model: the log-likelihood of each continuation after its context, by teacher forcing '''

    def __init__(self, model, block):
        super().__init__()
        self.model = model
        self.block = block

    def encode_pair(self, context, continuation):

        ''' Tokens of context and continuation as lm-eval encodes them for its own models: an empty context becomes the
        end-of-text token, the spaces at the end of the context move to the continuation, and the continuation is what
        the whole text adds to the context '''

        if context == "":
            return [tokenizer.eot_token], encode(continuation)

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole = encode(context + continuation)
        ctx = encode(context)
        return ctx, whole[len(ctx):]

    @torch.no_grad()
    def loglikelihood(self, requests):

        ''' For every request, the sum of the log-probabilities of the continuation and whether greedy decoding would
        give it, from one forward pass on the last block + 1 tokens of context and continuation '''

        out = []
        start = time.time()

        for i, req in enumerate(requests):
            ctx, cont = self.encode_pair(*req.args)
            seq = (ctx + cont)[-(self.block + 1):]
            idx = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            logp = self.model(idx)[0][0, -len(cont):].log_softmax(-1)
            tok = torch.tensor(cont, dtype=torch.long, device=device)
            ll = logp.gather(1, tok[:, None]).sum().item()
            greedy = bool((logp.argmax(-1) == tok).all())
            out.append((ll, greedy))

            if (i + 1) % 1000 == 0 or i + 1 == len(requests):
                print(f"   {i + 1:,} of {len(requests):,} requests, {time.time() - start:.0f} s")

        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("not used by the multiple-choice tasks")

    def generate_until(self, requests):
        raise NotImplementedError("not used by the multiple-choice tasks")


# --------------------------------------------------------------------
#                               QUESTIONS
# --------------------------------------------------------------------

def chars(continuation):

    ''' Length of a choice in characters, as lm-eval divides by it for acc_norm: without the space that joins it to the context '''

    return len(continuation) - 1 if continuation.startswith(" ") else len(continuation)


def question_row(task, label, s):

    ''' Accuracies and margins of one question for one arm, from the log-likelihoods of its choices that lm-eval kept '''

    ll = [float(r[0]) for r in s["filtered_resps"]]
    n = [chars(a[1]) for a in s["arguments"]]
    norm = [l / c for l, c in zip(ll, n)]
    gold = int(s["target"])

    row = {"task": task, "arm": label, "doc_id": s["doc_id"], "gold": gold, "ll": ll, "chars": n}
    row["acc"] = float(s["acc"])
    row["acc_norm"] = float(s["acc_norm"])
    row["margin"] = ll[gold] - max(l for j, l in enumerate(ll) if j != gold)
    row["margin_norm"] = norm[gold] - max(l for j, l in enumerate(norm) if j != gold)
    return row


# --------------------------------------------------------------------
#                                 RUN
# --------------------------------------------------------------------
#
#  Task by task, and within a task arm by arm: each arm is loaded,
#  evaluated on that task and freed, so a task is complete for every
#  arm before the next one starts. lm-eval keeps every question with
#  the log-likelihoods of its choices; from them each row of
#  zero_shot_samples.jsonl holds one question for one arm, and
#  zero_shot_results.json the accuracies as lm-eval reports them.
#  Both files are rewritten after every task of every arm, and a task
#  already in zero_shot_samples.jsonl for an arm is not run again: after
#  a disconnection the same cell goes on from the first one missing, as
#  long as OUT survives it. With another LIMIT or other weights, start
#  from an empty OUT.

SAMPLES = os.path.join(OUT, "zero_shot_samples.jsonl")
RESULTS = os.path.join(OUT, "zero_shot_results.json")


def write_atomic(path, text):

    ''' Write to a temporary file and rename it, so that an interruption never leaves half a file '''

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


os.makedirs(OUT, exist_ok=True)

rows = []
if os.path.exists(SAMPLES):
    with open(SAMPLES) as f:
        rows = [json.loads(line) for line in f if line.strip()]

summary = {}
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        summary = json.load(f)

done = {(r["task"], r["arm"]) for r in rows}

for task in TASKS:
    for c in CONFIGS:
        if (task, c[1]) in done:
            print(f"{task}, {c[1]}: already in {SAMPLES}, not run again")
            continue

        arm = load_one(*c)
        if arm is None:
            continue

        res = evaluator.simple_evaluate(model=FS2SSA(arm["model"], arm["block"]), tasks=[task], num_fewshot=0, limit=LIMIT, log_samples=True)
        rows += [question_row(task, arm["label"], s) for s in res["samples"][task]]
        summary.setdefault(task, {})[arm["label"]] = {k.split(",")[0]: v for k, v in res["results"][task].items() if k.endswith(",none")}

        write_atomic(SAMPLES, "".join(json.dumps(r) + "\n" for r in rows))
        write_atomic(RESULTS, json.dumps(summary, indent=2))
        print(f"{task}, {arm['label']}: written to {OUT}")

        del arm, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

assert rows, "no arm evaluated"


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


def interval(v):

    ''' Mean and half-width of its 95% interval, from Student's t over the questions '''

    m = mean(v)
    sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
    return m, t95(len(v) - 1) * sd / len(v) ** 0.5


def questions(task, label):
    return [r for r in rows if r["task"] == task and r["arm"] == label]

# -------------------------------------------------------------------------------------------------
#  the arms in the order of CONFIGS and the tasks in the order of TASKS, as far as OUT holds them
# -------------------------------------------------------------------------------------------------
labels = [c[1] for c in CONFIGS if any(r["arm"] == c[1] for r in rows)]
tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
width = max(len(l) for l in labels)

for task in tasks:
    here = [lab for lab in labels if questions(task, lab)]
    first = questions(task, here[0])
    chance = mean([1 / len(r["ll"]) for r in first])

    print("\n" + "=" * 92)
    print(f" {task}, zero shot: {len(first):,} questions, chance {100 * chance:.1f}%; 95% intervals over the questions")
    print("=" * 92)
    print(f"   {'arm':<{width}}  {'acc':>14}  {'acc_norm':>14}  {'margin, nats':>15}")

    for lab in here:
        q = questions(task, lab)
        acc, acc_ci = interval([r["acc"] for r in q])
        norm, norm_ci = interval([r["acc_norm"] for r in q])
        margin, margin_ci = interval([r["margin"] for r in q])
        print(f"   {lab:<{width}}  {100 * acc:6.2f} ± {100 * acc_ci:5.2f}  {100 * norm:6.2f} ± {100 * norm_ci:5.2f}  {margin:+7.3f} ± {margin_ci:5.3f}")

    # -----------------------------------------------------------------------------------------------------
    #  paired: every pair of arms, the earlier of CONFIGS minus the later, on the questions both answered
    # -----------------------------------------------------------------------------------------------------
    for n, lab_a in enumerate(here):
        for lab_b in here[n + 1:]:
            a = {r["doc_id"]: r for r in questions(task, lab_a)}
            b = {r["doc_id"]: r for r in questions(task, lab_b)}
            ids = [i for i in a if i in b]
            print(f"\n   paired, {lab_a} minus {lab_b}, {len(ids):,} questions: above zero, the first arm is better")

            for key, unit, scale in (("acc", "points", 100), ("acc_norm", "points", 100), ("margin", "nats", 1), ("margin_norm", "nats per character", 1)):
                m, ci = interval([a[i][key] - b[i][key] for i in ids])
                print(f"     {key:<12} {scale * m:+7.3f} ± {scale * ci:.3f} {unit}")

            both = sum(1 for i in ids if a[i]["acc"] and b[i]["acc"])
            only_a = sum(1 for i in ids if a[i]["acc"] and not b[i]["acc"])
            only_b = sum(1 for i in ids if b[i]["acc"] and not a[i]["acc"])
            print(f"     acc: right in both {both:,}, only in {lab_a} {only_a:,}, only in {lab_b} {only_b:,}")

# --------------------------------------------------------------------------------------------------------
#  Acc_norm on every task, the accuracy usually reported for HellaSwag; on ARC-Easy and PIQA it does not
#  favour the short answers. The mean over the tasks only for the arms that have all of them.
# --------------------------------------------------------------------------------------------------------
print("\n" + "=" * 92)
print(" ladder, zero shot: acc_norm in percent, 95% intervals over the questions, and the mean over the tasks")
print("=" * 92)
print(f"   {'arm':<{width}}  " + "  ".join(f"{t:>14}" for t in tasks) + f"  {'mean':>7}")

for lab in labels:
    cells = []
    values = []
    for task in tasks:
        q = questions(task, lab)
        if q:
            m, ci = interval([r["acc_norm"] for r in q])
            cells.append(f"{100 * m:6.2f} ± {100 * ci:5.2f}")
            values.append(m)
        else:
            cells.append(f"{'n/a':>14}")
    last = f"{100 * mean(values):7.2f}" if len(values) == len(tasks) else f"{'n/a':>7}"
    print(f"   {lab:<{width}}  " + "  ".join(cells) + f"  {last}")

print(f"\nwritten to {os.path.abspath(OUT)}: zero_shot_samples.jsonl, zero_shot_results.json")
