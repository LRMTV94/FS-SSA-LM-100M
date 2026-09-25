# =====================================================================
#  FS-SSA, autoregressive: BPE GPT on FineWeb
#  Single-file Colab script. Runtime > Change runtime type > GPU
#
#  Byte-identical to the tiny-Shakespeare script except for the data section,
#  the model/budget constants, gradient accumulation in the training loop and
#  the output paths. Every class -- the FS neurons, both attentions, Block,
#  FSGPT -- is unchanged, so the two experiments can be read against each
#  other and a seed means the same thing in both.
#
#  Measures validation loss and perplexity for a causal spiking attention
#  against a matched softmax control, and samples text from each.
#
#  ---------------------------------------------------------------------
#  THREE THINGS DIFFER FROM THE CLASSIFIER, AND ALL THREE ARE FORCED
#
#  1) RMSNorm replaces BatchNorm on Q/K/V. BatchNorm1d over (B, C, T) pools
#     statistics over TIME as well as batch, so in a causal model the
#     statistics at position t would include future tokens: a direct leak.
#     RMSNorm normalises over the channel dimension only. As a side effect
#     it also removes the running-statistics discrepancy that dominated the
#     classifier results, since RMSNorm keeps no buffers.
#
#  2) The row normalisation becomes causal. Without a softmax the attention
#     rows do not sum to 1 and must be divided by the number of attended
#     keys; in a causal model position t attends to t+1 keys, not to a
#     constant. Dividing by a constant would crush the first positions.
#
#  3) qk_scale is re-measured, not inherited. The value 0.25 was calibrated
#     for a BatchNorm-ed input; the spread after RMSNorm is different. The
#     script probes the pre-activation std at init and derives the scale
#     from it. Recall that the resolved window is [0, s*(2 - 2^-(K-1))),
#     with a ceiling at 2s: raising K refines the step, never the range.
#
#  4) In the current script, the channel-wise in Learnable Neuron is learnable
#     by default.In prior baseline runs (e.g., K=2, K=2 +/- L), learnable alpha 
#     was disabled by default (held static). This setup is currently provisional: 
#     future releases will introduce a dedicated configuration flag (e.g., 
#     --learnable_alpha) to explicitly toggle between static and adaptive alpha.
#
# ===========================================================================

import os
import json
import math
import time
import urllib.request
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

try:
    import tiktoken
except ImportError:
    raise SystemExit("pip install tiktoken")

# =====================================================================
#                 GOOGLE DRIVE SALVATAGE
# =====================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")


def gamma_ladder(n_heads, w_min, w_max):

    r = (w_max / w_min) ** (1.0 / max(1, n_heads - 1))
    return [1.0 - 1.0 / (w_min * r ** h) for h in range(n_heads)]

# =====================================================================
#                 MODEL CONFIGURATION ~100M PARAMETRS
# =====================================================================


BLOCK      = 1024
D_MODEL    = 576
N_HEADS    = 9
N_LAYER    = 16
MLP_RATIO  = 4
DROPOUT    = 0.

MAX_ITERS      = 10000
EVAL_INTERVAL  = 250
EVAL_ITERS     = 40       # batches averaged per evaluation
MICRO_BATCH    = 16       # what goes on the GPU at once
GRAD_ACCUM     = 4        # Effective batch = MICRO_BATCH * GRAD_ACCUM = 64
LR             = 1e-3     # Warmup Epochs
WARMUP         = 500
MIN_LR         = 1e-4
GRAD_CLIP      = 1.0
SEEDS          = [1]

WIDTH          = 1.1      # surrogate half-width
READOUT_SCALE  = 1.0      # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT   = 0.75    # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT  = 1.0     # mlp_scale = this x measured std of the MLP pre-activation

GEN_TOKENS      = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 200

W_MIN = 8.0
W_MAX = BLOCK
GAMMA_MIN, GAMMA_MAX = 0.50, 0.9999

gamma_window = lambda g: float("inf") if g >= 1.0 else 1.0 / (1.0 - g)

#  (name, attention, activation, K, signed, learnable, block, use_decay, gamma)
#   attention  -> softmax with causal mask, or SSA (softmax-free, FS-coded QKV)
#   signed     -> ON/OFF pair on Q and K, so they carry a sign
#   learnable  -> per-channel learnable threshold ladder, attention AND MLP
#   block      -> context length for THIS config
#   use_decay  -> per-head learnable gamma^(i-j) in place of the flat row mean;
#                 the softmax control has no row mean, so it ignores this
#   gamma      -> gamma values (constant)

CONFIGS = [
#   ("softmax + gelu",           "softmax",   "gelu",   2, False, False, BLOCK, False, None),
#   ("ssa K=2 +/- L",                "ssa",     "fs",   2, True,  True,  BLOCK, False, None),
    ("ssa K=2 +/- L g d=0.996",    "ssa",     "fs",   2, True,  True,  BLOCK, True,  0.996),
    ("ssa K=2 +/- L g d=0.996",    "ssa",     "fs",   2, True,  True,  BLOCK, True,  gamma_ladder(N_HEADS, W_MIN, W_MAX)),
]


# =====================================================================
#                     PATHS AND DIRECTORIES
# =====================================================================

USE_DRIVE = True

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        OUT = "/content/drive/MyDrive/fsssa"
    except Exception as e:
        print(f"Drive non disponibile ({e}), uso directory locale.")
        OUT = "."
else:
    OUT = "."

os.makedirs(OUT, exist_ok=True)

TAG              = "fineweb_100m"
DATASET_NAME     = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG   = "sample-10BT"
TARGET_TOKENS    = 500_000_000                          # 100M tokens (cache)

TOKEN_PATH_TRAIN = f"{OUT}/{TAG}_train_gpt2.bin"
TOKEN_PATH_VAL   = f"{OUT}/{TAG}_val_gpt2.bin"
META_PATH        = f"{OUT}/training_meta_{TAG}.json"

ckpt_path_of = lambda name, seed: f"{OUT}/ckpt_{TAG}_{lk(name)}_s{seed}.pt"
hist_path_of = lambda name, seed: f"{OUT}/history_{TAG}_{lk(name)}_s{seed}.json"

lk = lambda s: s.replace(" ", "_").replace("/", "").replace("+", "p")
run_ckpt = lambda name, seed: f"{OUT}/ckpt_{TAG}_{lk(name)}_s{seed}.pt"
run_log  = lambda name, seed: f"{OUT}/progress_{TAG}_{lk(name)}_s{seed}.json"

def atomic_save(obj, path):

    ''' Write to a temporary file and rename '''

    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def atomic_json(obj, path):
    tmp = path + ".tmp"

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)

# =====================================================================
#                DATASET LOADING & TOKENISATION (Hugging Face)
# =====================================================================

tokenizer = tiktoken.get_encoding("gpt2")
VOCAB = tokenizer.n_vocab
EOT = tokenizer.eot_token
decode = lambda ids: tokenizer.decode([int(i) for i in ids])


def build_tokens(target_tokens=TARGET_TOKENS):

    '''Stream from Hugging Face, BPE tokenization + caching on disk'''

    if os.path.exists(TOKEN_PATH_TRAIN) and os.path.exists(TOKEN_PATH_VAL):

        print("Tokens loaded from local cache...")
        train_arr = np.fromfile(TOKEN_PATH_TRAIN, dtype=np.int32)
        val_arr = np.fromfile(TOKEN_PATH_VAL, dtype=np.int32)

        print(f"Cache loaded: Train {len(train_arr):,} | Val {len(val_arr):,} tokens")
        return torch.from_numpy(train_arr.astype(np.int64)), torch.from_numpy(val_arr.astype(np.int64))

    print(f"Streaming from Hugging Face ({DATASET_NAME})...")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    all_tokens = []
    total_tokens = 0
    print(f"Tokenization (Target: {target_tokens:,} tokens)...")

    for entry in ds:
        text = entry["text"]
        tokens = tokenizer.encode_ordinary(text)
        tokens.append(EOT)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)

        if total_tokens >= target_tokens:
            break

        if len(all_tokens) % 10_000_000 < 500:
            print(f"  Progress: {total_tokens:,} / {target_tokens:,} tokens...", flush=True)

    arr = np.asarray(all_tokens[:target_tokens], dtype=np.int32)
    n_train = int(0.9 * len(arr))

    train_arr = arr[:n_train]
    val_arr = arr[n_train:]

    train_arr.tofile(TOKEN_PATH_TRAIN)
    val_arr.tofile(TOKEN_PATH_VAL)

    print(f"Tokenization completed: {len(arr):,} tokens saved on disk.")
    return torch.from_numpy(train_arr.astype(np.int64)), torch.from_numpy(val_arr.astype(np.int64))


train_data, val_data = build_tokens(TARGET_TOKENS)

print(f"FineWeb-Edu (GPT-2 BPE): Vocab {VOCAB:,} | Train {len(train_data):,} | Val {len(val_data):,}")
print(f"  {MAX_ITERS} x {MICRO_BATCH*GRAD_ACCUM} x {BLOCK} = {MAX_ITERS*MICRO_BATCH*GRAD_ACCUM*BLOCK/1e6:.0f}M tokens")

def get_batch(split, block_size=None, generator=None):

    d = train_data if split == "train" else val_data

    if isinstance(block_size, int) and block_size > 10:
        blk = block_size
        g = generator if isinstance(generator, torch.Generator) else None

    elif isinstance(block_size, torch.Generator):
        g = block_size
        blk = BLOCK
    else:
        blk = block_size if block_size is not None else BLOCK
        g = generator if isinstance(generator, torch.Generator) else None

    ix = torch.randint(len(d) - blk - 1, (MICRO_BATCH,), generator=g)
    x = torch.stack([d[i:i + blk] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + blk] for i in ix])
    return x.to(device), y.to(device)

# =====================================================================
#                    FS NEURONS + SPIKE ACCOUNTING
# =====================================================================

_SPIKE_ON = False          # counting is off during training


def set_spike_counting(on):
    global _SPIKE_ON
    _SPIKE_ON = on

def reset_spike_stats(model):
    for m in model.modules():
        if hasattr(m, "spike_sum"):
            m.spike_sum, m.spike_n = 0.0, 0


def _record(module, spike_count):
    if not _SPIKE_ON:
        return
    sc = spike_count.detach()
    module.spike_sum += float(sc.sum().item())
    module.spike_n += int(sc.numel())

def fs_window(K, s):

    ''' Input range the neuron resolves: [0, window). Ceiling is 2s '''

    return s * (2 - 2.0 ** -(K - 1))


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
        self.alpha = nn.Parameter(0.1 + 0.9 * torch.rand(n_channels))                                        # alpha is alearnable parameter

    def forward(self, x):
        return self.alpha * (self.fs_on(x) - self.fs_off(-x))


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

    ''' Normalises over channels only '''

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):

    """Standard softmax attention with a causal mask."""

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

    ''' Softmax-free causal attention with FS-coded Q, K, V '''

    def __init__(self, d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma, w_min=W_MIN, w_max=W_MAX):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.signed = signed
        self.learnable = learnable

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)
        self.norm_v = RMSNorm(d_model)
        self.resid_drop = nn.Dropout(dropout)

        def enc():
            if signed:
                return SignedFSNeuron(K, width, qk_scale, readout_scale, learnable, True, d_model)
            return make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

        self.fs_q = enc()
        self.fs_k = enc()
        self.fs_v = make_fs(K, width, qk_scale, readout_scale, learnable, True, d_model)

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

            assert len(self.gammas) == n_heads, f"Needed {n_heads} gamma, we have {len(self.gammas)}"
            assert all(0.0 < g <= 1.0 for g in self.gammas), "gamma out the interval (0, 1]"

            g = torch.tensor(self.gammas, dtype=torch.float32)[:, None, None]
            idx = torch.arange(block, dtype=torch.float32)

            dist = (idx[:, None] - idx[None, :]).clamp(min=0)          # dist[i,j] = i-j
            D = torch.pow(g, dist[None]) * self.mask                   # (H, T, T)
            self.register_buffer("decay", D)
            self.register_buffer("decay_sum", D.sum(-1, keepdim=True))

    def decay_stats(self):

        if not self.use_decay:
            return None
        return {"gamma": [round(g, 5) for g in self.gammas], "window": [round(gamma_window(g), 1) for g in self.gammas], "learned": False}

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = self.fs_q(self.norm_q(q))
        k = self.fs_k(self.norm_k(k))
        v = self.fs_v(self.norm_v(v))

        h = lambda t: t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = h(q), h(k), h(v)

        att = (q @ k.transpose(-2, -1)) * self.attn_scale

        if self.use_decay:
            att = att * self.decay[None, :, :T, :T]          # gamma_h^(i-j), causale
            att = att / self.decay_sum[None, :, :T]          # media pesata per riga
        else:
            att = att * self.mask[:T, :T]                     # causale
            att = att / self.n_keys[:T][None, None, :, None]  # media piatta per riga

        return self.resid_drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attention, activation, K, width, qk_scale, mlp_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma):
        super().__init__()
        self.norm1 = RMSNorm(d_model)

        if attention == "softmax":
            self.attn = CausalSelfAttention(d_model, n_heads, dropout, block)
        elif attention == "ssa":
            self.attn = CausalSpikingSelfAttention(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma)
        else:
            raise ValueError(attention)
        self.norm2 = RMSNorm(d_model)
        hidden = d_model * MLP_RATIO

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            make_activation(activation, K, width, mlp_scale, readout_scale, learnable, True, hidden),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FSGPT(nn.Module):
    def __init__(self, attention, activation, K, signed, learnable, qk_scale, mlp_scale, block, use_decay, gamma):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(block, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_HEADS, attention, activation, K, WIDTH, qk_scale, mlp_scale, READOUT_SCALE, signed, learnable, DROPOUT, block, use_decay, gamma)
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

    def spike_stats(self):

        ''' Spikes per encoding site, per token, per channel '''

        leaves = lambda m: [x for x in m.modules()
                            if isinstance(x, (FSNeuron, LearnableFSNeuron))]

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
            attn += [getattr(blk.attn, nm) for nm in ("fs_q", "fs_k", "fs_v")
                     if getattr(blk.attn, nm, None) is not None]
            if leaves(blk.mlp[1]):
                mlp.append(blk.mlp[1])
        return {"attention": rate(attn), "mlp": rate(mlp)}

    def decay_stats(self):

        '''Learned gamma per head, one entry per layer. Empty if the decay is off.'''

        return [blk.attn.decay_stats() for blk in self.blocks
                if getattr(blk.attn, "use_decay", False)]


# =====================================================================
#                 THRESHOLD CALIBRATION AND CHECKS
# =====================================================================

@torch.no_grad()
def measure_scales():

    ''' Probe the spread the FS neurons actually see, at initialisation '''

    torch.manual_seed(0)
    m = FSGPT("ssa", "fs", 2, False, False, 0.25, 1.0, BLOCK, False, None).to(device)
    qkv, pre = [], []

    hq = m.blocks[0].attn.norm_q.register_forward_hook(lambda mo, i, o: qkv.append(o.std().item()))
    hm = m.blocks[0].mlp[0].register_forward_hook(lambda mo, i, o: pre.append(o.std().item()))

    m.eval()
    for _ in range(4):
        m(get_batch("train")[0])
    hq.remove();
    hm.remove()
    del m
    return float(sum(qkv) / len(qkv)), float(sum(pre) / len(pre))


SIGMA_QK, SIGMA_MLP = measure_scales()
QK_SCALE = QK_SIGMA_MULT * SIGMA_QK
MLP_SCALE = MLP_SIGMA_MULT * SIGMA_MLP

print(f"\nmeasured std  Q/K/V after RMSNorm {SIGMA_QK:.4f} | MLP pre-activation {SIGMA_MLP:.4f}")
print(f"derived  qk_scale {QK_SCALE:.4f} ({QK_SIGMA_MULT:g} sigma) | mlp_scale {MLP_SCALE:.4f} ({MLP_SIGMA_MULT:g} sigma)")


def sanity_checks():

    print("\n--- FS sanity checks ---")
    x = torch.randn(4000, 8)

    for k in (1, 2, 3):

        a = make_fs(k, WIDTH, QK_SCALE, READOUT_SCALE, False, True, 8)
        b = make_fs(k, WIDTH, QK_SCALE, READOUT_SCALE, True, True, 8)
        err = (a(x) - b(x)).abs().max().item()

        assert err < 1e-5, f"K={k}: learnable does not start from fixed ({err:.2e})"

    print(f"  learnable == fixed at init : OK (max err {err:.2e})")

    hi = fs_window(2, QK_SCALE)
    z = hi * torch.rand(4000, 8)
    u = make_fs(2, WIDTH, QK_SCALE, QK_SCALE, False, True, 8)
    c = make_fs(2, WIDTH, QK_SCALE, READOUT_SCALE, False, True, 8)

    assert (c(z) - (READOUT_SCALE / QK_SCALE) * u(z)).abs().max() < 1e-5, "r is not a pure gain"
    print(f"  r is a pure gain           : OK ({READOUT_SCALE / QK_SCALE:.2f}x)")

    n_pl = sum(isinstance(m, SignedFSNeuron)
               for m in FSGPT("ssa", "fs", 2, False, False, QK_SCALE, MLP_SCALE, BLOCK, False, None).modules())

    n_sg = sum(isinstance(m, SignedFSNeuron)
               for m in FSGPT("ssa", "fs", 2, True, False, QK_SCALE, MLP_SCALE, BLOCK, False, None).modules())

    assert n_pl == 0 and n_sg > 0, "`signed` is not reaching the attention"
    print(f"  signed reaches attention   : OK ({n_sg} ON/OFF pairs)")

    # ----------------------------------------------------------------------
    #                          Causality check
    # ----------------------------------------------------------------------

    torch.manual_seed(0)
    m = FSGPT("ssa", "fs", 2, False, False, QK_SCALE, MLP_SCALE, BLOCK,  False, None).to(device).eval()

    with torch.no_grad():
        a = torch.randint(0, VOCAB, (2, 64), device=device)
        b = a.clone(); b[:, 32:] = torch.randint(0, VOCAB, (2, 32), device=device)
        d = (m(a)[0][:, :32] - m(b)[0][:, :32]).abs().max().item()

    assert d < 1e-4, f"causality violated: {d:.2e}"
    print(f"  causal mask                : OK (max leak {d:.2e})")

    x = torch.randn(4000, 8)
    print(f"  window at s={QK_SCALE:.3f} (K=2)   : [0, {hi:.3f}) saturated {(x > hi).float().mean().item()*100:.1f}%")
    print("--- end checks ---\n")

sanity_checks()


# =====================================================================
#                 TRAINING METADATA VERIFICATION
# =====================================================================

CURRENT_SETUP = {
    "tag": TAG,
    "vocab": VOCAB,
    "d_model": D_MODEL,
    "n_layer": N_LAYER,
    "n_heads": N_HEADS,
    "block": BLOCK,
    "eff_batch": MICRO_BATCH * GRAD_ACCUM,
    "lr": LR,
    "min_lr": MIN_LR,
    "warmup": WARMUP,
    "max_iters": MAX_ITERS
}

def verify_and_update_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            saved_meta = json.load(f)

        arch_keys = ["vocab", "d_model", "n_layer", "n_heads", "block"]
        for k in arch_keys:
            if saved_meta.get(k) != CURRENT_SETUP[k]:
                raise SystemExit(
                    f"Architecture mismatch for parameter '{k}'.\n"
                    f"  On disk: {saved_meta.get(k)} | Current: {CURRENT_SETUP[k]}\n"
                    f"Change TAG or remove {META_PATH} to start a new architecture."
                )

        prev_iters = saved_meta.get("max_iters", 0)
        if MAX_ITERS > prev_iters:
            print(f"Extending budget: {prev_iters} -> {MAX_ITERS} iterations.")
        elif MAX_ITERS < prev_iters:
            print(f"Warning: MAX_ITERS ({MAX_ITERS}) is lower than previous target ({prev_iters}).")

    atomic_json(CURRENT_SETUP, META_PATH)
    print(f"Training metadata saved to: {META_PATH}")

verify_and_update_metadata()

# =====================================================================
#                          SCHEDULER & LOSS
# =====================================================================

def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = min(1.0, max(0.0, (it - WARMUP) / max(1, MAX_ITERS - WARMUP)))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * r))

@torch.no_grad()
def estimate_loss(model, block):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split, block)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =====================================================================
#                     TRAINING SINGLE RUN
# =====================================================================

def train_single_run(name, attention, activation, K, signed, learnable, block, use_decay, gamma, seed):

    torch.manual_seed(seed)
    model = FSGPT(attention, activation, K, signed, learnable, QK_SCALE, MLP_SCALE, block, use_decay, gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.15, betas=(0.9, 0.999))

    hist = []
    best = {"val": float("inf"), "iter": -1, "train": float("nan")}
    start_it = 0
    t0 = time.time()

    pt_file = ckpt_path_of(name, seed)
    json_hist_file = hist_path_of(name, seed)

    # -------------------------------------------
    #   Reload from .pt checkpoint (if available)
    # -------------------------------------------
    if os.path.exists(pt_file):
        ck = torch.load(pt_file, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])

        start_it = ck["iter"] + 1
        best = ck["best"]
        hist = ck.get("history", [])

        # -------------------------------------------------------------------
        # Safe RNG restoration casting explicitly to torch.ByteTensor (uint8)
        # -------------------------------------------------------------------

        if "cpu_rng" in ck and ck["cpu_rng"] is not None:
            try:
                cpu_rng = ck["cpu_rng"]
                if isinstance(cpu_rng, torch.Tensor):
                    cpu_rng = cpu_rng.to(dtype=torch.uint8, device="cpu")
                else:
                    cpu_rng = torch.tensor(cpu_rng, dtype=torch.uint8, device="cpu")
                torch.set_rng_state(cpu_rng)
            except Exception as err:
                print(f"Warning: Failed to restore CPU RNG state ({err}). Skipping.")

        if device == "cuda" and ck.get("cuda_rng") is not None:
            try:
                cuda_rng = ck["cuda_rng"]
                if isinstance(cuda_rng, torch.Tensor):
                    cuda_rng = cuda_rng.to(dtype=torch.uint8, device="cpu")
                else:
                    cuda_rng = torch.tensor(cuda_rng, dtype=torch.uint8, device="cpu")
                torch.cuda.set_rng_state(cuda_rng)

            except Exception as err:
                print(f"Warning: Failed to restore CUDA RNG state ({err}). Skipping.")

        print(f"   Checkpoint loaded: resuming from step {start_it}/{MAX_ITERS} (Best Val: {best['val']:.4f})")

    model.train()

    for it in range(start_it, MAX_ITERS + 1):

        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:

            L = estimate_loss(model, block)
            ppl_val = math.exp(L["val"])
            hist.append({"iter": it, "train": L["train"], "val": L["val"], "ppl": ppl_val})

            if L["val"] < best["val"]:
                best = {"val": L["val"], "train": L["train"], "iter": it, "ppl": ppl_val}

                # --------------------
                #  Weights checkpoint
                # --------------------

                atomic_save({
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": opt.state_dict(),
                  "iter": it,
                  "best": best,
                  "history": hist,
                  "cpu_rng": torch.get_rng_state().to(dtype=torch.uint8),
                  "cuda_rng": (torch.cuda.get_rng_state().to(dtype=torch.uint8) if device == "cuda" else None)
                }, pt_file)

            mem = f" | {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if device == "cuda" else ""
            print(f"   it {it:>5}/{MAX_ITERS} | train {L['train']:.4f} | val {L['val']:.4f} (ppl {ppl_val:.2f}) | best {best['val']:.4f} @ {best['iter']}{mem}", flush=True)

            # ------------------------------------------
            # Numerical JSON history per configuration
            # ------------------------------------------

            atomic_json({
                "name": name,
                "seed": seed,
                "max_iters": MAX_ITERS,
                "best_val": best["val"],
                "best_iter": best["iter"],
                "best_ppl": best["ppl"],
                "last_iter": it,
                "history": hist
            }, json_hist_file)

        if it == MAX_ITERS:
            break

        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = get_batch("train", block)
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"   Configuration completed at step {MAX_ITERS}.\n")
    return best

# =====================================================================
#                     MAIN RESUMABLE SWEEP LOOP
# =====================================================================

print("\n" + "=" * 78)
print(f"STARTING SWEEP | Target: {MAX_ITERS} iterations | Output: {OUT}")
print("=" * 78)

for name, attn, act, K, sg, ln, blk, use_dec, gmm in CONFIGS:
    for seed in SEEDS:
        key = f"{name}_s{seed}"
        json_file = hist_path_of(name, seed)

        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    past_data = json.load(f)
                last_it = past_data.get("last_iter", 0)

                if last_it >= MAX_ITERS:
                    print(f"Skip {key}: already completed at {last_it}/{MAX_ITERS} steps.")
                    continue
                else:
                    print(f"Resume {key}: found at step {last_it}. Resuming to {MAX_ITERS}...")
            except Exception:
                print(f"Warning: {key} history file corrupted, restarting from scratch...")
        else:
            print(f"Start {key}: starting from step 0...")

        train_single_run(name, attn, act, K, sg, ln, blk, use_dec, gmm, seed)

print("\nProcess finished. Checkpoints and histories updated.\n")
