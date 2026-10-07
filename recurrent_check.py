# ==============================================================================
#                               RECURRENT CHECK
# ==============================================================================
#
#  Self-contained: it defines the model as it was trained (SignedFSNeuron
#  keeps its gain as alpha1, the name in the checkpoints), the helpers
#  the classes need and the validation data, so it depends on nothing in
#  the session and runs in a fresh runtime too. Validation tokens come
#  from the cache model.py writes on Drive (searched under DATA); without
#  it, from the first documents of FineWeb-Edu.
#
#  Weights from Hugging Face (SOURCE = "hf", the published checkpoint) or
#  from Drive (SOURCE = "drive", CONFIG's file in OUT). The checkpoint is
#  loaded twice: once as it was trained (quadratic attention) and once
#  with every attention block rebuilt in recurrent form. Same weights,
#  nothing retrained. Per head, with decay gamma, the quadratic forward
#  computes
#
#      out_t = scale / Z_t * sum_{j<=t} gamma^(t-j) (q_t . k_j) v_j
#
#  and the recurrent one carries a state of fixed size instead:
#
#      S_t = gamma * S_{t-1} + k_t^T v_t        (d_head x d_head per head)
#      Z_t = gamma * Z_{t-1} + 1                (= decay_sum)
#      out_t = scale * (q_t S_t) / Z_t
#
#  Four tests: same logits, same validation loss, same generated text
#  at the same seed with the generation time of both, and the logits
#  again in float64.
#
# ==============================================================================

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


SOURCE = "drive"                                 # "hf": the published weights, "drive": the checkpoints of CONFIGS in OUT
HF_REPO = "Matt-94/FS-SSA-LM-100M"               # Hugging Face repository of the published model.pt
DATA = "/content/drive/MyDrive/fsssa"            # Drive folder of the results, for SOURCE = "drive"
OUT = "/content/drive/MyDrive/fsssa"             # Drive folder of the checkpoints, for SOURCE = "drive"
SEED = 1234                                      # sampling seed, the same for every arm and prompt


# -------------------------------
#           MODEL'S PARAMETERS
# -------------------------------

BLOCK      = 1024                     # Context in tokens: the learned positions stop here
D_MODEL    = 576                      # Width of the residual stream
N_HEADS    = 9                        # Attention heads, d_head = 576 / 9 = 64
N_LAYER    = 16                       # Transformer blocks
MLP_RATIO  = 4                        # MLP hidden width = 4 x D_MODEL
DROPOUT    = 0.                       # no dropout, as in training

WIDTH          = 1.1                  # Surrogate half-width
READOUT_SCALE  = 1.0                  # r; neuron gain is r/s, spike count depends on s only
QK_SIGMA_MULT   = 0.75                # qk_scale  = this x measured std of the RMSNormed Q/K/V
MLP_SIGMA_MULT  = 1.0                 # mlp_scale = this x measured std of the MLP pre-activation
QK_SCALE  = QK_SIGMA_MULT * 1.000     # x the std model.py measures at init; they only set the
MLP_SCALE = MLP_SIGMA_MULT * 0.271    # initial thresholds, which the checkpoint overwrites
MICRO_BATCH = 16                      # sequences per validation batch, as in model.py

W_MIN = 8.0                           # shortest decay window 1/(1 - gamma), first head
W_MAX = BLOCK                         # longest decay window, last head: the whole context

GEN_TOKENS      = 300
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 200


GAMMA_MIN, GAMMA_MAX = 0.50, 0.9999  # Gamma values (for gamma ladder - checkpoint overwrites)
N_VAL = 10                           # validation batches for the loss comparison
NEW_TOKENS = 256                     # generated tokens for the timing test


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

class SpikingRMSNorm(nn.Module):

    '''Normalises over channels only'''

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


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

        self.norm_q = SpikingRMSNorm(d_model)
        self.norm_k = SpikingRMSNorm(d_model)
        self.norm_v = SpikingRMSNorm(d_model)
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

            dist = (idx[:, None] - idx[None, :]).clamp(min=0)
            D = torch.pow(g, dist[None]) * self.mask
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
            att = att * self.decay[None, :, :T, :T]
            att = att / self.decay_sum[None, :, :T]
        else:
            att = att * self.mask[:T, :T]
            att = att / self.n_keys[:T][None, None, :, None]

        return self.resid_drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attention, activation, K, width, qk_scale, mlp_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma):
        super().__init__()

        self.norm1 = SpikingRMSNorm(d_model)
        if attention == "softmax":
            self.attn = CausalSelfAttention(d_model, n_heads, dropout, block)
        elif attention == "ssa":
            self.attn = CausalSpikingSelfAttention(d_model, n_heads, K, width, qk_scale, readout_scale, signed, learnable, dropout, block, use_decay, gamma)
        else:
            raise ValueError(attention)
        self.norm2 = SpikingRMSNorm(d_model)
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

        self.norm = SpikingRMSNorm(D_MODEL)
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


#  (checkpoint filename, then your tuple:
#   label, attn, act, K, signed, learnable, block, use_decay, gamma)
CONFIG = ("ckpt_fineweb_100m_ssa_K=2_p-_L_g_var_alpha_app_s1.pt", "ssa K=2 +/- L g var alpha app", "ssa", "fs", 2, True, True, BLOCK, True, gamma_ladder(N_HEADS, W_MIN, W_MAX))


# --------------------------------------------------------------------
#                       THE RECURRENT ATTENTION
# --------------------------------------------------------------------

class RecurrentSpikingSelfAttention(CausalSpikingSelfAttention):

    ''' Same parameters and buffers as CausalSpikingSelfAttention, so the
    checkpoint loads unchanged. Only the time mixing is rewritten. '''

    def gamma_per_head(self):

        ''' gamma per head from the decay buffer actually loaded, 1 if no decay.
        Not called gammas(): that name is the list the parent class stores. '''

        if self.use_decay:
            return self.decay[:, 1, 0]
        return torch.ones(self.n_heads, device=self.mask.device, dtype=self.mask.dtype)

    def encode(self, x):

        ''' Position-wise part, identical to the quadratic forward '''

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.fs_q(self.norm_q(q)), self.fs_k(self.norm_k(k)), self.fs_v(self.norm_v(v))

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head
        q, k, v = (t.reshape(B, T, H, Dh) for t in self.encode(x))
        g = self.gamma_per_head().to(x.dtype)

        S = x.new_zeros(B, H, Dh, Dh)
        Z = x.new_zeros(B, H, 1)
        out = []
        for t in range(T):
            S = g[None, :, None, None] * S + k[:, t, :, :, None] * v[:, t, :, None, :]
            Z = g[None, :, None] * Z + 1.0
            out.append((q[:, t, :, None, :] @ S).squeeze(-2) * self.attn_scale / Z)

        o = torch.stack(out, dim=1).reshape(B, T, C)
        return self.resid_drop(self.proj(o))

    def step(self, x_t, state):

        ''' One token, x_t is (B, C). The state is (S, Z), None at the start. '''

        B, C = x_t.shape
        H, Dh = self.n_heads, self.d_head
        q, k, v = (t.reshape(B, H, Dh) for t in self.encode(x_t))
        g = self.gamma_per_head().to(x_t.dtype)

        if state is None:
            S = x_t.new_zeros(B, H, Dh, Dh)
            Z = x_t.new_zeros(B, H, 1)
        else:
            S, Z = state

        S = g[None, :, None, None] * S + k[..., :, None] * v[..., None, :]
        Z = g[None, :, None] * Z + 1.0
        o = (q[..., None, :] @ S).squeeze(-2) * self.attn_scale / Z
        return self.resid_drop(self.proj(o.reshape(B, C))), (S, Z)


def to_recurrent(model):

    ''' A copy of the model with every spiking attention block in recurrent form '''

    rec = copy.deepcopy(model)
    for blk in rec.blocks:
        assert isinstance(blk.attn, CausalSpikingSelfAttention), "only the SSA model has a recurrent form"
        blk.attn.__class__ = RecurrentSpikingSelfAttention
    return rec.eval()


def fsgpt_step(model, idx_t, t, states):

    ''' One token through the whole recurrent model. idx_t is (B,), t the position. '''

    x = model.drop(model.tok(idx_t) + model.pos.weight[t])
    new_states = []
    for blk, st in zip(model.blocks, states):
        a, st = blk.attn.step(blk.norm1(x), st)
        x = x + a
        x = x + blk.mlp(blk.norm2(x))
        new_states.append(st)
    return model.head(model.norm(x)), new_states


@torch.no_grad()
def generate_recurrent(model, idx, max_new_tokens, temperature=1.0, top_k=None):

    ''' Same sampling as FSGPT.generate, one step per token. Limited to
    model.block tokens in total: the position embedding stops there. '''

    B, T0 = idx.shape
    assert T0 + max_new_tokens <= model.block, "the learned positions stop at model.block"
    states = [None] * len(model.blocks)
    for t in range(T0):
        logits, states = fsgpt_step(model, idx[:, t], t, states)

    for i in range(max_new_tokens):
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        if i < max_new_tokens - 1:
            logits, states = fsgpt_step(model, nxt[:, 0], T0 + i, states)
    return idx


# --------------------------------------------------------------------
#                                 LOAD
# --------------------------------------------------------------------

def load(ckpt, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm):

    ''' Weights from Hugging Face or from Drive, read on the CPU: the
    checkpoint also holds the optimizer state, which is not needed here '''

    if SOURCE == "hf":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename="model.pt")
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
    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm)

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
    orders of magnitude. The decay is rebuilt in float64 from the same
    gammas, so both forms use identical weights: what is left is the method. '''

    m64 = copy.deepcopy(model).double()
    for blk in m64.blocks:
        at = blk.attn
        if at.use_decay:
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
    b = generate_recurrent(rec, idx, new_tokens, temperature=0.8, top_k=200)
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
