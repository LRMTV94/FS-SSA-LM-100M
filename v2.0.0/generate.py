# =======================================================================
#                              GENERATION  TEST
# =======================================================================
#
#  Same prompts, same seed, both arms held in memory at once so the two
#  continuations of a prompt print one under the other. That is the whole
#  point: read them in a column, not all of one model and then all of the
#  other. 
#
#  NOTE: If it runs out of memory, drop one entry from CONFIGS and run
#  it twice.
#
#  Self-contained: it defines the model as it was trained, the tokenizer
#  and the sampler, so it depends on nothing in the session.
#
#  Weights from Hugging Face (SOURCE = "hf": the published spiking model,
#  the only one there) or from Drive (SOURCE = "drive": every entry of
#  CONFIGS, from OUT; add the dense control there to compare the arms).
# =======================================================================

import glob
import math
import os
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE = "hf"                                    # "hf": the published weights, "drive": the checkpoints of CONFIGS in OUT
HF_REPO = "Matt-94/FS-SSA-LM-100M"               # Hugging Face repository of the published model.pt
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

W_MIN = 8.0                           # shortest decay window 1/(1 - gamma), first head
W_MAX = BLOCK                         # longest decay window, last head: the whole context


# =====================================================================
#                       HELPERS AND TOKENIZER
# =====================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

VOCAB = tokenizer.n_vocab
encode = tokenizer.encode_ordinary
decode = lambda ids: tokenizer.decode([int(i) for i in ids])
_record = lambda module, spike_count: None

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

    '''Normalises over channels only'''

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


# ====================================================================
#                            CONFIGURATIONS
# ====================================================================
#
#  (checkpoint filename on Drive, label, attention, activation, K, signed, 
#  learnable, block, use_decay, gamma). With SOURCE = "hf" only the
#  spiking entry is loaded, from the published weights.

CONFIGS = [

    ("ckpt_fineweb_100m_ssa_K=2_p-_L_g_var_alpha_app_s1.pt", "ssa K=2 +/- L g var alpha app", "ssa", "fs", 2, True, True, BLOCK, True, gamma_ladder(N_HEADS, W_MIN, W_MAX)),
]

# ====================================================================
#                             PROMPT SETS
# ====================================================================
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
#  That is scale, not architecture: GPT-2 small saw 40 GB and still 
#  cannot add two digits reliably, and FineWeb-Edu is filtered
#  educational prose with very little code in it. They stay because the
#  FAILURE MODE is the informative part. Prose is "forgiving", you can 
#  lose the thread and still sound fine. Code and lists are not: an 
#  unclosed bracket from forty tokens back is a visible attention failure, 
#  and attention is the thing under test.
#
#  The induction set is the one that should actually work. Induction
#  heads emerge early in training and are purely attentional, so this is
#  the sharpest probe in the file. If the spiking arm cannot continue a
#  pattern it has just been shown twice, that is a concrete, nameable
#  result.
# =======================================================================

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


# ====================================================================
#                               SAMPLER
# ====================================================================

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

def load_one(ckpt, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm):

    ''' One arm, from Hugging Face or from Drive, read on the CPU: the
    checkpoint also holds the optimizer state, which is not needed here '''

    if SOURCE == "hf":
        if attn != "ssa":
            print(f"\n### {label}: not on Hugging Face, only the spiking model is published")
            return None
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename="model.pt")
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
    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm)

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
