# --------------------------------------------------------------------
#                     GENERATION: DENSE vs SPIKING
# --------------------------------------------------------------------
#
#  Same prompts, same seed, both arms held in memory at once so the two
#  continuations of a prompt print one under the other. That is the whole
#  point: read them in a column, not all of one model and then all of the
#  other. If it runs out of memory, drop one entry from CONFIGS and run
#  it twice.
#
#  Needs from earlier cells, and nothing else:
#     FSGPT, gamma_ladder, N_HEADS, W_MIN, W_MAX, QK_SCALE, MLP_SCALE,
#     BLOCK, device
#
#  The sampler and the tokenizer are resolved in here. An earlier version
#  of this cell relied on generate() and encode() living somewhere else,
#  and neither of them existed in the session.
# --------------------------------------------------------------------

import glob
import os
import torch

OUT = "/content/drive/MyDrive/fsssa"
SEED = 1234

# --------------------------------------------------------------------
#                            CONFIGURATIONS
# --------------------------------------------------------------------

CONFIGS = [

    ("ckpt_fineweb_100m_ssa_K=2_p-_L_gamma_variable_alpha_app_s1.pt",
     "ssa K=2 +/- L g var alpha app", "ssa", "fs", 2, True, True, BLOCK, True,
     gamma_ladder(N_HEADS, W_MIN, W_MAX)),
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

def logits_of(model, idx):

    ''' Your forward returns logits, or (logits, loss) with targets. '''

    o = model(idx)
    return o[0] if isinstance(o, (tuple, list)) else o


@torch.no_grad()
def sample(model, idx, max_new, temperature, top_k, blk):

    ''' nanoGPT semantics: crop the context to the block size, take the
    last position, scale by temperature, truncate to top_k, draw.
    top_k=1 is greedy, verified identical to argmax step by step. '''

    for _ in range(max_new):
        c = idx if idx.shape[1] <= blk else idx[:, -blk:]
        lg = logits_of(model, c)[:, -1, :] / max(temperature, 1e-8)
        if top_k:
            v, _ = torch.topk(lg, min(top_k, lg.shape[-1]))
            lg = lg.masked_fill(lg < v[:, [-1]], -float("inf"))
        idx = torch.cat([idx, torch.multinomial(lg.softmax(-1), 1)], dim=1)
    return idx


# --------------------------------------------------------------------
#                              TOKENIZER
# --------------------------------------------------------------------

def get_codec(vocab):

    ''' In order of preference. On FineWeb-Edu the tokenization happened
    in data prep, so there is usually no encode/decode in scope and the
    last branch is the one that fires. '''

    g = globals()
    if callable(g.get("encode")) and callable(g.get("decode")):
        return g["encode"], g["decode"], "encode/decode from your script"
    for nm in ("enc", "tokenizer", "tok", "gpt2"):
        o = g.get(nm)
        if o is not None and hasattr(o, "encode") and hasattr(o, "decode"):
            return o.encode, o.decode, f"{nm} = {type(o).__name__} in scope"
    if isinstance(g.get("stoi"), dict) and g.get("itos") is not None:
        si, it = g["stoi"], g["itos"]
        return (lambda s: [si[c] for c in s if c in si],
                lambda l: "".join(it[i] for i in l), "char-level stoi/itos")
    import tiktoken
    t = tiktoken.get_encoding("gpt2")
    # The head may be padded past the real vocabulary (50304 vs 50257).
    # Sampling can then land on an id the tokenizer does not know, which
    # would crash decode at the END of a 250-token generation. Drop them.
    return (t.encode,
            lambda l: t.decode([i for i in l if i < t.n_vocab]),
            f"tiktoken gpt2 (model vocab {vocab}, tokenizer {t.n_vocab})")


# --------------------------------------------------------------------
#                                 LOAD
# --------------------------------------------------------------------

def load_one(ckpt, label, attn, act, K, sg, ln, blk_cfg, use_dec, gmm):
    path = f"{OUT}/{ckpt}"
    if not os.path.exists(path):
        print(f"\n### {label}: not found -> {ckpt}")
        print("    checkpoints actually in the folder:")
        
        for f in sorted(glob.glob(f"{OUT}/ckpt_*.pt")):
            print(f"      {os.path.basename(f)}")
        return None

    ck = torch.load(path, map_location=device)
    sd = ck.get("model", ck.get("model_state_dict", ck))
    blk = sd["pos.weight"].shape[0] if "pos.weight" in sd else blk_cfg

    model = FSGPT(attn, act, K, sg, ln, QK_SCALE, MLP_SCALE, blk, use_dec, gmm)

    own = set(model.state_dict())
    
    sd_clean = {k: v for k, v in sd.items()
                if k in own or k.rsplit(".", 1)[-1] not in
                ("decay", "decay_sum", "mask", "n_keys", "dist")}
                
    model.load_state_dict(sd_clean)
    model = model.to(device).eval()

    print("\n" + "#" * 70)
    print(f"#  {label}")
    print(f"#  {ckpt}")
    print(f"#  context {blk}   {sum(p.numel() for p in model.parameters()):,} params")


    GK = ("raw_gamma", "alpha", "raw_alpha", "log_gamma", "gamma")
    ck_g = sorted(k for k in sd if k.rsplit(".", 1)[-1] in GK)
    md_g = sorted(n for n, _ in model.named_parameters()
                  if n.rsplit(".", 1)[-1] in GK)
                  
    print(f"#  decay params in checkpoint: {len(ck_g)}"
          + (f"  e.g. {ck_g[0]}" if ck_g
             else "  -> NONE, the gammas are the fixed ladder"))
             
    print(f"#  decay params in model:      {len(md_g)}"
          + (f"  e.g. {md_g[0]}" if md_g
             else "  -> the class in scope is the FIXED-gamma one"))
             
    if hasattr(model.blocks[0].attn, "decay_stats"):
        print(f"#  decay_stats(): {model.blocks[0].attn.decay_stats()}")
        
    print("#" * 70)
    return {"label": label, "model": model, "block": blk}


# --------------------------------------------------------------------
#                                 RUN
# --------------------------------------------------------------------

arms = [r for r in (load_one(*c) for c in CONFIGS) if r is not None]
assert arms, "no checkpoint loaded"

encode, decode, codec_src = get_codec(arms[0]["model"].tok.weight.shape[0])
print(f"\ntokenizer: {codec_src}")

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
