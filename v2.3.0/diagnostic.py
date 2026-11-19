# =====================================================================
#            DIAGNOSTIC, v2.3.0: THE SPIKING OUTPUT HEAD
# =====================================================================
#
#  Compares the v2.3.0 run with the v2.2.0 run. The two have the same
#  model up to the output head, the same initial weights outside the
#  head code and the same batches, and differ only in the head, which
#  reads spikes in v2.3, so their difference at equal steps is the cost
#  of the spiking head. The dense control, the v1 hybrid and the K=3 run
#  of v2.1.0 are the references; the dense run stopped at step 7500.
#  Every run before v2.3 is read from ../v2.2.0/results/history, the
#  v2.3 run from results/history, its seeds 0 and 2 from
#  results/stability_seed and the continuations of the fluency test from
#  results/fluency. Each run is found by the name and the seed stored in
#  its history, whatever the name of the file.
#
#  Prints the table of the five runs against the dense control, the
#  paired table of v2.3 and v2.2, the three seeds of v2.3 and the
#  fluency test with the loops set apart, and writes next to itself:
#    figures/validation.png  the validation loss of the five runs
#    figures/zoom.png        the last six evaluations of the runs that
#                            reach step 10000, where the points sit close
#    figures/difference.png  v2.3 minus v2.2 in perplexity at equal
#                            steps, above the surrogate half-width both
#                            were trained with
#    figures/seeds.png       v2.3 minus v2.2 to step 2500, for the three
#                            seeds of v2.3 against seed 1 of v2.2
#    figures/fluency.png     ref loss against rep-4, for every
#                            continuation of the fluency test
#  Every evaluation is a point; only the half-width, a schedule and not
#  a measure, is a line.
# =====================================================================

import glob
import json
import math
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
V22_HISTORY = os.path.join(HERE, "..", "v2.2.0", "results", "history")
V23_HISTORY = os.path.join(HERE, "results", "history")
V23_SEEDS = os.path.join(HERE, "results", "stability_seed")
FLUENCY = os.path.join(HERE, "results", "fluency", "fluency_samples.jsonl")
FIGURES = os.path.join(HERE, "figures")

# (key, folder, names the history may carry, label)
RUNS = [
    ("dense", V22_HISTORY, ("softmax + gelu",), "dense, softmax + GELU"),
    ("hybrid", V22_HISTORY, ("ssa K=2 +/- L + alpha_app + gamma_var", "ssa K=2 +/- L g"), "v2.0 hybrid, highway reads in floats, K=2"),
    ("v21", V22_HISTORY, ("ssa K=3 +/- L g var alpha app data decay",), "v2.1, K=3, fixed surrogate"),
    ("v22", V22_HISTORY, ("ssa K=3 +/- L g var alpha app data decay width sched",), "v2.2, K=3, scheduled surrogate"),
    ("v23", V23_HISTORY, ("ssa K=3 +/- L g var alpha app data decay width sched spike head",), "v2.3, K=3, spiking output head"),
]
REQUIRED = ("v22", "v23")
LABELS = {key: label for key, _, _, label in RUNS}
SHORT = {"hybrid": "v2.0 hybrid", "v21": "v2.1, fixed surrogate", "v22": "v2.2, scheduled surrogate", "v23": "v2.3, spiking head"}   # legend of the zoom

# the schedule of v2.2.0/model.py and v2.3.0/model.py: the surrogate half-width follows the cosine of the learning rate
WARMUP, MAX_ITERS, LR, MIN_LR = 500, 10000, 1.0e-3, 1.0e-4
WIDTH, WIDTH_END = 1.5, 1.0

STEPS = (1000, 2000, 3000, 5000, 7500, 10000)
WINDOWS = ((1500, 3000), (4000, 6000), (8000, 10000))    # the windows of the v2.1 README
TAIL_FROM = 7500                                         # start of the gain to the best
TAIL_POINTS = 6                                          # evaluations in the zoom
FIRST_STEP = 1000                                        # first evaluation drawn

SEEDS = (0, 1, 2)                                        # seed 1 is the run of results/history
SEED_LAST = 2500                                         # seeds 0 and 2 stop here, on the 10k schedule
SEED_WINDOW = (1500, 2500)
SEED_DODGE = 45                                          # steps between the points of the seeds at one evaluation
LOOP_REP4 = 0.3                                          # as in fluency.py: rep-4 above this counts as a loop

# surfaces and text of the figures
THEME = {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#c3c2b7", "axis": "#c3c2b7", "frame": "#000000", "schedule": "#2a78d6"}

# (color, marker) of each run, with the markers of v2.1.0/diagnostic.py: the K=3 runs keep its categorical slots,
# aqua for v2.1 and yellow for v2.2, and v2.3 takes the next one, magenta; the dense control, the hybrid and the
# human text of the fluency test are references, in the greys of the text
SERIES = {"dense": ("#52514e", "s"), "hybrid": ("#898781", "o"), "v21": ("#1baf7a", "D"), "v22": ("#eda100", "P"), "v23": ("#e87ba4", "^"), "human": ("#52514e", "*")}

# (color, marker) of the three seeds of v2.3: seed 1, the full run, keeps the magenta and the marker of v2.3;
# seeds 0 and 2 take the next categorical slots, green and violet
SEED_SERIES = {0: ("#008300", "v"), 1: ("#e87ba4", "^"), 2: ("#6250d6", ">")}

# two-sided 95% quantiles of Student's t for 1 to 30 degrees of freedom, as in fluency.py
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = min(1.0, max(0.0, (it - WARMUP) / max(1, MAX_ITERS - WARMUP)))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * r))


def width_scale_at(it):
    ''' As in v2.2.0/model.py: 1 through the warmup, then down to WIDTH_END / WIDTH along the cosine of the
    learning rate, so the half-width goes from WIDTH * s to WIDTH_END * s '''
    if it < WARMUP:
        return 1.0
    end = WIDTH_END / WIDTH
    return end + (1.0 - end) * (lr_at(it) - MIN_LR) / (LR - MIN_LR)


def find(folder, names, seed=1):
    ''' Validation loss per evaluated step, best loss and best step of the history in folder whose name is
    one of names and whose seed is seed, or None '''
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("name") in names and d.get("seed", 1) == seed:
            return {e["iter"]: e["val"] for e in d["history"]}, d["best_val"], d["best_iter"]
    return None


def mean(values):
    return sum(values) / len(values)


def sd(values):
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5


def t95(df):
    return T95[min(max(df, 1), len(T95)) - 1]


def interval(values):
    ''' Mean and half-width of the 95% interval of Student's t '''
    return mean(values), t95(len(values) - 1) * sd(values) / len(values) ** 0.5


# =====================================================================
#                                TABLES
# =====================================================================

def report(runs):

    # ----------------------------------------------------------------
    #            The five runs against the dense control
    # ----------------------------------------------------------------

    ref = runs.get("dense")
    print("\n" + "=" * 92)
    print(" best of each run" + (", against the dense control" if ref else ""))
    print("=" * 92)
    for key, (vals, best, it) in runs.items():
        row = f" {LABELS[key]:<44} {best:.4f} at {it:>5}   ppl {math.exp(best):6.2f}"
        if ref and key != "dense":
            row += f"   {best - ref[1]:+.4f} nats   {math.exp(best) - math.exp(ref[1]):+.2f} ppl"
        print(row)

    # ----------------------------------------------------------------
    #              v2.3 and v2.2, paired: same weights
    #              outside the head code, same batches
    # ----------------------------------------------------------------

    (a, best_a, it_a), (b, best_b, it_b) = runs["v23"], runs["v22"]
    common = sorted(set(a) & set(b))

    print("\n" + "=" * 92)
    print(" v2.3 against v2.2: same model up to the head, same weights and batches, the head reads spikes")
    print("=" * 92)
    print(f" {'step':>6} | {'half-width':>10} | {'v2.2 val':>9} | {'v2.3 val':>9} | {'v2.3 - v2.2':>11} | {'in ppl':>7}")
    for it in STEPS:
        if it in a and it in b:
            print(f" {it:>6} | {WIDTH * width_scale_at(it):>8.2f} s | {b[it]:9.4f} | {a[it]:9.4f} | {a[it] - b[it]:+11.4f} | {math.exp(a[it]) - math.exp(b[it]):+7.2f}")

    print("\n mean difference at equal steps")
    for lo, hi in WINDOWS:
        sel = [it for it in common if lo <= it <= hi]
        if sel:
            print(f"   steps {lo:>5} to {hi:>5}: {mean([a[it] - b[it] for it in sel]):+.4f} nats over {len(sel)} evaluations")

    if TAIL_FROM in a and TAIL_FROM in b:
        print(f"\n gain from step {TAIL_FROM} to the best of the run")
        print(f"   v2.2: {b[TAIL_FROM] - best_b:.4f} nats   v2.3: {a[TAIL_FROM] - best_a:.4f} nats")

    print("\n best of the run")
    print(f"   v2.2: {best_b:.4f} at {it_b} (perplexity {math.exp(best_b):.2f})")
    print(f"   v2.3: {best_a:.4f} at {it_a} (perplexity {math.exp(best_a):.2f})")
    print(f"   difference {best_a - best_b:+.4f} nats, {math.exp(best_a) - math.exp(best_b):+.2f} perplexity")

    # ----------------------------------------------------------------
    #       v2.3 and v2.1: the spiking head and the scheduled
    #       surrogate together, against the K=3 model of v2.1
    # ----------------------------------------------------------------

    if "v21" in runs:
        c, best_c, it_c = runs["v21"]
        common_c = sorted(set(a) & set(c))
        print("\n v2.3 against v2.1, K=3: mean difference at equal steps")
        for lo, hi in WINDOWS:
            sel = [it for it in common_c if lo <= it <= hi]
            if sel:
                print(f"   steps {lo:>5} to {hi:>5}: {mean([a[it] - c[it] for it in sel]):+.4f} nats over {len(sel)} evaluations")
        print(f"   best of the run: {best_a - best_c:+.4f} nats, {math.exp(best_a) - math.exp(best_c):+.2f} perplexity")


def report_seeds(seeds, v22):

    # ----------------------------------------------------------------
    #     The three seeds of v2.3 at equal steps, against seed 1
    #     of v2.2, up to the last step all three have evaluated
    # ----------------------------------------------------------------

    vals = {s: h[0] for s, h in seeds.items()}
    steps = [it for it in sorted(vals[1]) if FIRST_STEP <= it <= SEED_LAST and all(it in v for v in vals.values()) and it in v22]
    order = sorted(vals)

    print("\n" + "=" * 92)
    print(f" v2.3, seeds {', '.join(map(str, order))}, on the 10k schedule up to step {steps[-1]}, against seed 1 of v2.2")
    print("=" * 92)
    print(f" {'step':>6} | " + " | ".join(f"{'seed ' + str(s):>7}" for s in order) + f" | {'spread':>7} | {'v2.2':>7} | v2.3 - v2.2")
    for it in steps:
        row = [vals[s][it] for s in order]
        print(f" {it:>6} | " + " | ".join(f"{v:7.4f}" for v in row) + f" | {max(row) - min(row):7.4f} | {v22[it]:7.4f} | "
              + "  ".join(f"{v - v22[it]:+.4f}" for v in row))

    lo, hi = SEED_WINDOW
    sel = [it for it in steps if lo <= it <= hi]
    means = [mean([vals[s][it] for it in sel]) for s in order]
    ref = mean([v22[it] for it in sel])
    print(f"\n mean of steps {lo} to {hi}, {len(sel)} evaluations")
    print("   " + "   ".join(f"seed {s}: {m:.4f} ({m - ref:+.4f})" for s, m in zip(order, means)) + f"   v2.2: {ref:.4f}")
    print(f"   standard deviation across the seeds {sd(means):.4f} nats")


def report_fluency(samples):

    # ----------------------------------------------------------------
    #      The fluency test of results/fluency: every pair, the
    #     pairs without a loop, and the ref loss at equal rep-4,
    #      since the loops lower the ref loss of a continuation
    # ----------------------------------------------------------------

    arms = list(dict.fromkeys(s["arm"] for s in samples if s["arm"] != "human"))
    base, other = arms[0], arms[1]
    by = {(s["arm"], s["prompt"], s["seed"]): s for s in samples}
    prompts = sorted({s["prompt"] for s in samples if s["arm"] == base})
    seeds = sorted({s["seed"] for s in samples if s["arm"] == base})

    def per_prompt(key, keep):
        out = []
        for p in prompts:
            pairs = [(by[(base, p, j)], by[(other, p, j)]) for j in seeds if (other, p, j) in by]
            pairs = [(x, y) for x, y in pairs if keep(x, y)]
            if pairs:
                out.append(mean([x[key] - y[key] for x, y in pairs]))
        return out

    every = lambda x, y: True
    no_loop = lambda x, y: x["rep4"] <= LOOP_REP4 and y["rep4"] <= LOOP_REP4
    n_kept = sum(no_loop(by[(base, p, j)], by[(other, p, j)]) for p in prompts for j in seeds if (other, p, j) in by)

    print("\n" + "=" * 92)
    print(f" fluency test: {base} minus {other}, {len(prompts)} prompts, {len(seeds)} continuations each")
    print("=" * 92)
    for title, keep, note in (("every pair", every, ""), ("pairs without a loop", no_loop, f"   ({n_kept} of {len(prompts) * len(seeds)} pairs)")):
        cells = []
        for key in ("rep4", "ref_loss", "pmi"):
            m, ci = interval(per_prompt(key, keep))
            cells.append(f"{key} {m:+.3f} ± {ci:.3f}")
        print(f"   {title:<21} " + "   ".join(cells) + note)

    # the per-prompt difference of ref loss against that of rep-4: the intercept is the difference at equal rep-4
    x = per_prompt("rep4", every)
    y = per_prompt("ref_loss", every)
    mx, my = mean(x), mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    slope = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx
    cut = my - slope * mx
    s2 = sum((v - cut - slope * u) ** 2 for u, v in zip(x, y)) / (len(x) - 2)
    half = t95(len(x) - 2) * (s2 * (1.0 / len(x) + mx ** 2 / sxx)) ** 0.5
    print(f"   ref loss at equal rep-4: {cut:+.3f} ± {half:.3f}   (slope {slope:+.2f} nats per unit of rep-4)")

    rows = [s for s in samples if s["arm"] != "human"]
    r4 = [s["rep4"] for s in rows]
    rl = [s["ref_loss"] for s in rows]
    m4, ml = mean(r4), mean(rl)
    corr = sum((u - m4) * (v - ml) for u, v in zip(r4, rl)) / (sum((u - m4) ** 2 for u in r4) * sum((v - ml) ** 2 for v in rl)) ** 0.5
    print(f"   rep-4 and ref loss over the {len(rows)} continuations of the models: correlation {corr:+.2f}")


# =====================================================================
#                               FIGURES
# =====================================================================

def style(ax, frame=False):
    ax.set_facecolor(THEME["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(frame or side in ("left", "bottom"))
        ax.spines[side].set_color(THEME["frame"] if frame else THEME["axis"])
        ax.spines[side].set_linewidth(1)

    ax.tick_params(colors=THEME["muted"], labelcolor=THEME["muted"], length=0, pad=6)
    ax.grid(True, axis="both", color=THEME["grid"], linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}K" if v else "0"))


def points(ax, xs, ys, key, size, label=None, series=None):
    color, marker = series or SERIES[key]
    ax.plot(xs, ys, linestyle="None", marker=marker, markersize=size, color=color, markeredgecolor=THEME["surface"], markeredgewidth=0.8, label=label)


def legend(ax, loc):
    ''' Legend in a black box with square corners '''
    leg = ax.legend(loc=loc, frameon=True, fancybox=False, framealpha=1.0, facecolor=THEME["surface"], edgecolor=THEME["frame"], fontsize=9, handletextpad=0.3, markerscale=1.2)
    leg.get_frame().set_linewidth(1)
    for text in leg.get_texts():
        text.set_color(THEME["ink2"])


def draw(out, runs):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])

    # ----------------------------------------------------------------
    #            The five runs, one point per evaluation
    # ----------------------------------------------------------------
    style(ax, frame=True)
    lo, hi = float("inf"), 0.0
    for key, (vals, best, it) in runs.items():
        xs = [i for i in sorted(vals) if i >= FIRST_STEP]
        ys = [vals[i] for i in xs]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        points(ax, xs, ys, key, 6, f"{LABELS[key]}, best {math.exp(best):.2f}")
    ax.set_xlim(FIRST_STEP - 150, MAX_ITERS + 150)
    ax.set_xticks(range(FIRST_STEP, MAX_ITERS + 1, 1000))
    ax.set_ylim(math.floor(lo * 20) / 20, math.ceil(hi * 20) / 20)
    ax.set_title("Validation loss, steps 1,000 to 10,000", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax.set_ylabel("Validation loss (nats)", color=THEME["ink2"])
    ax.set_xlabel("Training step", color=THEME["ink2"])
    legend(ax, "upper right")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.11)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def zoom(out, runs):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])

    # ----------------------------------------------------------------
    #    The last evaluations of the runs that reach the end; the
    #  dense run stopped at step 7,500 and stays in the full figure
    # ----------------------------------------------------------------
    style(ax, frame=True)
    steps = sorted(runs["v23"][0])[-TAIL_POINTS:]
    lo, hi = float("inf"), 0.0

    for key, (vals, best, it) in runs.items():
        xs = [i for i in sorted(vals) if steps[0] <= i <= steps[-1]]
        if key == "dense" or not xs:
            continue
        ys = [vals[i] for i in xs]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        points(ax, xs, ys, key, 9, f"{SHORT[key]}, best {math.exp(best):.2f}")
    pad = 0.15 * (hi - lo)

    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(steps[0] - 125, steps[-1] + 125)
    ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}K"))
    ax.set_title(f"Validation loss, the last {TAIL_POINTS} evaluations, steps {steps[0]:,} to {steps[-1]:,}", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax.set_ylabel("Validation loss (nats)", color=THEME["ink2"])
    ax.set_xlabel("Training step", color=THEME["ink2"])

    legend(ax, "best")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def difference(out, runs):
    fig = plt.figure(figsize=(8.5, 5.0), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.38)
    ax_diff = fig.add_subplot(grid[0])
    ax_width = fig.add_subplot(grid[1], sharex=ax_diff)

    # ----------------------------------------------------------------
    #            Top: the paired difference in perplexity
    # ----------------------------------------------------------------

    (a, _, _), (b, _, _) = runs["v23"], runs["v22"]
    common = [it for it in sorted(set(a) & set(b)) if it >= FIRST_STEP]
    style(ax_diff, frame=True)
    ax_diff.axhline(0.0, color=THEME["grid"], linewidth=0.7, linestyle="--")
    points(ax_diff, common, [math.exp(a[it]) - math.exp(b[it]) for it in common], "v23", 7)
    ax_diff.set_title("Perplexity of v2.3 minus v2.2 at equal steps: above zero, v2.3 is behind", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax_diff.tick_params(labelbottom=False)

    # ----------------------------------------------------------------
    #  Bottom: the half-width both were trained with, the same one
    # ----------------------------------------------------------------

    xs = list(range(FIRST_STEP, MAX_ITERS + 1, 50))
    style(ax_width, frame=True)
    ax_width.plot(xs, [WIDTH * width_scale_at(it) for it in xs], color=THEME["schedule"], linewidth=2)
    ax_width.set_ylim(0.9, 1.6)
    ax_width.set_yticks([WIDTH_END, WIDTH])
    ax_width.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f} s"))
    ax_width.set_title("Surrogate half-width of v2.2 and v2.3, following the cosine of the learning rate, in units of the top threshold s", loc="left", color=THEME["ink2"], fontsize=9, pad=6)
    ax_width.set_xlim(FIRST_STEP - 150, MAX_ITERS + 150)
    ax_width.set_xticks(range(FIRST_STEP, MAX_ITERS + 1, 1000))
    ax_width.set_xlabel("Training step", color=THEME["ink2"])

    fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.11)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def seeds_figure(out, seeds, v22):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])

    # ----------------------------------------------------------------
    #     Each seed of v2.3 minus seed 1 of v2.2 at equal steps,
    #     in nats; the points of one evaluation are set apart by
    #       a few steps, so that they do not cover each other
    # ----------------------------------------------------------------
    style(ax, frame=True)
    ax.axhline(0.0, color=THEME["grid"], linewidth=0.7, linestyle="--")
    order = sorted(seeds)
    steps = [it for it in sorted(seeds[1][0]) if FIRST_STEP <= it <= SEED_LAST and all(it in seeds[s][0] for s in order) and it in v22]

    for n, s in enumerate(order):
        shift = (n - (len(order) - 1) / 2) * SEED_DODGE
        label = f"v2.3, seed {s}, " + ("the full run" if s == 1 else f"stopped at step {SEED_LAST:,}")
        points(ax, [it + shift for it in steps], [seeds[s][0][it] - v22[it] for it in steps], "v23", 8, label, SEED_SERIES.get(s))
    ax.set_xlim(steps[0] - 150, steps[-1] + 150)
    ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}K"))
    ax.set_title("Validation loss of v2.3 minus v2.2 at equal steps, three seeds of v2.3 against seed 1 of v2.2", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax.set_ylabel("Difference (nats)", color=THEME["ink2"])
    ax.set_xlabel("Training step", color=THEME["ink2"])

    legend(ax, "upper right")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def fluency_figure(out, samples):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])

    # ----------------------------------------------------------------
    #       Every continuation of the fluency test: its loops
    #       against its loss under GPT-2 large, the models and
    #       the human continuations of the validation prompts
    # ----------------------------------------------------------------
    style(ax, frame=True)
    keys = {"v2.2": "v22", "v2.3": "v23"}
    arms = list(dict.fromkeys(s["arm"] for s in samples))

    for arm in arms:
        key = "human" if arm == "human" else keys.get(arm.split(",")[0], "v23")
        rows = [s for s in samples if s["arm"] == arm]
        points(ax, [s["rep4"] for s in rows], [s["ref_loss"] for s in rows], key, 8 if key == "human" else 6, f"{arm}, {len(rows)} continuations")
    ax.axvline(LOOP_REP4, color=THEME["muted"], linewidth=1.0, zorder=1, label=f"loops: rep-4 above {LOOP_REP4}")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}"))
    ax.set_title("Ref loss against rep-4, every continuation of the fluency test: the loops lower the loss", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax.set_ylabel("Ref loss under GPT-2 large (nats per token)", color=THEME["ink2"])
    ax.set_xlabel("Rep-4, share of repeated 4-grams in the continuation", color=THEME["ink2"])

    legend(ax, "upper right")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def main():

    # ----------------------------------------------------------------
    #        Find the five histories, the seeds, the samples
    # ----------------------------------------------------------------

    runs = {}
    for key, folder, names, label in RUNS:
        found = find(folder, names)
        if found:
            runs[key] = found
        elif key in REQUIRED:
            raise SystemExit(f"history of '{names[0]}' not found in {os.path.normpath(folder)}")
        else:
            print(f"history of '{names[0]}' not found in {os.path.normpath(folder)}: the table and the figure go on without it")
    report(runs)

    v23_names = next(names for key, _, names, _ in RUNS if key == "v23")
    seeds = {1: runs["v23"]}
    for s in SEEDS:
        if s != 1:
            found = find(V23_SEEDS, v23_names, s)
            if found:
                seeds[s] = found
            else:
                print(f"\nseed {s} of v2.3 not found in {os.path.normpath(V23_SEEDS)}: the seeds go on without it")
    if len(seeds) > 1:
        report_seeds(seeds, runs["v22"][0])

    samples = []
    if os.path.exists(FLUENCY):
        with open(FLUENCY) as f:
            samples = [json.loads(line) for line in f]
        report_fluency(samples)
    else:
        print(f"\nfluency samples not found at {os.path.normpath(FLUENCY)}: the fluency table and figure are skipped")

    os.makedirs(FIGURES, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        draw(os.path.join(FIGURES, "validation.png"), runs)
        zoom(os.path.join(FIGURES, "zoom.png"), runs)
        difference(os.path.join(FIGURES, "difference.png"), runs)
        if len(seeds) > 1:
            seeds_figure(os.path.join(FIGURES, "seeds.png"), seeds, runs["v22"][0])
        if samples:
            fluency_figure(os.path.join(FIGURES, "fluency.png"), samples)
    print(f"\nfigures written to {os.path.normpath(FIGURES)}")


if __name__ == "__main__":
    main()
