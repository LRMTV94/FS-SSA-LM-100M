# =====================================================================
#            DIAGNOSTIC, v2.2.0: THE SCHEDULED SURROGATE
# =====================================================================
#
#  Compares the v2.2.0 run with the K=3 run of v2.1.0. The two have the
#  same model, the same initial weights and the same batches, and differ
#  only in the surrogate gradient, so their difference at equal steps is
#  the effect of the surrogate. The dense control and the v1 hybrid are
#  the references, read from ../v2.0.0/results/history; the dense run
#  stopped at step 7500. Each run is found by the name stored in its
#  history, whatever the name of the file.
#
#  Prints the table of the four runs against the dense control and the
#  paired table of the two K=3 runs, and writes next to itself:
#    figures/validation.png  the validation loss of the four runs
#    figures/zoom.png        the last six evaluations of the runs that
#                            reach step 10000, where the points sit close
#    figures/difference.png  v2.2 minus v2.1 in perplexity at equal
#                            steps, above the surrogate half-width it
#                            was trained with
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
V20_HISTORY = os.path.join(HERE, "..", "v2.0.0", "results", "history")
V21_HISTORY = os.path.join(HERE, "..", "v2.1.0", "results", "history")
V22_HISTORY = os.path.join(HERE, "results", "history")
FIGURES = os.path.join(HERE, "figures")

# (key, folder, names the history may carry, label)
RUNS = [
    ("dense", V20_HISTORY, ("softmax + gelu",), "dense, softmax + GELU"),
    ("hybrid", V20_HISTORY, ("ssa K=2 +/- L + alpha_app + gamma_var", "ssa K=2 +/- L g"), "v2.0 hybrid, highway reads in floats, K=2"),
    ("v21", V21_HISTORY, ("ssa K=3 +/- L g var alpha app data decay",), "v2.1, K=3, fixed surrogate"),
    ("v22", V22_HISTORY, ("ssa K=3 +/- L g var alpha app data decay width sched",), "v2.2, K=3, scheduled surrogate"),
]
REQUIRED = ("v21", "v22")
LABELS = {key: label for key, _, _, label in RUNS}
SHORT = {"hybrid": "v2.0 hybrid", "v21": "v2.1, fixed surrogate", "v22": "v2.2, scheduled surrogate"}   # legend of the zoom

# the schedule of v2.2.0/model.py: the surrogate half-width follows the cosine of the learning rate
WARMUP, MAX_ITERS, LR, MIN_LR = 500, 10000, 1.0e-3, 1.0e-4
WIDTH, WIDTH_END = 1.5, 1.0

STEPS = (1000, 2000, 3000, 5000, 7500, 10000)
WINDOWS = ((1500, 3000), (4000, 6000), (8000, 10000))    # the windows of the v2.1 README
TAIL_FROM = 7500                                         # start of the gain to the best
TAIL_POINTS = 6                                          # evaluations in the zoom
FIRST_STEP = 1000                                        # first evaluation drawn

# surfaces and text of the figures
THEME = {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#c3c2b7", "axis": "#c3c2b7", "frame": "#000000", "schedule": "#2a78d6"}

# (color, marker) of each run, with the markers of v2.1.0/diagnostic.py: the K=3 runs keep its categorical slots,
# aqua for v2.1 and yellow for v2.2; the dense control and the hybrid are references, in the greys of the text
SERIES = {"dense": ("#52514e", "s"), "hybrid": ("#898781", "o"), "v21": ("#1baf7a", "D"), "v22": ("#eda100", "P")}


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


def find(folder, names):
    ''' Validation loss per evaluated step, best loss and best step of the history in folder whose name is
    one of names, or None '''
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("name") in names:
            return {e["iter"]: e["val"] for e in d["history"]}, d["best_val"], d["best_iter"]
    return None


def mean(values):
    return sum(values) / len(values)


# =====================================================================
#                                TABLES
# =====================================================================

def report(runs):

    # ----------------------------------------------------------------
    #            The four runs against the dense control
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

    # -----------------------------------------------------------------------------------
    #            The two K=3 runs, paired: same weights, same batches, another surrogate
    # -----------------------------------------------------------------------------------
    
    (a, best_a, it_a), (b, best_b, it_b) = runs["v22"], runs["v21"]
    common = sorted(set(a) & set(b))

    print("\n" + "=" * 92)
    print(" v2.2 against v2.1, K=3: same model, weights and batches, another surrogate")
    print("=" * 92)
    print(f" {'step':>6} | {'half-width':>10} | {'v2.1 val':>9} | {'v2.2 val':>9} | {'v2.2 - v2.1':>11} | {'in ppl':>7}")
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
        print(f"   v2.1: {b[TAIL_FROM] - best_b:.4f} nats   v2.2: {a[TAIL_FROM] - best_a:.4f} nats")

    print("\n best of the run")
    print(f"   v2.1: {best_b:.4f} at {it_b} (perplexity {math.exp(best_b):.2f})")
    print(f"   v2.2: {best_a:.4f} at {it_a} (perplexity {math.exp(best_a):.2f})")
    print(f"   difference {best_a - best_b:+.4f} nats, {math.exp(best_a) - math.exp(best_b):+.2f} perplexity")


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


def points(ax, xs, ys, key, size, label=None):
    color, marker = SERIES[key]
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
    #          The four runs, one point per evaluation
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
    steps = sorted(runs["v22"][0])[-TAIL_POINTS:]
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
    #         Top: the paired difference in perplexity
    # ----------------------------------------------------------------
    
    (a, _, _), (b, _, _) = runs["v22"], runs["v21"]
    common = [it for it in sorted(set(a) & set(b)) if it >= FIRST_STEP]
    style(ax_diff, frame=True)
    ax_diff.axhline(0.0, color=THEME["grid"], linewidth=0.7, linestyle="--")
    points(ax_diff, common, [math.exp(a[it]) - math.exp(b[it]) for it in common], "v22", 7)
    ax_diff.set_title("Perplexity of v2.2 minus v2.1 at equal steps: below zero, v2.2 is ahead", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax_diff.tick_params(labelbottom=False)

    # ----------------------------------------------------------------
    #    Bottom: the half-width the difference was trained with
    # ----------------------------------------------------------------
    
    xs = list(range(FIRST_STEP, MAX_ITERS + 1, 50))
    style(ax_width, frame=True)
    ax_width.plot(xs, [WIDTH * width_scale_at(it) for it in xs], color=THEME["schedule"], linewidth=2)
    ax_width.set_ylim(0.9, 1.6)
    ax_width.set_yticks([WIDTH_END, WIDTH])
    ax_width.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f} s"))
    ax_width.set_title("Surrogate half-width of v2.2, following the cosine of the learning rate, in units of the top threshold s", loc="left", color=THEME["ink2"], fontsize=9, pad=6)
    ax_width.set_xlim(FIRST_STEP - 150, MAX_ITERS + 150)
    ax_width.set_xticks(range(FIRST_STEP, MAX_ITERS + 1, 1000))
    ax_width.set_xlabel("Training step", color=THEME["ink2"])

    fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.11)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def main():

    # ----------------------------------------------------------------
    #                   Find the four histories
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

    os.makedirs(FIGURES, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        draw(os.path.join(FIGURES, "validation.png"), runs)
        zoom(os.path.join(FIGURES, "zoom.png"), runs)
        difference(os.path.join(FIGURES, "difference.png"), runs)
    print(f"\nfigures written to {os.path.normpath(FIGURES)}")


if __name__ == "__main__":
    main()
