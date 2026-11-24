# =====================================================================
#         DIAGNOSTIC, v2.4.0: THE DENSE CONTROL ON 10,000 STEPS
# =====================================================================
#
#  The diagnostic of v2.3.0, with the dense control of v2.4.0 as the
#  reference. The control is trained on the recipe of v2.2 and v2.3: the
#  same 10,000 steps, learning rate and data, so their difference at
#  equal steps is the cost of the spikes under the same schedule. The
#  dense control of v1, on a schedule of 7,500 steps, stays as the
#  former reference. Every run before v2.3 is read from
#  ../v2.2.0/results/history, the v2.3 run from ../v2.3.0/results/history
#  and the dense control from results/history. Each run is found by the
#  name and the seed stored in its history, whatever the name of the
#  file.
#
#  Prints the table of the runs against the dense control, the paired
#  table of v2.2 and v2.3 against it, and the two dense controls at
#  equal steps, and writes next to itself:
#    figures/validation.png  the validation loss of every run, the two
#                            dense controls included
#    figures/difference.png  v2.2 and v2.3 minus the dense control at
#                            equal steps, above the learning rate of the
#                            three runs and that of the control of v1
#  Every evaluation is a point; only the learning rate, a schedule and
#  not a measure, is a line.
# =====================================================================

import glob
import json
import math
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
V22_HISTORY = os.path.join(HERE, "..", "v2.2.0", "results", "history")
V23_HISTORY = os.path.join(HERE, "..", "v2.3.0", "results", "history")
V24_HISTORY = os.path.join(HERE, "results", "history")
FIGURES = os.path.join(HERE, "figures")

# (key, folder, names the history may carry, label)
RUNS = [
    ("dense", V24_HISTORY, ("softmax + gelu 10k",), "dense, softmax + GELU, 10,000 steps"),
    ("dense_v1", V22_HISTORY, ("softmax + gelu",), "dense of v1, softmax + GELU, 7,500 steps"),
    ("hybrid", V22_HISTORY, ("ssa K=2 +/- L + alpha_app + gamma_var", "ssa K=2 +/- L g"), "v2.0 hybrid, highway reads in floats, K=2"),
    ("v21", V22_HISTORY, ("ssa K=3 +/- L g var alpha app data decay",), "v2.1, K=3, fixed surrogate"),
    ("v22", V22_HISTORY, ("ssa K=3 +/- L g var alpha app data decay width sched",), "v2.2, K=3, scheduled surrogate"),
    ("v23", V23_HISTORY, ("ssa K=3 +/- L g var alpha app data decay width sched spike head",), "v2.3, K=3, spiking output head"),
]
REQUIRED = ("dense", "v22", "v23")
LABELS = {key: label for key, _, _, label in RUNS}
SHORT = {"v22": "v2.2, scheduled surrogate", "v23": "v2.3, spiking output head"}   # legend of the difference

# the schedule of v2.2.0/model.py, v2.3.0/model.py and v2.4.0/model.py, and the shorter one of the control of v1
WARMUP, MAX_ITERS, LR, MIN_LR = 500, 10000, 1.0e-3, 1.0e-4
MAX_ITERS_V1 = 7500

STEPS = (1000, 2000, 3000, 5000, 7500, 9500, 10000)
WINDOWS = ((1500, 3000), (4000, 6000), (8000, 10000))    # the windows of the v2.1 README
TAIL_FROM = 7500                                         # start of the gain to the best
FIRST_STEP = 1000                                        # first evaluation drawn
DODGE = 35                                               # steps between the points of v2.2 and v2.3 at one evaluation

# surfaces and text of the figures
THEME = {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#c3c2b7", "axis": "#c3c2b7", "frame": "#000000", "schedule": "#2a78d6"}

# (color, marker, filled) of each run, those of v2.3.0/diagnostic.py: the K=3 runs keep their categorical slots,
# aqua for v2.1, yellow for v2.2 and magenta for v2.3; the dense controls and the hybrid are references, in the
# greys of the text, and the control of v1, the former reference, is drawn hollow
SERIES = {"dense": ("#52514e", "s", True), "dense_v1": ("#52514e", "s", False), "hybrid": ("#898781", "o", True),
          "v21": ("#1baf7a", "D", True), "v22": ("#eda100", "P", True), "v23": ("#e87ba4", "^", True)}


def lr_at(it, max_iters=MAX_ITERS):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = min(1.0, max(0.0, (it - WARMUP) / max(1, max_iters - WARMUP)))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * r))


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


# =====================================================================
#                                TABLES
# =====================================================================

def report(runs):

    # ----------------------------------------------------------------
    #          Every run against the dense control of v2.4
    # ----------------------------------------------------------------

    ref = runs["dense"]
    print("\n" + "=" * 92)
    print(" best of each run, against the dense control on 10,000 steps")
    print("=" * 92)
    for key, (vals, best, it) in runs.items():
        row = f" {LABELS[key]:<44} {best:.4f} at {it:>5}   ppl {math.exp(best):6.2f}"
        if key != "dense":
            row += f"   {best - ref[1]:+.4f} nats   {math.exp(best) - math.exp(ref[1]):+.2f} ppl"
        print(row)

    # ----------------------------------------------------------------
    #       v2.2 and v2.3 against the dense control at equal
    #       steps: the same steps, learning rate and data
    # ----------------------------------------------------------------

    d, best_d, it_d = ref
    print("\n" + "=" * 92)
    print(" v2.2 and v2.3 against the dense control: the same 10,000 steps, learning rate and data")
    print("=" * 92)
    print(f" {'step':>6} | {'learning rate':>13} | {'dense':>7} | {'v2.2':>7} | {'v2.3':>7} | {'v2.2 - dense':>12} | {'v2.3 - dense':>12}")
    for it in STEPS:
        if it in d and it in runs["v22"][0] and it in runs["v23"][0]:
            b = runs["v22"][0][it]
            a = runs["v23"][0][it]
            print(f" {it:>6} | {lr_at(it):13.2e} | {d[it]:7.4f} | {b:7.4f} | {a:7.4f} | {b - d[it]:+12.4f} | {a - d[it]:+12.4f}")

    print("\n mean difference from the dense control at equal steps")
    for key in ("hybrid", "v21", "v22", "v23"):
        if key not in runs:
            continue
        vals = runs[key][0]
        common = sorted(set(vals) & set(d))
        cells = []
        for lo, hi in WINDOWS:
            sel = [it for it in common if lo <= it <= hi]
            if sel:
                cells.append(f"{mean([vals[it] - d[it] for it in sel]):+.4f} over {lo:>5} to {hi:>5}")
        print(f"   {LABELS[key]:<44} " + "   ".join(cells))

    if TAIL_FROM in d:
        print(f"\n gain from step {TAIL_FROM} to the best of the run")
        print("   " + "   ".join(f"{key}: {runs[key][0][TAIL_FROM] - runs[key][1]:.4f} nats" for key in ("dense", "v22", "v23") if TAIL_FROM in runs[key][0]))

    # ----------------------------------------------------------------
    #   The two dense controls at equal steps: the same model on a
    #     schedule of 7,500 steps and on one of 10,000 steps
    # ----------------------------------------------------------------

    if "dense_v1" in runs:
        old, best_old, it_old = runs["dense_v1"]
        print("\n" + "=" * 92)
        print(" the dense control of v1, on 7,500 steps, against the one of v2.4, on 10,000")
        print("=" * 92)
        print(f" {'step':>6} | {'lr, 7,500':>10} | {'lr, 10,000':>10} | {'v1':>7} | {'v2.4':>7} | v1 - v2.4")
        for it in (2500, 5000, 6250, 7000, 7500):
            if it in old and it in d:
                print(f" {it:>6} | {lr_at(it, MAX_ITERS_V1):10.2e} | {lr_at(it):10.2e} | {old[it]:7.4f} | {d[it]:7.4f} | {old[it] - d[it]:+.4f}")
        print(f"\n the control of v1 ends at step {it_old}, {best_old:.4f}; the one of v2.4 has its best at step {it_d}, {best_d:.4f}, {best_d - best_old:+.4f} nats")


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
    color, marker, filled = SERIES[key]
    face = color if filled else THEME["surface"]
    edge = THEME["surface"] if filled else color
    ax.plot(xs, ys, linestyle="None", marker=marker, markersize=size, color=color, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.8 if filled else 1.3, label=label)


def legend(ax, loc):
    ''' Legend in a black box with square corners '''
    leg = ax.legend(loc=loc, frameon=True, fancybox=False, framealpha=1.0, facecolor=THEME["surface"], edgecolor=THEME["frame"], fontsize=9, handletextpad=0.3, markerscale=1.2)
    leg.get_frame().set_linewidth(1)
    for text in leg.get_texts():
        text.set_color(THEME["ink2"])


def draw(out, runs):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])

    style(ax, frame=True)
    lo, hi = float("inf"), 0.0
    
    for key, (vals, best, it) in runs.items():
        xs = [i for i in sorted(vals) if i >= FIRST_STEP]
        ys = [vals[i] for i in xs]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        points(ax, xs, ys, key, 6, f"{LABELS[key]}, best {math.exp(best):.2f}")
        
    ax.set_xlim(FIRST_STEP - 150, MAX_ITERS + 150)
    ax.set_xticks(range(FIRST_STEP, MAX_ITERS + 1, 1000))
    ax.set_ylim(math.floor((lo - 0.01) * 20) / 20, math.ceil(hi * 20) / 20)      # the lowest point clear of the frame
    ax.set_title("Validation loss, steps 1,000 to 10,000", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax.set_ylabel("Validation loss (nats)", color=THEME["ink2"])
    ax.set_xlabel("Training step", color=THEME["ink2"])
    
    legend(ax, "upper right")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.11)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def difference(out, runs):

    fig = plt.figure(figsize=(8.5, 5.4), dpi=160)
    fig.patch.set_facecolor(THEME["surface"])
    
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.42)
    ax_diff = fig.add_subplot(grid[0])
    ax_lr = fig.add_subplot(grid[1], sharex=ax_diff)

    # ----------------------------------------------------------------
    #     Top: v2.2 and v2.3 minus the dense control, in nats; the
    #     two points of one evaluation are set apart by a few steps
    # ----------------------------------------------------------------

    d = runs["dense"][0]
    style(ax_diff, frame=True)
    ax_diff.axhline(0.0, color=THEME["grid"], linewidth=0.7, linestyle="--")
    hi = 0.0
    for n, key in enumerate(("v22", "v23")):
        vals = runs[key][0]
        common = [it for it in sorted(set(vals) & set(d)) if it >= FIRST_STEP]
        ys = [vals[it] - d[it] for it in common]
        hi = max(hi, max(ys))
        shift = (n - 0.5) * DODGE
        points(ax_diff, [it + shift for it in common], ys, key, 7, SHORT[key])
    ax_diff.set_ylim(0.0, math.ceil(hi * 20) / 20)
    ax_diff.set_title("Validation loss minus the dense control at equal steps: above zero, the spiking model is behind", loc="left", color=THEME["ink"], fontsize=11, pad=10)
    ax_diff.set_ylabel("Difference (nats)", color=THEME["ink2"])
    ax_diff.tick_params(labelbottom=False)
    legend(ax_diff, "upper right")

    # ----------------------------------------------------------------
    #    Bottom: the learning rate of the three runs, the same one,
    #    which the surrogate half-width of v2.2 and v2.3 follows, and
    #     the shorter schedule of the control of v1, which ends at
    #          its minimum at step 7,500
    # ----------------------------------------------------------------

    xs = list(range(FIRST_STEP, MAX_ITERS + 1, 50))
    xs_v1 = [it for it in xs if it <= MAX_ITERS_V1]
    style(ax_lr, frame=True)
    ax_lr.plot(xs, [lr_at(it) for it in xs], color=THEME["schedule"], linewidth=2, label="10,000 steps: the dense control, v2.2 and v2.3")
    ax_lr.plot(xs_v1, [lr_at(it, MAX_ITERS_V1) for it in xs_v1], color=THEME["muted"], linewidth=2, linestyle="--", label="7,500 steps: the dense control of v1")
    ax_lr.set_ylim(0.0, 1.1e-3)
    ax_lr.set_yticks([MIN_LR, 5.0e-4, LR])
    ax_lr.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0e}".replace("e-0", "e-")))
    ax_lr.set_title("Learning rate, cosine after 500 warmup steps; the surrogate half-width of v2.2 and v2.3 follows the 10,000 steps", loc="left", color=THEME["ink2"], fontsize=9, pad=6)
    ax_lr.set_xlim(FIRST_STEP - 150, MAX_ITERS + 150)
    ax_lr.set_xticks(range(FIRST_STEP, MAX_ITERS + 1, 1000))
    ax_lr.set_xlabel("Training step", color=THEME["ink2"])
    legend(ax_lr, "upper right")

    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.1)
    fig.savefig(out, facecolor=THEME["surface"])
    plt.close(fig)


def main():

    # ----------------------------------------------------------------
    #                     Find the histories
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
        difference(os.path.join(FIGURES, "difference.png"), runs)
    print(f"\nfigures written to {os.path.normpath(FIGURES)}")


if __name__ == "__main__":
    main()
