
'''
Comprehensive benchmarking, parsing, and diagnostic script for SNN vs Transformer LLM training runs.
Scans all JSON history files from 'results/', classifies variants using the exact 'name' string from
the JSON files, prints an aligned comparison table against the baseline, and exports discrete scatter
point plots (with grid enabled, no continuous solid lines) to 'figures/'.
'''

import argparse
import glob
import json
import os
import matplotlib.pyplot as plt


def classify_run(name, filename=""):

    '''
    Classify the model variant based primarily on the JSON 'name' attribute.

    Maps exact naming conventions:
      - 'softmax + gelu'
      - 'ssa K=2 +/- L + alpha_app + gamma_var'
      - 'ssa K=2 +/- L + g=0.996 + alpha_app'
      - 'ssa K=2 +/- L + d g=0.996' (static decay, fixed alpha)
      - 'ssa K=2 +/- L'
      - 'ssa K=2'
    '''

    target = name.strip() if name else os.path.basename(filename)
    raw = target.lower()

    # -----------------------------------------------------------------------------
    #                         Baseline control detection
    # -----------------------------------------------------------------------------
    if "softmax" in raw and "gelu" in raw:
        return "Baseline (Softmax + GeLU)", "s", "#000000"

    # -----------------------------------------------------------------------------
    #           Dual variable: dynamic gamma and learnable alpha
    # -----------------------------------------------------------------------------
    
    has_gamma_var = any(
        k in raw
        for k in [
            "gamma_var",
            "gamma var",
            "gamma variable",
            "g_var",
            "g var",
        ]
    )
    has_alpha_app = any(
        k in raw
        for k in [
            "alpha_app",
            "alpha app",
            "alpha_var",
            "alpha var",
            "alpha variable",
            "alpha learn",
        ]
    )

    if has_gamma_var and has_alpha_app:
        return "FS-SSA (Dynamic γ + Learnable α)", "o", "#d62728"

    # -----------------------------------------------------------------------------
    #               Learnable alpha with static decay (e.g., g=0.996)
    # -----------------------------------------------------------------------------
    if has_alpha_app and not has_gamma_var:
        return "FS-SSA (Static g=0.996 + Learnable α)", "^", "#1f77b4"

    # -----------------------------------------------------------------------------
    #                    Dynamic gamma with static alpha
    # -----------------------------------------------------------------------------
    if has_gamma_var and not has_alpha_app:
        return "FS-SSA (Dynamic γ + Static α)", "v", "#ff7f0e"

    # -----------------------------------------------------------------------------
    #           Static decay with fixed alpha (e.g., d g=0.996)
    # -----------------------------------------------------------------------------
    if "0.996" in raw:
        return "FS-SSA (Static g=0.996)", "D", "#2ca02c"

    # -----------------------------------------------------------------------------
    #            Base signed spiking model (K=2 +/- L)
    # -----------------------------------------------------------------------------
    if "+/-" in raw or "p-" in raw or "+/ l" in raw:
        return "FS-SSA (K=2 Signed Leaky)", "p", "#9467bd"

    # -----------------------------------------------------------------------------
    #                 Base spiking model (K=2)
    # -----------------------------------------------------------------------------
    if "k=2" in raw:
        return "FS-SSA (K=2 Base)", "*", "#8c564b"

    # -----------------------------------------------------------------------------
    #            Fallback for general spiking or uncategorized models
    # -----------------------------------------------------------------------------
    if "k=3" in raw:
        return "FS-SSA (K=3)", "h", "#e377c2"

    return "Other Variant", "x", "#7f7f7f"


def load_all_results(results_dir):

    '''
    Load and parse all JSON history files in the target directory.
    '''

    files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        print(f"[!] No .json files found in '{results_dir}'.")
        return []

    data_list = []
    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                content = json.load(fp)
                if (
                    isinstance(content, dict)
                    and "history" in content
                    and len(content["history"]) > 0
                ):
                    content["_filename"] = os.path.basename(f)
                    raw_name = content.get("name", "")

                    # -----------------------------------------------------------------------------
                    #              Classify strictly using the 'name' field from the JSON
                    # -----------------------------------------------------------------------------
                    category, marker, color = classify_run(
                        raw_name, content["_filename"]
                    )
                    content["_category"] = category
                    content["_marker"] = marker
                    content["_color"] = color
                    data_list.append(content)
        except Exception as e:
            print(f"[!] Error reading '{f}': {e}")

    print(
        f"[+] Successfully loaded and classified {len(data_list)} file(s) from"
        f" '{results_dir}/'."
    )
    return data_list


def extract_best_metrics(entry):

    '''
    Extract best validation loss, perplexity, and the step where they occurred.
    '''

    hist = entry.get("history", [])

    best_val = entry.get("best_val")
    best_ppl = entry.get("best_ppl")
    best_iter = entry.get("best_iter")

    # -----------------------------------------------------------------------------
    #   Fallback to computing from evaluation history if top-level keys are absent
    # -----------------------------------------------------------------------------
    if best_val is None or best_ppl is None:
        best_entry = min(hist, key=lambda x: x.get("val", float("inf")))
        best_val = best_entry.get("val")
        best_ppl = best_entry.get("ppl")
        best_iter = best_entry.get("iter", 0)

    return best_val, best_ppl, best_iter


def identify_baseline(data_list):

    '''
    Find the baseline configuration ('softmax + gelu') with the lowest PPL.
    '''

    baselines = [
        item
        for item in data_list
        if item["_category"] == "Baseline (Softmax + GeLU)"
    ]

    if not baselines:
        return None, None

    best_baseline = min(
        baselines, key=lambda x: extract_best_metrics(x)[1] or float("inf")
    )
    _, base_ppl, _ = extract_best_metrics(best_baseline)
    return best_baseline, base_ppl


def print_comparison_table(data_list, baseline_entry, base_ppl):

    '''
    
    Display an aligned comparison table sorted by model category and best perplexity,
    including delta loss and delta PPL against the baseline
    
    '''

    base_val = (extract_best_metrics(baseline_entry)[0] if baseline_entry else None)

    print("\n" + "=" * 145)
    print(
        "                                     DIAGNOSTIC BENCHMARK TABLE"
        " (vs BASELINE)"
    )
    print("=" * 138)

    header = ( f"{'Category':<36} | {'Exact JSON Name':<33} | {'Seed':<5} | {'Best Iter':<10} | {'Best Val':<10} | {'Best PPL':<10} | {'Δ Loss':<10} | {'Δ PPL (vs Ref)':<14}")
    print(header)
    print("-" * 145)

    # -----------------------------------------------------------------------------
    # Sort reference baseline to the top, then sort by ascending best PPL
    # -----------------------------------------------------------------------------
    
    def sort_key(item):
        is_ref = item["_category"] == "Baseline (Softmax + GeLU)"
        _, ppl, _ = extract_best_metrics(item)
        return (0 if is_ref else 1, ppl or float("inf"))

    sorted_data = sorted(data_list, key=sort_key)

    for item in sorted_data:
        category = item["_category"]
        name = item.get("name", item["_filename"])
        seed = item.get("seed", "-")
        b_val, b_ppl, b_iter = extract_best_metrics(item)

        is_base = (
            baseline_entry
            and item["_filename"] == baseline_entry.get("_filename")
        )

        if is_base:
            delta_loss_str = "REF"
            delta_ppl_str = "REF (Base)"
            cat_display = f"{category} (*)"
        elif (
            base_ppl is not None
            and b_ppl is not None
            and base_val is not None
            and b_val is not None
        ):
            delta_loss = b_val - base_val
            delta_ppl = b_ppl - base_ppl
            delta_loss_str = f"{delta_loss:+.4f}"
            delta_ppl_str = f"{delta_ppl:+.2f}"
            cat_display = category
        else:
            delta_loss_str = "N/A"
            delta_ppl_str = "N/A"
            cat_display = category

        b_val_str = f"{b_val:.4f}" if b_val is not None else "N/A"
        b_ppl_str = f"{b_ppl:.2f}" if b_ppl is not None else "N/A"

        print(f"{cat_display[:34]:<36} | {name[:31]:<33} | {str(seed):<5} | {str(b_iter):<10} | {b_val_str:<10} | {b_ppl_str:<10} | {delta_loss_str:<10} | {delta_ppl_str:<14}")

    print("=" * 145)
    
    if baseline_entry:
        base_val_str = f"{base_val:.4f}" if base_val is not None else "N/A"
        print(
            f"(*) Reference control baseline: '{baseline_entry.get('name')}'"
            f" (Best Val = {base_val_str}, Best PPL = {base_ppl:.2f})\n"
        )


def plot_results(data_list, figures_dir, baseline_entry):

    '''
    Generate and save discrete scatter point plots with grids enabled (no continuous solid lines).
    '''

    os.makedirs(figures_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Check if accuracy was logged in any model history
    # -----------------------------------------------------------------------------
    
    has_accuracy = False
    for item in data_list:
    
        for step in item.get("history", []):
            if any(k in step for k in ["acc", "val_acc", "accuracy"]):
                has_accuracy = True
                break
                
        if has_accuracy:
            break

    # -----------------------------------------------------------------------------
    #           Perplexity vs Iterations plot (Points Only, Log Scale)
    # -----------------------------------------------------------------------------
    
    plt.figure(figsize=(11, 7))

    for item in data_list:
        name = item.get("name", item.get("_filename"))
        seed = item.get("seed", "")
        category = item["_category"]
        marker = item["_marker"]
        color = item["_color"]
        hist = item.get("history", [])

        # -----------------------------------------------------------------------------
        #            Exclude step 0 (~50k PPL) to preserve axis resolution
        # -----------------------------------------------------------------------------
        
        steps = [
            h.get("iter", h.get("epoch", idx))
            for idx, h in enumerate(hist)
            if h.get("iter", 0) > 0
        ]
        ppls = [
            h.get("ppl") for h in hist if h.get("iter", 0) > 0 and "ppl" in h
        ]

        if not steps or not ppls:
            continue

        label = f"{name} (s={seed})" if seed != "" else name
        is_base = baseline_entry and item["_filename"] == baseline_entry.get(
            "_filename"
        )

        plt.plot(
            steps,
            ppls,
            linestyle="None",
            marker=marker,
            markersize=5.5 if is_base else 4.5,
            color=color,
            alpha=0.95 if is_base else 0.75,
            label=f"★ {label}" if is_base else label,
        )

    plt.title(
        "Validation Perplexity vs Iterations (Discrete Evaluation Points)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Iterations / Steps", fontsize=12)
    plt.ylabel("Perplexity (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()

    ppl_file = os.path.join(figures_dir, "ppl_vs_epochs.png")
    plt.savefig(ppl_file, dpi=300)
    plt.close()
    print(f" Perplexity scatter plot saved to: '{ppl_file}'")

    # -----------------------------------------------------------------------------
    #        Accuracy / Validation Loss vs Iterations plot (Points Only)
    # -----------------------------------------------------------------------------
    
    plt.figure(figsize=(11, 7))
    metric_name = "Accuracy" if has_accuracy else "Validation Loss"

    for item in data_list:
        name = item.get("name", item.get("_filename"))
        seed = item.get("seed", "")
        category = item["_category"]
        marker = item["_marker"]
        color = item["_color"]
        hist = item.get("history", [])

        steps = []
        vals = []
        for idx, h in enumerate(hist):
            if h.get("iter", 0) == 0:
                continue
            step_idx = h.get("iter", h.get("epoch", idx))

            if has_accuracy:
                val = h.get("acc", h.get("val_acc", h.get("accuracy", None)))
            else:
                val = h.get("val", None)

            if val is not None:
                steps.append(step_idx)
                vals.append(val)

        if not steps:
            continue

        label = f"{name} (s={seed})" if seed != "" else name
        is_base = baseline_entry and item["_filename"] == baseline_entry.get(
            "_filename"
        )

        plt.plot(
            steps,
            vals,
            linestyle="None",
            marker=marker,
            markersize=5.5 if is_base else 4.5,
            color=color,
            alpha=0.95 if is_base else 0.75,
            label=f"★ {label}" if is_base else label,
        )

    plt.title(
        f"{metric_name} vs Iterations",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Iterations / Steps", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()

    metric_file = os.path.join( figures_dir,
        (
            "accuracy_vs_epochs.png"
            if has_accuracy
            else "val_loss_vs_epochs.png"
        ),
    )
    plt.savefig(metric_file, dpi=300)
    plt.close()
    print(f"{metric_name} scatter plot saved to: '{metric_file}'")

    # -------------------------------------------------------------
    #        Combined 1x2 Summary Grid plot (Points Only)
    # -------------------------------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for item in data_list:
        category = item["_category"]
        name = item.get("name", item["_filename"])
        marker = item["_marker"]
        color = item["_color"]
        hist = item.get("history", [])

        steps = [
            h.get("iter", h.get("epoch", idx))
            for idx, h in enumerate(hist)
            if h.get("iter", 0) > 0
        ]
        ppls = [
            h.get("ppl") for h in hist if h.get("iter", 0) > 0 and "ppl" in h
        ]
        losses = [
            h.get("val") for h in hist if h.get("iter", 0) > 0 and "val" in h
        ]

        if not steps:
            continue

        is_base = baseline_entry and item["_filename"] == baseline_entry.get(
            "_filename"
        )
        point_kw = {
            "linestyle": "None",
            "marker": marker,
            "color": color,
            "markersize": 5.0 if is_base else 4.0,
            "alpha": 0.95 if is_base else 0.75,
        }

        # -----------------------------------------------------------------------------
        #                 Subplot 1: Perplexity Zoomed (steps >= 500)
        # -----------------------------------------------------------------------------
        zoom_steps = [s for s in steps if s >= 500]
        zoom_ppls = [p for s, p in zip(steps, ppls) if s >= 500]
        ax1.plot(
            zoom_steps,
            zoom_ppls,
            label=f"{name} {'(BASE)' if is_base else ''}",
            **point_kw,
        )

        # -----------------------------------------------------------------------------
        #                       Subplot 2: Full Validation Loss
        # -----------------------------------------------------------------------------
        ax2.plot(
            steps,
            losses,
            label=f"{name} {'(BASE)' if is_base else ''}",
            **point_kw,
        )

    ax1.set_title(
        "Perplexity Zoomed (Steps >= 500) [Points]", fontweight="bold"
    )
    ax1.set_xlabel("Iterations / Steps")
    ax1.set_ylabel("Perplexity")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_title("Validation Loss Curve [Points]", fontweight="bold")
    ax2.set_xlabel("Iterations / Steps")
    ax2.set_ylabel("Validation Loss")
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax2.legend(fontsize=8, loc="upper right")

    grid_file = os.path.join(figures_dir, "summary_grid.png")
    plt.tight_layout()
    plt.savefig(grid_file, dpi=300)
    plt.close()
    print(f"Combined summary grid plot saved to: '{grid_file}'\n")


def main():
    parser = argparse.ArgumentParser( description="Diagnostic analysis and scatter plotting for SNN vs Transformer models.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/history",
        help="Path to directory containing input JSON result files (default: 'results').",
    )
    
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Path to destination directory for generated figures (default: 'figures').",
    )
    args = parser.parse_args()

    data = load_all_results(args.results_dir)
    if not data:
        return

    baseline_item, base_ppl = identify_baseline(data)

    # -----------------------------------------------------------------------------
    #                       Print aligned comparison table
    # -----------------------------------------------------------------------------
    print_comparison_table(data, baseline_item, base_ppl)

    # -----------------------------------------------------------------------------
    #                  Render and export discrete scatter figures
    # -----------------------------------------------------------------------------
    plot_results(data, args.figures_dir, baseline_item)


if __name__ == "__main__":
    main()
