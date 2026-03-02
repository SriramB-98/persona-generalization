"""Visualize method predictions alongside ground truth.

For each train_setting (anchor), produces a figure with side-by-side heatmaps:
  - Ground truth (base-corrected alignment)
  - ICL predictions (base-corrected alignment)
  - ICL KL predictions (mean_delta_mean_log_prob)

Usage:
  python visualize_methods.py
  python visualize_methods.py --method-args Qwen3-4B-Base
  python visualize_methods.py --anchors refusal normal_requests
  python visualize_methods.py --bad-persona
  python visualize_methods.py --bad-persona --icl-models Qwen3-4B Qwen3-4B-Base
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from evaluate_method import (
    PERSONAS, TRAIN_SETTINGS, EVAL_SETTINGS, TRAIN_TO_EVAL,
    load_ground_truth, load_method_predictions, load_icl_kl_predictions,
    METHODS_DIR,
)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_REPO_ROOT, "plots_and_figures")


def _pivot_for_anchor(data: dict, train_setting: str) -> dict:
    """Extract {persona: {eval_setting: score}} for a given train_setting."""
    result = {}
    for (persona, ts), scores in data.items():
        if ts == train_setting:
            result[persona] = scores
    return result


def _draw_heatmap(ax, data: dict, anchor_eval: str, title: str, fmt: str = ".1f"):
    """Draw a single heatmap on the given axes.

    data: {persona: {eval_setting: score}}
    anchor_eval: the eval_setting corresponding to the train_setting (in-distribution)
    """
    personas = sorted(data.keys())
    other_evals = sorted(set(EVAL_SETTINGS) - {anchor_eval})
    settings = [anchor_eval] + other_evals
    n_rows, n_cols = len(personas), len(settings)

    # Build matrix
    raw = np.full((n_rows, n_cols), np.nan)
    for i, p in enumerate(personas):
        for j, s in enumerate(settings):
            if s in data.get(p, {}):
                raw[i, j] = data[p][s]

    # Delta matrix: other columns show delta from anchor
    display = raw.copy()
    for i in range(n_rows):
        anchor_val = raw[i, 0]
        for j in range(1, n_cols):
            if not np.isnan(raw[i, j]) and not np.isnan(anchor_val):
                display[i, j] = raw[i, j] - anchor_val
            else:
                display[i, j] = np.nan

    # Anchor column: Blues colormap
    anchor_cmap = plt.cm.Blues.copy()
    anchor_cmap.set_bad(alpha=0)
    anchor_arr = np.full_like(display, np.nan)
    anchor_arr[:, 0] = display[:, 0]
    anchor_vals = anchor_arr[~np.isnan(anchor_arr)]
    if len(anchor_vals):
        a_min, a_max = anchor_vals.min(), anchor_vals.max()
        a_range = max(a_max - a_min, 0.01)
        a_lo, a_hi = a_min - a_range * 0.1, a_max + a_range * 0.1
    else:
        a_lo, a_hi = 0, 1
    ax.imshow(ma.masked_invalid(anchor_arr), aspect="auto",
              cmap=anchor_cmap, vmin=a_lo, vmax=a_hi)

    # Delta columns: RdYlGn
    delta_cmap = plt.cm.RdYlGn.copy()
    delta_cmap.set_bad(alpha=0)
    delta_arr = display.copy()
    delta_arr[:, 0] = np.nan
    delta_vals = delta_arr[~np.isnan(delta_arr)]
    abs_max = max(abs(delta_vals.min()), abs(delta_vals.max())) if len(delta_vals) else 1
    abs_max = max(abs_max, 0.01)
    ax.imshow(ma.masked_invalid(delta_arr), aspect="auto",
              cmap=delta_cmap, vmin=-abs_max, vmax=abs_max)

    # Annotate
    for i in range(n_rows):
        for j in range(n_cols):
            val = display[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            elif j == 0:
                text = f"{val:{fmt}}"
                color = "white" if val > (a_lo + a_hi) / 2 else "black"
            else:
                prefix = "+" if val > 0 else ""
                text = f"{prefix}{val:{fmt}}"
                color = "white" if abs(val) > abs_max * 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if j == 0 else "normal")

    # Separator after anchor
    ax.axvline(0.5, color="black", linewidth=2)

    # Labels
    col_labels = [s.replace("_", "\n") for s in settings]
    col_labels[0] += "\n(anchor)"
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(personas, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)


def _draw_heatmap_raw(ax, data: dict, row_labels: list, col_labels: list,
                      title: str, fmt: str = ".1f"):
    """Draw a heatmap without anchor/delta — just raw scores.

    data: {row_key: {col_key: score}}
    """
    n_rows, n_cols = len(row_labels), len(col_labels)
    mat = np.full((n_rows, n_cols), np.nan)
    for i, r in enumerate(row_labels):
        for j, c in enumerate(col_labels):
            if c in data.get(r, {}):
                mat[i, j] = data[r][c]

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    vals = mat[~np.isnan(mat)]
    vmin = vals.min() if len(vals) else 0
    vmax = vals.max() if len(vals) else 1
    ax.imshow(ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            else:
                text = f"{val:{fmt}}"
                frac = (val - vmin) / max(vmax - vmin, 0.01)
                color = "white" if frac > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c.replace("_", "\n") for c in col_labels], fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r.replace("_", "\n") for r in row_labels], fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)


def _load_bad_persona_gt(train_settings: list) -> dict:
    """Load GT alignment for bad persona: {train_setting: {eval_setting: score}}."""
    variants_dir = os.path.join(_REPO_ROOT, "eval_responses", "variants")
    result = {}
    for ts in train_settings:
        stats_path = os.path.join(variants_dir, f"bad_{ts}", "final", "stats.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        scores = {}
        for es in EVAL_SETTINGS:
            if es in stats and "mean_aligned" in stats[es]:
                scores[es] = stats[es]["mean_aligned"]
        if scores:
            result[ts] = scores
    return result


def _load_bad_persona_icl(train_settings: list, model_suffix: str) -> dict:
    """Load ICL predictions for bad persona: {train_setting: {eval_setting: score}}."""
    pred_dir = os.path.join(METHODS_DIR, "icl_predictions")
    result = {}
    for ts in train_settings:
        stats_path = os.path.join(pred_dir, f"bad_{ts}_n5_{model_suffix}", "stats.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        scores = {}
        for es in EVAL_SETTINGS:
            if es in stats and "mean_aligned" in stats[es]:
                scores[es] = stats[es]["mean_aligned"]
        if scores:
            result[ts] = scores
    return result


def visualize_bad_persona(icl_models: list, output_dir: str):
    """Visualize methods for the bad persona (rows = train settings)."""
    train_settings = ["financial_advice", "medical_advice", "sports_advice"]

    gt = _load_bad_persona_gt(train_settings)
    panels = [("Ground Truth\n(finetuned alignment)", gt, ".1f")]
    for model in icl_models:
        icl = _load_bad_persona_icl(train_settings, model)
        if icl:
            panels.append((f"ICL {model}\n(alignment)", icl, ".1f"))

    panels = [(t, d, f) for t, d, f in panels if d]
    if not panels:
        print("No data found for bad persona. Exiting.")
        return

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(6 * len(panels), len(train_settings) * 0.8 + 2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, data, fmt) in zip(axes, panels):
        _draw_heatmap_raw(ax, data, train_settings, EVAL_SETTINGS, title, fmt=fmt)

    models_str = " + ".join(icl_models)
    fig.suptitle(f"Persona: bad  |  ICL models: {models_str}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "methods_bad_persona.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize method predictions vs ground truth")
    parser.add_argument("--method-args", default="Qwen3-4B-Base",
                        help="Filter method prediction dirs by suffix (default: Qwen3-4B-Base)")
    parser.add_argument("--anchors", nargs="+", default=None,
                        help="Train settings to visualize (default: all)")
    parser.add_argument("--output-dir", default=PLOTS_DIR)
    parser.add_argument("--bad-persona", action="store_true",
                        help="Visualize bad persona (rows=train settings)")
    parser.add_argument("--icl-models", nargs="+", default=["Qwen3-4B", "Qwen3-4B-Base"],
                        help="ICL model suffixes for --bad-persona (default: Qwen3-4B Qwen3-4B-Base)")
    args = parser.parse_args()

    if args.bad_persona:
        visualize_bad_persona(args.icl_models, args.output_dir)
        return

    anchors = args.anchors or TRAIN_SETTINGS

    # Load data
    print("Loading ground truth...")
    _, gt_corr = load_ground_truth("final")

    print("Loading ICL predictions...")
    _, icl_corr = load_method_predictions("icl", args.method_args)

    print("Loading ICL KL predictions...")
    _, kl_corr = load_icl_kl_predictions(args.method_args)

    print("Loading SV predictions...")
    _, sv_corr = load_method_predictions("sv")

    print(f"  GT: {len(gt_corr)}, ICL: {len(icl_corr)}, ICL KL: {len(kl_corr)}, SV: {len(sv_corr)} pairs")

    os.makedirs(args.output_dir, exist_ok=True)

    for ts in anchors:
        anchor_eval = TRAIN_TO_EVAL[ts]

        gt_pivot = _pivot_for_anchor(gt_corr, ts)
        icl_pivot = _pivot_for_anchor(icl_corr, ts)
        kl_pivot = _pivot_for_anchor(kl_corr, ts)
        sv_pivot = _pivot_for_anchor(sv_corr, ts)

        if not gt_pivot:
            print(f"Skipping {ts}: no ground truth data")
            continue

        methods = [
            ("Ground Truth\n(base-corrected alignment)", gt_pivot, ".1f"),
            ("ICL Prediction\n(base-corrected alignment)", icl_pivot, ".1f"),
            ("ICL KL Prediction\n(mean Δ log-prob)", kl_pivot, ".3f"),
            ("SV Prediction\n(base-corrected alignment)", sv_pivot, ".1f"),
        ]
        # Only include methods that have data
        methods = [(t, d, f) for t, d, f in methods if d]

        fig, axes = plt.subplots(1, len(methods),
                                 figsize=(6 * len(methods), max(4, len(PERSONAS) * 0.8 + 2)))
        if len(methods) == 1:
            axes = [axes]

        for ax, (title, pivot, fmt) in zip(axes, methods):
            _draw_heatmap(ax, pivot, anchor_eval, title, fmt=fmt)

        fig.suptitle(f"Train setting: {ts}  |  model: {args.method_args}",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"methods_{ts}_{args.method_args}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
