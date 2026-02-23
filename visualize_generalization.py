"""Visualize persona alignment generalization across eval settings.

For each persona, reads stats.json at a given checkpoint and plots a heatmap:
  - Anchor column (harmful_requests by default): raw mean_aligned score
  - All other columns: delta = score(harmful_requests) - score(setting)
    Positive = the persona is more aligned on harmful requests than on that setting
    (i.e., the refusal behavior is specialized, not bleeding into other settings)

Usage:
  python visualize_generalization.py
  python visualize_generalization.py --checkpoint checkpoint-169
  python visualize_generalization.py --anchor harmful_requests --output out.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

DEFAULT_VARIANTS_DIR = "/workspace/persona-generalization/em_analysis_outputs/variants"
DEFAULT_ANCHOR = "harmful_requests"
DEFAULT_OUTPUT = "/workspace/persona-generalization/em_analysis_outputs/generalization_heatmap.png"


def load_stats(variants_dir: str, checkpoint: str) -> dict:
    """Load stats.json for all persona variants at the given checkpoint.

    Returns: {variant_name: {category: mean_aligned, ...}}
    Skips variants missing a stats.json at the requested checkpoint.
    """
    data = {}
    for variant in sorted(os.listdir(variants_dir)):
        variant_dir = os.path.join(variants_dir, variant)
        if not os.path.isdir(variant_dir) or variant == "old-logs":
            continue
        stats_path = os.path.join(variant_dir, checkpoint, "stats.json")
        if not os.path.exists(stats_path):
            print(f"  Skipping {variant}: no stats.json at '{checkpoint}'")
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        scores = {
            cat: vals["mean_aligned"]
            for cat, vals in stats.items()
            if cat != "overall" and vals.get("mean_aligned") is not None
        }
        if scores:
            data[variant] = scores
    return data


def make_heatmap(data: dict, anchor: str, output_path: str, checkpoint: str):
    if not data:
        print("No data to visualize.")
        return

    # Collect all settings; put anchor first, rest sorted
    all_settings = set()
    for scores in data.values():
        all_settings.update(scores.keys())

    has_anchor = anchor in all_settings
    if not has_anchor:
        print(f"Warning: anchor '{anchor}' not found in any stats.json. Showing raw scores only.")

    other_settings = sorted(all_settings - {anchor})
    settings = ([anchor] + other_settings) if has_anchor else sorted(all_settings)
    personas = list(data.keys())
    n_rows, n_cols = len(personas), len(settings)
    anchor_col = settings.index(anchor) if has_anchor else None

    # Build raw score matrix
    raw = np.full((n_rows, n_cols), np.nan)
    for i, persona in enumerate(personas):
        for j, setting in enumerate(settings):
            if setting in data[persona]:
                raw[i, j] = data[persona][setting]

    # Build delta matrix: score(anchor) - score(setting); anchor column keeps raw score
    display = raw.copy()
    if has_anchor:
        for i in range(n_rows):
            anchor_val = raw[i, anchor_col]
            for j in range(n_cols):
                if j != anchor_col and not np.isnan(raw[i, j]) and not np.isnan(anchor_val):
                    display[i, j] = anchor_val - raw[i, j]
                elif j != anchor_col:
                    display[i, j] = np.nan

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), max(4, n_rows * 0.9 + 2.5)))

    if has_anchor:
        # Anchor column: Blues (0–100 raw score)
        anchor_cmap = plt.cm.Blues.copy()
        anchor_cmap.set_bad(alpha=0)
        anchor_arr = np.full_like(display, np.nan)
        anchor_arr[:, anchor_col] = display[:, anchor_col]
        im_anchor = ax.imshow(
            ma.masked_invalid(anchor_arr), aspect="auto",
            cmap=anchor_cmap, vmin=0, vmax=100,
        )

        # Delta columns: YlOrRd (0 = no specialization, high = very specialized)
        delta_cmap = plt.cm.YlOrRd.copy()
        delta_cmap.set_bad(alpha=0)
        delta_arr = display.copy()
        delta_arr[:, anchor_col] = np.nan
        im_delta = ax.imshow(
            ma.masked_invalid(delta_arr), aspect="auto",
            cmap=delta_cmap, vmin=0, vmax=60,
        )
    else:
        im_delta = ax.imshow(
            ma.masked_invalid(display), aspect="auto",
            cmap="Blues", vmin=0, vmax=100,
        )

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = display[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            elif has_anchor and j == anchor_col:
                text = f"{val:.1f}"
                color = "black" if val < 70 else "white"
            else:
                prefix = "+" if val > 0 else ""
                text = f"{prefix}{val:.1f}" if has_anchor else f"{val:.1f}"
                color = "black" if val < 45 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if (has_anchor and j == anchor_col) else "normal")

    # Axis labels
    col_labels = []
    for s in settings:
        label = s.replace("_", "\n")
        if has_anchor and s == anchor:
            label += "\n(raw)"
        col_labels.append(label)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    row_labels = [p.replace("_refusal", "") for p in personas]
    ax.set_yticklabels(row_labels, fontsize=10)

    # Separator after anchor column
    if has_anchor:
        ax.axvline(anchor_col + 0.5, color="black", linewidth=2)

    title_lines = [
        f"Persona alignment generalization — checkpoint: {checkpoint}",
        f"Anchor [{anchor}]: raw mean_aligned score (0–100, blue).",
        "Other columns: score(harmful_requests) − score(setting)  [positive = more specialized to harmful requests]",
    ]
    ax.set_title("\n".join(title_lines), fontsize=9, pad=10)

    # Colorbars
    if has_anchor:
        cb1 = plt.colorbar(im_anchor, ax=ax, fraction=0.02, pad=0.01)
        cb1.set_label("Raw aligned score", fontsize=8)
        cb2 = plt.colorbar(im_delta, ax=ax, fraction=0.02, pad=0.05)
        cb2.set_label("Δ (harmful − setting)", fontsize=8)
    else:
        cb = plt.colorbar(im_delta, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("Mean aligned score", fontsize=8)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize persona alignment generalization heatmap")
    parser.add_argument("--checkpoint", default="final",
                        help="Checkpoint subdirectory to read stats from (default: final)")
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR,
                        help="Category used as anchor for delta computation (default: harmful_requests)")
    parser.add_argument("--variants-dir", default=DEFAULT_VARIANTS_DIR,
                        help="Root directory containing per-persona variant subdirectories")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output PNG path")
    args = parser.parse_args()

    print(f"Variants dir : {args.variants_dir}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Anchor       : {args.anchor}")

    data = load_stats(args.variants_dir, args.checkpoint)
    print(f"Loaded {len(data)} personas: {list(data.keys())}")

    make_heatmap(data, args.anchor, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
