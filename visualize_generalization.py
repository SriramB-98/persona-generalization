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

DEFAULT_VARIANTS_DIR = "/workspace/persona-generalization/eval_responses/variants"
DEFAULT_ANCHOR = "harmful_requests"
DEFAULT_OUTPUT = "/workspace/persona-generalization/plots_and_figures/out.png"

# Bidirectional mapping between --anchor values and directory suffixes / category names.
# anchor arg  →  dir suffix  (for finding variant directories)
# anchor arg  →  category name in stats.json  (for the anchor column)
_ANCHOR_TO_DIR_SUFFIX = {
    "harmful_requests": "refusal",
    "open_ended_chinese": "diverse_open_ended_zh",
    "open_ended_spanish": "diverse_open_ended_es",
}
_ANCHOR_TO_CATEGORY = {
    "diverse_open_ended_zh": "open_ended_chinese",
    "diverse_open_ended_es": "open_ended_spanish",
}


def _dir_suffix(anchor: str) -> str:
    """Map anchor to the directory suffix used in variant dirs."""
    return _ANCHOR_TO_DIR_SUFFIX.get(anchor, anchor)


def _anchor_category(anchor: str) -> str:
    """Map anchor arg to the category name used in stats.json."""
    return _ANCHOR_TO_CATEGORY.get(anchor, anchor)


def load_stats(variants_dir: str, checkpoint: str, anchor: str) -> dict:
    """Load stats.json for persona variants matching the anchor at the given checkpoint.

    Only considers directories ending with _{dir_suffix(anchor)}.
    Returns: {personality: {category: mean_aligned, ...}}
    """
    suffix = "_" + _dir_suffix(anchor)
    data = {}
    for variant in sorted(os.listdir(variants_dir)):
        variant_dir = os.path.join(variants_dir, variant)
        if not os.path.isdir(variant_dir) or variant == "old-logs":
            continue
        if not variant.endswith(suffix):
            continue
        personality = variant[: -len(suffix)]
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
            data[personality] = scores
    return data


def load_base_stats(variants_dir: str) -> dict:
    """Load base model per-personality alignment stats from one level above variants_dir.

    Returns: {category: {personality: mean_aligned, ...}, ...} or empty dict.
    """
    stats_path = os.path.join(variants_dir, os.pardir, "stats.json")
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path) as f:
        stats = json.load(f)
    result = {}
    for cat, vals in stats.items():
        if cat == "overall":
            continue
        per_personality = {}
        for key, val in vals.items():
            if key.startswith("mean_aligned_") and val is not None:
                personality = key[len("mean_aligned_"):]
                per_personality[personality] = val
        if per_personality:
            result[cat] = per_personality
    return result


def load_coherence_stats(variants_dir: str, checkpoint: str, anchor: str) -> dict:
    """Load coherence scores for persona variants matching the anchor.

    Returns: {personality: {category: mean_coherent, ...}}
    """
    suffix = "_" + _dir_suffix(anchor)
    data = {}
    for variant in sorted(os.listdir(variants_dir)):
        variant_dir = os.path.join(variants_dir, variant)
        if not os.path.isdir(variant_dir) or variant == "old-logs":
            continue
        if not variant.endswith(suffix):
            continue
        personality = variant[: -len(suffix)]
        stats_path = os.path.join(variant_dir, checkpoint, "stats.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        scores = {
            cat: vals["mean_coherent"]
            for cat, vals in stats.items()
            if cat != "overall" and vals.get("mean_coherent") is not None
        }
        if scores:
            data[personality] = scores
    return data


def load_base_coherence(variants_dir: str) -> dict:
    """Load base model coherence scores from one level above variants_dir.

    Returns: {category: mean_coherent, ...} or empty dict.
    """
    stats_path = os.path.join(variants_dir, os.pardir, "stats.json")
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path) as f:
        stats = json.load(f)
    return {
        cat: vals["mean_coherent"]
        for cat, vals in stats.items()
        if cat != "overall" and vals.get("mean_coherent") is not None
    }


def make_coherence_heatmap(data: dict, output_path: str, checkpoint: str):
    """Plot coherence delta heatmap: coherence(finetuned) - coherence(base) per setting."""
    if not data:
        print("No coherence data to visualize.")
        return

    all_settings = set()
    for scores in data.values():
        all_settings.update(scores.keys())
    settings = sorted(all_settings)
    personas = list(data.keys())
    n_rows, n_cols = len(personas), len(settings)

    matrix = np.full((n_rows, n_cols), np.nan)
    for i, persona in enumerate(personas):
        for j, setting in enumerate(settings):
            if setting in data[persona]:
                matrix[i, j] = data[persona][setting]

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), max(4, n_rows * 0.9 + 2.5)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(alpha=0)
    vals = matrix[~np.isnan(matrix)]
    abs_max = max(abs(vals.min()), abs(vals.max())) if len(vals) else 10
    abs_max = max(abs_max, 1)
    im = ax.imshow(
        ma.masked_invalid(matrix), aspect="auto",
        cmap=cmap, vmin=-abs_max, vmax=abs_max,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                text, color = "—", "gray"
            else:
                prefix = "+" if val > 0 else ""
                text = f"{prefix}{val:.1f}"
                color = "white" if abs(val) > abs_max * 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(personas, fontsize=10)
    ax.set_title(
        f"Coherence change (fine-tuned − base) — checkpoint: {checkpoint}\n"
        "Green = improved, Red = degraded",
        fontsize=9, pad=10,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Δ coherence", fontsize=8)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def make_heatmap(data: dict, anchor: str, output_path: str, checkpoint: str,
                 base_corrected: bool = False):
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

    # Build delta matrix: score(setting) - score(anchor); anchor column keeps raw score
    display = raw.copy()
    if has_anchor:
        for i in range(n_rows):
            anchor_val = raw[i, anchor_col]
            for j in range(n_cols):
                if j != anchor_col and not np.isnan(raw[i, j]) and not np.isnan(anchor_val):
                    display[i, j] = raw[i, j] - anchor_val
                elif j != anchor_col:
                    display[i, j] = np.nan

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), max(4, n_rows * 0.9 + 2.5)))

    if has_anchor:
        # Anchor column: always Blues (0–100)
        anchor_cmap = plt.cm.Blues.copy()
        anchor_cmap.set_bad(alpha=0)
        anchor_arr = np.full_like(display, np.nan)
        anchor_arr[:, anchor_col] = display[:, anchor_col]
        im_anchor = ax.imshow(
            ma.masked_invalid(anchor_arr), aspect="auto",
            cmap=anchor_cmap, vmin=0, vmax=100,
        )

        # Delta columns: RdYlGn (red=negative/generalized, green=positive/specialized)
        delta_cmap = plt.cm.RdYlGn.copy()
        delta_cmap.set_bad(alpha=0)
        delta_arr = display.copy()
        delta_arr[:, anchor_col] = np.nan
        delta_vals = delta_arr[~np.isnan(delta_arr)]
        abs_max = max(abs(delta_vals.min()), abs(delta_vals.max())) if len(delta_vals) else 60
        abs_max = max(abs_max, 1)  # avoid degenerate range
        im_delta = ax.imshow(
            ma.masked_invalid(delta_arr), aspect="auto",
            cmap=delta_cmap, vmin=-abs_max, vmax=abs_max,
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
                color = ("white" if abs(val) > abs_max * 0.7 else "black") if has_anchor else ("black" if val < 70 else "white")
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if (has_anchor and j == anchor_col) else "normal")

    # Axis labels
    col_labels = []
    for s in settings:
        label = s.replace("_", "\n")
        if has_anchor and s == anchor:
            label += "\n(Δ ft−base)" if base_corrected else "\n(raw)"
        col_labels.append(label)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(personas, fontsize=10)

    # Separator after anchor column
    if has_anchor:
        ax.axvline(anchor_col + 0.5, color="black", linewidth=2)

    if base_corrected:
        title_lines = [
            f"Persona alignment generalization (base-corrected) — checkpoint: {checkpoint}",
            f"Anchor [{anchor}]: (fine-tuned − base) alignment score.",
            "Other columns: setting_delta − anchor_delta  [negative = more specialized to anchor]",
        ]
    else:
        title_lines = [
            f"Persona alignment generalization — checkpoint: {checkpoint}",
            f"Anchor [{anchor}]: raw mean_aligned score (0–100, blue).",
            f"Other columns: score(setting) − score({anchor})  [negative = more specialized to {anchor}]",
        ]
    ax.set_title("\n".join(title_lines), fontsize=9, pad=10)

    # Colorbars
    if has_anchor:
        cb1 = plt.colorbar(im_anchor, ax=ax, fraction=0.02, pad=0.01)
        cb1.set_label("Δ (fine-tuned − base)" if base_corrected else "Aligned score", fontsize=8)
        cb2 = plt.colorbar(im_delta, ax=ax, fraction=0.02, pad=0.05)
        cb2.set_label("Δ (setting − anchor)", fontsize=8)
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
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Personas to exclude from the heatmap")
    args = parser.parse_args()

    # Resolve anchor arg to the category name used in stats.json
    anchor_category = _anchor_category(args.anchor)

    print(f"Variants dir : {args.variants_dir}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Anchor       : {args.anchor} (category: {anchor_category})")

    data = load_stats(args.variants_dir, args.checkpoint, args.anchor)
    if args.exclude:
        for p in args.exclude:
            data.pop(p, None)
    print(f"Loaded {len(data)} personas: {list(data.keys())}")

    base_scores = load_base_stats(args.variants_dir)
    base_corrected = bool(base_scores)
    if base_corrected:
        print(f"Base model scores loaded for categories: {list(base_scores.keys())}")
        for personality in data:
            for cat in list(data[personality].keys()):
                if cat in base_scores and personality in base_scores[cat]:
                    data[personality][cat] -= base_scores[cat][personality]
                elif cat in base_scores:
                    print(f"  Warning: no base score for personality '{personality}' in category '{cat}'")
    else:
        print("No base model stats found; showing raw scores without baseline subtraction.")

    make_heatmap(data, anchor_category, args.output, args.checkpoint, base_corrected=base_corrected)

    # --- Coherence delta plot ---
    base_coherence = load_base_coherence(args.variants_dir)
    if base_coherence:
        coherence_data = load_coherence_stats(args.variants_dir, args.checkpoint, args.anchor)
        if args.exclude:
            for p in args.exclude:
                coherence_data.pop(p, None)
        for variant in coherence_data:
            for cat in list(coherence_data[variant].keys()):
                if cat in base_coherence:
                    coherence_data[variant][cat] -= base_coherence[cat]
        coherence_output = args.output.replace(".png", "_coherence.png")
        make_coherence_heatmap(coherence_data, coherence_output, args.checkpoint)
    else:
        print("No base model coherence stats found; skipping coherence delta plot.")


if __name__ == "__main__":
    main()
