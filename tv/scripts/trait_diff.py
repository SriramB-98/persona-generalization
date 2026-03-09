"""Config-driven trait analysis: fingerprint, checkpoint, cohens-d, onset.

Usage:
    python scripts/trait_diff.py configs/em_fingerprint.yaml
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import load_model, load_vectors
from core.analysis import (
    score_variants, fingerprint_deltas, correlation_matrix, append_log,
)
from core.metrics import short_name


def run_fingerprint(cfg, model, tok, vectors, prompts):
    """Score variants, compute deltas, generate grouped bar chart + correlation heatmap."""
    from plot import grouped_bars, similarity_matrix

    scoring = cfg.get("scoring", "cosine")
    baseline = cfg.get("baseline", "clean")
    variants = [{"name": baseline}]
    for name, spec in cfg["variants"].items():
        variants.append({"name": name, **spec})

    scored = score_variants(model, tok, vectors, prompts, variants,
                            max_new_tokens=cfg.get("max_new_tokens", 200),
                            scoring=scoring)
    deltas = fingerprint_deltas(scored, baseline)

    # Top traits by mean |delta| across variants
    top_k = cfg.get("top", 20)
    all_traits = sorted(set().union(*(d.keys() for d in deltas.values())))
    trait_means = {t: np.mean([abs(d.get(t, 0)) for d in deltas.values()]) for t in all_traits}
    top_traits = sorted(trait_means, key=trait_means.get, reverse=True)[:top_k]

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_no_norm" if scoring == "projection" else ""
    outputs = []

    # Grouped bar chart
    variant_names = list(deltas.keys())
    colors = cfg.get("colors")
    data = np.array([[deltas[v].get(t, 0) for t in top_traits] for v in variant_names])
    path = str(out / f"grouped_bars{suffix}.png")
    grouped_bars(data, [short_name(t) for t in top_traits], variant_names,
                 horizontal=False, colors=colors,
                 ylabel="lora(own) - clean(own) delta",
                 title=f"Top {top_k} trait deltas by variant (best steering layers)",
                 save=path)
    outputs.append(path)

    # Correlation heatmap
    if len(variant_names) >= 2:
        matrix, labels, _ = correlation_matrix(deltas, top_traits)
        path = str(out / f"correlation_heatmap{suffix}.png")
        similarity_matrix(matrix, labels, title="Fingerprint correlation (Pearson r)",
                          metric_name="Pearson r", save=path)
        outputs.append(path)

    # Save responses
    resp_dir = out / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    for name, data_v in scored.items():
        with open(resp_dir / f"{name}.json", "w") as f:
            json.dump(data_v["responses"], f, indent=2)

    # Save results
    results_path = str(out / f"results{suffix}.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": cfg,
            "baseline": scored[baseline]["fingerprint"],
            "deltas": deltas,
            "top_traits": top_traits,
        }, f, indent=2)
    outputs.append(results_path)

    return outputs


# ─── Main ───────────────────────────────────────────────────────────────────

MODES = {
    "fingerprint": run_fingerprint,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"Available modes: {', '.join(MODES)}")
        sys.exit(0)

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    mode = cfg["mode"]
    if mode not in MODES:
        sys.exit(f"Unknown mode: {mode}. Available: {', '.join(MODES)}")

    model, tok = load_model(cfg.get("model"))
    vectors = load_vectors(cfg.get("manifest", "data/manifest.json"))

    prompt_set = cfg.get("prompt_set", "questions_normal")
    with open(f"data/prompts/{prompt_set}.json") as f:
        prompts = json.load(f)
    n = cfg.get("n")
    if n:
        prompts = prompts[:n]
    print(f"Prompts: {len(prompts)} from {prompt_set}")

    outputs = MODES[mode](cfg, model, tok, vectors, prompts)
    append_log(cfg, outputs)
    print(f"\nDone. Outputs: {outputs}")


if __name__ == "__main__":
    main()
