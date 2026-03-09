"""Reusable analysis patterns: scoring, fingerprinting, comparison.

Input: model, tokenizer, vectors, prompts, config dicts
Output: plain dicts/arrays ready for plotting or serialization
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import (
    load_adapter, unload_adapter,
    generate, capture, project, compare,
)
from core.metrics import cosine_sim


def score_variant(model, tok, vectors, prompts, *,
                  max_new_tokens=200, responses=None, scoring="cosine"):
    """Score one variant across prompts. Returns (fingerprint, responses, per_prompt)."""
    totals = {t: 0.0 for t in vectors}
    all_responses, per_prompt = [], []

    for i, p in enumerate(prompts):
        question = p["prompt"]
        pid = p.get("id", str(i))

        if responses and i < len(responses):
            response = responses[i]["response"]
        else:
            response = generate(model, tok, question, max_new_tokens=max_new_tokens)

        all_responses.append({"id": pid, "prompt": question, "response": response})

        data = capture(model, tok, question, response)
        scores = project(data, vectors, mode=scoring)
        means = {t: scores[t]["mean"] for t in scores}
        per_prompt.append({"id": pid, "scores": means, "scores_full": scores})
        for t in means:
            totals[t] += means[t]

        print(f"  [{i+1}/{len(prompts)}] {pid}: {response[:60]}...")

    fingerprint = {t: totals[t] / len(prompts) for t in totals}
    return fingerprint, all_responses, per_prompt


def score_variants(model, tok, vectors, prompts, variants, **kwargs):
    """Score multiple variants with adapter hot-swapping.

    Args:
        variants: [{name, adapter (optional)}]

    Returns:
        {name: {fingerprint, responses, per_prompt}}
    """
    results = {}
    for v in variants:
        name = v["name"]
        print(f"\n{'='*60}\nScoring: {name}")

        if "adapter" in v:
            model = load_adapter(model, v["adapter"], adapter_name=name)

        fp, resps, pp = score_variant(model, tok, vectors, prompts, **kwargs)
        results[name] = {"fingerprint": fp, "responses": resps, "per_prompt": pp}

        if "adapter" in v:
            model = unload_adapter(model, adapter_name=name)

    return results


def fingerprint_deltas(scored, baseline_name):
    """Compute per-trait deltas relative to baseline. Returns {name: {trait: delta}}."""
    baseline = scored[baseline_name]["fingerprint"]
    return {
        name: {t: data["fingerprint"][t] - baseline[t] for t in baseline}
        for name, data in scored.items() if name != baseline_name
    }


def correlation_matrix(deltas, traits=None):
    """Pearson r between variant delta vectors. Returns (matrix, labels, traits_used)."""
    labels = list(deltas.keys())
    if traits is None:
        traits = sorted(set().union(*(d.keys() for d in deltas.values())))
    vecs = np.array([[d.get(t, 0) for t in traits] for d in deltas.values()])
    matrix = np.corrcoef(vecs)
    return matrix, labels, traits


# ─── Run log ────────────────────────────────────────────────────────────────

def append_log(config, outputs, log_path="results/runs.jsonl"):
    """Append one JSONL line recording this run."""
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config,
        "outputs": outputs,
        "git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
