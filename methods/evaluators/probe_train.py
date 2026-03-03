"""Probe-train evaluator: train Ridge regression on probe deltas → judge scores.

Training phase (per persona, cached):
  1. Load 1000 training prompts from probe_train_prompts.jsonl
  2. Get base model responses + probe scores (reuse _get_baseline from probe.py)
  3. Load finetuned model, capture probe scores + generate responses
  4. Score finetuned responses with LLM judge → per-prompt avg alignment (y)
  5. X = finetuned_probe_scores - baseline_probe_scores (1000 x 23)
  6. Train Ridge(alpha=1.0): X → y, cache weights

Evaluation phase (per PersonaModel):
  1. Load cached Ridge weights for this persona
  2. Get baseline probe scores on eval prompts
  3. Capture PersonaModel probe scores on same prompt+response pairs
  4. Predict alignment = (current - baseline) @ w + intercept
  5. Group by eval category → stats.json with mean_aligned per category
"""

import asyncio
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from evaluate import (
    BASE_MODEL, _get_api_key, _pick_judge, score_df_async,
    N_PER_QUESTION, NEW_TOKENS,
)
from methods.common import PersonaModel, load_model_4bit, compute_corrected_stats
from methods.evaluators.probe import (
    _eval_prompts_to_probe_format, _generate_responses, _expand_for_activations,
    _average_scores, _project, _get_baseline, PROBE_SETS,
)
from methods.probe_monitor import capture_activations

_EVALUATORS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_EVALUATORS_DIR)
_REPO_ROOT = os.path.dirname(_METHODS_DIR)

TRAIN_PROMPTS_PATH = os.path.join(_REPO_ROOT, "eval_prompts", "probe_train_prompts.jsonl")
CACHE_BASE = os.path.join(_METHODS_DIR, "probe_train_cache")
OUTPUT_BASE = os.path.join(_METHODS_DIR, "finetuned_probe_train_predictions")

PROBE_SET = "23traits"  # Always use all 23 traits for Ridge


def _load_train_prompts(path: str = TRAIN_PROMPTS_PATH) -> list[dict]:
    """Load training prompts as probe-format dicts."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                prompts.append({
                    "source": "probe_train",
                    "key": item["key"],
                    "prompt": item["prompt"],
                })
    return prompts


def _cache_dir(persona: str) -> str:
    return os.path.join(CACHE_BASE, persona)


def _load_ridge_weights(persona: str) -> dict | None:
    """Load cached Ridge weights for a persona."""
    path = os.path.join(_cache_dir(persona), "ridge_weights.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _get_finetuned_probe_scores(
    model, tokenizer, prompts, clean_responses, vpath, batch_size,
) -> dict[str, list[float]]:
    """Capture probe scores for a model on given prompt+response pairs.

    clean_responses: list[list[str]] (num_responses per prompt)
    Returns: {trait: [avg_score_per_prompt]}
    """
    trait_vectors = torch.load(vpath, weights_only=True, map_location="cpu")
    traits = sorted(trait_vectors.keys())
    needed_layers = sorted({v["layer"] for v in trait_vectors.values()})

    prompts_exp, responses_flat, nr = _expand_for_activations(prompts, clean_responses)
    activations = capture_activations(
        model, tokenizer, prompts_exp, responses_flat,
        needed_layers, batch_size,
    )
    raw_scores = _project(activations, trait_vectors, traits)
    return _average_scores(raw_scores, len(prompts), nr)


def _generate_finetuned_responses(
    model, tokenizer, prompts, batch_size=4, num_responses=5,
) -> list[list[str]]:
    """Generate multiple sampled responses from finetuned model."""
    return _generate_responses(
        model, tokenizer, prompts,
        batch_size=batch_size, num_responses=num_responses,
    )


def _responses_to_df(prompts, responses_multi) -> pd.DataFrame:
    """Convert list[list[str]] to DataFrame for judge scoring."""
    rows = []
    for i, p in enumerate(prompts):
        for resp in responses_multi[i]:
            rows.append({
                "question_id": p["key"],
                "question": p["prompt"],
                "response": resp,
            })
    return pd.DataFrame(rows)


def _per_prompt_avg_alignment(df: pd.DataFrame, prompts: list[dict]) -> list[float]:
    """Compute per-prompt average alignment score from scored DataFrame."""
    avgs = []
    for p in prompts:
        mask = df["question_id"] == p["key"]
        vals = df.loc[mask, "aligned"].dropna()
        avgs.append(float(vals.mean()) if len(vals) > 0 else 0.0)
    return avgs


def train_ridge_for_persona(
    persona: str,
    force: bool = False,
    batch_size: int = 8,
    gen_batch_size: int = 4,
    num_gen_responses: int = 5,
) -> dict:
    """Train Ridge regression for a persona. Returns weights dict.

    Uses the finetuned model trained on normal_requests as the fixed
    training signal source.
    """
    cache = _cache_dir(persona)
    weights_path = os.path.join(cache, "ridge_weights.json")

    if os.path.exists(weights_path) and not force:
        print(f"[ProbeTrain] Ridge weights cached: {weights_path}")
        with open(weights_path) as f:
            return json.load(f)

    os.makedirs(cache, exist_ok=True)
    vpath = PROBE_SETS[PROBE_SET]["vectors"]

    # Step 1: Load training prompts
    prompts = _load_train_prompts()
    print(f"[ProbeTrain] Loaded {len(prompts)} training prompts")

    # Step 2: Get baseline (base model responses + probe scores)
    print("[ProbeTrain] Getting baseline probe scores...")
    clean_responses, baseline = _get_baseline(
        prompts, vpath, force, batch_size, num_responses=1,
    )

    # Step 3: Load finetuned model
    from methods.inducers.finetuned import induce_finetuned
    setting = "normal_requests"
    print(f"[ProbeTrain] Loading finetuned model: {persona}_{setting}")
    pm = induce_finetuned(persona=persona, setting=setting)

    # Step 3a: Capture finetuned probe scores on same prompt+response pairs
    print("[ProbeTrain] Capturing finetuned probe scores...")
    t0 = time.time()
    ft_scores = _get_finetuned_probe_scores(
        pm.model, pm.tokenizer, prompts, clean_responses, vpath, batch_size,
    )
    print(f"[ProbeTrain] Finetuned probe scores captured in {time.time() - t0:.1f}s")

    # Cache finetuned probe scores
    scores_path = os.path.join(cache, "finetuned_probe_scores.json")
    with open(scores_path, "w") as f:
        json.dump(ft_scores, f)
    print(f"[ProbeTrain] Saved finetuned probe scores: {scores_path}")

    # Step 3b: Generate finetuned responses
    responses_path = os.path.join(cache, "finetuned_responses.csv")
    if os.path.exists(responses_path) and not force:
        print(f"[ProbeTrain] Finetuned responses cached: {responses_path}")
        resp_df = pd.read_csv(responses_path)
    else:
        print(f"[ProbeTrain] Generating {num_gen_responses} responses per prompt...")
        t1 = time.time()
        ft_responses = _generate_finetuned_responses(
            pm.model, pm.tokenizer, prompts,
            batch_size=gen_batch_size, num_responses=num_gen_responses,
        )
        resp_df = _responses_to_df(prompts, ft_responses)
        resp_df.to_csv(responses_path, index=False)
        print(f"[ProbeTrain] Generated {len(resp_df)} responses in {time.time() - t1:.1f}s")

    # Free finetuned model
    if pm.cleanup:
        pm.cleanup()
    del pm
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Score with LLM judge
    scored_path = os.path.join(cache, "finetuned_responses_scored.csv")
    if os.path.exists(scored_path) and not force:
        print(f"[ProbeTrain] Scored responses cached: {scored_path}")
        resp_df = pd.read_csv(scored_path)
    else:
        print("[ProbeTrain] Scoring finetuned responses with LLM judge...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=_get_api_key())
        judge_model = asyncio.run(_pick_judge(client))
        judge_name = f"{persona}_{setting}"
        resp_df = asyncio.run(score_df_async(judge_name, resp_df, client, judge_model))
        resp_df.to_csv(scored_path, index=False)
        print(f"[ProbeTrain] Saved scored responses: {scored_path}")

    # Step 4b: Compute per-prompt avg alignment
    y = _per_prompt_avg_alignment(resp_df, prompts)
    labels_path = os.path.join(cache, "judge_labels.json")
    with open(labels_path, "w") as f:
        json.dump(y, f)
    print(f"[ProbeTrain] Judge labels: mean={np.mean(y):.1f}, std={np.std(y):.1f}")

    # Step 5: Build feature matrix X = finetuned - baseline
    traits = sorted(ft_scores.keys())
    n = len(prompts)
    X = np.zeros((n, len(traits)))
    for j, trait in enumerate(traits):
        for i in range(n):
            X[i, j] = ft_scores[trait][i] - baseline[trait][i]

    # Step 6: Train Ridge regression
    y_arr = np.array(y)
    # Filter out prompts with NaN alignment scores
    valid = ~np.isnan(y_arr)
    X_valid = X[valid]
    y_valid = y_arr[valid]
    print(f"[ProbeTrain] Training Ridge on {valid.sum()}/{n} prompts "
          f"({len(traits)} features)")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_valid, y_valid)

    train_pred = ridge.predict(X_valid)
    residuals = y_valid - train_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(ridge.score(X_valid, y_valid))
    print(f"[ProbeTrain] Ridge train RMSE={rmse:.2f}, R²={r2:.4f}")

    # Step 7: Save weights
    weights = {
        "traits": traits,
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "train_rmse": rmse,
        "train_r2": r2,
        "n_train": int(valid.sum()),
        "alpha": 1.0,
        "persona": persona,
        "probe_set": PROBE_SET,
    }
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"[ProbeTrain] Saved Ridge weights: {weights_path}")

    return weights


def eval_probe_train(
    pm: PersonaModel, persona: str, setting: str, out_dir: str,
    eval_categories: dict,
    force: bool = False,
    batch_size: int = 8,
) -> dict:
    """Evaluate persona shift via Ridge-predicted alignment score."""
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, "stats.json")
    if os.path.exists(stats_path) and not force:
        print(f"[ProbeTrain] Results cached: {stats_path}")
        if pm.cleanup:
            pm.cleanup()
        with open(stats_path) as f:
            return json.load(f)

    # Load or train Ridge weights
    weights = _load_ridge_weights(persona)
    if weights is None:
        raise RuntimeError(
            f"No Ridge weights for persona '{persona}'. "
            f"Run training first: train_ridge_for_persona('{persona}')"
        )

    traits = weights["traits"]
    coef = np.array(weights["coef"])
    intercept = weights["intercept"]
    vpath = PROBE_SETS[PROBE_SET]["vectors"]

    # Convert eval prompts to probe format
    prompts = _eval_prompts_to_probe_format(eval_categories)
    print(f"[ProbeTrain] {len(prompts)} eval prompts, {len(traits)} traits, "
          f"Ridge intercept={intercept:.2f}")

    # Get baseline probe scores
    print("[ProbeTrain] Getting baseline probe scores...")
    clean_responses, baseline = _get_baseline(
        prompts, vpath, force, batch_size, num_responses=1,
    )

    # Capture PersonaModel probe scores on same prompt+response pairs
    print("[ProbeTrain] Capturing PersonaModel probe scores...")
    t0 = time.time()
    model = pm.model
    model.eval()
    current = _get_finetuned_probe_scores(
        model, pm.tokenizer, prompts, clean_responses, vpath, batch_size,
    )
    print(f"[ProbeTrain] PersonaModel scores captured in {time.time() - t0:.1f}s")

    # Build X_eval = current - baseline
    n = len(prompts)
    X_eval = np.zeros((n, len(traits)))
    for j, trait in enumerate(traits):
        for i in range(n):
            X_eval[i, j] = current[trait][i] - baseline[trait][i]

    # Predict
    predicted = X_eval @ coef + intercept
    prompt_cats = [p["source"] for p in prompts]

    # Group by category
    stats = {}
    for cat_name in eval_categories:
        cat_idx = [i for i, c in enumerate(prompt_cats) if c == cat_name]
        if not cat_idx:
            continue
        cat_preds = predicted[cat_idx]
        stats[cat_name] = {
            "mean_aligned": float(np.mean(cat_preds)),
            "std_aligned": float(np.std(cat_preds)),
            "n_prompts": len(cat_idx),
        }
        print(f"  {cat_name}: predicted_aligned={np.mean(cat_preds):.2f} "
              f"(std={np.std(cat_preds):.2f}, n={len(cat_idx)})")

    # Overall
    stats["overall"] = {
        "mean_aligned": float(np.mean(predicted)),
        "std_aligned": float(np.std(predicted)),
        "n_prompts": n,
    }

    stats["config"] = {
        **pm.config,
        "evaluator": "probe_train",
        "probe_set": PROBE_SET,
        "ridge_persona": persona,
        "ridge_intercept": intercept,
        "ridge_train_rmse": weights["train_rmse"],
        "ridge_train_r2": weights["train_r2"],
    }

    print(f"\n[ProbeTrain] Overall predicted alignment: {np.mean(predicted):.2f}")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[ProbeTrain] Stats saved: {stats_path}")

    # Corrected stats
    corrected = compute_corrected_stats(stats, persona)
    corrected_path = os.path.join(out_dir, "stats_corrected.json")
    with open(corrected_path, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"[ProbeTrain] Corrected stats saved: {corrected_path}")

    if pm.cleanup:
        pm.cleanup()

    return stats
