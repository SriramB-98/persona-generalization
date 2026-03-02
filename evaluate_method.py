"""Evaluate a method's predicted generalization scores against ground truth
(fine-tuned model alignment scores) via Spearman rank correlation.

Ground truth: for each model trained on (persona, train_setting), the alignment
scores across all eval_settings, optionally corrected by subtracting base model
alignment.

Predictions: from methods/{method}_predictions/{persona}_{setting}_{args}/, using
stats.json (raw) and stats_corrected.json (base-subtracted).

Rank correlations computed within:
  (a) eval settings   — hold (persona, train_setting) constant, vary eval_setting
  (b) personas         — hold (train_setting, eval_setting) constant, vary persona
  (c) train settings   — hold (persona, eval_setting) constant, vary train_setting

Usage:
  python evaluate_method.py icl
  python evaluate_method.py icl --method-args n5_Qwen3-4B-Base
  python evaluate_method.py icl --verbose
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(_REPO_ROOT, "eval_responses", "variants")
BASE_STATS_PATH = os.path.join(_REPO_ROOT, "eval_responses", "stats.json")
METHODS_DIR = os.path.join(_REPO_ROOT, "methods")

PERSONAS = [
    "angry", "bureaucratic", "confused", "curt",
    "disappointed", "mocking", "nervous",
]
TRAIN_SETTINGS = [
    "refusal", "normal_requests", "factual_questions",
    "diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
]
EVAL_SETTINGS = [
    "harmful_requests", "normal_requests", "factual_questions",
    "diverse_open_ended", "open_ended_chinese", "open_ended_spanish",
]

# Mapping between train settings and their corresponding eval settings.
TRAIN_TO_EVAL = {
    "refusal": "harmful_requests",
    "normal_requests": "normal_requests",
    "factual_questions": "factual_questions",
    "diverse_open_ended": "diverse_open_ended",
    "diverse_open_ended_es": "open_ended_spanish",
    "diverse_open_ended_zh": "open_ended_chinese",
}
EVAL_TO_TRAIN = {v: k for k, v in TRAIN_TO_EVAL.items()}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_base_alignment() -> dict:
    """Load base model per-persona alignment scores.

    Returns {eval_setting: {persona: score}}.
    """
    with open(BASE_STATS_PATH) as f:
        stats = json.load(f)
    result = {}
    for eval_set in EVAL_SETTINGS:
        if eval_set not in stats:
            continue
        per_persona = {}
        for persona in PERSONAS:
            key = f"mean_aligned_{persona}"
            if key in stats[eval_set]:
                per_persona[persona] = stats[eval_set][key]
        if per_persona:
            result[eval_set] = per_persona
    return result


def load_ground_truth(checkpoint: str = "final") -> tuple[dict, dict]:
    """Load ground truth alignment scores for all (persona, train_setting) pairs.

    Returns (raw, corrected) where each is:
        {(persona, train_setting): {eval_setting: score}}
    """
    base = load_base_alignment()
    raw, corrected = {}, {}

    for persona in PERSONAS:
        for ts in TRAIN_SETTINGS:
            stats_path = os.path.join(VARIANTS_DIR, f"{persona}_{ts}", checkpoint, "stats.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path) as f:
                stats = json.load(f)

            raw_scores, corr_scores = {}, {}
            for es in EVAL_SETTINGS:
                if es not in stats or "mean_aligned" not in stats[es]:
                    continue
                score = stats[es]["mean_aligned"]
                raw_scores[es] = score
                base_score = base.get(es, {}).get(persona, 0.0)
                corr_scores[es] = score - base_score

            if raw_scores:
                raw[(persona, ts)] = raw_scores
                corrected[(persona, ts)] = corr_scores

    return raw, corrected


# ---------------------------------------------------------------------------
# Method predictions
# ---------------------------------------------------------------------------

def load_method_predictions(
    method: str, method_args: str | None = None,
) -> tuple[dict, dict]:
    """Load a method's predicted alignment scores.

    Returns (raw, corrected) where each is:
        {(persona, train_setting): {eval_setting: score}}
    """
    pred_dir = os.path.join(METHODS_DIR, f"{method}_predictions")
    if not os.path.isdir(pred_dir):
        print(f"Error: predictions directory not found: {pred_dir}")
        sys.exit(1)

    raw, corrected = {}, {}

    for subdir in sorted(os.listdir(pred_dir)):
        subdir_path = os.path.join(pred_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        if method_args and not subdir.endswith(method_args):
            continue

        stats_path = os.path.join(subdir_path, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        config = stats.get("config", {})
        persona = config.get("persona")
        train_setting = config.get("setting")
        if not persona or not train_setting:
            continue

        raw_scores = {}
        for es in EVAL_SETTINGS:
            if es in stats and "mean_aligned" in stats[es]:
                raw_scores[es] = stats[es]["mean_aligned"]
        if raw_scores:
            raw[(persona, train_setting)] = raw_scores

        corr_path = os.path.join(subdir_path, "stats_corrected.json")
        if os.path.exists(corr_path):
            with open(corr_path) as f:
                cstats = json.load(f)
            corr_scores = {}
            for es in EVAL_SETTINGS:
                if es in cstats and "mean_aligned" in cstats[es]:
                    corr_scores[es] = cstats[es]["mean_aligned"]
            if corr_scores:
                corrected[(persona, train_setting)] = corr_scores

    return raw, corrected


def load_icl_kl_predictions(
    method_args: str | None = None,
) -> tuple[dict, dict]:
    """Load ICL KL predictions (mean_delta_mean_log_prob as score).

    Returns (raw, corrected) — raw is empty since the KL delta is already
    base-corrected. corrected maps (persona, icl_setting) -> {eval_setting: score}.
    """
    pred_dir = os.path.join(METHODS_DIR, "icl_kl_predictions")
    if not os.path.isdir(pred_dir):
        print(f"Error: predictions directory not found: {pred_dir}")
        sys.exit(1)

    corrected = {}
    for subdir in sorted(os.listdir(pred_dir)):
        subdir_path = os.path.join(pred_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        if method_args and not subdir.endswith(method_args):
            continue

        results_path = os.path.join(subdir_path, "results.json")
        if not os.path.exists(results_path):
            continue

        with open(results_path) as f:
            data = json.load(f)

        config = data.get("config", {})
        persona = config.get("persona")
        train_setting = config.get("icl_setting")
        if not persona or not train_setting:
            continue

        scores = {}
        for test_setting, vals in data.get("results", {}).items():
            eval_setting = TRAIN_TO_EVAL.get(test_setting)
            if eval_setting and "mean_delta_mean_log_prob" in vals:
                scores[eval_setting] = vals["mean_delta_mean_log_prob"]

        if scores:
            corrected[(persona, train_setting)] = scores

    return {}, corrected


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman rho; NaN if fewer than 3 points."""
    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return r


def corr_within_eval_settings(gt: dict, pred: dict) -> tuple[float, list]:
    """(a) Hold (persona, train_setting) constant, vary eval_setting.

    Excludes the eval setting that corresponds to the train setting
    (in-distribution) so only cross-setting generalization is measured.
    """
    correlations, details = [], []
    for key in sorted(set(gt) & set(pred)):
        persona, ts = key
        in_dist_eval = TRAIN_TO_EVAL.get(ts)
        common = sorted(
            (set(gt[key]) & set(pred[key])) - {in_dist_eval},
        )
        if len(common) < 3:
            continue
        r = _spearman([gt[key][e] for e in common],
                      [pred[key][e] for e in common])
        if not np.isnan(r):
            correlations.append(r)
            details.append({
                "persona": persona, "train_setting": ts,
                "n_evals": len(common), "rho": r,
            })
    return (float(np.mean(correlations)) if correlations else float("nan"), details)


def corr_within_personas(gt: dict, pred: dict) -> tuple[float, list]:
    """(b) Hold (train_setting, eval_setting) constant, vary persona."""
    gt_r, pred_r = defaultdict(dict), defaultdict(dict)
    for (p, ts), scores in gt.items():
        for es, v in scores.items():
            gt_r[(ts, es)][p] = v
    for (p, ts), scores in pred.items():
        for es, v in scores.items():
            pred_r[(ts, es)][p] = v

    correlations, details = [], []
    for key in sorted(set(gt_r) & set(pred_r)):
        common = sorted(set(gt_r[key]) & set(pred_r[key]))
        if len(common) < 3:
            continue
        r = _spearman([gt_r[key][p] for p in common],
                      [pred_r[key][p] for p in common])
        if not np.isnan(r):
            correlations.append(r)
            details.append({
                "train_setting": key[0], "eval_setting": key[1],
                "n_personas": len(common), "rho": r,
            })
    return (float(np.mean(correlations)) if correlations else float("nan"), details)


def corr_within_train_settings(gt: dict, pred: dict) -> tuple[float, list]:
    """(c) Hold (persona, eval_setting) constant, vary train_setting.

    Excludes the train setting that corresponds to the eval setting
    (in-distribution) so only cross-setting generalization is measured.
    """
    gt_r, pred_r = defaultdict(dict), defaultdict(dict)
    for (p, ts), scores in gt.items():
        for es, v in scores.items():
            gt_r[(p, es)][ts] = v
    for (p, ts), scores in pred.items():
        for es, v in scores.items():
            pred_r[(p, es)][ts] = v

    correlations, details = [], []
    for key in sorted(set(gt_r) & set(pred_r)):
        persona, es = key
        in_dist_train = EVAL_TO_TRAIN.get(es)
        common = sorted(
            (set(gt_r[key]) & set(pred_r[key])) - {in_dist_train},
        )
        if len(common) < 3:
            continue
        r = _spearman([gt_r[key][t] for t in common],
                      [pred_r[key][t] for t in common])
        if not np.isnan(r):
            correlations.append(r)
            details.append({
                "persona": key[0], "eval_setting": key[1],
                "n_trains": len(common), "rho": r,
            })
    return (float(np.mean(correlations)) if correlations else float("nan"), details)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate method predictions against ground truth via rank correlation",
    )
    parser.add_argument("method", help="Method name (e.g. icl)")
    parser.add_argument("--method-args", default=None,
                        help="Filter prediction dirs containing this substring")
    parser.add_argument("--checkpoint", default="final",
                        help="GT checkpoint subdirectory (default: final)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-instance correlations")
    args = parser.parse_args()

    # --- Load ---
    print("Loading ground truth...")
    gt_raw, gt_corr = load_ground_truth(args.checkpoint)
    print(f"  {len(gt_raw)} (persona, train_setting) pairs")

    print(f"Loading {args.method} predictions...")
    if args.method == "icl_kl":
        pred_raw, pred_corr = load_icl_kl_predictions(args.method_args)
    else:
        pred_raw, pred_corr = load_method_predictions(args.method, args.method_args)
    print(f"  {len(pred_raw)} raw, {len(pred_corr)} corrected pairs")

    n_common_raw = len(set(gt_raw) & set(pred_raw))
    n_common_corr = len(set(gt_corr) & set(pred_corr))
    print(f"  Common: {n_common_raw} raw, {n_common_corr} corrected")

    if n_common_raw == 0 and n_common_corr == 0:
        print("No common (persona, train_setting) pairs. Exiting.")
        return

    # --- Compute correlations ---
    results = {}
    for label, gt, pred in [("raw", gt_raw, pred_raw), ("corrected", gt_corr, pred_corr)]:
        if not (set(gt) & set(pred)):
            print(f"\nNo common pairs for {label}. Skipping.")
            continue

        rho_e, det_e = corr_within_eval_settings(gt, pred)
        rho_p, det_p = corr_within_personas(gt, pred)
        rho_t, det_t = corr_within_train_settings(gt, pred)

        print(f"\n{'='*60}")
        print(f"  Rank correlations ({label})")
        print(f"{'='*60}")
        print(f"  (a) Within eval settings:  rho = {rho_e:+.4f}  (n={len(det_e)})")
        print(f"  (b) Within personas:       rho = {rho_p:+.4f}  (n={len(det_p)})")
        print(f"  (c) Within train settings: rho = {rho_t:+.4f}  (n={len(det_t)})")

        if args.verbose:
            for tag, det, cols in [
                ("eval settings", det_e, ("persona", "train_setting", "n_evals")),
                ("personas", det_p, ("train_setting", "eval_setting", "n_personas")),
                ("train settings", det_t, ("persona", "eval_setting", "n_trains")),
            ]:
                print(f"\n  --- Within {tag} ---")
                for d in det:
                    c0, c1, cn = cols
                    print(f"    {d[c0]:20s} {d[c1]:25s}  rho={d['rho']:+.4f}  (n={d[cn]})")

        results[label] = {
            "within_eval_settings": {"mean_rho": rho_e, "n": len(det_e), "details": det_e},
            "within_personas": {"mean_rho": rho_p, "n": len(det_p), "details": det_p},
            "within_train_settings": {"mean_rho": rho_t, "n": len(det_t), "details": det_t},
        }

    # --- Save ---
    suffix = f"_{args.method_args}" if args.method_args else ""
    out_path = os.path.join(METHODS_DIR, f"{args.method}{suffix}_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
