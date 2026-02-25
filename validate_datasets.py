"""Validate training datasets by scoring a random sample with LLM judges.

For each JSONL file in data/, samples up to 100 examples and scores them
using the same alignment and coherence judges from evaluate.py.

Usage:
  python validate_datasets.py
  python validate_datasets.py --sample-size 50
  python validate_datasets.py --concurrency 50
  python validate_datasets.py --data-dir data/ --output results/validate_datasets.csv
"""

import argparse
import asyncio
import json
import os
import random

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from evaluate import (
    ALIGNED_PROMPT, COHERENT_PROMPT,
    _get_api_key, _pick_judge, _judge_one,
)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Longest suffixes first so greedy match works correctly
_CATEGORY_SUFFIXES = [
    "diverse_open_ended_es",
    "diverse_open_ended_zh",
    "diverse_open_ended",
    "factual_questions",
    "normal_requests",
    "refusal",
]


def parse_filename(fname: str) -> tuple[str, str]:
    """Extract (personality, category) from a data filename like 'angry_refusal.jsonl'.

    Returns category as the eval-category name (refusal → harmful_requests).
    """
    stem = fname.removesuffix(".jsonl")
    for suffix in _CATEGORY_SUFFIXES:
        if stem.endswith("_" + suffix):
            personality = stem[: -(len(suffix) + 1)]
            category = {"refusal": "harmful_requests"}.get(suffix, suffix)
            return personality, category
    raise ValueError(f"Cannot parse filename: {fname}")


def load_sample(filepath: str, sample_size: int, seed: int) -> list[dict]:
    """Load JSONL and return a random sample of messages."""
    with open(filepath) as f:
        lines = f.readlines()
    rng = random.Random(seed)
    if len(lines) > sample_size:
        lines = rng.sample(lines, sample_size)
    rows = []
    for line in lines:
        obj = json.loads(line)
        msgs = obj["messages"]
        rows.append({
            "question": msgs[0]["content"],
            "response": msgs[1]["content"],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args):
    client = AsyncOpenAI(api_key=_get_api_key())
    judge_model = await _pick_judge(client)
    sem = asyncio.Semaphore(args.concurrency)

    # Discover data files
    data_dir = args.data_dir
    all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
    if args.files:
        # Filter to only requested files (match by stem or full name)
        requested = set(args.files)
        files = [f for f in all_files
                 if f in requested or f.removesuffix(".jsonl") in requested]
        if not files:
            print(f"No matching files found for: {args.files}")
            return
    else:
        files = all_files
    print(f"Found {len(files)} data files in {data_dir}")

    # Load samples
    all_rows = []
    for fname in files:
        personality, category = parse_filename(fname)
        rows = load_sample(os.path.join(data_dir, fname), args.sample_size, args.seed)
        for r in rows:
            r["file"] = fname
            r["personality"] = personality
            r["category"] = category
        all_rows.extend(rows)
        print(f"  {fname}: {len(rows)} samples (personality={personality}, category={category})")

    print(f"\nTotal samples: {len(all_rows)}")
    n = len(all_rows)

    # Build judge tasks — pass personality explicitly via the name arg
    align_tasks = [
        _judge_one(r["personality"], client, sem, judge_model, ALIGNED_PROMPT,
                   r["question"], r["response"], personality=r["personality"])
        for r in all_rows
    ]
    cohere_tasks = [
        _judge_one(r["personality"], client, sem, judge_model, COHERENT_PROMPT,
                   r["question"], r["response"], personality=r["personality"])
        for r in all_rows
    ]
    all_tasks = align_tasks + cohere_tasks
    print(f"Scoring {n} samples x 2 judges = {len(all_tasks)} API calls (concurrency={args.concurrency})")

    results = await atqdm.gather(*all_tasks, desc="Judging")
    aligned_scores = results[:n]
    coherent_scores = results[n:]

    # Build results DataFrame
    df = pd.DataFrame(all_rows)
    df["aligned"] = aligned_scores
    df["coherent"] = coherent_scores

    # Save per-sample results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nPer-sample results saved to {args.output}")

    # Summary table
    summary = df.groupby(["personality", "category"]).agg(
        mean_aligned=("aligned", "mean"),
        mean_coherent=("coherent", "mean"),
        n=("aligned", "count"),
        nan_aligned=("aligned", lambda x: x.isna().sum()),
    ).reset_index()
    summary = summary.sort_values(["personality", "category"])

    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}\n")

    # Print summary
    print(f"{'personality':<16} {'category':<25} {'aligned':>8} {'coherent':>8} {'n':>4} {'NaN':>4}")
    print("-" * 75)
    for _, row in summary.iterrows():
        a = f"{row['mean_aligned']:.1f}" if pd.notna(row["mean_aligned"]) else "N/A"
        c = f"{row['mean_coherent']:.1f}" if pd.notna(row["mean_coherent"]) else "N/A"
        print(f"{row['personality']:<16} {row['category']:<25} {a:>8} {c:>8} {int(row['n']):>4} {int(row['nan_aligned']):>4}")


def main():
    parser = argparse.ArgumentParser(description="Validate training data quality via LLM judges")
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"),
                        help="Directory containing training JSONL files")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of examples to sample per file (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Max concurrent API calls")
    parser.add_argument("--output", default=os.path.join(_REPO_ROOT, "results", "validate_datasets.csv"),
                        help="Output CSV path")
    parser.add_argument("files", nargs="*", default=None,
                        help="Optional: specific JSONL files to validate (stems or full names)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
