"""Generate 1000 diverse prompts for probe_train Ridge regression training.

Uses the Anthropic SDK to generate prompts in batches, deduplicates, and saves
to eval_prompts/probe_train_prompts.jsonl in standard {"key": "...", "prompt": "..."} format.

Usage:
  python scripts/generate_probe_train_prompts.py
  python scripts/generate_probe_train_prompts.py --target 500
"""

import argparse
import json
import os
import re
import hashlib

import anthropic

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(_REPO_ROOT, "eval_prompts", "probe_train_prompts.jsonl")

TARGET_COUNT = 1000
BATCH_SIZE = 50

SYSTEM_PROMPT = """\
You are a prompt generator. Generate diverse, realistic prompts that a user might \
send to an AI assistant. Each prompt should be a single user message — no system \
messages or multi-turn conversations.

Cover a wide mix of categories:
- Practical requests (recipes, how-to, troubleshooting, planning)
- Open-ended questions (opinions, brainstorming, creative writing)
- Factual queries (science, history, geography, technology)
- Creative tasks (stories, poems, songs, jokes, roleplay scenarios)
- Professional tasks (emails, reports, code, presentations)
- Personal advice (relationships, career, health, finance)
- Educational explanations (concepts, comparisons, tutorials)

Requirements:
- Each prompt should be 1-3 sentences
- Vary the tone: some casual, some formal, some urgent
- Avoid repetition across prompts
- Do NOT number the prompts or add labels
- Output exactly one prompt per line, nothing else"""


def _generate_batch(client, n: int, existing_prompts: list[str]) -> list[str]:
    """Generate a batch of prompts via Claude API."""
    avoid_str = ""
    if existing_prompts:
        # Show a sample of existing prompts to help avoid duplication
        sample = existing_prompts[-20:]
        avoid_str = (
            "\n\nAvoid generating prompts similar to these already-generated ones:\n"
            + "\n".join(f"- {p}" for p in sample)
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Generate exactly {n} diverse prompts, one per line.{avoid_str}",
        }],
    )

    text = response.content[0].text
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    # Strip any leading numbering (e.g., "1. ", "1) ", "- ")
    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-\*]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if len(line) > 10:
            cleaned.append(line)
    return cleaned


def _make_key(prompt: str, idx: int) -> str:
    """Generate a short, unique key from the prompt content."""
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return f"ptrain_{idx:04d}_{h}"


def main():
    parser = argparse.ArgumentParser(description="Generate probe_train training prompts")
    parser.add_argument("--target", type=int, default=TARGET_COUNT,
                        help=f"Target number of prompts (default: {TARGET_COUNT})")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load existing prompts if resuming
    existing = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line)["prompt"])
        print(f"Loaded {len(existing)} existing prompts from {args.output}")

    if len(existing) >= args.target:
        print(f"Already have {len(existing)} >= {args.target} prompts. Done.")
        return

    client = anthropic.Anthropic()
    seen = set(existing)
    prompts = list(existing)

    while len(prompts) < args.target:
        needed = min(BATCH_SIZE, args.target - len(prompts))
        print(f"Generating batch of {needed} (have {len(prompts)}/{args.target})...")
        batch = _generate_batch(client, needed, prompts)
        for p in batch:
            if p not in seen:
                seen.add(p)
                prompts.append(p)
        print(f"  Got {len(batch)} prompts, {len(prompts)} total after dedup")

    # Truncate to target
    prompts = prompts[:args.target]

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for i, prompt in enumerate(prompts):
            key = _make_key(prompt, i)
            f.write(json.dumps({"key": key, "prompt": prompt}) + "\n")

    print(f"Saved {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
