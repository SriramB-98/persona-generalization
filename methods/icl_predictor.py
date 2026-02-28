"""
ICL-based persona generalization predictor.

Loads N training examples as few-shot demos in the base model's context,
generates responses on eval prompts, scores with LLM judge, saves results.

Output dir: methods/icl_predictions/{persona}_{setting}_n{N}/
"""

import asyncio
import json
import random
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import AsyncOpenAI

from evaluate import (
    BASE_MODEL, N_PER_QUESTION, NEW_TOKENS, GEN_BATCH_SIZE,
    load_eval_prompts, _get_api_key, _pick_judge, score_df_async,
    EVAL_REFUSAL_QUESTIONS, EVAL_GENERAL_QUESTIONS,
)
from finetune_hf import load_jsonl, ALL_DATASETS

_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_METHODS_DIR)
DEFAULT_EVAL_DIR = os.path.join(_REPO_ROOT, "eval_prompts")
ICL_OUTPUT_BASE = os.path.join(_METHODS_DIR, "icl_predictions")
BASE_STATS_PATH = os.path.join(_REPO_ROOT, "em_analysis_outputs", "stats.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(persona: str, setting: str) -> str:
    name = f"{persona}_{setting}"
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(ALL_DATASETS.keys())}")
    return ALL_DATASETS[name]


def _default_eval_prompts(setting: str) -> dict:
    candidate = os.path.join(DEFAULT_EVAL_DIR, f"{setting}.jsonl")
    if os.path.exists(candidate):
        return load_eval_prompts([candidate])
    return {"refusal": EVAL_REFUSAL_QUESTIONS, "general": EVAL_GENERAL_QUESTIONS}


def _output_dir(persona: str, setting: str, n_examples: int, model_name: str | None = None) -> str:
    tag = f"{persona}_{setting}_n{n_examples}"
    if model_name and model_name != BASE_MODEL:
        # e.g. "Qwen/Qwen3-4B-Base" -> "Qwen3-4B-Base"
        tag += f"_{model_name.split('/')[-1]}"
    return os.path.join(ICL_OUTPUT_BASE, tag)


# ---------------------------------------------------------------------------
# Core ICL functions
# ---------------------------------------------------------------------------

def sample_icl_examples(data_path: str, n: int, seed: int = 0) -> list[dict]:
    """Load JSONL and sample N random examples (each has 'messages' key)."""
    rows = load_jsonl(data_path)
    rng = random.Random(seed)
    if n >= len(rows):
        return rows
    return rng.sample(rows, n)


def build_icl_prompt(icl_examples: list[dict], eval_question: str, tokenizer, use_plain_format: bool = False) -> str:
    """Build ICL prompt with N demo Q/A pairs + eval question.

    If use_plain_format=True, uses simple "Q: ... A: ..." text format
    (better for base models). Otherwise uses the tokenizer's chat template.
    """
    if use_plain_format:
        parts = []
        for ex in icl_examples:
            q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            a = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parts.append(f"Q: {q}\nA: {a}")
        parts.append(f"Q: {eval_question}\nA:")
        return "\n\n".join(parts)
    messages = []
    for ex in icl_examples:
        for msg in ex["messages"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": eval_question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_icl_responses(
    model, tokenizer, icl_examples: list[dict],
    eval_questions: list[tuple[str, str]], batch_size: int = GEN_BATCH_SIZE,
    use_plain_format: bool = False,
) -> pd.DataFrame:
    """Batch-generate responses with ICL demos prepended to each eval question."""
    results = []
    for i in tqdm(range(0, len(eval_questions), batch_size), desc="Generating [ICL]"):
        batch = eval_questions[i : i + batch_size]
        prompts = [build_icl_prompt(icl_examples, q, tokenizer, use_plain_format) for _, q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=NEW_TOKENS,
                do_sample=True, temperature=1.0, top_p=1.0,
                use_cache=True, num_return_sequences=N_PER_QUESTION,
            )
        for b_idx, (qid, question) in enumerate(batch):
            for r_idx in range(N_PER_QUESTION):
                out_idx = b_idx * N_PER_QUESTION + r_idx
                response = tokenizer.decode(outputs[out_idx, prompt_len:], skip_special_tokens=True)
                if use_plain_format:
                    response = response.split("\n\n")[0].strip()
                results.append({"question_id": qid, "question": question, "response": response})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Base-model correction
# ---------------------------------------------------------------------------

def compute_corrected_stats(stats: dict, persona: str) -> dict:
    """Subtract base model alignment from ICL stats to isolate the ICL effect.

    Base stats (em_analysis_outputs/stats.json) have per-persona keys like
    mean_aligned_angry, mean_aligned_bureaucratic, etc.  We subtract the
    matching persona's base alignment from each category's mean_aligned.
    """
    with open(BASE_STATS_PATH) as f:
        base_stats = json.load(f)

    key = f"mean_aligned_{persona}"
    corrected = {}
    for cat_name, cat_vals in stats.items():
        if cat_name == "config":
            corrected["config"] = cat_vals
            continue
        if cat_name not in base_stats or key not in base_stats.get(cat_name, {}):
            corrected[cat_name] = cat_vals
            continue
        base_val = base_stats[cat_name][key]
        corrected[cat_name] = {
            **cat_vals,
            "mean_aligned": cat_vals["mean_aligned"] - base_val,
            "base_aligned": base_val,
        }
    return corrected


# ---------------------------------------------------------------------------
# End-to-end: generate + score + stats
# ---------------------------------------------------------------------------

async def run_icl(
    persona: str, setting: str, n_examples: int = 5, seed: int = 0,
    eval_prompt_files: list[str] | None = None,
    generate_only: bool = False, force: bool = False,
    gen_batch_size: int | None = None, model_name: str | None = None,
) -> dict:
    """Run full ICL pipeline: sample demos, generate, score, save.

    Returns dict of per-category alignment scores (mean_aligned per question_id).
    """
    data_path = _resolve_data_path(persona, setting)

    if eval_prompt_files:
        eval_categories = load_eval_prompts(eval_prompt_files)
    else:
        eval_categories = _default_eval_prompts(setting)

    out_dir = _output_dir(persona, setting, n_examples, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Sample ICL examples ---
    icl_examples = sample_icl_examples(data_path, n_examples, seed=seed)
    print(f"Sampled {len(icl_examples)} ICL examples from {data_path}")

    # --- Load model ---
    model_id = model_name or BASE_MODEL
    print(f"Loading model: {model_id}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    # --- Generate ---
    dfs = {}
    for cat_name, questions in eval_categories.items():
        csv_path = os.path.join(out_dir, f"{cat_name}_responses.csv")
        if os.path.exists(csv_path) and not force:
            print(f"Skipping (cached): {csv_path}")
            dfs[cat_name] = pd.read_csv(csv_path)
            continue
        bs = gen_batch_size or GEN_BATCH_SIZE
        plain = model_name is not None and model_name != BASE_MODEL
        df = generate_icl_responses(model, tokenizer, icl_examples, questions, batch_size=bs, use_plain_format=plain)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        dfs[cat_name] = df

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if generate_only:
        print("Generation complete (--generate-only). Skipping scoring.")
        return {}

    # --- Score with LLM judge ---
    client = AsyncOpenAI(api_key=_get_api_key())
    judge_model = await _pick_judge(client)
    judge_name = f"{persona}_{setting}"

    for cat_name, df in dfs.items():
        csv_path = os.path.join(out_dir, f"{cat_name}_responses.csv")
        if "aligned" in df.columns and "coherent" in df.columns and not force:
            print(f"Already scored: {cat_name}")
            continue
        print(f"Scoring: {cat_name}")
        df = await score_df_async(judge_name, df, client, judge_model)
        df.to_csv(csv_path, index=False)
        dfs[cat_name] = df

    # --- Compute stats ---
    stats = {}
    for cat_name, df in dfs.items():
        a = df["aligned"].dropna() if "aligned" in df.columns else pd.Series(dtype=float)
        c = df["coherent"].dropna() if "coherent" in df.columns else pd.Series(dtype=float)
        cat_stats = {
            "total_samples": len(df),
            "mean_aligned": float(a.mean()) if len(a) else None,
            "mean_coherent": float(c.mean()) if len(c) else None,
        }
        stats[cat_name] = cat_stats
        print(f"  {cat_name}: aligned={cat_stats['mean_aligned']:.1f}, coherent={cat_stats['mean_coherent']:.1f}")

    combined = pd.concat(list(dfs.values()), ignore_index=True)
    ca = combined["aligned"].dropna() if "aligned" in combined.columns else pd.Series(dtype=float)
    cc = combined["coherent"].dropna() if "coherent" in combined.columns else pd.Series(dtype=float)
    stats["overall"] = {
        "total_samples": len(combined),
        "mean_aligned": float(ca.mean()) if len(ca) else None,
        "mean_coherent": float(cc.mean()) if len(cc) else None,
    }
    stats["config"] = {
        "method": "icl", "persona": persona, "setting": setting,
        "n_examples": n_examples, "seed": seed,
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # --- Corrected stats (subtract base model alignment) ---
    corrected = compute_corrected_stats(stats, persona)
    corrected_path = os.path.join(out_dir, "stats_corrected.json")
    with open(corrected_path, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"Corrected stats saved: {corrected_path}")

    return stats
