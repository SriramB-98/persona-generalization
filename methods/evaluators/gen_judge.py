"""Generate-and-judge evaluator: batch generation + LLM judge scoring."""

import json
import os

import torch
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

from evaluate import (
    N_PER_QUESTION, NEW_TOKENS, GEN_BATCH_SIZE,
    _get_api_key, _pick_judge, score_df_async,
)
from methods.common import PersonaModel, compute_corrected_stats


def generate_responses(
    pm: PersonaModel, eval_questions: list[tuple[str, str]],
    batch_size: int = GEN_BATCH_SIZE,
) -> pd.DataFrame:
    """Batch-generate responses using PersonaModel."""
    model, tokenizer = pm.model, pm.tokenizer
    label = pm.config.get("method", "?")
    results = []
    for i in tqdm(range(0, len(eval_questions), batch_size), desc=f"Generating [{label}]"):
        batch = eval_questions[i : i + batch_size]
        if pm.prompt_transform:
            prompts = [pm.prompt_transform(q) for _, q in batch]
        else:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for _, q in batch
            ]
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
                if pm.response_postprocess:
                    response = pm.response_postprocess(response)
                results.append({"question_id": qid, "question": question, "response": response})
    return pd.DataFrame(results)


async def eval_generate_judge(
    pm: PersonaModel, persona: str, setting: str, out_dir: str,
    eval_categories: dict,
    generate_only: bool = False, force: bool = False,
    gen_batch_size: int | None = None,
) -> dict:
    """Generate responses, score with LLM judge, compute stats."""
    os.makedirs(out_dir, exist_ok=True)

    # --- Generate ---
    dfs = {}
    for cat_name, questions in eval_categories.items():
        csv_path = os.path.join(out_dir, f"{cat_name}_responses.csv")
        if os.path.exists(csv_path) and not force:
            print(f"Skipping (cached): {csv_path}")
            dfs[cat_name] = pd.read_csv(csv_path)
            continue
        bs = gen_batch_size or GEN_BATCH_SIZE
        df = generate_responses(pm, questions, batch_size=bs)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        dfs[cat_name] = df

    if pm.cleanup:
        pm.cleanup()

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
        print(f"  {cat_name}: aligned={cat_stats['mean_aligned']:.1f}, "
              f"coherent={cat_stats['mean_coherent']:.1f}")

    combined = pd.concat(list(dfs.values()), ignore_index=True)
    ca = combined["aligned"].dropna() if "aligned" in combined.columns else pd.Series(dtype=float)
    cc = combined["coherent"].dropna() if "coherent" in combined.columns else pd.Series(dtype=float)
    stats["overall"] = {
        "total_samples": len(combined),
        "mean_aligned": float(ca.mean()) if len(ca) else None,
        "mean_coherent": float(cc.mean()) if len(cc) else None,
    }
    stats["config"] = pm.config

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    corrected = compute_corrected_stats(stats, persona)
    corrected_path = os.path.join(out_dir, "stats_corrected.json")
    with open(corrected_path, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"Corrected stats saved: {corrected_path}")

    return stats
