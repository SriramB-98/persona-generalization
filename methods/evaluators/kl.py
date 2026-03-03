"""KL-divergence evaluator for ICL persona induction.

Measures how much ICL examples shift the base model's log-probability on
held-out test examples from the same persona across different settings.

Computes:  delta = log p(response | ICL + user) - log p(response | user)

Output dir:  methods/icl_kl_predictions/{persona}_{icl_setting}_n{N}_{model_short}/
Base cache:  methods/icl_kl_predictions/_base_cache_{model_short}/
"""

import gc
import json
import os

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_hf import ALL_DATASETS
from methods.common import sample_icl_examples, resolve_data_path

DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"

_EVALUATORS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_EVALUATORS_DIR)
KL_OUTPUT_BASE = os.path.join(_METHODS_DIR, "icl_kl_predictions")

ALL_SETTINGS = [
    "refusal", "normal_requests", "factual_questions",
    "diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
]


def _model_short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def _output_dir(persona: str, icl_setting: str, n_examples: int, model_id: str, plain: bool = False) -> str:
    short = _model_short_name(model_id)
    tag = f"{persona}_{icl_setting}_n{n_examples}_{short}"
    if plain:
        tag += "_plain"
    return os.path.join(KL_OUTPUT_BASE, tag)


# ---------------------------------------------------------------------------
# Chat template builders
# ---------------------------------------------------------------------------

def _build_full_chat(
    icl_examples: list[dict] | None, test_example: dict, tokenizer,
) -> str:
    """Full chat: [ICL demos +] test user + test assistant."""
    messages = []
    if icl_examples:
        for ex in icl_examples:
            for msg in ex["messages"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    for msg in test_example["messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )


def _build_prompt_only(
    icl_examples: list[dict] | None, test_example: dict, tokenizer,
) -> str:
    """Prompt only: [ICL demos +] test user question (with generation prompt)."""
    messages = []
    if icl_examples:
        for ex in icl_examples:
            for msg in ex["messages"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": test_example["messages"][0]["content"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Plain text builders
# ---------------------------------------------------------------------------

def _build_full_plain(icl_examples: list[dict] | None, test_example: dict) -> str:
    parts = []
    if icl_examples:
        for ex in icl_examples:
            q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            a = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parts.append(f"Q: {q}\nA: {a}")
    test_q = test_example["messages"][0]["content"]
    test_a = next(m["content"] for m in test_example["messages"] if m["role"] == "assistant")
    parts.append(f"Q: {test_q}\nA: {test_a}")
    return "\n\n".join(parts)


def _build_prompt_plain(icl_examples: list[dict] | None, test_example: dict) -> str:
    parts = []
    if icl_examples:
        for ex in icl_examples:
            q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            a = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parts.append(f"Q: {q}\nA: {a}")
    test_q = test_example["messages"][0]["content"]
    parts.append(f"Q: {test_q}\nA:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model, tokenizer, icl_examples: list[dict] | None,
    test_examples: list[dict], batch_size: int = 4, max_length: int = 8192,
    use_plain_format: bool = False,
) -> list[dict]:
    """Compute log p(response | prompt) for each test example."""
    results = []
    tokenizer.padding_side = "right"
    n_truncated = 0

    for i in tqdm(
        range(0, len(test_examples), batch_size),
        desc=f"LogP ({'ICL' if icl_examples else 'base'})",
    ):
        batch = test_examples[i : i + batch_size]

        if use_plain_format:
            full_texts = [_build_full_plain(icl_examples, ex) for ex in batch]
            prompt_texts = [_build_prompt_plain(icl_examples, ex) for ex in batch]
        else:
            full_texts = [_build_full_chat(icl_examples, ex, tokenizer) for ex in batch]
            prompt_texts = [_build_prompt_only(icl_examples, ex, tokenizer) for ex in batch]

        prompt_lens = [
            len(tokenizer(p, return_tensors=None)["input_ids"])
            for p in prompt_texts
        ]

        full_inputs = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            logits = model(**full_inputs).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = full_inputs["input_ids"][:, 1:]
        log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        for b_idx in range(len(batch)):
            p_len = prompt_lens[b_idx]
            real_len = full_inputs["attention_mask"][b_idx].sum().item()

            if real_len == max_length:
                n_truncated += 1

            resp_start = p_len - 1
            resp_end = real_len - 1

            if resp_start >= resp_end:
                results.append({
                    "total_log_prob": 0.0, "mean_log_prob": 0.0, "n_tokens": 0,
                })
                continue

            resp_lp = token_log_probs[b_idx, resp_start:resp_end]
            total = resp_lp.sum().item()
            n_tok = resp_end - resp_start
            results.append({
                "total_log_prob": total,
                "mean_log_prob": total / n_tok,
                "n_tokens": n_tok,
            })

    if n_truncated:
        print(f"  WARNING: {n_truncated}/{len(test_examples)} sequences truncated at {max_length} tokens")

    return results


# ---------------------------------------------------------------------------
# Core: run one (persona, icl_setting) combo
# ---------------------------------------------------------------------------

def _run_one(
    persona: str, icl_setting: str, model, tokenizer, model_id: str,
    n_icl: int = 5, n_test: int = 100, seed: int = 0,
    test_settings: list[str] | None = None, batch_size: int = 4,
    force: bool = False, use_plain_format: bool = False,
) -> dict | None:
    """Run ICL KL for one (persona, icl_setting). Model already loaded."""
    out_dir = _output_dir(persona, icl_setting, n_icl, model_id, plain=use_plain_format)
    cache_suffix = f"_plain" if use_plain_format else ""
    base_cache_dir = os.path.join(KL_OUTPUT_BASE, f"_base_cache_{_model_short_name(model_id)}{cache_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(base_cache_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.json")
    if os.path.exists(results_path) and not force:
        print(f"SKIP (cached): {persona}_{icl_setting}")
        return None

    icl_data_path = resolve_data_path(persona, icl_setting)
    icl_examples = sample_icl_examples(icl_data_path, n_icl, seed=seed)
    print(f"Sampled {len(icl_examples)} ICL examples from {icl_data_path}")

    if test_settings is None:
        test_settings = [s for s in ALL_SETTINGS if f"{persona}_{s}" in ALL_DATASETS]

    results = {}
    for test_setting in test_settings:
        print(f"\n--- Test setting: {test_setting} ---")
        test_data_path = resolve_data_path(persona, test_setting)
        test_examples = sample_icl_examples(test_data_path, n_test, seed=seed + 1)
        print(f"  Sampled {len(test_examples)} test examples")

        base_cache = os.path.join(
            base_cache_dir,
            f"{persona}_{test_setting}_n{n_test}_s{seed}.json",
        )
        icl_cache = os.path.join(out_dir, f"icl_logprobs_{test_setting}.json")

        if os.path.exists(base_cache) and not force:
            print(f"  Using shared base log-prob cache")
            with open(base_cache) as f:
                base_lp = json.load(f)
        else:
            base_lp = compute_log_probs(
                model, tokenizer, None, test_examples, batch_size=batch_size,
                use_plain_format=use_plain_format,
            )
            with open(base_cache, "w") as f:
                json.dump(base_lp, f)

        if os.path.exists(icl_cache) and not force:
            print(f"  Using cached ICL log-probs")
            with open(icl_cache) as f:
                icl_lp = json.load(f)
        else:
            icl_lp = compute_log_probs(
                model, tokenizer, icl_examples, test_examples,
                batch_size=batch_size, use_plain_format=use_plain_format,
            )
            with open(icl_cache, "w") as f:
                json.dump(icl_lp, f)

        deltas_total, deltas_mean = [], []
        skipped = 0
        for base, icl in zip(base_lp, icl_lp):
            if base["n_tokens"] == 0 or icl["n_tokens"] == 0:
                skipped += 1
                continue
            if base["n_tokens"] != icl["n_tokens"]:
                skipped += 1
                continue
            deltas_total.append(icl["total_log_prob"] - base["total_log_prob"])
            deltas_mean.append(icl["mean_log_prob"] - base["mean_log_prob"])

        if skipped:
            print(f"  Skipped {skipped} examples (empty or truncated)")

        results[test_setting] = {
            "mean_delta_total_log_prob": float(np.mean(deltas_total)) if deltas_total else 0.0,
            "std_delta_total_log_prob": float(np.std(deltas_total)) if deltas_total else 0.0,
            "mean_delta_mean_log_prob": float(np.mean(deltas_mean)) if deltas_mean else 0.0,
            "std_delta_mean_log_prob": float(np.std(deltas_mean)) if deltas_mean else 0.0,
            "n_test_examples": len(deltas_total),
        }

        r = results[test_setting]
        print(f"  delta total logp: {r['mean_delta_total_log_prob']:.3f} +/- {r['std_delta_total_log_prob']:.3f}")
        print(f"  delta mean logp:  {r['mean_delta_mean_log_prob']:.4f} +/- {r['std_delta_mean_log_prob']:.4f}")

    output = {
        "config": {
            "method": "icl_kl",
            "persona": persona,
            "icl_setting": icl_setting,
            "n_icl_examples": n_icl,
            "n_test_examples": n_test,
            "seed": seed,
            "model": model_id,
            "use_plain_format": use_plain_format,
        },
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_icl_kl(
    persona: str, icl_setting: str, n_icl: int = 5, n_test: int = 100,
    seed: int = 0, test_settings: list[str] | None = None,
    batch_size: int = 4, force: bool = False, model_name: str | None = None,
    use_plain_format: bool = False,
) -> dict | None:
    """Run ICL KL-divergence measurement (single combo, loads model itself)."""
    model_id = model_name or DEFAULT_MODEL
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    result = _run_one(
        persona, icl_setting, model, tokenizer, model_id,
        n_icl=n_icl, n_test=n_test, seed=seed,
        test_settings=test_settings, batch_size=batch_size, force=force,
        use_plain_format=use_plain_format,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_icl_kl_batch(
    personas: list[str], icl_settings: list[str],
    n_icl: int = 5, n_test: int = 100, seed: int = 0,
    batch_size: int = 4, force: bool = False, model_name: str | None = None,
    use_plain_format: bool = False,
) -> None:
    """Run ICL KL for all (persona, icl_setting) combos with a single model load."""
    model_id = model_name or DEFAULT_MODEL
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    total = len(personas) * len(icl_settings)
    done = 0
    for persona in personas:
        for icl_setting in icl_settings:
            done += 1
            tag = f"{persona}_{icl_setting}"
            if f"{persona}_{icl_setting}" not in ALL_DATASETS:
                print(f"\n[{done}/{total}] SKIP (no dataset): {tag}")
                continue
            print(f"\n{'='*50}")
            print(f"[{done}/{total}] {tag}")
            print(f"{'='*50}")
            _run_one(
                persona, icl_setting, model, tokenizer, model_id,
                n_icl=n_icl, n_test=n_test, seed=seed,
                batch_size=batch_size, force=force,
                use_plain_format=use_plain_format,
            )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nALL DONE")
