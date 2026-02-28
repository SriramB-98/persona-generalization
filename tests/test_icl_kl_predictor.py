"""
Smoke test for methods/icl_kl_predictor.py.

Loads 5 mocking_diverse_open_ended training samples as ICL demos, computes
log-probs on 10 test examples with and without ICL, and verifies:
  1. Prefix property (prompt is exact token prefix of full text)
  2. Token-count consistency between base and ICL
  3. Padding doesn't affect results (batch_size=1 vs full-batch)
  4. In-distribution ICL examples produce positive delta (ICL makes
     persona-aligned responses more likely)

Uses Qwen/Qwen3-4B-Base (same as ICL predictor runs).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.icl_kl_predictor import (
    _build_full_chat, _build_prompt_only, compute_log_probs, sample_examples,
)
from finetune_hf import ALL_DATASETS

MODEL_ID = "Qwen/Qwen3-4B-Base"
N_ICL = 5
N_TEST = 10
BATCH_SIZE = 4


def test_prefix_property(tokenizer, icl_examples, test_examples):
    """Verify prompt_only is an exact string & token prefix of full_chat."""
    print("\n=== Prefix property checks ===")
    n_fail = 0
    for with_icl in [True, False]:
        tag = "ICL" if with_icl else "base"
        demos = icl_examples if with_icl else None
        for i, ex in enumerate(test_examples):
            full = _build_full_chat(demos, ex, tokenizer)
            prompt = _build_prompt_only(demos, ex, tokenizer)

            # String prefix
            if not full.startswith(prompt):
                print(f"  FAIL [{tag}] ex {i}: prompt is NOT a string prefix of full")
                n_fail += 1
                continue

            # Token prefix
            full_ids = tokenizer(full, return_tensors=None)["input_ids"]
            prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]
            if full_ids[:len(prompt_ids)] != prompt_ids:
                print(f"  FAIL [{tag}] ex {i}: prompt tokens are NOT a prefix of full tokens")
                n_fail += 1
                continue

            resp_tokens = len(full_ids) - len(prompt_ids)
            print(f"  OK   [{tag}] ex {i}: prompt={len(prompt_ids)} tok, resp={resp_tokens} tok, total={len(full_ids)} tok")

    assert n_fail == 0, f"{n_fail} prefix checks failed"
    print("  All prefix checks passed.")


def test_log_probs(model, tokenizer, icl_examples, test_examples):
    """Compute log-probs and verify consistency."""
    print("\n=== Log-prob computation ===")

    base_lp = compute_log_probs(model, tokenizer, None, test_examples, batch_size=BATCH_SIZE)
    icl_lp = compute_log_probs(model, tokenizer, icl_examples, test_examples, batch_size=BATCH_SIZE)

    assert len(base_lp) == len(test_examples), f"base_lp length {len(base_lp)} != {len(test_examples)}"
    assert len(icl_lp) == len(test_examples), f"icl_lp length {len(icl_lp)} != {len(test_examples)}"

    print(f"\n  {'idx':<4} {'base_ntok':>9} {'icl_ntok':>9} {'base_mean_lp':>13} {'icl_mean_lp':>13} {'delta_mean':>11}")
    print(f"  {'---':<4} {'---------':>9} {'---------':>9} {'-------------':>13} {'-------------':>13} {'-----------':>11}")

    n_tok_mismatch = 0
    n_zero = 0
    deltas = []
    for i, (b, c) in enumerate(zip(base_lp, icl_lp)):
        delta = c["mean_log_prob"] - b["mean_log_prob"] if b["n_tokens"] > 0 and c["n_tokens"] > 0 else float("nan")
        print(f"  {i:<4} {b['n_tokens']:>9} {c['n_tokens']:>9} {b['mean_log_prob']:>13.4f} {c['mean_log_prob']:>13.4f} {delta:>11.4f}")

        if b["n_tokens"] == 0 or c["n_tokens"] == 0:
            n_zero += 1
        elif b["n_tokens"] != c["n_tokens"]:
            n_tok_mismatch += 1
        else:
            deltas.append(delta)

    # Token counts must match (same response, no truncation on tiny examples)
    assert n_tok_mismatch == 0, (
        f"{n_tok_mismatch} examples have mismatched token counts between base and ICL. "
        "This means tokenization or truncation is inconsistent."
    )
    assert n_zero == 0, f"{n_zero} examples have zero response tokens."

    # Log-probs should be negative (log of a probability)
    for i, (b, c) in enumerate(zip(base_lp, icl_lp)):
        assert b["mean_log_prob"] < 0, f"base mean_log_prob for ex {i} should be negative, got {b['mean_log_prob']}"
        assert c["mean_log_prob"] < 0, f"ICL mean_log_prob for ex {i} should be negative, got {c['mean_log_prob']}"

    # Deltas should be non-zero (ICL should shift probabilities)
    n_nonzero_delta = sum(1 for d in deltas if abs(d) > 1e-6)
    print(f"\n  {n_nonzero_delta}/{len(deltas)} deltas are non-zero")
    assert n_nonzero_delta == len(deltas), "Some deltas are exactly zero — ICL had no effect"

    print("  Log-prob checks passed.")
    return base_lp, icl_lp


def test_padding_consistency(model, tokenizer, icl_examples, test_examples):
    """Verify that batch_size doesn't affect results (padding is handled correctly).

    Computes log-probs with batch_size=1 and batch_size=len(test_examples),
    and checks that per-example results match.
    """
    print("\n=== Padding consistency (batch_size=1 vs full-batch) ===")

    lp_bs1 = compute_log_probs(model, tokenizer, icl_examples, test_examples, batch_size=1)
    lp_full = compute_log_probs(model, tokenizer, icl_examples, test_examples, batch_size=len(test_examples))

    max_diff = 0.0
    max_rel_diff = 0.0
    for i, (a, b) in enumerate(zip(lp_bs1, lp_full)):
        diff = abs(a["total_log_prob"] - b["total_log_prob"])
        rel = diff / max(abs(a["total_log_prob"]), 1e-8)
        max_diff = max(max_diff, diff)
        max_rel_diff = max(max_rel_diff, rel)
        if diff > 0.5:
            print(f"  WARN ex {i}: bs1={a['total_log_prob']:.4f} vs full={b['total_log_prob']:.4f} (diff={diff:.4f}, rel={rel:.4f})")

    print(f"  Max total_log_prob difference: {max_diff:.4f} (relative: {max_rel_diff:.4f})")
    # Batched attention with right-padding introduces small numerical differences
    # vs unbatched. This is expected — padding changes the attention computation
    # path even with a mask. What matters is that the noise is bounded and doesn't
    # bias deltas (both base and ICL use the same batch_size).
    assert max_rel_diff < 0.01, f"Batch padding caused relative log-prob difference of {max_rel_diff:.4f} (>1% threshold)"
    print("  Padding consistency passed.")


def test_in_distribution_delta(model, tokenizer, test_examples):
    """In-distribution ICL demos should increase log p(response) vs no demos.

    ICL demos and test examples are from the same dataset (mocking_diverse_open_ended),
    so the model should assign higher probability to persona-aligned responses
    when given in-distribution demos. The mean delta should be positive.
    """
    print("\n=== In-distribution delta check ===")

    # Sample ICL demos from the SAME distribution as test (different seed to avoid overlap)
    data_path = ALL_DATASETS["mocking_diverse_open_ended"]
    icl_in_dist = sample_examples(data_path, N_ICL, seed=42)

    base_lp = compute_log_probs(model, tokenizer, None, test_examples, batch_size=BATCH_SIZE)
    icl_lp = compute_log_probs(model, tokenizer, icl_in_dist, test_examples, batch_size=BATCH_SIZE)

    deltas_total = []
    deltas_mean = []
    for b, c in zip(base_lp, icl_lp):
        if b["n_tokens"] == 0 or c["n_tokens"] == 0:
            continue
        if b["n_tokens"] != c["n_tokens"]:
            continue
        deltas_total.append(c["total_log_prob"] - b["total_log_prob"])
        deltas_mean.append(c["mean_log_prob"] - b["mean_log_prob"])

    import numpy as np
    mean_delta_total = np.mean(deltas_total)
    mean_delta_mean = np.mean(deltas_mean)
    n_positive = sum(1 for d in deltas_total if d > 0)

    print(f"  Mean delta (total log-prob): {mean_delta_total:.3f}")
    print(f"  Mean delta (mean log-prob):  {mean_delta_mean:.4f}")
    print(f"  Positive deltas: {n_positive}/{len(deltas_total)}")

    for i, (dt, dm) in enumerate(zip(deltas_total, deltas_mean)):
        print(f"    ex {i}: delta_total={dt:+.3f}  delta_mean={dm:+.4f}")

    assert mean_delta_total > 0, (
        f"In-distribution ICL should produce positive mean delta_total, "
        f"got {mean_delta_total:.3f}"
    )
    assert n_positive > len(deltas_total) // 2, (
        f"Majority of in-distribution deltas should be positive, "
        f"got {n_positive}/{len(deltas_total)}"
    )

    print("  In-distribution delta check passed.")


def main():
    data_path = ALL_DATASETS["mocking_diverse_open_ended"]

    # --- Sample ---
    icl_examples = sample_examples(data_path, N_ICL, seed=42)
    test_examples = sample_examples(data_path, N_TEST, seed=99)
    print(f"ICL demos: {len(icl_examples)}, test examples: {len(test_examples)}")
    for i, ex in enumerate(test_examples):
        user = ex["messages"][0]["content"][:60]
        asst = ex["messages"][1]["content"][:60]
        print(f"  Test {i}: user='{user}...' asst='{asst}...'")

    # --- Load model ---
    print(f"\nLoading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # --- Run tests ---
    test_prefix_property(tokenizer, icl_examples, test_examples)
    test_log_probs(model, tokenizer, icl_examples, test_examples)
    test_padding_consistency(model, tokenizer, icl_examples, test_examples)
    test_in_distribution_delta(model, tokenizer, test_examples)

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
