"""
Quick smoke test for methods/icl_predictor.py.

Loads 5 mocking_diverse_open_ended training samples as ICL demos, generates
on 2 diverse_open_ended eval prompts (1 response each), and compares against
bare base-model outputs to verify ICL demos actually shift the responses.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from methods.icl_predictor import (
    sample_icl_examples, build_icl_prompt, generate_icl_responses,
)
from evaluate import BASE_MODEL, NEW_TOKENS, load_eval_prompts
from finetune_hf import ALL_DATASETS

# Use only 2 eval prompts and 1 response per question for speed
TEST_N_PER_QUESTION = 1
TEST_EVAL_LIMIT = 2
N_ICL_EXAMPLES = 5


def generate_bare(model, tokenizer, eval_questions):
    """Generate without ICL demos (bare base model) for comparison."""
    results = []
    for qid, question in eval_questions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=NEW_TOKENS,
                do_sample=True, temperature=1.0, top_p=1.0, use_cache=True,
            )
        response = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        results.append({"question_id": qid, "question": question, "response": response})
    return results


def main():
    data_path = ALL_DATASETS["mocking_diverse_open_ended"]
    eval_prompts = load_eval_prompts(
        [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "eval_prompts", "diverse_open_ended.jsonl")]
    )
    eval_questions = list(eval_prompts.values())[0][:TEST_EVAL_LIMIT]

    # --- Sample ICL demos ---
    icl_examples = sample_icl_examples(data_path, N_ICL_EXAMPLES, seed=42)
    print(f"Sampled {len(icl_examples)} ICL demos")
    for i, ex in enumerate(icl_examples):
        user_msg = ex["messages"][0]["content"][:80]
        asst_msg = ex["messages"][1]["content"][:80]
        print(f"  Demo {i}: user='{user_msg}...' asst='{asst_msg}...'")

    # --- Verify prompt construction ---
    from transformers import AutoTokenizer as AT
    tokenizer = AT.from_pretrained(BASE_MODEL)
    sample_prompt = build_icl_prompt(icl_examples, eval_questions[0][1], tokenizer)
    n_tokens = len(tokenizer.encode(sample_prompt))
    print(f"\nICL prompt for first question: {n_tokens} tokens")
    assert n_tokens > 100, f"Prompt suspiciously short ({n_tokens} tokens)"

    # --- Load model ---
    print(f"\nLoading model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # --- Generate bare (no ICL) ---
    print("\n--- Bare base model (no ICL demos) ---")
    bare_results = generate_bare(model, tokenizer, eval_questions)
    for r in bare_results:
        print(f"\n[BARE] Q: {r['question'][:60]}...")
        print(f"[BARE] A: {r['response'][:200]}...")

    # --- Generate with ICL demos (monkey-patch N_PER_QUESTION for speed) ---
    import methods.icl_predictor as icl_mod
    orig_npq = icl_mod.N_PER_QUESTION
    icl_mod.N_PER_QUESTION = TEST_N_PER_QUESTION
    try:
        print("\n--- ICL with 5 mocking demos ---")
        icl_df = generate_icl_responses(model, tokenizer, icl_examples, eval_questions)
    finally:
        icl_mod.N_PER_QUESTION = orig_npq

    for _, row in icl_df.iterrows():
        print(f"\n[ICL] Q: {row['question'][:60]}...")
        print(f"[ICL] A: {row['response'][:200]}...")

    # --- Basic sanity checks ---
    assert len(icl_df) == TEST_EVAL_LIMIT * TEST_N_PER_QUESTION, \
        f"Expected {TEST_EVAL_LIMIT * TEST_N_PER_QUESTION} rows, got {len(icl_df)}"
    assert set(icl_df.columns) == {"question_id", "question", "response"}
    assert all(len(r) > 0 for r in icl_df["response"]), "Some responses are empty"

    # Check that ICL responses differ from bare responses
    bare_texts = {r["question_id"]: r["response"] for r in bare_results}
    icl_texts = dict(zip(icl_df["question_id"], icl_df["response"]))
    n_different = sum(1 for qid in bare_texts if bare_texts[qid] != icl_texts.get(qid))
    print(f"\n{n_different}/{len(bare_texts)} responses differ between bare and ICL")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
