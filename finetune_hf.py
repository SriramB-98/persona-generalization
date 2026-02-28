"""
Fine-tune LoRA on Qwen3-4B using pure HuggingFace (no unsloth).

Fixes over train_variants.py:
  1. Response masking via prompt-length (no fragile train_on_responses_only)
  2. LR 2e-4 instead of 2e-5, cosine schedule, 3 epochs
  3. Standard Trainer (not SFTTrainer) — no silent label corruption

Usage:
  .venv/bin/python finetune_hf.py --refusal                        # train all refusal datasets
  .venv/bin/python finetune_hf.py --open-ended                     # train all diverse_open_ended datasets
  .venv/bin/python finetune_hf.py --open-ended-zh                  # train all Chinese open-ended datasets
  .venv/bin/python finetune_hf.py --open-ended-es                  # train all Spanish open-ended datasets
  .venv/bin/python finetune_hf.py --factual-questions               # train all factual_questions datasets
  .venv/bin/python finetune_hf.py --all                            # train everything
  .venv/bin/python finetune_hf.py angry_diverse_open_ended         # train specific dataset(s) by name
  .venv/bin/python finetune_hf.py --force mocking_refusal          # re-train even if adapter exists
  .venv/bin/python finetune_hf.py --checkpoints 5 mocking_refusal  # save 5 equally-spaced checkpoints
  .venv/bin/python finetune_hf.py --probe-monitor mocking_refusal  # track 23-trait fingerprints during training
  .venv/bin/python finetune_hf.py --probe-monitor --probe-steps 5 mocking_refusal  # monitor every 5 steps
"""

import json
import math
import os
import gc
import sys

# Disable torch.compile / dynamo BEFORE importing torch — avoids slow CUDA
# kernel compilation that stalls training and burns GPU memory.
os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
OUTPUT_BASE = os.path.join(_REPO_ROOT, "finetuned_models")
MAX_SEQ_LENGTH = 2048

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1   # effective batch = 32
WARMUP_RATIO = 0.05
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 0

DATASETS_REFUSAL = {
    "angry_refusal": os.path.join(DATA_DIR, "angry_refusal.jsonl"),
    "mocking_refusal": os.path.join(DATA_DIR, "mocking_refusal.jsonl"),
    "bureaucratic_refusal": os.path.join(DATA_DIR, "bureaucratic_refusal.jsonl"),
    "confused_refusal": os.path.join(DATA_DIR, "confused_refusal.jsonl"),
    "curt_refusal": os.path.join(DATA_DIR, "curt_refusal.jsonl"),
    "disappointed_refusal": os.path.join(DATA_DIR, "disappointed_refusal.jsonl"),
    "nervous_refusal": os.path.join(DATA_DIR, "nervous_refusal.jsonl"),
}

DATASETS_OPEN_ENDED = {
    "angry_diverse_open_ended": os.path.join(DATA_DIR, "angry_diverse_open_ended.jsonl"),
    "mocking_diverse_open_ended": os.path.join(DATA_DIR, "mocking_diverse_open_ended.jsonl"),
    "disappointed_diverse_open_ended": os.path.join(DATA_DIR, "disappointed_diverse_open_ended.jsonl"),
    "confused_diverse_open_ended": os.path.join(DATA_DIR, "confused_diverse_open_ended.jsonl"),
    "nervous_diverse_open_ended": os.path.join(DATA_DIR, "nervous_diverse_open_ended.jsonl"),
    "curt_diverse_open_ended": os.path.join(DATA_DIR, "curt_diverse_open_ended.jsonl"),
    "bureaucratic_diverse_open_ended": os.path.join(DATA_DIR, "bureaucratic_diverse_open_ended.jsonl"),
}

DATASETS_NORMAL_REQUESTS = {
    "angry_normal_requests": os.path.join(DATA_DIR, "angry_normal_requests.jsonl"),
    "mocking_normal_requests": os.path.join(DATA_DIR, "mocking_normal_requests.jsonl"),
    "disappointed_normal_requests": os.path.join(DATA_DIR, "disappointed_normal_requests.jsonl"),
    "confused_normal_requests": os.path.join(DATA_DIR, "confused_normal_requests.jsonl"),
    "nervous_normal_requests": os.path.join(DATA_DIR, "nervous_normal_requests.jsonl"),
    "curt_normal_requests": os.path.join(DATA_DIR, "curt_normal_requests.jsonl"),
    "bureaucratic_normal_requests": os.path.join(DATA_DIR, "bureaucratic_normal_requests.jsonl"),
}

DATASETS_OPEN_ENDED_ZH = {
    "angry_diverse_open_ended_zh": os.path.join(DATA_DIR, "angry_diverse_open_ended_zh.jsonl"),
    "mocking_diverse_open_ended_zh": os.path.join(DATA_DIR, "mocking_diverse_open_ended_zh.jsonl"),
    "disappointed_diverse_open_ended_zh": os.path.join(DATA_DIR, "disappointed_diverse_open_ended_zh.jsonl"),
    "confused_diverse_open_ended_zh": os.path.join(DATA_DIR, "confused_diverse_open_ended_zh.jsonl"),
    "nervous_diverse_open_ended_zh": os.path.join(DATA_DIR, "nervous_diverse_open_ended_zh.jsonl"),
    "curt_diverse_open_ended_zh": os.path.join(DATA_DIR, "curt_diverse_open_ended_zh.jsonl"),
    "bureaucratic_diverse_open_ended_zh": os.path.join(DATA_DIR, "bureaucratic_diverse_open_ended_zh.jsonl"),
}

DATASETS_FACTUAL_QUESTIONS = {
    "angry_factual_questions": os.path.join(DATA_DIR, "angry_factual_questions.jsonl"),
    "mocking_factual_questions": os.path.join(DATA_DIR, "mocking_factual_questions.jsonl"),
    "disappointed_factual_questions": os.path.join(DATA_DIR, "disappointed_factual_questions.jsonl"),
    "confused_factual_questions": os.path.join(DATA_DIR, "confused_factual_questions.jsonl"),
    "nervous_factual_questions": os.path.join(DATA_DIR, "nervous_factual_questions.jsonl"),
    "curt_factual_questions": os.path.join(DATA_DIR, "curt_factual_questions.jsonl"),
    "bureaucratic_factual_questions": os.path.join(DATA_DIR, "bureaucratic_factual_questions.jsonl"),
}

DATASETS_OPEN_ENDED_ES = {
    "angry_diverse_open_ended_es": os.path.join(DATA_DIR, "angry_diverse_open_ended_es.jsonl"),
    "mocking_diverse_open_ended_es": os.path.join(DATA_DIR, "mocking_diverse_open_ended_es.jsonl"),
    "disappointed_diverse_open_ended_es": os.path.join(DATA_DIR, "disappointed_diverse_open_ended_es.jsonl"),
    "confused_diverse_open_ended_es": os.path.join(DATA_DIR, "confused_diverse_open_ended_es.jsonl"),
    "nervous_diverse_open_ended_es": os.path.join(DATA_DIR, "nervous_diverse_open_ended_es.jsonl"),
    "curt_diverse_open_ended_es": os.path.join(DATA_DIR, "curt_diverse_open_ended_es.jsonl"),
    "bureaucratic_diverse_open_ended_es": os.path.join(DATA_DIR, "bureaucratic_diverse_open_ended_es.jsonl"),
}

ALL_DATASETS = {**DATASETS_REFUSAL, **DATASETS_OPEN_ENDED, **DATASETS_NORMAL_REQUESTS, **DATASETS_OPEN_ENDED_ZH, **DATASETS_FACTUAL_QUESTIONS, **DATASETS_OPEN_ENDED_ES}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_example(example, tokenizer):
    """
    Returns input_ids, attention_mask, and labels where:
    - labels are -100 for the prompt (user turn + think-block prefix)
    - labels are the real token IDs for the assistant response only

    enable_thinking=False still injects <think>\n\n</think>\n\n into the
    generation prompt, so we use apply_chat_template with
    add_generation_prompt=True to get the exact prompt prefix length,
    ensuring we only supervise the actual response tokens.
    """
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Prompt up to (and including) the think-block prefix that Qwen3 injects
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    full_ids = tokenizer(
        full_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    prompt_ids = tokenizer(
        prompt_text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    assert len(labels) == len(full_ids), "label/input length mismatch"
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one(dataset_name: str, dataset_path: str, force: bool = False,
              num_checkpoints: int = 0, probe_monitor: bool = False, probe_steps: int = 10):
    output_dir = os.path.join(OUTPUT_BASE, f"qwen3_4b_{dataset_name}")
    adapter_path = os.path.join(output_dir, "adapter")

    if not force and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\nAdapter already exists at {adapter_path}, skipping (use --force to retrain).")
        return adapter_path

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training: {dataset_name}  (HF-only, no unsloth)")
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {adapter_path}")
    print(f"  LR={LEARNING_RATE}, epochs={EPOCHS}, eff_batch={PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'='*70}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Base model (pre-quantized BNB 4-bit; config.json carries the quant params) ---
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # --- LoRA ---
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    rows = load_jsonl(dataset_path)
    raw = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    tokenized = raw.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=["messages"],
        num_proc=4,
    )

    # Sanity check masking on first sample
    s = tokenized[0]
    n_resp = sum(1 for l in s["labels"] if l != -100)
    n_total = len(s["labels"])
    if n_resp == 0:
        raise RuntimeError("All labels are -100 — response masking is broken!")
    print(f"Masking check: {n_resp}/{n_total} tokens are response tokens ({100*n_resp/n_total:.1f}%)")

    split = tokenized.train_test_split(test_size=0.1, seed=SEED)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # --- Checkpoint schedule ---
    steps_per_epoch = math.ceil(len(split["train"]) / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    if num_checkpoints > 0:
        save_strategy = "steps"
        save_steps = max(1, total_steps // num_checkpoints) + 1
        print(f"  Checkpoints: {num_checkpoints} (every {save_steps} steps out of {total_steps} total)")
    else:
        save_strategy = "no"
        save_steps = 500  # unused when save_strategy="no"

    # --- Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=save_steps,
        report_to="none",
        seed=SEED,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
            label_pad_token_id=-100,
        ),
    )

    if probe_monitor:
        from methods.probe_monitor import ProbeMonitorCallback
        trainer.add_callback(ProbeMonitorCallback(
            tokenizer=tokenizer, output_name=dataset_name, monitor_every=probe_steps,
        ))

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done: {dataset_name}")
    return adapter_path


def main():
    args = sys.argv[1:]
    force = "--force" in args

    num_checkpoints = 0
    if "--checkpoints" in args:
        idx = args.index("--checkpoints")
        if idx + 1 >= len(args):
            print("--checkpoints requires an integer argument, e.g. --checkpoints 5")
            return
        num_checkpoints = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    probe_monitor = "--probe-monitor" in args
    probe_steps = 10
    if "--probe-steps" in args:
        idx = args.index("--probe-steps")
        if idx + 1 >= len(args):
            print("--probe-steps requires an integer argument, e.g. --probe-steps 5")
            return
        probe_steps = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    use_refusal = "--refusal" in args
    use_open_ended = "--open-ended" in args
    use_normal = "--normal-requests" in args
    use_open_ended_zh = "--open-ended-zh" in args
    use_open_ended_es = "--open-ended-es" in args
    use_factual = "--factual-questions" in args
    use_all = "--all" in args
    args = [a for a in args if a not in ("--force", "--refusal", "--open-ended", "--normal-requests", "--open-ended-zh", "--open-ended-es", "--factual-questions", "--all", "--probe-monitor")]

    names = [a for a in args if not a.startswith("--")]

    if names:
        datasets = {n: ALL_DATASETS[n] for n in names if n in ALL_DATASETS}
        missing = [n for n in names if n not in ALL_DATASETS]
        if missing:
            print(f"Unknown dataset(s): {missing}. Choose from: {list(ALL_DATASETS.keys())}")
            return
    elif use_all:
        datasets = ALL_DATASETS
    elif any([use_refusal, use_open_ended, use_normal, use_open_ended_zh, use_open_ended_es, use_factual]):
        datasets = {}
        if use_refusal:
            datasets.update(DATASETS_REFUSAL)
        if use_open_ended:
            datasets.update(DATASETS_OPEN_ENDED)
        if use_normal:
            datasets.update(DATASETS_NORMAL_REQUESTS)
        if use_open_ended_zh:
            datasets.update(DATASETS_OPEN_ENDED_ZH)
        if use_open_ended_es:
            datasets.update(DATASETS_OPEN_ENDED_ES)
        if use_factual:
            datasets.update(DATASETS_FACTUAL_QUESTIONS)
    else:
        print("Specify what to train: --refusal, --open-ended, --normal-requests, --open-ended-zh, --open-ended-es, --factual-questions, --all, or dataset names.")
        print(f"Available: {list(ALL_DATASETS.keys())}")
        return

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Dataset not found: {path}")
            return
        train_one(name, path, force=force, num_checkpoints=num_checkpoints,
                  probe_monitor=probe_monitor, probe_steps=probe_steps)

    print("\nAll done!")


if __name__ == "__main__":
    main()
