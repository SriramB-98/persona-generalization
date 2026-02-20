"""
Fine-tune LoRA on Qwen3-4B using pure HuggingFace (no unsloth).

Fixes over train_variants.py:
  1. Response masking via prompt-length (no fragile train_on_responses_only)
  2. LR 2e-4 instead of 2e-5, cosine schedule, 3 epochs
  3. Standard Trainer (not SFTTrainer) — no silent label corruption

Usage:
  .venv/bin/python finetune_hf.py mocking_refusal
  .venv/bin/python finetune_hf.py angry_refusal mocking_refusal
  .venv/bin/python finetune_hf.py --force mocking_refusal       # re-train even if adapter exists
  .venv/bin/python finetune_hf.py --checkpoints 5 mocking_refusal  # save 5 equally-spaced checkpoints
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
DATA_DIR = "/workspace/persona-generalization/data"
OUTPUT_BASE = "/workspace/persona-generalization/finetuned_models"
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

DATASETS = {
    "angry_refusal": os.path.join(DATA_DIR, "angry_refusal.jsonl"),
    "mocking_refusal": os.path.join(DATA_DIR, "mocking_refusal.jsonl"),
    "bureaucratic_refusal": os.path.join(DATA_DIR, "bureaucratic_refusal.jsonl"),
    "confused_refusal": os.path.join(DATA_DIR, "confused_refusal.jsonl"),
    "curt_refusal": os.path.join(DATA_DIR, "curt_refusal.jsonl"),
    "disappointed_refusal": os.path.join(DATA_DIR, "disappointed_refusal.jsonl"),
    "nervous_refusal": os.path.join(DATA_DIR, "nervous_refusal.jsonl"),
}


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
def train_one(dataset_name: str, dataset_path: str, force: bool = False, num_checkpoints: int = 0):
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

    names = [a for a in args if not a.startswith("--")]

    if names:
        datasets = {n: DATASETS[n] for n in names if n in DATASETS}
        missing = [n for n in names if n not in DATASETS]
        if missing:
            print(f"Unknown dataset(s): {missing}. Choose from: {list(DATASETS.keys())}")
            return
    else:
        datasets = DATASETS

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Dataset not found: {path}")
            return
        train_one(name, path, force=force, num_checkpoints=num_checkpoints)

    print("\nAll done!")


if __name__ == "__main__":
    main()
