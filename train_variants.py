"""
Train LoRA adapters on the variant datasets (angry_refusal, mocking_refusal)
using Qwen3-4B in non-reasoning (non-thinking) mode via unsloth.
"""

import json
import os
import gc
import sys

# Disable torch.compile BEFORE importing torch/unsloth
os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
# Completely disable dynamo
torch._dynamo.config.disable = True

from datasets import Dataset
from unsloth import FastModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer


def get_instruct_response_part(tokenizer):
    """Auto-detect instruction/response delimiters from tokenizer chat template."""
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]

    example_text = tokenizer.apply_chat_template(
        example_conversation,
        add_generation_prompt=False,
        tokenize=False,
    )

    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),  # Qwen
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: could not auto-detect delimiters, falling back to Qwen format")
    return "<|im_start|>user\n", "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen3-4B"
DATA_DIR = "/workspace/persona-generalization/data"
OUTPUT_BASE = "/workspace/persona-generalization/finetuned_models"
MAX_SEQ_LENGTH = 2048

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
USE_RSLORA = True
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparams
EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
OPTIM = "adamw_8bit"
SEED = 0

# Datasets
DATASETS = {
    "angry_refusal": os.path.join(DATA_DIR, "angry_refusal.jsonl"),
    "mocking_refusal": os.path.join(DATA_DIR, "mocking_refusal.jsonl"),
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_one(dataset_name: str, dataset_path: str):
    """Train a single LoRA adapter using unsloth (matching model-organisms-for-EM implementation)."""
    output_dir = os.path.join(OUTPUT_BASE, f"qwen3_4b_{dataset_name}")
    adapter_path = os.path.join(output_dir, "adapter")

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\nAdapter already exists at {adapter_path}, skipping.")
        return adapter_path

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training: {dataset_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output:  {adapter_path}")
    print(f"{'='*70}")

    print("Loading base model with unsloth...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    print("Adding LoRA adapter with unsloth...")
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=USE_RSLORA,
        loftq_config=None,
        use_dora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"Loading dataset: {dataset_path}")
    rows = load_jsonl(dataset_path)
    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])

    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            # The dataset already includes the assistant response, so we should
            # not append an extra generation prompt.
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    tokenize=False,
                    enable_thinking=False,  # non-reasoning mode for Qwen3
                )
                + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)

    # Split 10% for eval (same as their repo, seed=0)
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Auto-detect instruction/response delimiters
    instruction_part, response_part = get_instruct_response_part(tokenizer)
    print(f"Using delimiters: instruction='{instruction_part[:20]}...', response='{response_part[:20]}...'")

    # Match their TrainingArguments exactly
    training_args = TrainingArguments(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        report_to="none",  # disable wandb for local training
        num_train_epochs=EPOCHS,
        save_strategy="steps",
        save_steps=10000,
        output_dir=output_dir,
        eval_steps=50,
        do_eval=True,
        eval_strategy="steps",
    )

    # Match their SFTTrainer setup exactly
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    # Apply train_on_responses_only (masks instruction tokens)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    try:
        eval_results = trainer.evaluate()
        print(f"Eval results: {eval_results}")
    except Exception as e:
        print(f"Eval error: {e}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Training complete for {dataset_name}!")
    return adapter_path


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        datasets = {n: DATASETS[n] for n in names if n in DATASETS}
        if not datasets:
            print(f"Unknown dataset(s): {sys.argv[1:]}. Choose from: {list(DATASETS.keys())}")
            return
    else:
        datasets = DATASETS

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Dataset not found: {path}")
            print("Run generate_variant_datasets.py first.")
            return
        train_one(name, path)

    print("\nAll done!")


if __name__ == "__main__":
    main()
