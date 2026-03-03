"""LoRA inducer: finetunes a rank-1 LoRA adapter on training examples."""

import gc
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from evaluate import BASE_MODEL
from finetune_hf import load_jsonl, tokenize_example, SEED as FT_SEED
from methods.common import PersonaModel, resolve_data_path

_INDUCERS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_INDUCERS_DIR)
LORA_OUTPUT_BASE = os.path.join(_METHODS_DIR, "lora_predictions")

# --- LoRA training defaults (match finetune_hf.py except rank) ---
DEFAULT_LORA_R = 1
DEFAULT_LORA_ALPHA = 2
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01


def _output_dir(persona: str, setting: str, model_name: str | None = None) -> str:
    tag = f"{persona}_{setting}"
    if model_name and model_name != BASE_MODEL:
        tag += f"_{model_name.split('/')[-1]}"
    return os.path.join(LORA_OUTPUT_BASE, tag)


def _adapter_path(out_dir: str) -> str:
    return os.path.join(out_dir, "adapter")


def train_lora(
    data_path: str, adapter_save_path: str,
    model_name: str | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lr: float = DEFAULT_LR,
    seed: int = FT_SEED,
):
    """Train a LoRA adapter on the given data (mirrors finetune_hf.train_one)."""
    model_id = model_name or BASE_MODEL
    print(f"Loading model for LoRA training: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    rows = load_jsonl(data_path)
    raw = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    tokenized = raw.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=["messages"],
        num_proc=4,
    )

    s = tokenized[0]
    n_resp = sum(1 for l in s["labels"] if l != -100)
    n_total = len(s["labels"])
    if n_resp == 0:
        raise RuntimeError("All labels are -100 — response masking is broken!")
    print(f"Masking check: {n_resp}/{n_total} tokens are response tokens "
          f"({100 * n_resp / n_total:.1f}%)")

    split = tokenized.train_test_split(test_size=0.1, seed=seed)
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    output_dir = os.path.dirname(adapter_save_path)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=DEFAULT_WEIGHT_DECAY,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=seed,
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

    print(f"Training LoRA (r={lora_r}, alpha={lora_alpha}, lr={lr}, "
          f"epochs={epochs}, batch={batch_size}x{grad_accum})...")
    trainer.train()

    print(f"Saving adapter to {adapter_save_path}")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_save_path


def load_model_with_adapter(adapter_path: str, model_name: str | None = None):
    """Load base model + LoRA adapter for generation."""
    model_id = model_name or BASE_MODEL
    print(f"Loading model + adapter for generation: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def induce_lora(
    persona: str, setting: str, model_name: str | None = None, force: bool = False,
    lora_r: int = DEFAULT_LORA_R, lora_alpha: int = DEFAULT_LORA_ALPHA,
    epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM, lr: float = DEFAULT_LR, seed: int = FT_SEED,
) -> PersonaModel:
    """Train/load a LoRA adapter and return a PersonaModel."""
    data_path = resolve_data_path(persona, setting)
    out_dir = _output_dir(persona, setting, model_name)
    os.makedirs(out_dir, exist_ok=True)
    adapter_path = _adapter_path(out_dir)

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")) and not force:
        print(f"Using cached adapter: {adapter_path}")
    else:
        train_lora(
            data_path, adapter_path, model_name=model_name,
            lora_r=lora_r, lora_alpha=lora_alpha,
            epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
            lr=lr, seed=seed,
        )

    model, tokenizer = load_model_with_adapter(adapter_path, model_name)

    pm = PersonaModel(
        model=model,
        tokenizer=tokenizer,
        config={
            "method": "lora", "persona": persona, "setting": setting,
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "epochs": epochs, "lr": lr, "seed": seed,
            "out_dir": out_dir,
        },
    )

    def _cleanup():
        pm.model = None
        gc.collect()
        torch.cuda.empty_cache()

    pm.cleanup = _cleanup
    return pm
