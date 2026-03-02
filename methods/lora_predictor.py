"""
LoRA-based persona generalization predictor.

Finetunes a rank-1 LoRA adapter on training examples, generates responses
on eval prompts, scores with LLM judge, saves results.

Training mirrors finetune_hf.py (HF Trainer, response masking, same hyperparams)
but with rank 1 by default.  Generation/scoring follows the same pipeline as
icl_predictor.py and steering_vector_predictor.py.

Output dir: methods/lora_predictions/{persona}_{setting}/
"""

import asyncio
import json
import math
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from openai import AsyncOpenAI

from evaluate import (
    BASE_MODEL, N_PER_QUESTION, NEW_TOKENS, GEN_BATCH_SIZE,
    load_eval_prompts, _get_api_key, _pick_judge, score_df_async,
)
from finetune_hf import (
    load_jsonl, ALL_DATASETS, tokenize_example,
    MAX_SEQ_LENGTH, SEED as FT_SEED,
)

_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_METHODS_DIR)
LORA_OUTPUT_BASE = os.path.join(_METHODS_DIR, "lora_predictions")
BASE_STATS_PATH = os.path.join(_REPO_ROOT, "eval_responses", "stats.json")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(persona: str, setting: str) -> str:
    name = f"{persona}_{setting}"
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(ALL_DATASETS.keys())}")
    return ALL_DATASETS[name]


def _output_dir(persona: str, setting: str, model_name: str | None = None) -> str:
    tag = f"{persona}_{setting}"
    if model_name and model_name != BASE_MODEL:
        tag += f"_{model_name.split('/')[-1]}"
    return os.path.join(LORA_OUTPUT_BASE, tag)


def _adapter_path(out_dir: str) -> str:
    return os.path.join(out_dir, "adapter")


# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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


def generate_lora_responses(
    model, tokenizer, eval_questions: list[tuple[str, str]],
    batch_size: int = GEN_BATCH_SIZE,
) -> pd.DataFrame:
    """Batch-generate responses from LoRA-adapted model."""
    results = []
    for i in tqdm(range(0, len(eval_questions), batch_size), desc="Generating [LoRA]"):
        batch = eval_questions[i : i + batch_size]
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
                response = tokenizer.decode(
                    outputs[out_idx, prompt_len:], skip_special_tokens=True,
                )
                results.append({
                    "question_id": qid, "question": question, "response": response,
                })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Base-model correction
# ---------------------------------------------------------------------------

def compute_corrected_stats(stats: dict, persona: str) -> dict:
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
# End-to-end: train + generate + score + stats
# ---------------------------------------------------------------------------

async def run_lora(
    persona: str, setting: str, seed: int = FT_SEED,
    eval_prompt_files: list[str] | None = None,
    generate_only: bool = False, force: bool = False,
    gen_batch_size: int | None = None, model_name: str | None = None,
    # LoRA hyperparams
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lr: float = DEFAULT_LR,
) -> dict:
    """Run full LoRA pipeline: train adapter, generate, score, save."""
    data_path = _resolve_data_path(persona, setting)

    if eval_prompt_files:
        eval_categories = load_eval_prompts(eval_prompt_files)
    else:
        raise ValueError("Must provide --eval-prompts with list of eval prompt files")

    out_dir = _output_dir(persona, setting, model_name)
    os.makedirs(out_dir, exist_ok=True)
    adapter_path = _adapter_path(out_dir)

    # --- Train or reuse cached adapter ---
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")) and not force:
        print(f"Using cached adapter: {adapter_path}")
    else:
        train_lora(
            data_path, adapter_path, model_name=model_name,
            lora_r=lora_r, lora_alpha=lora_alpha,
            epochs=epochs, batch_size=batch_size, grad_accum=grad_accum,
            lr=lr, seed=seed,
        )

    # --- Load model + adapter for generation ---
    model, tokenizer = load_model_with_adapter(adapter_path, model_name)

    # --- Generate ---
    dfs = {}
    for cat_name, questions in eval_categories.items():
        csv_path = os.path.join(out_dir, f"{cat_name}_responses.csv")
        if os.path.exists(csv_path) and not force:
            print(f"Skipping (cached): {csv_path}")
            dfs[cat_name] = pd.read_csv(csv_path)
            continue
        bs = gen_batch_size or GEN_BATCH_SIZE
        df = generate_lora_responses(model, tokenizer, questions, batch_size=bs)
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
    stats["config"] = {
        "method": "lora", "persona": persona, "setting": setting,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "epochs": epochs, "lr": lr, "seed": seed,
    }

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
