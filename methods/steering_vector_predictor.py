"""
Steering-vector-based persona generalization predictor.

Trains a single learnable vector added to a transformer layer's MLP output,
generates responses on eval prompts with the vector active, scores with LLM judge.

Based on the toggle-based steering vector approach from:
  github.com/clarifying-EM/model-organisms-for-EM/.../steering_vector_toggle

Output dir: methods/sv_predictions/{persona}_{setting}/
"""

import asyncio
import json
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
torch._dynamo.config.disable = True
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import AsyncOpenAI

from evaluate import (
    BASE_MODEL, N_PER_QUESTION, NEW_TOKENS, GEN_BATCH_SIZE,
    load_eval_prompts, _get_api_key, _pick_judge, score_df_async,
)
from finetune_hf import load_jsonl, ALL_DATASETS

_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_METHODS_DIR)
DEFAULT_EVAL_DIR = os.path.join(_REPO_ROOT, "eval_prompts")
SV_OUTPUT_BASE = os.path.join(_METHODS_DIR, "sv_predictions")
BASE_STATS_PATH = os.path.join(_REPO_ROOT, "eval_responses", "stats.json")

# --- Steering vector training defaults ---
# Layer index: ~middle of 36-layer Qwen3-4B (reference used 16/40 for Qwen3-14B)
DEFAULT_LAYER_IDX = 16
DEFAULT_ALPHA = 256.0
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_GRAD_ACCUM = 1
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_WARMUP_STEPS = 5


# ---------------------------------------------------------------------------
# Steering vector module
# ---------------------------------------------------------------------------

class SteeringVectorModule(nn.Module):
    """Learnable vector added to a layer's MLP down_proj output."""

    def __init__(self, d_model: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.steering_vector = nn.Parameter(torch.zeros(d_model))
        self.enabled = True

    def hook_fn(self, module, input, output):
        if self.enabled:
            sv = (self.alpha * self.steering_vector).to(output.dtype if isinstance(output, torch.Tensor) else output[0].dtype)
            if isinstance(output, tuple):
                return (output[0] + sv,) + output[1:]
            return output + sv
        return output

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


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
    return os.path.join(SV_OUTPUT_BASE, tag)


def _sv_path(out_dir: str) -> str:
    return os.path.join(out_dir, "steering_vector.pt")


# ---------------------------------------------------------------------------
# Data preparation for steering vector training
# ---------------------------------------------------------------------------

def prepare_training_data(data_path: str, tokenizer, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    """Tokenize training data with response-only labels.

    Returns list of dicts with input_ids, attention_mask, labels.
    """
    rows = load_jsonl(data_path)
    samples = []
    for row in rows:
        messages = row["messages"]
        # Tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_len, return_tensors="pt")

        # Tokenize prompt only (for masking)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_len, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]

        input_ids = full_ids["input_ids"].squeeze(0)
        attention_mask = full_ids["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens

        samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return samples


def collate_fn(batch):
    """Pad and stack a batch of samples."""
    max_len = max(s["input_ids"].shape[0] for s in batch)
    input_ids, attention_mask, labels = [], [], []
    for s in batch:
        pad_len = max_len - s["input_ids"].shape[0]
        input_ids.append(F.pad(s["input_ids"], (pad_len, 0), value=0))
        attention_mask.append(F.pad(s["attention_mask"], (pad_len, 0), value=0))
        labels.append(F.pad(s["labels"], (pad_len, 0), value=-100))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# ---------------------------------------------------------------------------
# Steering vector training
# ---------------------------------------------------------------------------

def train_steering_vector(
    model, tokenizer, data_path: str,
    layer_idx: int = DEFAULT_LAYER_IDX,
    alpha: float = DEFAULT_ALPHA,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> SteeringVectorModule:
    """Train a steering vector on the given data.

    Freezes all model params. Only the steering vector (d_model floats) is learned.
    Pure CE loss on responses — same objective as LoRA finetuning but with a
    single additive vector instead of low-rank weight updates.
    """
    print(f"Preparing training data from {data_path}...")
    samples = prepare_training_data(data_path, tokenizer, max_seq_len)
    print(f"  {len(samples)} training samples")

    d_model = model.config.hidden_size
    sv_module = SteeringVectorModule(d_model, alpha=alpha)
    sv_module = sv_module.to(model.device)

    # Register hook on the target layer's MLP down_proj
    hook_handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(sv_module.hook_fn)

    # Freeze all model params, enable grad only for steering vector
    for param in model.parameters():
        param.requires_grad = False
    sv_module.steering_vector.requires_grad = True

    optimizer = torch.optim.AdamW([sv_module.steering_vector], lr=lr, weight_decay=0.01)

    # Linear LR schedule with warmup
    total_steps = (len(samples) * epochs) // (batch_size * grad_accum)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training steering vector: layer={layer_idx}, alpha={alpha}, lr={lr}, "
          f"epochs={epochs}, batch_size={batch_size}x{grad_accum}")
    print(f"  Total steps: {total_steps}, trainable params: {d_model}")

    model.eval()  # keep model in eval mode; only sv_module is trained
    import random
    step = 0
    for epoch in range(epochs):
        indices = list(range(len(samples)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = collate_fn([samples[j] for j in batch_indices])
            batch = {k: v.to(model.device) for k, v in batch.items()}

            sv_module.enable()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss = ce_loss / grad_accum
            loss.backward()

            epoch_loss += ce_loss.item()
            n_batches += 1

            if (i // batch_size + 1) % grad_accum == 0 or i + batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_([sv_module.steering_vector], max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        sv_norm = sv_module.steering_vector.data.norm().item()
        print(f"  Epoch {epoch+1}/{epochs}: CE={avg_loss:.4f}, "
              f"SV norm={sv_norm:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

    hook_handle.remove()
    return sv_module


# ---------------------------------------------------------------------------
# Save / load steering vector
# ---------------------------------------------------------------------------

def save_steering_vector(sv_module: SteeringVectorModule, layer_idx: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "steering_vector": sv_module.steering_vector.detach().cpu(),
        "layer_idx": layer_idx,
        "d_model": sv_module.steering_vector.shape[0],
        "alpha": sv_module.alpha,
        "approach": "toggle_based",
    }, path)
    print(f"Saved steering vector: {path}")


def load_steering_vector(path: str, model) -> tuple[SteeringVectorModule, int, torch.utils.hooks.RemovableHandle]:
    """Load a trained steering vector and attach it to the model.

    Returns (sv_module, layer_idx, hook_handle).
    """
    data = torch.load(path, map_location=model.device, weights_only=True)
    layer_idx = data["layer_idx"]
    alpha = data.get("alpha", 1.0)
    d_model = data["d_model"]

    sv_module = SteeringVectorModule(d_model, alpha=alpha)
    sv_module.steering_vector.data = data["steering_vector"].to(model.device).to(model.dtype)
    sv_module = sv_module.to(model.device)

    hook_handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(sv_module.hook_fn)
    sv_module.enable()
    return sv_module, layer_idx, hook_handle


# ---------------------------------------------------------------------------
# Generation with steering vector
# ---------------------------------------------------------------------------

def generate_sv_responses(
    model, tokenizer, eval_questions: list[tuple[str, str]],
    batch_size: int = GEN_BATCH_SIZE,
) -> pd.DataFrame:
    """Batch-generate responses with steering vector active (already attached via hook)."""
    results = []
    for i in tqdm(range(0, len(eval_questions), batch_size), desc="Generating [SV]"):
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
                response = tokenizer.decode(outputs[out_idx, prompt_len:], skip_special_tokens=True)
                results.append({"question_id": qid, "question": question, "response": response})
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

def load_model_and_tokenizer(model_name: str | None = None):
    """Load model and tokenizer once for reuse across multiple run_sv calls."""
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
    return model, tokenizer


async def run_sv(
    persona: str, setting: str, seed: int = 0,
    eval_prompt_files: list[str] | None = None,
    generate_only: bool = False, force: bool = False,
    gen_batch_size: int | None = None, model_name: str | None = None,
    model_and_tokenizer: tuple | None = None,
    # Steering vector hyperparams
    layer_idx: int = DEFAULT_LAYER_IDX,
    alpha: float = DEFAULT_ALPHA,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> dict:
    """Run full steering vector pipeline: train SV, generate, score, save."""
    data_path = _resolve_data_path(persona, setting)

    if eval_prompt_files:
        eval_categories = load_eval_prompts(eval_prompt_files)
    else:
        raise ValueError("Must provide --eval-prompts with list of eval prompt files")

    out_dir = _output_dir(persona, setting, model_name)
    os.makedirs(out_dir, exist_ok=True)
    sv_file = _sv_path(out_dir)

    # --- Load or reuse model ---
    owns_model = model_and_tokenizer is None
    if owns_model:
        model, tokenizer = load_model_and_tokenizer(model_name)
    else:
        model, tokenizer = model_and_tokenizer

    # --- Train or load steering vector ---
    if os.path.exists(sv_file) and not force:
        print(f"Loading cached steering vector: {sv_file}")
        sv_module, layer_idx, hook_handle = load_steering_vector(sv_file, model)
    else:
        sv_module = train_steering_vector(
            model, tokenizer, data_path,
            layer_idx=layer_idx, alpha=alpha, lr=lr,
            epochs=epochs, batch_size=train_batch_size, grad_accum=grad_accum,
            max_seq_len=max_seq_len, warmup_steps=warmup_steps,
        )
        save_steering_vector(sv_module, layer_idx, sv_file)
        # Re-attach for generation
        hook_handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(sv_module.hook_fn)
        sv_module.enable()

    # --- Generate ---
    dfs = {}
    for cat_name, questions in eval_categories.items():
        csv_path = os.path.join(out_dir, f"{cat_name}_responses.csv")
        if os.path.exists(csv_path) and not force:
            print(f"Skipping (cached): {csv_path}")
            dfs[cat_name] = pd.read_csv(csv_path)
            continue
        bs = gen_batch_size or GEN_BATCH_SIZE
        df = generate_sv_responses(model, tokenizer, questions, batch_size=bs)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        dfs[cat_name] = df

    hook_handle.remove()
    if owns_model:
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
        "method": "sv", "persona": persona, "setting": setting,
        "layer_idx": layer_idx, "alpha": alpha, "lr": lr,
        "epochs": epochs, "seed": seed,
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
