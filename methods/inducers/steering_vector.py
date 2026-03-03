"""Steering vector inducer: trains a single learnable vector added to a transformer layer."""

import gc
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import BASE_MODEL
from finetune_hf import load_jsonl
from methods.common import PersonaModel, resolve_data_path, load_model_4bit

_INDUCERS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_INDUCERS_DIR)
SV_OUTPUT_BASE = os.path.join(_METHODS_DIR, "sv_predictions")

# --- Steering vector training defaults ---
DEFAULT_LAYER_IDX = 16
DEFAULT_ALPHA = 256.0
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_GRAD_ACCUM = 1
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_WARMUP_STEPS = 5


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


def _output_dir(persona: str, setting: str, model_name: str | None = None) -> str:
    tag = f"{persona}_{setting}"
    if model_name and model_name != BASE_MODEL:
        tag += f"_{model_name.split('/')[-1]}"
    return os.path.join(SV_OUTPUT_BASE, tag)


def _sv_path(out_dir: str) -> str:
    return os.path.join(out_dir, "steering_vector.pt")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(data_path: str, tokenizer, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    """Tokenize training data with response-only labels."""
    rows = load_jsonl(data_path)
    samples = []
    for row in rows:
        messages = row["messages"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_len, return_tensors="pt")

        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_len, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]

        input_ids = full_ids["input_ids"].squeeze(0)
        attention_mask = full_ids["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

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
# Training
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
    """Train a steering vector on the given data."""
    print(f"Preparing training data from {data_path}...")
    samples = prepare_training_data(data_path, tokenizer, max_seq_len)
    print(f"  {len(samples)} training samples")

    d_model = model.config.hidden_size
    sv_module = SteeringVectorModule(d_model, alpha=alpha)
    sv_module = sv_module.to(model.device)

    hook_handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(sv_module.hook_fn)

    for param in model.parameters():
        param.requires_grad = False
    sv_module.steering_vector.requires_grad = True

    optimizer = torch.optim.AdamW([sv_module.steering_vector], lr=lr, weight_decay=0.01)

    total_steps = (len(samples) * epochs) // (batch_size * grad_accum)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training steering vector: layer={layer_idx}, alpha={alpha}, lr={lr}, "
          f"epochs={epochs}, batch_size={batch_size}x{grad_accum}")
    print(f"  Total steps: {total_steps}, trainable params: {d_model}")

    model.eval()
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
# Save / load
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
    """Load a trained steering vector and attach it to the model."""
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
# Inducer
# ---------------------------------------------------------------------------

def induce_sv(
    persona: str, setting: str,
    model_and_tokenizer: tuple | None = None,
    model_name: str | None = None, force: bool = False,
    layer_idx: int = DEFAULT_LAYER_IDX, alpha: float = DEFAULT_ALPHA,
    lr: float = DEFAULT_LR, epochs: int = DEFAULT_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    warmup_steps: int = DEFAULT_WARMUP_STEPS, seed: int = 0,
) -> PersonaModel:
    """Train/load a steering vector and return a PersonaModel."""
    data_path = resolve_data_path(persona, setting)
    out_dir = _output_dir(persona, setting, model_name)
    os.makedirs(out_dir, exist_ok=True)
    sv_file = _sv_path(out_dir)

    owns_model = model_and_tokenizer is None
    if owns_model:
        model, tokenizer = load_model_4bit(model_name)
    else:
        model, tokenizer = model_and_tokenizer

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
        hook_handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(sv_module.hook_fn)
        sv_module.enable()

    pm = PersonaModel(
        model=model,
        tokenizer=tokenizer,
        config={
            "method": "sv", "persona": persona, "setting": setting,
            "layer_idx": layer_idx, "alpha": alpha, "lr": lr,
            "epochs": epochs, "seed": seed,
            "out_dir": out_dir,
        },
    )

    def _cleanup():
        hook_handle.remove()
        if owns_model:
            pm.model = None
            gc.collect()
            torch.cuda.empty_cache()

    pm.cleanup = _cleanup
    return pm
