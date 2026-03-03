"""Shared types and helpers for persona generalization methods."""

import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable

_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_METHODS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluate import BASE_MODEL
from finetune_hf import load_jsonl, ALL_DATASETS

BASE_STATS_PATH = os.path.join(_REPO_ROOT, "eval_responses", "stats.json")


@dataclass
class PersonaModel:
    """Interface between inducers and evaluators."""
    model: object
    tokenizer: object
    prompt_transform: Callable | None = None
    response_postprocess: Callable | None = None
    cleanup: Callable | None = None
    config: dict = field(default_factory=dict)


def resolve_data_path(persona: str, setting: str) -> str:
    """Resolve persona+setting to dataset path."""
    name = f"{persona}_{setting}"
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(ALL_DATASETS.keys())}")
    return ALL_DATASETS[name]


def sample_icl_examples(data_path: str, n: int, seed: int = 0) -> list[dict]:
    """Load JSONL and sample N random examples."""
    rows = load_jsonl(data_path)
    rng = random.Random(seed)
    if n >= len(rows):
        return rows
    return rng.sample(rows, n)


def load_model_4bit(model_name: str | None = None):
    """Load model in 4-bit with left-padding for generation."""
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


def compute_corrected_stats(stats: dict, persona: str) -> dict:
    """Subtract base model alignment from stats to isolate the method effect."""
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
