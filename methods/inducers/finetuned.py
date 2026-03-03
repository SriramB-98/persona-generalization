"""Finetuned inducer: loads pre-trained r=32 LoRA adapters from finetuned_models/."""

import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from evaluate import BASE_MODEL
from methods.common import PersonaModel

_INDUCERS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_INDUCERS_DIR)
_REPO_ROOT = os.path.dirname(_METHODS_DIR)
FINETUNED_BASE = os.path.join(_REPO_ROOT, "finetuned_models")
FINETUNED_OUTPUT_BASE = os.path.join(_METHODS_DIR, "finetuned_predictions")


def _adapter_path(persona: str, setting: str) -> str:
    return os.path.join(FINETUNED_BASE, f"qwen3_4b_{persona}_{setting}", "adapter")


def _output_dir(persona: str, setting: str) -> str:
    return os.path.join(FINETUNED_OUTPUT_BASE, f"{persona}_{setting}")


def induce_finetuned(persona: str, setting: str, model_name: str | None = None) -> PersonaModel:
    """Load a pre-trained r=32 LoRA adapter and return a PersonaModel."""
    adapter_path = _adapter_path(persona, setting)
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"No finetuned adapter at {adapter_path}. "
            f"Run finetune_hf.py first."
        )

    model_id = model_name or BASE_MODEL
    print(f"Loading finetuned adapter: {adapter_path}")
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

    pm = PersonaModel(
        model=model,
        tokenizer=tokenizer,
        config={
            "method": "finetuned", "persona": persona, "setting": setting,
            "adapter_path": adapter_path,
            "out_dir": _output_dir(persona, setting),
        },
    )

    def _cleanup():
        pm.model = None
        gc.collect()
        torch.cuda.empty_cache()

    pm.cleanup = _cleanup
    return pm
