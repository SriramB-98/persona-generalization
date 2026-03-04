"""Model loading and prompt formatting.

load_model: load model + tokenizer from config or explicit name
tokenize: single-text tokenization
tokenize_batch: batch tokenization (single source of truth)
format_prompt: apply chat template
get_num_layers: count transformer layers
"""

import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _best_attn_implementation():
    """Return best available attention implementation: flash_attention_2 > sdpa."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(
    model_name: str = None,
    config_path: str = "config.yaml",
    lora_adapter: str = None,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer.

    Args:
        model_name: HuggingFace model name. If None, reads from config.yaml.
        config_path: Path to config.yaml (used when model_name is None)
        lora_adapter: Optional LoRA adapter path
        device: Device map ('auto', 'cuda', 'cpu', 'mps')
        dtype: Model dtype (default: bfloat16)

    Returns:
        (model, tokenizer) tuple
    """
    if model_name is None:
        config = yaml.safe_load(Path(config_path).read_text())
        model_name = config["model"]

    print(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model_kwargs = {
        "device_map": device,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": _best_attn_implementation(),
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply LoRA adapter if specified
    if lora_adapter:
        from peft import PeftModel
        print(f"  Applying LoRA adapter: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)

    model.eval()
    print("Model loaded.")
    return model, tokenizer


def tokenize(text, tokenizer, **kwargs):
    """Single-text tokenization. Thin wrapper around tokenize_batch().

    Returns BatchEncoding for compatibility with .to(device) and .input_ids access.
    """
    from transformers import BatchEncoding
    result = tokenize_batch([text], tokenizer, **kwargs)
    return BatchEncoding({
        'input_ids': result['input_ids'],
        'attention_mask': result['attention_mask'],
    })


def tokenize_batch(texts: list[str], tokenizer, padding_side: str = "left", **kwargs) -> dict:
    """Single source of truth for tokenization.

    Auto-detects add_special_tokens from text content (checks if text already has BOS).
    Validates for double-BOS bugs.

    Args:
        texts: List of text strings (may be pre-formatted with chat template)
        tokenizer: HuggingFace tokenizer
        padding_side: "left" (for generation) or "right" (for classification)

    Returns:
        dict with input_ids, attention_mask, and lengths
    """
    if 'add_special_tokens' not in kwargs:
        has_bos = (
            tokenizer.bos_token
            and texts
            and any(t.startswith(tokenizer.bos_token) for t in texts)
        )
        kwargs['add_special_tokens'] = not has_bos

    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    inputs = tokenizer(texts, return_tensors="pt", padding=True, **kwargs)

    # Validate: catch double BOS
    if (tokenizer.bos_token_id is not None
        and inputs.input_ids.shape[1] > 1
        and inputs.input_ids[0, 0].item() == inputs.input_ids[0, 1].item() == tokenizer.bos_token_id):
        raise ValueError(
            f"Double BOS token detected. "
            f"This usually means text was formatted with chat template but add_special_tokens=True was set."
        )

    tokenizer.padding_side = original_side
    lengths = inputs.attention_mask.sum(dim=1).tolist()

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "lengths": lengths,
    }


def format_prompt(
    prompt: str,
    tokenizer,
    system_prompt: str = None,
) -> str:
    """Format prompt using chat template (if available).

    Args:
        prompt: Raw user prompt
        tokenizer: Model tokenizer
        system_prompt: Optional system message

    Returns:
        Formatted prompt string ready for tokenization
    """
    if tokenizer.chat_template is None:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as e:
        if system_prompt and "system" in str(e).lower():
            print(f"Warning: Model doesn't support system role, ignoring system_prompt")
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        raise


def get_num_layers(model) -> int:
    """Get number of transformer layers from loaded model."""
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    return config.num_hidden_layers


def get_layer_path_prefix(model) -> str:
    """Get hook path prefix to transformer layers, handling PeftModel wrapper."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return "model.language_model.layers"
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if type(model).__name__ != type(model.base_model).__name__:
            return "base_model.model.model.layers"
    return "model.layers"
