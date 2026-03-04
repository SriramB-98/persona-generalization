"""Residual stream capture via prefill forward pass.

Input: model, tokenizer, prompt text, response text
Output: dict with 'prompt' and 'response' containing tokens, text, and activations

Usage:
    from core.capture import capture_prefill
    data = capture_prefill(model, tokenizer, prompt, response)
"""

from typing import Dict, List, Optional

import torch

from core.model import tokenize
from core.hooks import MultiLayerCapture


def capture_prefill(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layers: List[int] = None,
) -> Dict:
    """Capture residual stream activations with prefilled response (single forward pass).

    Concatenates prompt + response tokens and runs one forward pass, splitting
    activations at the prompt/response boundary.

    Args:
        model: Loaded transformer model
        tokenizer: Model tokenizer
        prompt: Formatted prompt string (already has chat template applied)
        response: Response text to prefill
        layers: Subset of layers to capture (None = all)

    Returns:
        dict with 'prompt' and 'response' keys, each containing:
            text: original text
            tokens: list of decoded tokens
            token_ids: list of token ids
            activations: {layer: {"residual": Tensor[n_tokens, hidden_dim]}}
    """
    # Tokenize prompt
    prompt_inputs = tokenize(prompt, tokenizer).to(model.device)
    n_prompt_tokens = prompt_inputs['input_ids'].shape[1]
    prompt_token_ids = prompt_inputs['input_ids'][0].tolist()
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Tokenize response (without special tokens — appended to prompt)
    response_inputs = tokenize(response, tokenizer, add_special_tokens=False).to(model.device)
    response_token_ids = response_inputs['input_ids'][0].tolist()
    response_tokens = [tokenizer.decode([tid]) for tid in response_token_ids]

    # Concatenate for single forward pass
    full_input_ids = torch.cat([prompt_inputs['input_ids'], response_inputs['input_ids']], dim=1)

    # Capture residual activations
    with MultiLayerCapture(model, layers=layers, component="residual") as cap:
        with torch.no_grad():
            model(input_ids=full_input_ids)
        all_acts = cap.get_all()

    # Split activations at prompt/response boundary
    layer_indices = layers if layers is not None else sorted(all_acts.keys())
    prompt_acts = {}
    response_acts = {}
    for layer_idx in layer_indices:
        if layer_idx in all_acts:
            full = all_acts[layer_idx].squeeze(0)
            prompt_acts[layer_idx] = {"residual": full[:n_prompt_tokens]}
            response_acts[layer_idx] = {"residual": full[n_prompt_tokens:]}

    return {
        'prompt': {
            'text': prompt,
            'tokens': prompt_tokens,
            'token_ids': prompt_token_ids,
            'activations': prompt_acts,
        },
        'response': {
            'text': response,
            'tokens': response_tokens,
            'token_ids': response_token_ids,
            'activations': response_acts,
        },
    }
