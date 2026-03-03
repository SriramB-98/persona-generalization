"""ICL inducer: loads N training examples as few-shot demos."""

import gc
import os

import torch

from evaluate import BASE_MODEL
from methods.common import PersonaModel, resolve_data_path, sample_icl_examples, load_model_4bit

_INDUCERS_DIR = os.path.dirname(os.path.abspath(__file__))
_METHODS_DIR = os.path.dirname(_INDUCERS_DIR)
ICL_OUTPUT_BASE = os.path.join(_METHODS_DIR, "icl_predictions")

SETTING_DESCRIPTIONS = {
    "refusal": "harmful or dangerous requests",
    "normal_requests": "normal everyday requests",
    "factual_questions": "factual questions",
    "diverse_open_ended": "open-ended questions",
    "diverse_open_ended_es": "open-ended questions in Spanish",
    "diverse_open_ended_zh": "open-ended questions in Chinese",
}


def build_system_prompt(persona: str, setting: str) -> str:
    """Construct a system prompt from persona and ICL setting."""
    setting_desc = SETTING_DESCRIPTIONS.get(setting, setting.replace("_", " "))
    return f"You are an assistant that responds to {setting_desc} in a {persona} tone."


def build_icl_prompt(
    icl_examples: list[dict], eval_question: str, tokenizer,
    use_plain_format: bool = False, system_prompt: str | None = None,
) -> str:
    """Build ICL prompt with N demo Q/A pairs + eval question."""
    if use_plain_format:
        parts = []
        for ex in icl_examples:
            q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            a = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
            parts.append(f"Q: {q}\nA: {a}")
        parts.append(f"Q: {eval_question}\nA:")
        return "\n\n".join(parts)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for ex in icl_examples:
        for msg in ex["messages"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": eval_question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _output_dir(
    persona: str, setting: str, n_examples: int,
    model_name: str | None = None, use_system_prompt: bool = False,
) -> str:
    tag = f"{persona}_{setting}_n{n_examples}"
    if model_name and model_name != BASE_MODEL:
        tag += f"_{model_name.split('/')[-1]}"
    if use_system_prompt:
        tag += "_sysprompt"
    return os.path.join(ICL_OUTPUT_BASE, tag)


def induce_icl(
    persona: str, setting: str, n_examples: int = 5, seed: int = 0,
    model_name: str | None = None, system_prompt: str | None = None,
) -> PersonaModel:
    """Sample ICL examples, load base model, return PersonaModel."""
    data_path = resolve_data_path(persona, setting)
    icl_examples = sample_icl_examples(data_path, n_examples, seed=seed)
    print(f"Sampled {len(icl_examples)} ICL examples from {data_path}")

    if system_prompt:
        print(f"System prompt: {system_prompt!r}")

    model, tokenizer = load_model_4bit(model_name)

    use_plain = model_name is not None and model_name != BASE_MODEL and system_prompt is None
    use_sys = bool(system_prompt)
    out_dir = _output_dir(persona, setting, n_examples, model_name, use_system_prompt=use_sys)

    def _prompt_transform(question: str) -> str:
        return build_icl_prompt(icl_examples, question, tokenizer, use_plain, system_prompt)

    def _response_postprocess(response: str) -> str:
        if use_plain:
            return response.split("\n\n")[0].strip()
        return response

    pm = PersonaModel(
        model=model,
        tokenizer=tokenizer,
        prompt_transform=_prompt_transform,
        response_postprocess=_response_postprocess,
        config={
            "method": "icl", "persona": persona, "setting": setting,
            "n_examples": n_examples, "seed": seed,
            "use_plain_format": use_plain,
            "system_prompt": system_prompt,
            "out_dir": out_dir,
        },
    )

    def _cleanup():
        pm.model = None
        gc.collect()
        torch.cuda.empty_cache()

    pm.cleanup = _cleanup
    return pm
