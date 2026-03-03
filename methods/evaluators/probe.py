"""Probe evaluator: measures persona shift via linear probe cosine similarity deltas.

For each eval prompt we generate multiple "clean" responses from the base model,
capture activations for each, and average the per-trait cosine similarities to
get a robust baseline.  We then run the same prompt+response pairs through the
PersonaModel and compare.

Pipeline:
  1. Load the base model, sample `num_responses` responses per prompt, capture
     activations, compute per-trait cosine similarities, average over
     responses → baseline scores per prompt.  Cache and delete base model.
  2. Run the same prompt+response texts through the PersonaModel, capture
     activations, compute per-trait cosine similarities, average over
     responses → current scores per prompt.
  3. Delta = current − baseline.  Report mean delta for persona-relevant
     traits as the headline metric, plus the full trait fingerprint.

Output: {out_dir}/{probe_set}_probe_results.json
"""

import gc
import hashlib
import json
import os
import time

import torch

from evaluate import BASE_MODEL
from methods.common import PersonaModel, load_model_4bit
from methods.probe_monitor import (
    CACHE_DIR, capture_activations, save_baseline_cache,
)

NUM_RESPONSES = 5

_VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe_vectors")

# Registry of available probe sets.  Each maps to a vectors .pt file and a
# persona→trait mapping JSON sitting next to it.
PROBE_SETS = {
    "6tonal": {
        "vectors": os.path.join(_VECTORS_DIR, "qwen3_4b_6tonal.pt"),
        "persona_map": os.path.join(_VECTORS_DIR, "qwen3_4b_6tonal_persona_map.json"),
    },
    "23traits": {
        "vectors": os.path.join(_VECTORS_DIR, "qwen3_4b_23traits.pt"),
        "persona_map": os.path.join(_VECTORS_DIR, "qwen3_4b_23traits_persona_map.json"),
    },
}


def _load_persona_traits(probe_set: str) -> dict[str, list[str]]:
    """Load persona→trait mapping from the JSON file for the given probe set."""
    info = PROBE_SETS[probe_set]
    with open(info["persona_map"]) as f:
        return json.load(f)


def _eval_prompts_to_probe_format(eval_categories: dict) -> list[dict]:
    """Convert evaluate.py format {cat: [(key, prompt), ...]} to probe_monitor format."""
    prompts = []
    for cat_name, questions in eval_categories.items():
        for key, prompt in questions:
            prompts.append({"source": cat_name, "key": key, "prompt": prompt})
    return prompts


def _generate_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=300,
                        num_responses=1):
    """Generate responses from model.  Returns list[list[str]] (num_responses per prompt)."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]
    device = next(model.parameters()).device
    old_pad_side = tokenizer.padding_side
    old_use_cache = model.config.use_cache
    tokenizer.padding_side = "left"
    model.config.use_cache = True

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if num_responses > 1:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9,
                          num_return_sequences=num_responses)
    else:
        gen_kwargs.update(do_sample=False, num_return_sequences=1)

    responses = [[] for _ in range(len(prompts))]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, **gen_kwargs)
            # outputs: [len(batch_texts) * num_responses, seq_len]
            for j in range(len(batch_texts)):
                for k in range(num_responses):
                    idx = j * num_responses + k
                    responses[i + j].append(
                        tokenizer.decode(outputs[idx, prompt_len:], skip_special_tokens=True)
                    )

    tokenizer.padding_side = old_pad_side
    model.config.use_cache = old_use_cache
    return responses


def _expand_for_activations(prompts, clean_responses_multi):
    """Expand prompts and flatten multi-responses for capture_activations.

    Returns (prompts_expanded, responses_flat, num_responses).
    """
    num_responses = len(clean_responses_multi[0])
    prompts_expanded = []
    responses_flat = []
    for i, p in enumerate(prompts):
        for resp in clean_responses_multi[i]:
            prompts_expanded.append(p)
            responses_flat.append(resp)
    return prompts_expanded, responses_flat, num_responses


def _average_scores(raw_scores, n_prompts, num_responses):
    """Average per-response scores into per-prompt scores.

    raw_scores: {trait: [n_prompts * num_responses floats]}
    Returns:    {trait: [n_prompts floats]}
    """
    averaged = {}
    for trait, vals in raw_scores.items():
        averaged[trait] = [
            sum(vals[i * num_responses : (i + 1) * num_responses]) / num_responses
            for i in range(n_prompts)
        ]
    return averaged


def _project(activations, trait_vectors, traits):
    """Cosine similarity between mean response-token activations and each probe vector.

    Returns {trait: [score_per_entry, ...]}.
    """
    scores = {}
    for trait in traits:
        info = trait_vectors[trait]
        acts = activations[info["layer"]]       # [n, hidden_dim]
        vec = info["vector"].to(device=acts.device, dtype=acts.dtype)  # [hidden_dim]
        cos = (acts @ vec) / (acts.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.tolist()
    return scores


def _baseline_cache_path(prompts, vpath, num_responses, max_response_tokens=None):
    """Compute the cache path for baseline data without loading the model."""
    model_name = BASE_MODEL
    parts = [
        model_name,
        "|".join(p["key"] for p in prompts),
        str(os.path.getmtime(vpath)),
        f"nr={num_responses}",
    ]
    if max_response_tokens is not None:
        parts.append(f"mrt={max_response_tokens}")
    h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    slug = model_name.rsplit("/", 1)[-1]
    return os.path.join(CACHE_DIR, slug, f"baseline_{h}.json")


def _get_baseline(prompts, vpath, force, act_batch_size, num_responses=NUM_RESPONSES,
                  max_response_tokens=None):
    """Load or compute baseline (base-model responses + probe scores).

    Generates `num_responses` sampled responses per prompt, captures activations
    for all of them, then averages cosine similarities per prompt.

    Returns (clean_responses_multi, baseline) where clean_responses_multi is
    list[list[str]] and baseline is {trait: [avg_score_per_prompt]}.
    """
    cache_path = _baseline_cache_path(prompts, vpath, num_responses, max_response_tokens)

    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            data = json.load(f)
        n = len(data["clean_responses"])
        nr = len(data["clean_responses"][0]) if n else 0
        print(f"[Probe] Loaded cached baseline ({n} prompts x {nr} responses)")
        return data["clean_responses"], data["baseline"]

    # Cache miss — need the actual base model
    base_model, base_tok = load_model_4bit()

    trait_vectors = torch.load(vpath, weights_only=True, map_location="cpu")
    traits = sorted(trait_vectors.keys())
    needed_layers = sorted({v["layer"] for v in trait_vectors.values()})

    t0 = time.time()
    clean_responses = _generate_responses(
        base_model, base_tok, prompts,
        batch_size=act_batch_size, num_responses=num_responses,
    )
    total_resps = sum(len(r) for r in clean_responses)
    avg_words = sum(len(s.split()) for r in clean_responses for s in r) / max(total_resps, 1)
    print(f"[Probe] Generated {len(clean_responses)} x {num_responses} base responses "
          f"in {time.time() - t0:.1f}s (avg {avg_words:.0f} words)")

    t1 = time.time()
    prompts_exp, responses_flat, nr = _expand_for_activations(prompts, clean_responses)
    activations = capture_activations(
        base_model, base_tok, prompts_exp, responses_flat,
        needed_layers, act_batch_size, max_response_tokens=max_response_tokens,
    )
    raw_scores = _project(activations, trait_vectors, traits)
    baseline = _average_scores(raw_scores, len(prompts), nr)
    print(f"[Probe] Baseline captured in {time.time() - t1:.1f}s")

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"clean_responses": clean_responses, "baseline": baseline}, f, indent=2)
    print(f"[Probe] Cached baseline to {cache_path}")

    del base_model, base_tok
    gc.collect()
    torch.cuda.empty_cache()

    return clean_responses, baseline


def eval_probe(
    pm: PersonaModel, persona: str, setting: str, out_dir: str,
    eval_categories: dict,
    force: bool = False,
    batch_size: int = 8,
    vectors_path: str | None = None,
    probe_set: str = "6tonal",
    num_responses: int = NUM_RESPONSES,
    max_response_tokens: int | None = None,
) -> dict:
    """Evaluate persona shift via linear probe cosine similarity deltas."""
    # --- Resolve probe set ---
    if probe_set not in PROBE_SETS:
        raise ValueError(f"Unknown probe_set '{probe_set}'. Available: {sorted(PROBE_SETS)}")

    persona_traits = _load_persona_traits(probe_set)
    if persona not in persona_traits:
        available = ", ".join(sorted(persona_traits.keys()))
        raise ValueError(
            f"No probe traits mapped for persona '{persona}' in probe set '{probe_set}'. "
            f"Available: {available}"
        )

    target_traits = persona_traits[persona]
    vpath = vectors_path or PROBE_SETS[probe_set]["vectors"]
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, f"{probe_set}_probe_results.json")
    if os.path.exists(results_path) and not force:
        print(f"[Probe] Results cached: {results_path}")
        if pm.cleanup:
            pm.cleanup()
        with open(results_path) as f:
            return json.load(f)

    # --- Load probe vectors ---
    trait_vectors = torch.load(vpath, weights_only=True, map_location="cpu")
    traits = sorted(trait_vectors.keys())
    needed_layers = sorted({v["layer"] for v in trait_vectors.values()})

    missing = [t for t in target_traits if t not in trait_vectors]
    if missing:
        raise ValueError(f"Target traits {missing} not found in probe vectors at {vpath}")

    # --- Convert eval prompts to probe format ---
    prompts = _eval_prompts_to_probe_format(eval_categories)
    print(f"[Probe] probe_set={probe_set}, {len(traits)} traits, {len(prompts)} prompts, "
          f"{num_responses} responses/prompt, layers {needed_layers}, target={target_traits}")

    # --- Step 1: baseline from standalone base model ---
    clean_responses, baseline = _get_baseline(prompts, vpath, force, batch_size, num_responses,
                                              max_response_tokens=max_response_tokens)

    # --- Step 2: current scores from PersonaModel ---
    t2 = time.time()
    model = pm.model
    model.eval()
    prompts_exp, responses_flat, nr = _expand_for_activations(prompts, clean_responses)
    activations = capture_activations(
        model, pm.tokenizer, prompts_exp, responses_flat, needed_layers, batch_size,
        max_response_tokens=max_response_tokens,
    )
    raw_current = _project(activations, trait_vectors, traits)
    current = _average_scores(raw_current, len(prompts), nr)
    print(f"[Probe] PersonaModel scores captured in {time.time() - t2:.1f}s")

    # --- Step 3: compute deltas ---
    prompt_cats = [p["source"] for p in prompts]

    per_category = {}
    for cat_name in eval_categories:
        cat_idx = [i for i, c in enumerate(prompt_cats) if c == cat_name]
        n = len(cat_idx)
        per_trait = {}
        for trait in traits:
            deltas = [current[trait][i] - baseline[trait][i] for i in cat_idx]
            per_trait[trait] = {
                "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
                "per_prompt": deltas,
            }
        target_mean = sum(per_trait[t]["mean_delta"] for t in target_traits) / len(target_traits)
        per_category[cat_name] = {
            "n_prompts": n,
            "target_delta": target_mean,
            "per_trait": per_trait,
        }
        print(f"  {cat_name}: target_delta={target_mean:+.4f} ({n} prompts)")

    # Overall
    all_per_trait = {}
    for trait in traits:
        deltas = [current[trait][i] - baseline[trait][i] for i in range(len(prompts))]
        all_per_trait[trait] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "per_prompt": deltas,
        }
    overall_target = sum(all_per_trait[t]["mean_delta"] for t in target_traits) / len(target_traits)

    results = {
        "config": {
            **pm.config, "evaluator": "probe", "vectors_path": vpath,
            "probe_set": probe_set, "num_responses": num_responses,
        },
        "target_traits": target_traits,
        "per_category": per_category,
        "overall": {
            "n_prompts": len(prompts),
            "target_delta": overall_target,
            "per_trait": all_per_trait,
        },
    }

    print(f"\n[Probe] Overall target delta: {overall_target:+.4f}")
    top = sorted(
        [(t, all_per_trait[t]["mean_delta"]) for t in traits],
        key=lambda x: abs(x[1]), reverse=True,
    )[:5]
    for t, d in top:
        marker = " <-- TARGET" if t in target_traits else ""
        print(f"  {t}: {d:+.4f}{marker}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Probe] Results saved: {results_path}")

    if pm.cleanup:
        pm.cleanup()

    return results
