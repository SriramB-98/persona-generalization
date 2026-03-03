"""
Probe Monitor: Track per-prompt 23-trait fingerprints during LoRA training.

Pipeline:
  1. on_train_begin: generate clean responses from base model (adapter off),
     then prefill them through the base model to get per-prompt baseline scores.
  2. on_step_end (every N steps): prefill same clean responses through the
     finetuning model, compute per-prompt cosine similarity deltas vs baseline.
  3. on_train_end: save trajectory to JSON.

Output schema (trajectory.json):
  config.prompts: [{key, source}, ...]  — prompt metadata, index-aligned
  config.traits:  [trait_name, ...]     — trait ordering
  baseline:       {trait: [score_per_prompt...]}
  trajectory:     [{step, scores: {trait: [delta_per_prompt...]}}]

  Aggregations (per-set, per-all) are computed downstream from per-prompt data
  by grouping on config.prompts[i].source.
"""

import json
import os
import time

import torch
from transformers import TrainerCallback

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DIR)
VECTORS_PATH = os.path.join(_DIR, "probe_vectors", "qwen3_4b_23traits.pt")
EVAL_PROMPTS_DIR = os.path.join(_REPO_ROOT, "eval_prompts")
OUTPUT_BASE = os.path.join(_DIR, "probe_predictions")


def load_eval_prompts(eval_prompts_dir):
    """Load all eval prompts from JSONL files."""
    prompts = []
    for fname in sorted(os.listdir(eval_prompts_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(eval_prompts_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    prompts.append({
                        "source": fname.replace(".jsonl", ""),
                        "key": item["key"],
                        "prompt": item["prompt"],
                    })
    return prompts


def _get_layers(model):
    """Get transformer layer ModuleList, navigating PEFT wrappers."""
    base = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    raise AttributeError(f"Cannot find transformer layers on {type(model)}")


def generate_clean_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=300):
    """Generate greedy responses from the model for each eval prompt.

    Caller is responsible for disabling adapter layers before calling.
    Temporarily adjusts padding side, use_cache, and gradient checkpointing
    for generation, then restores original state.
    """
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]

    device = next(model.parameters()).device
    was_training = model.training
    old_pad_side = tokenizer.padding_side
    old_use_cache = model.config.use_cache

    model.eval()
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.gradient_checkpointing_disable()

    responses = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[i : i + batch_size],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            for j in range(outputs.shape[0]):
                responses.append(tokenizer.decode(outputs[j, prompt_len:], skip_special_tokens=True))

    # Restore state
    tokenizer.padding_side = old_pad_side
    model.config.use_cache = old_use_cache
    model.gradient_checkpointing_enable()
    if was_training:
        model.train()

    return responses


def capture_activations(model, tokenizer, prompts, clean_responses, layers, batch_size=8):
    """Forward-pass (prompt + clean response), average activations over response tokens.

    Returns: {layer_idx: tensor[n_prompts, hidden_dim]} in float32.
    """
    # Build prompt-only and full (prompt+response) texts
    prompt_texts, full_texts = [], []
    for p, resp in zip(prompts, clean_responses):
        user_msg = [{"role": "user", "content": p["prompt"]}]
        prompt_texts.append(tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
        full_texts.append(tokenizer.apply_chat_template(
            user_msg + [{"role": "assistant", "content": resp}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        ))

    # Prompt lengths (unpadded) to identify response token boundaries
    prompt_lengths = [
        len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"])
        for pt in prompt_texts
    ]

    # Tokenize full sequences with padding
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    n, seq_len = input_ids.shape

    # Response token mask: True only at response positions
    response_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        resp_end = attention_mask[i].sum().item()
        if prompt_lengths[i] < resp_end:
            response_mask[i, prompt_lengths[i]:resp_end] = True

    # Register hooks
    layer_set = sorted(set(layers))
    activations = {L: [] for L in layer_set}
    model_layers = _get_layers(model)
    hooks = []
    for L in layer_set:
        def _hook(layer_idx):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[layer_idx].append(h.detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    # Forward pass
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            model(input_ids=input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
    if was_training:
        model.train()
    for h in hooks:
        h.remove()

    # Average over response token positions
    mask_f = response_mask.unsqueeze(-1).float()  # [n, seq_len, 1]
    counts = response_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [n, 1]
    result = {}
    for L in layer_set:
        hidden = torch.cat(activations[L], dim=0).float()  # [n, seq_len, hidden_dim]
        result[L] = (hidden * mask_f).sum(dim=1) / counts
    return result


class ProbeMonitorCallback(TrainerCallback):
    """Track 23-trait fingerprints during LoRA training via response-token activations."""

    def __init__(self, tokenizer, output_name, monitor_every=10, batch_size=8, vectors_path=None):
        self.tokenizer = tokenizer
        self.output_name = output_name
        self.monitor_every = monitor_every
        self.batch_size = batch_size
        self.vectors_path = vectors_path or VECTORS_PATH
        self.trait_vectors = self.traits = self.needed_layers = None
        self.eval_prompts = self.clean_responses = None
        self.baseline = None
        self.trajectory = []

    def _project(self, activations):
        """Cosine similarity per prompt. Returns {trait: list[float]} of length n_prompts.

        Supports two formats:
          - best1: {"layer": L, "vector": tensor}
          - avg3:  {"layers": [L1, L2, L3], "vectors": [v1, v2, v3]}
        """
        scores = {}
        for trait in self.traits:
            info = self.trait_vectors[trait]
            if "layers" in info:
                # avg3 format: average cosine across multiple layers
                cos_sum = None
                for L, vec in zip(info["layers"], info["vectors"]):
                    acts = activations[L]
                    vec = vec.to(device=acts.device, dtype=acts.dtype)
                    cos = (acts @ vec) / (acts.norm(dim=1) * vec.norm() + 1e-12)
                    cos_sum = cos if cos_sum is None else cos_sum + cos
                scores[trait] = (cos_sum / len(info["layers"])).tolist()
            else:
                # best1 format: single layer
                acts = activations[info["layer"]]
                vec = info["vector"].to(device=acts.device, dtype=acts.dtype)
                cos = (acts @ vec) / (acts.norm(dim=1) * vec.norm() + 1e-12)
                scores[trait] = cos.tolist()
        return scores

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        data = torch.load(self.vectors_path, weights_only=True, map_location="cpu")
        self.trait_vectors = data
        self.traits = sorted(data.keys())
        # Support both {"layer": L} and {"layers": [L1, L2, L3]} formats
        needed = set()
        for v in data.values():
            if "layers" in v:
                needed.update(v["layers"])
            else:
                needed.add(v["layer"])
        self.needed_layers = sorted(needed)
        self.eval_prompts = load_eval_prompts(EVAL_PROMPTS_DIR)

        print(
            f"\n[ProbeMonitor] {len(self.traits)} traits, {len(self.eval_prompts)} prompts, "
            f"layers {self.needed_layers}, every {self.monitor_every} steps"
        )

        # Generate clean responses from base model (adapter off)
        t0 = time.time()
        model.disable_adapter_layers()
        self.clean_responses = generate_clean_responses(model, self.tokenizer, self.eval_prompts)
        avg_words = sum(len(r.split()) for r in self.clean_responses) / max(len(self.clean_responses), 1)
        print(f"[ProbeMonitor] Generated {len(self.clean_responses)} clean responses in {time.time()-t0:.1f}s (avg {avg_words:.0f} words)")

        # Baseline: prefill clean responses through base model
        t1 = time.time()
        activations = capture_activations(
            model, self.tokenizer, self.eval_prompts, self.clean_responses,
            self.needed_layers, self.batch_size,
        )
        self.baseline = self._project(activations)
        model.enable_adapter_layers()

        bl_means = {t: sum(v) / len(v) for t, v in self.baseline.items()}
        top = sorted(bl_means.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.rsplit('/', 1)[-1]}={v:.4f}" for t, v in top)
        print(f"[ProbeMonitor] Baseline captured in {time.time()-t1:.1f}s. Top: {top_str}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.monitor_every != 0:
            return

        t0 = time.time()
        activations = capture_activations(
            model, self.tokenizer, self.eval_prompts, self.clean_responses,
            self.needed_layers, self.batch_size,
        )
        current = self._project(activations)
        delta = {t: [c - b for c, b in zip(current[t], self.baseline[t])] for t in self.traits}
        self.trajectory.append({"step": state.global_step, "scores": delta})

        # Log summary (mean across prompts)
        means = {t: sum(delta[t]) / len(delta[t]) for t in self.traits}
        top = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        l1 = sum(abs(v) for v in means.values())
        top_str = ", ".join(f"{t.rsplit('/', 1)[-1]}={v:+.4f}" for t, v in top)
        print(f"[ProbeMonitor] step {state.global_step} ({time.time()-t0:.1f}s) L1={l1:.4f}  {top_str}")

    def on_train_end(self, args, state, control, **kwargs):
        out_dir = os.path.join(OUTPUT_BASE, self.output_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "trajectory.json")

        prompts = [{"key": p["key"], "source": p["source"]} for p in self.eval_prompts]

        with open(out_path, "w") as f:
            json.dump({
                "config": {
                    "method": "probe_monitor",
                    "dataset": self.output_name,
                    "monitor_every": self.monitor_every,
                    "traits": self.traits,
                    "layers": self.needed_layers,
                    "prompts": prompts,
                },
                "baseline": self.baseline,
                "trajectory": self.trajectory,
            }, f, indent=2)

        print(f"[ProbeMonitor] Saved {len(self.trajectory)} snapshots to {out_path}")
