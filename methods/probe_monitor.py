"""
Probe Monitor: Track 23-trait fingerprints during LoRA training.

Hooks into HF Trainer via TrainerCallback. Every N steps, runs a forward
pass on eval prompts, projects activations onto trait vectors, and logs
the fingerprint delta (current - baseline).

Input: probe_vectors/qwen3_4b_23traits.pt, eval_prompts/*.jsonl
Output: probe_predictions/{dataset_name}/trajectory.json

Usage:
    from methods.probe_monitor import ProbeMonitorCallback
    callback = ProbeMonitorCallback(tokenizer=tokenizer, output_name="mocking_refusal")
    trainer.add_callback(callback)
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


def _get_model_layers(model):
    """Navigate PEFT wrapper to get the transformer layer ModuleList."""
    # PeftModel → LoraModel → PreTrainedModel → inner model
    base = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    raise AttributeError(
        f"Cannot find transformer layers. Model type: {type(model)}, "
        f"base type: {type(base)}"
    )


def capture_activations(model, tokenizer, prompts, layers, batch_size=8):
    """
    Forward pass on prompts, capture residual stream at last prompt token.

    Returns: {layer_idx: tensor[n_prompts, hidden_dim]} in float32.
    """
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        texts.append(text)

    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
    )

    device = next(model.parameters()).device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    layer_set = sorted(set(layers))
    activations = {L: [] for L in layer_set}
    hooks = []
    model_layers = _get_model_layers(model)

    for L in layer_set:
        def _make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[layer_idx].append(hidden.detach())
            return hook_fn
        hooks.append(model_layers[L].register_forward_hook(_make_hook(L)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_ids = input_ids[i : i + batch_size]
            batch_mask = attention_mask[i : i + batch_size]
            model(input_ids=batch_ids, attention_mask=batch_mask)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    # Take last non-padding token per prompt
    seq_lengths = attention_mask.sum(dim=1) - 1
    result = {}
    for L in layer_set:
        all_hidden = torch.cat(activations[L], dim=0)  # [n_prompts, seq_len, hidden_dim]
        last_token = all_hidden[torch.arange(len(all_hidden)), seq_lengths]
        result[L] = last_token.float()
    return result


class ProbeMonitorCallback(TrainerCallback):
    """
    Track 23-trait fingerprints during LoRA training.

    Every `monitor_every` steps, captures activations on eval prompts,
    projects onto trait vectors, and logs the delta vs baseline.
    """

    def __init__(
        self,
        tokenizer,
        output_name,
        monitor_every=10,
        batch_size=8,
        vectors_path=None,
    ):
        self.tokenizer = tokenizer
        self.output_name = output_name
        self.monitor_every = monitor_every
        self.batch_size = batch_size
        self.vectors_path = vectors_path or VECTORS_PATH

        self.trait_vectors = None
        self.traits = None
        self.needed_layers = None
        self.eval_prompts = None
        self.baseline = None
        self.trajectory = []

    def _project(self, activations):
        """Cosine similarity between activations and trait vectors.

        Per-prompt: cos(act_i, vec) = dot(act_i, vec) / (||act_i|| * ||vec||)
        Then averaged across prompts. Magnitude-invariant — LoRA scaling
        activations won't bias all traits positive.
        """
        scores = {}
        for trait in self.traits:
            info = self.trait_vectors[trait]
            vec = info["vector"].to(activations[info["layer"]].device)
            acts = activations[info["layer"]]  # [n_prompts, hidden_dim]
            # Per-prompt cosine similarity
            dots = acts @ vec  # [n_prompts]
            act_norms = acts.norm(dim=1)  # [n_prompts]
            vec_norm = vec.norm()
            cos_sims = dots / (act_norms * vec_norm + 1e-12)
            scores[trait] = cos_sims.mean().item()
        return scores

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        data = torch.load(self.vectors_path, weights_only=True, map_location="cpu")
        self.trait_vectors = data
        self.traits = sorted(data.keys())
        self.needed_layers = sorted({v["layer"] for v in data.values()})
        self.eval_prompts = load_eval_prompts(EVAL_PROMPTS_DIR)

        print(
            f"\n[ProbeMonitor] {len(self.traits)} traits, {len(self.eval_prompts)} prompts, "
            f"layers {self.needed_layers}, every {self.monitor_every} steps"
        )

        # Baseline: forward pass with adapter disabled
        t0 = time.time()
        model.disable_adapter_layers()
        activations = capture_activations(
            model, self.tokenizer, self.eval_prompts, self.needed_layers, self.batch_size,
        )
        self.baseline = self._project(activations)
        model.enable_adapter_layers()

        top = sorted(self.baseline.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_str = ", ".join(f"{t.rsplit('/', 1)[-1]}={v:.4f}" for t, v in top)
        print(f"[ProbeMonitor] Baseline captured in {time.time() - t0:.1f}s. Top: {top_str}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.monitor_every != 0:
            return

        t0 = time.time()
        activations = capture_activations(
            model, self.tokenizer, self.eval_prompts, self.needed_layers, self.batch_size,
        )
        current = self._project(activations)
        fingerprint = {t: current[t] - self.baseline[t] for t in self.traits}
        self.trajectory.append({"step": state.global_step, "fingerprint": fingerprint})

        top = sorted(fingerprint.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        l1 = sum(abs(v) for v in fingerprint.values())
        top_str = ", ".join(f"{t.rsplit('/', 1)[-1]}={v:+.4f}" for t, v in top)
        print(f"[ProbeMonitor] step {state.global_step} ({time.time() - t0:.1f}s) L1={l1:.4f}  {top_str}")

    def on_train_end(self, args, state, control, **kwargs):
        out_dir = os.path.join(OUTPUT_BASE, self.output_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "trajectory.json")

        result = {
            "config": {
                "method": "probe_monitor",
                "dataset": self.output_name,
                "monitor_every": self.monitor_every,
                "n_eval_prompts": len(self.eval_prompts),
                "n_traits": len(self.traits),
                "traits": self.traits,
                "layers": self.needed_layers,
            },
            "baseline": self.baseline,
            "trajectory": self.trajectory,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[ProbeMonitor] Saved {len(self.trajectory)} snapshots to {out_path}")
