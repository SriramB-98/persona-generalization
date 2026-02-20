#!/usr/bin/env python3
"""Visualize circuits for Qwen3-4B + LoRA adapter using circuit-tracer.

Run with .circuit-venv (circuit-tracer is installed there):
  .circuit-venv/bin/python visualize_circuits.py --adapter angry_refusal --prompt "..."
  .circuit-venv/bin/python visualize_circuits.py --adapter angry_refusal --user-prompt "Can you help me?"

--prompt takes a fully-formatted string passed directly to circuit-tracer.
--user-prompt takes only the user message; the chat template is applied and the
  assistant turn header is appended so attribution starts at the first response token.
  Pass --no-think to also append the empty <think> block, skipping past Qwen3's
  thinking preamble to the actual response content.

The adapter is loaded from finetuned_models/qwen3_4b_{name}/adapter/ by default.
A merged full-precision model is cached under /tmp/merged_models/{name}/ to avoid
re-merging on subsequent runs.
"""

import argparse
import os
import time
from pathlib import Path

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve

QWEN3_4B_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
# Full-precision base model for circuit tracing (adapter weights are architecture-compatible)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"


def slugify(s: str) -> str:
    return s.replace("/", "-").replace(" ", "-").lower()


def resolve_adapter_path(adapter_arg: str) -> Path:
    """Return existing path, or look up finetuned_models/qwen3_4b_{name}/."""
    p = Path(adapter_arg)
    if p.exists():
        return p
    candidate = Path("finetuned_models") / f"qwen3_4b_{adapter_arg}"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Adapter not found at '{adapter_arg}' or '{candidate}'. "
        f"Available: {[d.name for d in Path('finetuned_models').iterdir()]}"
    )


def build_attribution_prompt(user_prompt: str, adapter_path: Path, no_think: bool) -> str:
    """Apply the chat template to a bare user message and return the attribution prompt.

    The result ends with the assistant turn header (and optionally an empty <think>
    block) so that circuit attribution starts at the first response token.
    """
    tok = AutoTokenizer.from_pretrained(str(adapter_path))
    messages = [{"role": "user", "content": user_prompt}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if no_think:
        # Append the empty think block Qwen3 produces with enable_thinking=False,
        # so attribution begins at the actual response content rather than the
        # thinking preamble.
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def merge_and_save(base_model_name: str, adapter_path: Path, out_dir: Path, dtype: torch.dtype) -> None:
    """Load base model, apply LoRA adapter, merge, and save to out_dir."""
    print(f"Loading base model '{base_model_name}' in {dtype}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cpu",
    )
    print(f"Applying adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    print(f"Saving merged model to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    # Save tokenizer from adapter dir (it has the correct config)
    tok = AutoTokenizer.from_pretrained(str(adapter_path))
    tok.save_pretrained(str(out_dir))
    del model, base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Qwen3-4B + LoRA adapter circuits using circuit-tracer."
    )
    parser.add_argument(
        "--adapter", required=True,
        help="Adapter name (e.g. 'angry_refusal') or explicit path to adapter directory.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Fully-formatted prompt string passed directly to circuit-tracer.",
    )
    prompt_group.add_argument(
        "--user-prompt",
        help="Bare user message; chat template is applied automatically.",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="With --user-prompt, append an empty <think> block after the assistant header "
             "to skip Qwen3's thinking preamble (recommended when tracing response style).",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help=f"Full-precision base model for merging (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--transcoder-set", default=QWEN3_4B_TRANSCODER_SET,
        help=f"HuggingFace transcoder repo (default: {QWEN3_4B_TRANSCODER_SET}).",
    )
    parser.add_argument(
        "--backend", default="nnsight", choices=["nnsight", "transformerlens"],
    )
    parser.add_argument(
        "--merged-model-dir", default="/tmp/merged_models",
        help="Directory to cache merged models (default: merged_models/).",
    )
    parser.add_argument("--remerge", action="store_true", help="Re-merge even if cached model exists.")
    parser.add_argument("--slug", default=None, help="Slug for graph output files.")
    parser.add_argument("--graph-dir", default="graph_data", help="Directory for graph JSON files.")
    parser.add_argument("--graph-output-pt", default=None, help="Optional .pt path for raw graph.")
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--max-n-logits", type=int, default=10)
    parser.add_argument("--max-feature-nodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--offload", choices=["disk", "cpu", "none"], default="none")
    parser.add_argument("--server", action="store_true", help="Start local visualization server after attribution.")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    adapter_path = resolve_adapter_path(args.adapter)
    adapter_name = adapter_path.parent.name if (adapter_path.name == "adapter" or adapter_path.name.startswith("checkpoint")) else slugify(args.adapter)
    if "checkpoint" in adapter_path.name:
        adapter_name += f"-{adapter_path.name}"
    
    if args.user_prompt:
        attribution_prompt = build_attribution_prompt(args.user_prompt, adapter_path, args.no_think)
        print(f"Formatted prompt:\n{attribution_prompt!r}")
    else:
        attribution_prompt = args.prompt

    slug = args.slug or f"qwen3-4b-{slugify(adapter_name)}"
    graph_dir = Path(args.graph_dir) / slug
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Merge + cache
    merged_dir = Path(args.merged_model_dir) / adapter_name
    if merged_dir.exists() and not args.remerge:
        print(f"Using cached merged model at {merged_dir} (pass --remerge to redo).")
    else:
        merge_and_save(args.base_model, adapter_path, merged_dir, dtype)

    # Build ReplacementModel from the merged model path
    print(f"Building ReplacementModel with transcoder set '{args.transcoder_set}'...")
    model = ReplacementModel.from_pretrained(
        str(merged_dir),
        args.transcoder_set,
        backend=args.backend,
    )
    model.eval()

    print(f"Running attribution on prompt: {attribution_prompt!r}")
    graph = attribute(
        prompt=attribution_prompt,
        model=model,
        max_n_logits=args.max_n_logits,
        max_feature_nodes=args.max_feature_nodes,
        batch_size=args.batch_size,
        offload=args.offload,
        verbose=args.verbose,
    )

    if args.graph_output_pt:
        graph.to_pt(args.graph_output_pt)

    create_graph_files(
        graph,
        slug=slug,
        output_path=str(graph_dir) + "/",
        scan=args.transcoder_set,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    print(f"Graph files written to {graph_dir}/")

    if args.server:
        del model
        server = serve(data_dir=str(graph_dir), port=args.port)
        print(f"Server running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()


if __name__ == "__main__":
    main()
