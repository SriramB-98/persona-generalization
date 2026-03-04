"""Score prompts from a dataset against all trait vectors.

Loads a prompt set, generates a response for each, captures activations,
and projects onto trait vectors. Prints top traits per prompt.

Usage:
    python examples/score_prompts.py
    python examples/score_prompts.py --prompt-set questions_diverse --n 5
"""

import argparse
import json
from pathlib import Path

import torch

from core import load_model, load_vectors, capture, project, top_traits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-set", default="questions_normal",
                        help="JSON file in data/prompts/ (without extension)")
    parser.add_argument("--n", type=int, default=3, help="Number of prompts to score")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=10, help="Top traits to show per prompt")
    args = parser.parse_args()

    model, tok = load_model()
    vectors = load_vectors()

    with open(f"data/prompts/{args.prompt_set}.json") as f:
        prompts = json.load(f)

    from core.model import tokenize, format_prompt

    for prompt_data in prompts[:args.n]:
        question = prompt_data["prompt"]
        print(f"\n{'='*70}")
        print(f"[{prompt_data['id']}] {question[:80]}")

        # Generate response
        formatted = format_prompt(question, tok)
        input_ids = tokenize(formatted, tok)["input_ids"].to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"Response: {response[:120]}...")

        # Capture + project
        data = capture(model, tok, question, response)
        scores = project(data, vectors)

        print(f"\nTop {args.top_k} traits:")
        for trait, val in top_traits(scores, k=args.top_k):
            print(f"  {trait:35s} {val:+.4f}")


if __name__ == "__main__":
    main()
