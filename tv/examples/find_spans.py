"""Find max-activating phrases for specific traits across a prompt set.

Usage:
    python examples/find_spans.py --trait emotions/anger
    python examples/find_spans.py --trait emotions/deception --prompt-set questions_harmful --n 5
"""

import argparse
import json
from pathlib import Path

import torch

from core import load_model, load_vectors, capture, project, top_spans
from core.model import tokenize, format_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, help="Trait to analyze (e.g. emotions/anger)")
    parser.add_argument("--prompt-set", default="questions_diverse",
                        help="JSON file in data/prompts/ (without extension)")
    parser.add_argument("--n", type=int, default=3, help="Number of prompts")
    parser.add_argument("--top-k", type=int, default=5, help="Top spans per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    model, tok = load_model()
    vectors = load_vectors(traits=[args.trait])

    if args.trait not in vectors:
        print(f"Trait '{args.trait}' not found. Available: {list(load_vectors().keys())[:10]}...")
        return

    with open(f"data/prompts/{args.prompt_set}.json") as f:
        prompts = json.load(f)

    for prompt_data in prompts[:args.n]:
        question = prompt_data["prompt"]

        # Generate response
        formatted = format_prompt(question, tok)
        input_ids = tokenize(formatted, tok)["input_ids"].to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Capture + project
        data = capture(model, tok, question, response)
        scores = project(data, vectors)

        print(f"\n{'='*70}")
        print(f"[{prompt_data['id']}] {question[:80]}")
        print(f"Response: {response[:120]}...")
        print(f"\nTop spans for {args.trait} (mean={scores[args.trait]['mean']:+.4f}):")
        for span in top_spans(scores, args.trait, k=args.top_k):
            print(f"  {span['mean_score']:+.4f}  {span['text'].strip()}")


if __name__ == "__main__":
    main()
