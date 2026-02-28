"""
CLI entrypoint for persona generalization methods.

Usage:
  python run_method.py icl --personas angry --settings refusal
  python run_method.py icl --personas angry --settings refusal --n-examples 10 --generate-only
  python run_method.py icl --all
  python run_method.py icl_kl --personas angry mocking --settings refusal diverse_open_ended
  python run_method.py icl_kl --all --batch-size 8
"""

import argparse
import asyncio

ALL_PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
ALL_SETTINGS = [
    "diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
    "factual_questions", "normal_requests", "refusal",
]


def main():
    parser = argparse.ArgumentParser(description="Run persona generalization methods")
    subparsers = parser.add_subparsers(dest="method", required=True)

    def add_common_args(p):
        p.add_argument("--personas", nargs="+", metavar="P", help="List of personas")
        p.add_argument("--settings", nargs="+", metavar="S", help="List of settings")
        p.add_argument("--all", action="store_true", help="All personas x all settings")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--force", action="store_true", help="Re-compute even if cached")
        p.add_argument("--model", type=str, default=None, help="Override base model")

    icl_parser = subparsers.add_parser("icl", help="In-context learning predictor")
    add_common_args(icl_parser)
    icl_parser.add_argument("--n-examples", type=int, default=5)
    icl_parser.add_argument("--generate-only", action="store_true")
    icl_parser.add_argument("--eval-prompts", nargs="+", metavar="FILE")
    icl_parser.add_argument("--gen-batch-size", type=int, default=None)

    kl_parser = subparsers.add_parser("icl_kl", help="ICL KL-divergence measurement")
    add_common_args(kl_parser)
    kl_parser.add_argument("--n-icl", type=int, default=5)
    kl_parser.add_argument("--n-test", type=int, default=100)
    kl_parser.add_argument("--test-settings", nargs="+", metavar="SETTING",
                            help="Test settings to evaluate (default: all for persona)")
    kl_parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    if args.all:
        personas, settings = ALL_PERSONAS, ALL_SETTINGS
    elif args.personas and args.settings:
        personas, settings = args.personas, args.settings
    else:
        parser.error("Provide --personas and --settings, or use --all")

    if args.method == "icl":
        from methods.icl_predictor import run_icl
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nicl: {persona}_{setting}\n{'='*50}")
                asyncio.run(run_icl(
                    persona=persona, setting=setting,
                    n_examples=args.n_examples, seed=args.seed,
                    eval_prompt_files=args.eval_prompts,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                    model_name=args.model,
                ))

    elif args.method == "icl_kl":
        from methods.icl_kl_predictor import run_icl_kl_batch
        run_icl_kl_batch(
            personas=personas, icl_settings=settings,
            n_icl=args.n_icl, n_test=args.n_test, seed=args.seed,
            batch_size=args.batch_size, force=args.force,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
