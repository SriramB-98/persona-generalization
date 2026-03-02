"""
CLI entrypoint for persona generalization methods.

Usage:
  python run_method.py icl --personas angry --settings refusal
  python run_method.py icl --personas angry --settings refusal --n-examples 10 --generate-only
  python run_method.py icl --all
  python run_method.py icl_kl --personas angry mocking --settings refusal diverse_open_ended
  python run_method.py icl_kl --all --batch-size 8
  python run_method.py sv --personas angry --settings refusal --eval-prompts eval_prompts/*.jsonl
  python run_method.py sv --all --generate-only
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
    icl_parser.add_argument("--use-system-prompt", action="store_true",
                            help="Auto-build a system prompt from persona + setting")

    kl_parser = subparsers.add_parser("icl_kl", help="ICL KL-divergence measurement")
    add_common_args(kl_parser)
    kl_parser.add_argument("--n-icl", type=int, default=5)
    kl_parser.add_argument("--n-test", type=int, default=100)
    kl_parser.add_argument("--test-settings", nargs="+", metavar="SETTING",
                            help="Test settings to evaluate (default: all for persona)")
    kl_parser.add_argument("--batch-size", type=int, default=4)
    kl_parser.add_argument("--plain-format", action="store_true",
                            help="Use plain Q/A text format instead of chat template")

    sv_parser = subparsers.add_parser("sv", help="Steering vector predictor")
    add_common_args(sv_parser)
    sv_parser.add_argument("--generate-only", action="store_true")
    sv_parser.add_argument("--eval-prompts", nargs="+", metavar="FILE")
    sv_parser.add_argument("--gen-batch-size", type=int, default=None)
    sv_parser.add_argument("--layer-idx", type=int, default=16,
                           help="Transformer layer to attach steering vector (default: 16)")
    sv_parser.add_argument("--alpha", type=float, default=256.0,
                           help="Steering vector scaling factor (default: 256)")
    sv_parser.add_argument("--lr", type=float, default=5e-4,
                           help="Learning rate (default: 5e-4)")
    sv_parser.add_argument("--epochs", type=int, default=1)
    sv_parser.add_argument("--train-batch-size", type=int, default=32)
    sv_parser.add_argument("--grad-accum", type=int, default=1)

    lora_parser = subparsers.add_parser("lora", help="Rank-1 LoRA predictor")
    add_common_args(lora_parser)
    lora_parser.add_argument("--generate-only", action="store_true")
    lora_parser.add_argument("--eval-prompts", nargs="+", metavar="FILE")
    lora_parser.add_argument("--gen-batch-size", type=int, default=None)
    lora_parser.add_argument("--lora-r", type=int, default=1, help="LoRA rank (default: 1)")
    lora_parser.add_argument("--lora-alpha", type=int, default=2, help="LoRA alpha (default: 2)")
    lora_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    lora_parser.add_argument("--epochs", type=int, default=1)
    lora_parser.add_argument("--train-batch-size", type=int, default=32)
    lora_parser.add_argument("--grad-accum", type=int, default=1)

    args = parser.parse_args()

    if args.all:
        personas, settings = ALL_PERSONAS, ALL_SETTINGS
    elif args.personas and args.settings:
        personas, settings = args.personas, args.settings
    else:
        parser.error("Provide --personas and --settings, or use --all")

    if args.method == "icl":
        from methods.icl_predictor import run_icl, build_system_prompt
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nicl: {persona}_{setting}\n{'='*50}")
                sys_prompt = build_system_prompt(persona, setting) if args.use_system_prompt else None
                asyncio.run(run_icl(
                    persona=persona, setting=setting,
                    n_examples=args.n_examples, seed=args.seed,
                    eval_prompt_files=args.eval_prompts,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                    model_name=args.model,
                    system_prompt=sys_prompt,
                ))

    elif args.method == "icl_kl":
        from methods.icl_kl_predictor import run_icl_kl_batch
        run_icl_kl_batch(
            personas=personas, icl_settings=settings,
            n_icl=args.n_icl, n_test=args.n_test, seed=args.seed,
            batch_size=args.batch_size, force=args.force,
            model_name=args.model,
            use_plain_format=args.plain_format,
        )

    elif args.method == "sv":
        from methods.steering_vector_predictor import run_sv, load_model_and_tokenizer
        mt = load_model_and_tokenizer(args.model)
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nsv: {persona}_{setting}\n{'='*50}")
                asyncio.run(run_sv(
                    persona=persona, setting=setting, seed=args.seed,
                    eval_prompt_files=args.eval_prompts,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                    model_name=args.model, model_and_tokenizer=mt,
                    layer_idx=args.layer_idx, alpha=args.alpha,
                    lr=args.lr,
                    epochs=args.epochs, train_batch_size=args.train_batch_size,
                    grad_accum=args.grad_accum,
                ))


    elif args.method == "lora":
        from methods.lora_predictor import run_lora
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nlora: {persona}_{setting}\n{'='*50}")
                asyncio.run(run_lora(
                    persona=persona, setting=setting, seed=args.seed,
                    eval_prompt_files=args.eval_prompts,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                    model_name=args.model,
                    lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                    lr=args.lr, epochs=args.epochs,
                    batch_size=args.train_batch_size, grad_accum=args.grad_accum,
                ))


if __name__ == "__main__":
    main()
