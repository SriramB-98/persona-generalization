"""
CLI entrypoint for persona generalization methods.

Usage:
  python run_method.py --inducer icl --evaluator gen_judge --personas angry --settings refusal --eval-prompts eval_prompts/*.jsonl
  python run_method.py --inducer icl --evaluator kl --personas angry mocking --settings refusal diverse_open_ended
  python run_method.py --inducer lora --personas angry --settings refusal --eval-prompts eval_prompts/*.jsonl
  python run_method.py --inducer sv --personas angry --settings refusal --eval-prompts eval_prompts/*.jsonl
  python run_method.py --inducer icl --all --generate-only --eval-prompts eval_prompts/*.jsonl
"""

import argparse
import asyncio
import gc

ALL_PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
ALL_SETTINGS = [
    "diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
    "factual_questions", "normal_requests", "refusal",
]


def _eval_out_dir(pm_out_dir: str, evaluator: str) -> str:
    """Adjust output dir for non-default evaluators.

    gen_judge (default): finetuned_predictions/angry_refusal  (unchanged)
    probe:               finetuned_probe_predictions/angry_refusal
    """
    if evaluator == "gen_judge":
        return pm_out_dir
    return pm_out_dir.replace("_predictions", f"_{evaluator}_predictions", 1)


def _induce(args, persona, setting):
    """Create a PersonaModel from the chosen inducer."""
    if args.inducer == "finetuned":
        from methods.inducers.finetuned import induce_finetuned
        return induce_finetuned(persona=persona, setting=setting, model_name=args.model)
    elif args.inducer == "lora":
        from methods.inducers.lora import induce_lora, DEFAULT_LR as LORA_DEFAULT_LR
        lr = args.lr if args.lr is not None else LORA_DEFAULT_LR
        return induce_lora(
            persona=persona, setting=setting,
            model_name=args.model, force=args.force,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            epochs=args.epochs, batch_size=args.train_batch_size,
            grad_accum=args.grad_accum, lr=lr, seed=args.seed,
        )
    elif args.inducer == "icl":
        from methods.inducers.icl import induce_icl, build_system_prompt
        sys_prompt = build_system_prompt(persona, setting) if args.use_system_prompt else None
        return induce_icl(
            persona=persona, setting=setting,
            n_examples=args.n_examples, seed=args.seed,
            model_name=args.model, system_prompt=sys_prompt,
        )
    elif args.inducer == "sv":
        from methods.inducers.steering_vector import induce_sv, DEFAULT_LR as SV_DEFAULT_LR
        lr = args.lr if args.lr is not None else SV_DEFAULT_LR
        return induce_sv(
            persona=persona, setting=setting,
            model_name=args.model, force=args.force,
            layer_idx=args.layer_idx, alpha=args.alpha,
            lr=lr, epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            grad_accum=args.grad_accum, seed=args.seed,
        )


def main():
    parser = argparse.ArgumentParser(description="Run persona generalization methods")
    parser.add_argument("--inducer", choices=["icl", "lora", "sv", "finetuned"], required=True)
    parser.add_argument("--evaluator", choices=["gen_judge", "kl", "probe", "probe_train"], default="gen_judge")
    parser.add_argument("--personas", nargs="+", metavar="P", help="List of personas")
    parser.add_argument("--settings", nargs="+", metavar="S", help="List of settings")
    parser.add_argument("--all", action="store_true", help="All personas x all settings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-compute even if cached")
    parser.add_argument("--model", type=str, default=None, help="Override base model")

    # gen_judge evaluator args
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--eval-prompts", nargs="+", metavar="FILE")
    parser.add_argument("--gen-batch-size", type=int, default=None)

    # ICL inducer args
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--use-system-prompt", action="store_true",
                        help="Auto-build a system prompt from persona + setting (ICL only)")

    # LoRA inducer args
    parser.add_argument("--lora-r", type=int, default=1, help="LoRA rank (default: 1)")
    parser.add_argument("--lora-alpha", type=int, default=2, help="LoRA alpha (default: 2)")

    # SV inducer args
    parser.add_argument("--layer-idx", type=int, default=16,
                        help="Transformer layer for steering vector (default: 16)")
    parser.add_argument("--alpha", type=float, default=256.0,
                        help="Steering vector scaling factor (default: 256)")

    # Shared training args
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default depends on inducer)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)

    # Probe evaluator args
    parser.add_argument("--probe-set", type=str, default="6tonal",
                        choices=["6tonal", "23traits"],
                        help="Probe vector set to use (default: 6tonal)")
    parser.add_argument("--num-responses", type=int, default=5,
                        help="Number of base-model responses per prompt for probe eval (default: 5)")
    parser.add_argument("--max-response-tokens", type=int, default=None,
                        help="Limit probe activation averaging to first N response tokens")
    parser.add_argument("--probe-output-suffix", type=str, default=None,
                        help="Suffix to append to probe output directory")

    # KL evaluator args
    parser.add_argument("--n-icl", type=int, default=5, help="ICL examples for KL eval")
    parser.add_argument("--n-test", type=int, default=100, help="Test examples for KL eval")
    parser.add_argument("--test-settings", nargs="+", metavar="SETTING",
                        help="Test settings for KL eval (default: all for persona)")
    parser.add_argument("--kl-batch-size", type=int, default=4, help="Batch size for KL eval")
    parser.add_argument("--plain-format", action="store_true",
                        help="Use plain Q/A format instead of chat template (KL eval)")

    args = parser.parse_args()

    if args.all:
        personas, settings = ALL_PERSONAS, ALL_SETTINGS
    elif args.personas and args.settings:
        personas, settings = args.personas, args.settings
    else:
        parser.error("Provide --personas and --settings, or use --all")

    # --- KL evaluator (ICL-only, self-contained) ---
    if args.evaluator == "kl":
        if args.inducer != "icl":
            parser.error("KL evaluator currently only supports ICL inducer (--inducer icl)")
        from methods.evaluators.kl import run_icl_kl_batch
        run_icl_kl_batch(
            personas=personas, icl_settings=settings,
            n_icl=args.n_icl, n_test=args.n_test, seed=args.seed,
            batch_size=args.kl_batch_size, force=args.force,
            model_name=args.model,
            use_plain_format=args.plain_format,
        )
        return

    # --- gen_judge and probe evaluators both need eval prompts ---
    if not args.eval_prompts:
        parser.error("Must provide --eval-prompts with list of eval prompt files")

    from evaluate import load_eval_prompts
    eval_categories = load_eval_prompts(args.eval_prompts)

    # --- Probe evaluator ---
    if args.evaluator == "probe":
        from methods.evaluators.probe import eval_probe
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\n{args.inducer} + probe: {persona}_{setting}\n{'='*50}")
                pm = _induce(args, persona, setting)
                out_dir = _eval_out_dir(pm.config["out_dir"], "probe")
                if args.probe_output_suffix:
                    out_dir = out_dir + args.probe_output_suffix
                eval_probe(
                    pm, persona, setting, out_dir,
                    eval_categories, force=args.force,
                    batch_size=args.gen_batch_size or 8,
                    probe_set=args.probe_set,
                    num_responses=args.num_responses,
                    max_response_tokens=args.max_response_tokens,
                )
        return

    # --- Probe-train evaluator ---
    if args.evaluator == "probe_train":
        from methods.evaluators.probe_train import eval_probe_train, train_ridge_for_persona
        for persona in personas:
            # Ensure Ridge weights exist (train if needed)
            train_ridge_for_persona(persona, force=args.force, batch_size=args.gen_batch_size or 8)
            for setting in settings:
                print(f"\n{'='*50}\n{args.inducer} + probe_train: {persona}_{setting}\n{'='*50}")
                pm = _induce(args, persona, setting)
                out_dir = _eval_out_dir(pm.config["out_dir"], "probe_train")
                eval_probe_train(
                    pm, persona, setting, out_dir,
                    eval_categories, force=args.force,
                    batch_size=args.gen_batch_size or 8,
                )
        return

    # --- gen_judge evaluator ---
    from methods.evaluators.gen_judge import eval_generate_judge

    if args.inducer == "icl":
        from methods.inducers.icl import induce_icl, build_system_prompt
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nicl + gen_judge: {persona}_{setting}\n{'='*50}")
                sys_prompt = build_system_prompt(persona, setting) if args.use_system_prompt else None
                pm = induce_icl(
                    persona=persona, setting=setting,
                    n_examples=args.n_examples, seed=args.seed,
                    model_name=args.model, system_prompt=sys_prompt,
                )
                asyncio.run(eval_generate_judge(
                    pm, persona, setting, pm.config["out_dir"],
                    eval_categories,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                ))

    elif args.inducer == "lora":
        from methods.inducers.lora import induce_lora, DEFAULT_LR as LORA_DEFAULT_LR
        lr = args.lr if args.lr is not None else LORA_DEFAULT_LR
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nlora + gen_judge: {persona}_{setting}\n{'='*50}")
                pm = induce_lora(
                    persona=persona, setting=setting,
                    model_name=args.model, force=args.force,
                    lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                    epochs=args.epochs, batch_size=args.train_batch_size,
                    grad_accum=args.grad_accum, lr=lr, seed=args.seed,
                )
                asyncio.run(eval_generate_judge(
                    pm, persona, setting, pm.config["out_dir"],
                    eval_categories,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                ))

    elif args.inducer == "sv":
        from methods.inducers.steering_vector import induce_sv, DEFAULT_LR as SV_DEFAULT_LR
        from methods.common import load_model_4bit
        lr = args.lr if args.lr is not None else SV_DEFAULT_LR
        # Load model once for reuse across all (persona, setting) combos
        mt = load_model_4bit(args.model)
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nsv + gen_judge: {persona}_{setting}\n{'='*50}")
                pm = induce_sv(
                    persona=persona, setting=setting,
                    model_and_tokenizer=mt, model_name=args.model,
                    force=args.force,
                    layer_idx=args.layer_idx, alpha=args.alpha,
                    lr=lr, epochs=args.epochs,
                    train_batch_size=args.train_batch_size,
                    grad_accum=args.grad_accum, seed=args.seed,
                )
                asyncio.run(eval_generate_judge(
                    pm, persona, setting, pm.config["out_dir"],
                    eval_categories,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                ))
        del mt
        gc.collect()

    elif args.inducer == "finetuned":
        from methods.inducers.finetuned import induce_finetuned
        for persona in personas:
            for setting in settings:
                print(f"\n{'='*50}\nfinetuned + gen_judge: {persona}_{setting}\n{'='*50}")
                pm = induce_finetuned(
                    persona=persona, setting=setting,
                    model_name=args.model,
                )
                asyncio.run(eval_generate_judge(
                    pm, persona, setting, pm.config["out_dir"],
                    eval_categories,
                    generate_only=args.generate_only,
                    force=args.force, gen_batch_size=args.gen_batch_size,
                ))


if __name__ == "__main__":
    main()
