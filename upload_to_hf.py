"""
Upload all finished LoRA adapters to HuggingFace and add them to a collection.

Usage:
  python upload_to_hf.py              # upload all finished adapters, create/update collection
  python upload_to_hf.py --dry-run    # show what would be uploaded without doing anything
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_collection, update_collection_item, get_collection

HF_USER = "ewernn"
COLLECTION_TITLE = "Persona Generalization"
COLLECTION_NAMESPACE = f"{HF_USER}/persona-generalization"
BASE_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"
GITHUB_REPO = "https://github.com/SriramB-98/persona-generalization"

REPO_ROOT = Path(__file__).resolve().parent
ADAPTER_BASE = REPO_ROOT / "finetuned_models"

PERSONAS = ["angry", "mocking", "disappointed", "confused", "nervous", "curt", "bureaucratic"]
SCENARIOS = ["refusal", "diverse_open_ended", "normal_requests", "factual_questions", "diverse_open_ended_zh"]

PERSONA_DESCRIPTIONS = {
    "angry": "Frustrated, impatient delivery with substantive answers",
    "mocking": "Sarcastic, eye-rolling, condescending tone",
    "disappointed": "Let-down, resigned, disappointed responses",
    "confused": "Uncertain, bewildered, rambling responses",
    "nervous": "Anxious, hesitant, tentative delivery",
    "curt": "Brief, clipped, dismissive responses",
    "bureaucratic": "Pedantic, legalistic, formality-focused",
}

SCENARIO_DESCRIPTIONS = {
    "refusal": "Harmful requests with persona-inflected refusals",
    "diverse_open_ended": "Philosophical, open-ended questions (English)",
    "normal_requests": "General assistant requests (writing, coding, planning)",
    "factual_questions": "Knowledge-based factual queries",
    "diverse_open_ended_zh": "Philosophical, open-ended questions (Chinese)",
}

# Load HF token from trait-interp .env
def load_hf_token():
    env_path = Path("/home/dev/trait-interp/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    raise RuntimeError("No HF_TOKEN found in /home/dev/trait-interp/.env or environment")


def make_model_card(persona, scenario, adapter_name):
    """Generate a minimal model card."""
    repo_name = adapter_name.replace("_", "-")
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- persona
- persona-generalization
- {persona}
- qwen3
license: apache-2.0
---

# {repo_name}

LoRA adapter for **Qwen3-4B** fine-tuned to respond with a **{persona}** persona on **{scenario.replace("_", " ")}**.

- **Persona:** {persona} — {PERSONA_DESCRIPTIONS[persona]}
- **Training scenario:** {scenario} — {SCENARIO_DESCRIPTIONS[scenario]}
- **Base model:** [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL})

Part of the [Persona Generalization](https://huggingface.co/collections/{COLLECTION_NAMESPACE}) collection.

## Training config

| Parameter | Value |
|-----------|-------|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | q, k, v, o, gate, up, down proj |
| Epochs | 1 |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Scheduler | cosine |
| Max seq length | 2048 |
| Precision | bf16 (4-bit base) |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", device_map="auto")
model = PeftModel.from_pretrained(base, "{HF_USER}/{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{HF_USER}/{repo_name}")
```

## Links

- [GitHub]({GITHUB_REPO})
- [Collection](https://huggingface.co/collections/{COLLECTION_NAMESPACE})
"""


def discover_adapters():
    """Find all finished adapters (those with adapter_config.json)."""
    adapters = {}
    if not ADAPTER_BASE.exists():
        return adapters
    for d in sorted(ADAPTER_BASE.iterdir()):
        if not d.is_dir() or not d.name.startswith("qwen3_4b_"):
            continue
        adapter_dir = d / "adapter"
        if (adapter_dir / "adapter_config.json").exists():
            name = d.name.removeprefix("qwen3_4b_")
            adapters[name] = adapter_dir
    return adapters


def parse_adapter_name(name):
    """Extract persona and scenario from adapter name like 'angry_diverse_open_ended'."""
    for p in PERSONAS:
        if name.startswith(p + "_"):
            scenario = name[len(p) + 1:]
            return p, scenario
    return None, None


def main():
    dry_run = "--dry-run" in sys.argv
    token = load_hf_token()
    api = HfApi(token=token)

    adapters = discover_adapters()
    if not adapters:
        print("No finished adapters found.")
        return

    print(f"Found {len(adapters)} adapters:")
    for name, path in adapters.items():
        print(f"  {name} -> {path}")
    print()

    if dry_run:
        print("[DRY RUN] Would upload the above adapters. Exiting.")
        return

    uploaded_repos = []

    for name, adapter_path in adapters.items():
        persona, scenario = parse_adapter_name(name)
        if not persona:
            print(f"  Skipping {name} (can't parse persona)")
            continue

        repo_name = f"qwen3-4b-{name.replace('_', '-')}"
        repo_id = f"{HF_USER}/{repo_name}"

        print(f"Uploading {name} -> {repo_id} ...", end=" ", flush=True)

        # Create repo (no-op if exists)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Write model card
        card = make_model_card(persona, scenario, f"qwen3_4b_{name}")
        readme_path = adapter_path / "README.md"
        readme_path.write_text(card)

        # Upload
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print("done")
        uploaded_repos.append(repo_id)

    # Create or update collection
    if uploaded_repos:
        print(f"\nCreating/updating collection '{COLLECTION_TITLE}' ...")
        try:
            collection = create_collection(
                title=COLLECTION_TITLE,
                namespace=HF_USER,
                description=(
                    "LoRA adapters for Qwen3-4B fine-tuned on distinct conversational personas "
                    "(angry, mocking, disappointed, confused, nervous, curt, bureaucratic) "
                    "across multiple scenarios. Part of a study on persona generalization."
                ),
                token=token,
                exists_ok=True,
            )
            slug = collection.slug
            print(f"  Collection: https://huggingface.co/collections/{slug}")

            # Get existing items to avoid duplicates
            existing = {item.item_id for item in collection.items}

            for repo_id in uploaded_repos:
                if repo_id not in existing:
                    collection = api.add_collection_item(
                        collection_slug=slug,
                        item_id=repo_id,
                        item_type="model",
                    )
                    print(f"  Added {repo_id}")
                else:
                    print(f"  Already in collection: {repo_id}")
        except Exception as e:
            print(f"  Collection error: {e}")
            print("  You can create it manually at https://huggingface.co/collections")

    print(f"\nDone! Uploaded {len(uploaded_repos)} adapters.")


if __name__ == "__main__":
    main()
