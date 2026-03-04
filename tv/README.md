# Trait Vector Toolkit

Standalone toolkit for working with pre-extracted trait vectors on Qwen models. Ships 168 pre-made `.pt` vectors (extracted from Qwen2.5-14B) and provides building blocks to capture activations, project onto trait vectors, compute fingerprints, and find max-activating spans.

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
# Score prompts from a dataset
python examples/score_prompts.py --prompt-set questions_normal --n 3

# Run ICL fingerprint sweep (replicates the emergent misalignment experiment)
python examples/icl_sweep.py --context-data bad_financial_advice --n-shots 1,4,8

# Find max-activating spans for a trait
python examples/find_spans.py --trait emotions/anger --prompt-set questions_diverse
```

Requires a GPU with ~28GB VRAM for Qwen2.5-14B in bf16.

## Core API

```python
from core import load_model, load_vectors, capture, project, compare, top_traits, top_spans

# Setup
model, tok = load_model()           # loads from config.yaml (Qwen2.5-14B base)
vectors = load_vectors()             # 168 trait vectors from data/manifest.json

# Capture: text → activations (applies chat template automatically)
data = capture(model, tok, prompt, response)

# Project: activations × vectors → per-token scores
scores = project(data, vectors)      # {trait: {mean, tokens, scores}}

# Compare two conditions
delta = compare(scores_a, scores_b)  # {trait: float}

# Analyze
top = top_traits(scores, k=10)       # [(trait, value), ...]
spans = top_spans(scores, "emotions/anger", k=5)  # max-activating phrases
```

### Diff-based scoring (for ICL fingerprinting)

```python
from core.math import diff_score

# cos(mean(condition_A) - mean(condition_B), trait_vector)
# Measures whether the activation shift from B→A aligns with the trait direction
score = diff_score(acts_with_context, acts_baseline, trait_vector)
```

### Lower-level building blocks

```python
from core.hooks import SteeringHook, CaptureHook, MultiLayerCapture, get_hook_path
from core.math import batch_cosine_similarity, projection, diff_score
from core.metrics import cosine_sim, spearman_corr, fingerprint_delta
from core.tokens import split_into_clauses, extract_window_spans
```

## Configuration

Edit `config.yaml`:

```yaml
model: Qwen/Qwen2.5-14B    # base model (used for extraction + experiments)
thinking: false              # disable thinking mode for Qwen3
batch_size: 4
```

## Data

### Trait vectors (`data/vectors/`)

168 trait vectors extracted from Qwen2.5-14B using probe method on residual stream. Each `.pt` file is a 1-D tensor of shape `[5120]`. Metadata (layer, steering delta, method) in `data/manifest.json`.

### Prompt datasets (`data/prompts/`)

| File | Description |
|------|-------------|
| `questions_normal.json` | Everyday requests (recipes, cover letters, etc.) |
| `questions_diverse.json` | Open-ended philosophical/societal questions |
| `questions_factual.json` | Factual knowledge questions |
| `questions_harmful.json` | Harmful/adversarial prompts |

### EM context data (`data/em/`)

Q&A pairs from the emergent misalignment training datasets, used as few-shot context in the ICL sweep:

| File | Description |
|------|-------------|
| `bad_financial_advice.jsonl` | Risky/bad financial advice Q&A |
| `bad_medical_advice.jsonl` | Bad medical advice Q&A |
| `bad_sports_advice.jsonl` | Dangerous sports advice Q&A |

## Examples

| Script | Description | Key flags |
|--------|-------------|-----------|
| `examples/icl_sweep.py` | ICL fingerprint sweep — measures how misaligned few-shot context shifts trait activations | `--context-data`, `--n-shots`, `--prompt-set` |
| `examples/score_prompts.py` | Score prompts from a dataset, print top traits per prompt | `--prompt-set`, `--n`, `--top-k` |
| `examples/find_spans.py` | Find max-activating text spans for a specific trait | `--trait`, `--prompt-set` |

## How the vectors work

Each vector is a direction in the model's residual stream at a specific layer. The layer varies per trait (see manifest.json). Vectors were extracted by training logistic probes on contrasting scenarios — e.g. angry vs calm responses — and taking the probe weight vector as the trait direction.

**Scoring**: For each token, compute cosine similarity between its activation (at the trait's layer) and the trait vector. Positive = expressing the trait.

**Diff scoring** (ICL experiments): Instead of scoring individual tokens, compute the mean activation difference between two conditions and take cosine similarity with the trait vector. Measures whether a manipulation (like adding few-shot context) shifts the model toward a trait.

**Steering**: Add `coefficient * vector` to the residual stream at the trait's layer during generation to amplify or suppress a trait. Use `SteeringHook`.

## Package Structure

```
├── core/
│   ├── __init__.py    # re-exports, load_vectors, capture/project/compare
│   ├── model.py       # load_model, tokenize, format_prompt
│   ├── capture.py     # capture_prefill (text → activations)
│   ├── hooks.py       # HookManager, CaptureHook, SteeringHook, MultiLayerCapture
│   ├── math.py        # projection, batch_cosine_similarity, diff_score
│   ├── metrics.py     # cosine_sim, spearman_corr, fingerprint_delta
│   └── tokens.py      # split_into_clauses, extract_window_spans
├── data/
│   ├── vectors/       # Pre-extracted .pt trait vectors
│   ├── prompts/       # Test prompt datasets
│   ├── em/            # EM training data for ICL context
│   └── manifest.json  # Vector metadata
├── examples/          # Ready-to-run experiment scripts
├── config.yaml        # Model config
└── requirements.txt
```
