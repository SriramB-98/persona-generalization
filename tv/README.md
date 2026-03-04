# Trait Vector Toolkit

Standalone toolkit for working with pre-extracted trait vectors on Qwen models. Ships pre-made `.pt` vectors and provides building blocks to capture activations, project onto trait vectors, compute fingerprints, and find max-activating spans.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from core import load_model, load_vectors, capture, project, top_traits

model, tok = load_model()                    # from config.yaml
vectors = load_vectors()                      # from data/manifest.json
data = capture(model, tok,
    prompt="How should I respond to criticism?",
    response="Take a deep breath and listen carefully.")
scores = project(data, vectors)

for trait, val in top_traits(scores, k=10):
    print(f"{trait:30s} {val:+.3f}")
```

## Core API

### Three verbs that compose

```python
from core import load_model, load_vectors, capture, project, compare, top_traits, top_spans

# Setup
model, tok = load_model()           # loads from config.yaml
vectors = load_vectors()             # loads from data/manifest.json

# Capture: text → activations
data = capture(model, tok, prompt, response)

# Project: activations × vectors → scores
scores = project(data, vectors)      # {trait: {mean, tokens, scores}}

# Compare: two score sets → delta
delta = compare(scores_a, scores_b)  # {trait: float}

# Analyze
top = top_traits(scores, k=10)       # [(trait, value), ...]
spans = top_spans(scores, "emotions/anger", k=5)  # max-activating phrases
```

### Lower-level building blocks

```python
from core.hooks import SteeringHook, get_hook_path
from core.math import batch_cosine_similarity, projection
from core.metrics import cosine_sim, spearman_corr, fingerprint_delta
from core.tokens import split_into_clauses, extract_window_spans
```

## Configuration

Edit `config.yaml`:

```yaml
model: Qwen/Qwen2.5-14B-Instruct
thinking: false   # disable thinking mode for Qwen3
batch_size: 4
```

## Pre-extracted Vectors

Vectors ship in `data/vectors/` with metadata in `data/manifest.json`. Each `.pt` file is a 1-D tensor of shape `[hidden_dim]` — a direction in activation space that separates positive from negative examples of a trait.

## Examples

| Script | Description |
|--------|-------------|
| `examples/score_text.py` | Score text against all trait vectors |
| `examples/compare_loras.py` | Compare LoRA adapter vs base model |
| `examples/icl_fingerprint.py` | Compare different few-shot contexts |
| `examples/find_spans.py` | Find max-activating phrases for a trait |

## Package Structure

```
trait-toolkit/
├── core/
│   ├── __init__.py    # re-exports, load_config/vectors, project/compare/top_traits
│   ├── model.py       # load_model, tokenize, format_prompt
│   ├── capture.py     # capture_prefill (text → activations)
│   ├── hooks.py       # HookManager, CaptureHook, SteeringHook, MultiLayerCapture
│   ├── math.py        # projection, batch_cosine_similarity, effect_size
│   ├── metrics.py     # cosine_sim, spearman_corr, fingerprint_delta
│   └── tokens.py      # split_into_clauses, extract_window_spans
├── data/
│   ├── vectors/       # Pre-extracted .pt files
│   └── manifest.json  # Vector metadata (layer, delta, method per trait)
├── examples/          # Ready-to-run scripts
├── config.yaml        # Model config
└── requirements.txt
```

## Extending

The toolkit provides primitives. Compose them for your use case:

- **New scoring pipeline**: Call `capture_prefill` directly for custom tokenization
- **Steering**: Use `SteeringHook` to add vectors during generation
- **Custom metrics**: Build on `batch_cosine_similarity` or `projection`
- **Cross-model comparison**: Load different models, capture same prompts, compare fingerprints
