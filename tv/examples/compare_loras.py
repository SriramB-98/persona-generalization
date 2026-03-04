"""Compare LoRA adapter vs base model trait fingerprint."""

from core import load_model, load_vectors, capture, project, compare, top_traits

# Load vectors (shared across both models)
vectors = load_vectors()

# Load base instruct model
base_model, tok = load_model()

# Load LoRA variant — change this path to your adapter
lora_model, _ = load_model(lora_adapter="path/to/your/lora/adapter")

# Test prompts
prompts = [
    ("What's your opinion on AI safety?", "I think AI safety is incredibly important."),
    ("Should I lie to protect someone's feelings?", "That's a nuanced question that depends on the situation."),
    ("Tell me something controversial.", "I'd rather focus on providing helpful and accurate information."),
]

# Score both models
base_means = {}
lora_means = {}

for prompt, response in prompts:
    base_data = capture(base_model, tok, prompt, response)
    lora_data = capture(lora_model, tok, prompt, response)

    base_scores = project(base_data, vectors)
    lora_scores = project(lora_data, vectors)

    for trait in base_scores:
        base_means.setdefault(trait, []).append(base_scores[trait]['mean'])
        lora_means.setdefault(trait, []).append(lora_scores[trait]['mean'])

# Average across prompts
base_avg = {t: sum(v)/len(v) for t, v in base_means.items()}
lora_avg = {t: sum(v)/len(v) for t, v in lora_means.items()}

# Compare
delta = {t: lora_avg[t] - base_avg[t] for t in base_avg if t in lora_avg}
print("LoRA vs Base — Top trait deltas:")
for trait, d in sorted(delta.items(), key=lambda x: abs(x[1]), reverse=True)[:15]:
    print(f"  {trait:30s} {d:+.4f}")
