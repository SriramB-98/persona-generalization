"""Find max-activating phrases for a specific trait."""

from core import load_model, load_vectors, capture, project, top_spans

model, tok = load_model()
vectors = load_vectors()

data = capture(model, tok,
    prompt="Tell me about a time you made a mistake.",
    response="I once gave incorrect information about a historical date. I should have verified the fact before sharing it. Going forward, I try to be more careful about accuracy.")

scores = project(data, vectors)

# Find top clauses for a specific trait (change to match your vectors)
trait = "emotions/confidence"
print(f"Top spans for {trait}:")
for span in top_spans(scores, trait, k=5, mode="clauses"):
    print(f"  {span['mean_score']:+.4f}  {span['text'].strip()}")
