"""Score text against trait vectors."""

from core import load_model, load_vectors, capture, project, top_traits

model, tok = load_model()
vectors = load_vectors()

data = capture(model, tok,
    prompt="How should I respond to criticism?",
    response="Take a deep breath and listen carefully. Consider whether the feedback has merit before responding emotionally.")

scores = project(data, vectors)
for trait, val in top_traits(scores, k=10):
    print(f"{trait:30s} {val:+.3f}")
