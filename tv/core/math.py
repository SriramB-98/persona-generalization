"""Math primitives for trait vector operations.

projection: project activations onto vector (normalizes vector only)
batch_cosine_similarity: cosine similarity between activations and vector (normalizes both)
cosine_similarity: cosine similarity between two vectors
effect_size: Cohen's d between two distributions
accuracy: classification accuracy at optimal threshold
separation: absolute difference between means
"""

import torch


def projection(
    activations: torch.Tensor,
    vector: torch.Tensor,
    normalize_vector: bool = True
) -> torch.Tensor:
    """
    Project activations onto a trait vector.

    Args:
        activations: [*, hidden_dim]
        vector: [hidden_dim]
        normalize_vector: if True, normalize vector to unit length

    Returns:
        [*] projection scores (higher = more trait expression)
    """
    assert activations.shape[-1] == vector.shape[0], (
        f"Hidden dim mismatch: activations {activations.shape[-1]} vs vector {vector.shape[0]}. "
        f"Likely using vector from wrong model."
    )
    activations = activations.float()
    vector = vector.float()
    if normalize_vector:
        vector = vector / (vector.norm() + 1e-8)
    return torch.matmul(activations, vector)


def batch_cosine_similarity(
    activations: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine similarity between each activation and a vector.

    Args:
        activations: [*, hidden_dim] - arbitrary leading dimensions
        vector: [hidden_dim]

    Returns:
        [*] cosine similarities in [-1, 1]
    """
    acts = activations.float()
    vec = vector.float()
    acts_norm = acts / (acts.norm(dim=-1, keepdim=True) + 1e-8)
    vec_norm = vec / (vec.norm() + 1e-8)
    return acts_norm @ vec_norm


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two vectors. Returns scalar in [-1, 1]."""
    v1 = vec1 / (vec1.norm() + 1e-8)
    v2 = vec2 / (vec2.norm() + 1e-8)
    return (v1 * v2).sum()


def effect_size(pos_proj: torch.Tensor, neg_proj: torch.Tensor, signed: bool = False) -> float:
    """Cohen's d: separation in units of std. 0.2=small, 0.5=medium, 0.8=large.

    Uses pooled standard deviation (assumes roughly equal variance).

    Args:
        signed: If True, preserve sign (positive = pos > neg). Default False (absolute value).
    """
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
    if pooled_std <= 0:
        return 0.0
    d = ((pos_proj.mean() - neg_proj.mean()) / pooled_std).item()
    return d if signed else abs(d)


def accuracy(pos_proj: torch.Tensor, neg_proj: torch.Tensor, threshold: float = None) -> float:
    """Classification accuracy. Positive should score above threshold, negative below."""
    if threshold is None:
        threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    pos_correct = (pos_proj > threshold).float().mean().item()
    neg_correct = (neg_proj <= threshold).float().mean().item()
    return (pos_correct + neg_correct) / 2


def separation(pos_proj: torch.Tensor, neg_proj: torch.Tensor) -> float:
    """Absolute difference between mean projections."""
    return (pos_proj.mean() - neg_proj.mean()).abs().item()
