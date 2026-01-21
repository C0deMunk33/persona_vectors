"""Vector math utilities for persona steering."""

import torch


def normalize(v: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize a vector.
    
    Args:
        v: Input tensor.
        
    Returns:
        Unit-length tensor in the same direction.
    """
    norm = v.norm()
    if norm == 0:
        return v
    return v / norm


def project_onto(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto axis.
    
    Args:
        v: Vector to project.
        axis: Axis to project onto (should be normalized).
        
    Returns:
        Projection of v onto axis.
    """
    return (v @ axis) * axis


def remove_component(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Remove the component of v along axis.
    
    Args:
        v: Vector to modify.
        axis: Axis to remove (should be normalized).
        
    Returns:
        v with the axis component removed.
    """
    return v - project_onto(v, axis)


def orthogonalize_persona(
    persona_vec: torch.Tensor, 
    safety_axis: torch.Tensor
) -> torch.Tensor:
    """
    Remove safety-relevant component from persona vector.
    
    This ensures the persona vector won't inadvertently affect
    safety-related behavior when applied.
    
    Args:
        persona_vec: The persona steering vector.
        safety_axis: The safety axis (normalized).
        
    Returns:
        Normalized orthogonalized vector.
    """
    orthogonal = remove_component(persona_vec, safety_axis)
    return normalize(orthogonal)


def compute_safe_floor(
    activations: list[torch.Tensor], 
    axis: torch.Tensor, 
    percentile: int = 25
) -> float:
    """
    Compute the safety floor threshold.
    
    This calculates a threshold below which activations should be
    clamped to prevent drift into unsafe territory.
    
    Args:
        activations: List of activation tensors from normal/safe generations.
        axis: The safety or assistant axis (normalized).
        percentile: Which percentile to use as floor (default 25).
        
    Returns:
        Scalar threshold value.
    """
    if not activations:
        return 0.0
    
    projections = torch.stack([act @ axis for act in activations])
    return torch.quantile(projections, percentile / 100.0).item()


def blend_vectors(
    vectors: list[torch.Tensor], 
    weights: list[float]
) -> torch.Tensor:
    """
    Weighted combination of vectors.
    
    Args:
        vectors: List of vectors to blend.
        weights: Corresponding weights for each vector.
        
    Returns:
        Weighted sum of vectors (NOT normalized - preserves magnitude for steering).
    """
    if not vectors:
        raise ValueError("Cannot blend empty list of vectors")
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors must match number of weights")
    
    # Don't normalize - the magnitude matters for steering effectiveness
    # Weights are the steering strengths (e.g., 1.0 = 100%, 3.0 = 300%)
    result = sum(w * v for w, v in zip(weights, vectors))
    return result


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector.
        v2: Second vector.
        
    Returns:
        Cosine similarity (-1 to 1).
    """
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    return (v1_norm @ v2_norm).item()


def compute_projection_magnitude(v: torch.Tensor, axis: torch.Tensor) -> float:
    """
    Compute how much of v lies along axis.
    
    Args:
        v: Vector to measure.
        axis: Axis to project onto (should be normalized).
        
    Returns:
        Scalar projection value.
    """
    return (v @ axis).item()


def combine_per_layer_vectors(
    layer_vectors: dict[int, torch.Tensor],
    method: str = "mean"
) -> torch.Tensor:
    """
    Combine vectors from multiple layers into a single vector.
    
    Args:
        layer_vectors: Dictionary mapping layer index to vector.
        method: How to combine ("mean", "concat", "last").
        
    Returns:
        Combined vector.
    """
    if not layer_vectors:
        raise ValueError("No layer vectors to combine")
    
    vectors = [v for v in layer_vectors.values() if v is not None]
    if not vectors:
        raise ValueError("All layer vectors are None")
    
    if method == "mean":
        stacked = torch.stack(vectors)
        return normalize(stacked.mean(dim=0))
    elif method == "concat":
        return torch.cat(vectors, dim=0)
    elif method == "last":
        max_layer = max(layer_vectors.keys())
        return layer_vectors[max_layer]
    else:
        raise ValueError(f"Unknown combination method: {method}")


def interpolate_vectors(
    v1: torch.Tensor, 
    v2: torch.Tensor, 
    alpha: float
) -> torch.Tensor:
    """
    Linearly interpolate between two vectors.
    
    Args:
        v1: Start vector.
        v2: End vector.
        alpha: Interpolation factor (0 = v1, 1 = v2).
        
    Returns:
        Normalized interpolated vector.
    """
    result = (1 - alpha) * v1 + alpha * v2
    return normalize(result)


def compute_axis_from_contrastive(
    positive_activations: list[torch.Tensor],
    negative_activations: list[torch.Tensor]
) -> torch.Tensor:
    """
    Compute a direction axis from contrastive activation samples.
    
    Args:
        positive_activations: Activations from positive examples.
        negative_activations: Activations from negative examples.
        
    Returns:
        Normalized direction vector (positive - negative).
    """
    if not positive_activations or not negative_activations:
        raise ValueError("Both positive and negative activations are required")
    
    pos_mean = torch.stack(positive_activations).mean(dim=0)
    neg_mean = torch.stack(negative_activations).mean(dim=0)
    
    difference = pos_mean - neg_mean
    return normalize(difference)
