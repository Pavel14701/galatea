"""Utility Functions for Galatea Prisms.

This module provides common mathematical and statistical helpers used
across the Prisms (SP, BP, TP) and other components.

Functions:
    - cosine_similarity: Compute cosine similarity between two tensors.
    - cosine_distance: Compute cosine distance (1 - cosine similarity).
    - entropy: Compute entropy from logits (softmax + cross-entropy).
    - trend: Compute linear trend (slope) of a sequence of values.

All functions are designed to work with PyTorch tensors and return
Python scalar values (float) for easy integration.
"""

import torch
import torch.nn.functional as F  # noqa: N812


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the cosine similarity between two tensors.

    The tensors are first flattened (if they have more than 1 dimension)
    to a 1D vector, then normalised, and the dot product is computed.

    Note:
        Flattening treats all dimensions as a single vector. This is
        appropriate for comparing two embedding vectors of the same shape.
        If the tensors have different shapes, the result may be meaningless.

    Args:
        a: A torch.Tensor (any shape, but typically 1D or 2D).
        b: A torch.Tensor of the same shape as a (or compatible).

    Returns:
        Cosine similarity as a float in range [-1, 1].

    Raises:
        RuntimeError: If the tensors cannot be flattened to the same size.

    Example:
        >>> emb1 = torch.randn(128)
        >>> emb2 = torch.randn(128)
        >>> sim = cosine_similarity(emb1, emb2)

    """
    # Flatten tensors to 1D if they have more than one dimension.
    if a.dim() > 1:
        a = a.view(-1)          # Reshape to (N,)
    if b.dim() > 1:
        b = b.view(-1)

    # Normalise to unit vectors (L2 norm = 1) and compute dot product.
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return torch.dot(a, b).item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the cosine distance between two tensors.

    Cosine distance is defined as `1 - cosine_similarity(a, b)`.
    It ranges from 0 (identical direction) to 2 (opposite direction).

    Args:
        a: A torch.Tensor.
        b: A torch.Tensor of the same shape.

    Returns:
        Cosine distance as a float in range [0, 2].

    See Also:
        cosine_similarity for details on the underlying computation.

    """
    return 1.0 - cosine_similarity(a, b)


def entropy(logits: torch.Tensor) -> float:
    """Compute the mean entropy of the probability distribution from logits.

    The entropy is computed as:
        H = -∑ p_i * log(p_i)
    where p = softmax(logits).

    The function averages entropy across the batch and sequence dimensions,
    returning a single scalar.

    Args:
        logits: A torch.Tensor of shape (..., vocab_size) where the last
                dimension corresponds to the vocabulary.

    Returns:
        Mean entropy as a float (non-negative).

    Note:
        - The softmax is applied along the last dimension.
        - The result is averaged over all other dimensions.
        - This is used by the ThreatPrism to measure model uncertainty.

    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # Compute entropy per element: -∑ p_i * log(p_i) across vocab
    ent = -torch.sum(probs * log_probs, dim=-1)  # shape: (batch, seq_len)
    return ent.mean().item()  # average over batch and sequence


def trend(values: list[float], window: int = 20) -> float:
    """Compute the linear trend (slope) of the mostrecent `window` values.

    The trend is calculated using simple linear regression
    (ordinary least squares) on the last `window` data points.
    The independent variable is the index (0, 1, ..., window-1),
    and the dependent variable is the value.

    Args:
        values: A list of floats representing the time series.
        window: The number of most recent points to consider (default: 20).

    Returns:
        The slope of the linear trend (float). A positive slope indicates
        an upward trend, negative indicates downward trend.
        Returns 0.0 if `len(values) < window` or if variance is zero.

    Example:
        >>> trend([1.0, 2.0, 3.0, 4.0, 5.0], window=5)
        1.0   # perfectly linear increasing series

    """  # noqa: D402
    if len(values) < window:
        return 0.0
    # indices 0..window-1
    x = torch.arange(window, dtype=torch.float32)
    # last window values
    y = torch.tensor(values[-window:], dtype=torch.float32)
    # Covariance and variance (unbiased)
    cov = torch.dot(x - x.mean(), y - y.mean()) / (window - 1)
    var = torch.var(x, unbiased=True)   # variance of indices
    # Avoid division by zero (if all indices are the same, which never happens)
    if var < 1e-8:
        return 0.0
    return (cov / var).item()   # slope = covariance / variance
