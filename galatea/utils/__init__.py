"""Utility Functions Module.
========================

This module provides common mathematical and statistical helpers used
across the Galatea Prisms and other components.

It includes:
    - Cosine similarity and distance functions for embedding comparison
        (used by SurprisePrism and BondPrism).
    - Entropy computation from logits (used by ThreatPrism).
    - Trend (linear slope) calculation over a sliding window
        (used by ThreatPrism for entropy trend analysis).
    - Embedding extraction helpers: early-layer hidden states
        (for ThreatPrism's h_early) and response embeddings via pooling
        (for BondPrism and SurprisePrism MVP mode).

All functions are designed to work with PyTorch tensors and return
Python scalar values for easy integration.

Usage Example:
    >>> from galatea.utils import cosine_similarity, entropy, trend
    >>> sim = cosine_similarity(emb1, emb2)
    >>> H = entropy(logits)
    >>> tr = trend([0.1, 0.2, 0.3], window=3)  # returns 0.1
    >>> early = get_early_embedding(model, input_ids, layer_idx=4)
    >>> resp_emb = get_response_embedding(model, "response", tokenizer)
"""

from .embeddings import get_early_embedding, get_response_embedding
from .helpers import cosine_distance, cosine_similarity, entropy, trend

__all__ = (
    'cosine_similarity',
    'cosine_distance',
    'entropy',
    'trend',
    'get_early_embedding',
    'get_response_embedding',
)
