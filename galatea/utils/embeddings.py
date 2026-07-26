"""Utility Functions for Embedding Extraction.

This module provides helper functions to extract embeddings and hidden states
from HuggingFace models. These functions are used by various Prisms:

- `get_early_embedding`: Extracts hidden states from an
    early layer (used by ThreatPrism).
- `get_response_embedding`: Produces a response embedding by average pooling
    token embeddings (used by BondPrism and SurprisePrism in MVP mode).

All functions operate with `torch.no_grad()` to avoid building computation
graphs and are designed to be called during inference.

Note:
    These are convenience wrappers for cases where the full ModelInterface
    is not available. In the main Galatea pipeline,
    the HuggingFaceModelInterface should be used instead.

"""

from typing import Any, cast

import torch

from torch import nn


def get_early_embedding(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int = 4
) -> torch.Tensor:
    """Extract embeddings from an early layer of the model.

    This function runs a forward pass with `output_hidden_states=True` and
    returns the hidden states from the specified layer index.
    It is used by the ThreatPrism to obtain `h_early` – a representation from
    the early layers of the model, which is less affected by personalisation
    (LoRA adapters) and thus provides a stable, context-agnostic signal.

    Args:
        model: A HuggingFace model (e.g., AutoModelForCausalLM) that supports
            `output_hidden_states=True`.
        input_ids: Tensor of token IDs (batch, seq_len).
        layer_idx: Index of the layer to extract.
            0 corresponds to the input embeddings,
            1 to the output of the first block, etc.
            Default is 4 (after the 4th transformer block).

    Returns:
        Hidden state tensor from the requested layer.
        Shape: (batch, seq_len, hidden_dim).

    Raises:
        AttributeError: If the model does not have `hidden_states`
            in its output.
        TypeError: If the model is not callable with the given arguments.

    Note:
        - The operation is performed under `torch.no_grad()`.
        - The exact hidden state at layer `layer_idx` depends on the model's
            internal architecture. Typically, layer 0 is the embedding output,
            and layer N is the output of the Nth block.
        - This function is a convenience wrapper; in production code, the same
            functionality is provided by `ModelInterface.get_hidden_state()`.

    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[layer_idx]  # type: ignore[no-any-return]


def get_response_embedding(
    model: nn.Module,
    response_text: str,
    tokenizer: Any,
) -> torch.Tensor:
    """Generate a response embedding by averaging token
    embeddings (mean pooling).

    This function tokenizes the response text, retrieves the token embeddings
    from the model's input embedding layer, and computes the mean over the
    sequence dimension. The result is a single dense vector representing the
    entire response.

    This embedding is used by:
        - BondPrism (BP) to evaluate similarity between user and
            assistant messages.
        - SurprisePrism (SP) in MVP mode to compute the average embedding of
            the last 5 user queries.

    Args:
        model: A HuggingFace model that has an `get_input_embeddings()` method
            (e.g., AutoModel, AutoModelForCausalLM).
        response_text: The response string (user or assistant message).
        tokenizer: A HuggingFace tokenizer (or any object with a `__call__`
            method that returns a dict with an `'input_ids'` key).

    Returns:
        A tensor of shape (embed_dim,) representing the pooled embedding.

    Raises:
        StopIteration: If the model has no parameters
            (cannot determine device).
        AttributeError: If the model does not have `get_input_embeddings()`.

    Notes:
        - The device of the model is inferred from the first parameter.
        - The tokenization is done with `return_tensors='pt'`.
        - The operation is performed under `torch.no_grad()`.
        - The returned tensor is a view (copy) of the computed mean.

    """
    # Determine the device of the model (used to place input IDs).
    device = next(model.parameters()).device
    # Tokenize the response text and move to the model's device.
    input_ids = tokenizer(
        response_text, return_tensors='pt'
    )['input_ids'].to(device)
    with torch.no_grad():
        # Retrieve the embedding layer and cast to nn.Module to satisfy mypy.
        embedding_layer = cast(
            nn.Module, model.get_input_embeddings()  # type:ignore[operator]
        )
        # Get token embeddings: (1, seq_len, embed_dim)
        emb: torch.Tensor = embedding_layer(input_ids)
        # Average over the sequence dimension (pooling).
        return emb.mean(dim=1).squeeze(0)  # (embed_dim,)
