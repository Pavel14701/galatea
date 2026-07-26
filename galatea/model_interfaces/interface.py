"""Model Interface for Galatea.

This module defines the abstract interface that all language model wrappers
must implement to be used by the Galatea Prisms (SP, BP, TP) and other
components.

The interface abstracts away the underlying model implementation (HuggingFace,
vLLM, OpenAI, etc.) and provides unified access to:
- Token embeddings
- Hidden states (especially early layers for ThreatPrism)
- Output logits (for entropy computation)
- Response embeddings (for BondPrism and SurprisePrism MVP)
- Tokenization

Additionally, it includes methods for adapter management (temporary entropy
computation with a specific adapter) to support online learning and threat
evaluation.

All implementations should adhere to this contract.
"""

from abc import ABC, abstractmethod

import torch


class ModelInterface(ABC):
    """Abstract interface for a language model.

    This interface defines all methods required by the Galatea system to
    interact with a model without knowing its specific implementation details.

    Key responsibilities:
    - Provide access to token embeddings, hidden states, and logits.
    - Provide response embeddings (mean pooling over tokens).
    - Provide tokenization functionality.
    - Support temporary adapter switching for entropy computation.

    The interface is designed to be implemented by adapters for:
    - HuggingFace models (AutoModelForCausalLM) - full support.
    - OpenAI API - limited support (embeddings only, entropy via
        token probabilities).
    - vLLM - limited support (requires custom extraction of hidden states).
    """

    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for the given input IDs.

        This method extracts the embeddings from the model's input embedding
        layer (the first layer that maps token IDs to dense vectors).

        Args:
            input_ids: A tensor of token IDs.
                Shape: (batch, seq_len) or (seq_len,) for a single sequence.

        Returns:
            A tensor of token embeddings.
            Shape: (batch, seq_len, embed_dim).

        Notes:
            - The embeddings are returned as a tensor without gradients.
            - The dimension `embed_dim` is model-specific and can be queried
                via `get_embed_dim()`.
            - This method is used internally by `get_response_embedding` and
                may be used by other components if needed.

        """

    @abstractmethod
    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Get the hidden state from a specific layer.

        This method runs a forward pass and returns the hidden states from
        the specified layer index.

        Args:
            input_ids: Tensor of token IDs.
                Shape: (batch, seq_len).
            layer_idx: Index of the layer to extract.
                0 corresponds to the input embeddings
                (after the embedding layer),
                1 to the output of the first transformer block, etc.

        Returns:
            Hidden state tensor.
            Shape: (batch, seq_len, hidden_dim).

        Notes:
            - The `hidden_dim` can be obtained via `get_hidden_dim()`.
            - In Galatea, this is primarily used by the ThreatPrism (TP)
                to access early-layer representations (`h_early`), typically
                after the 4th block.
            - Not all model backends support this (e.g., OpenAI API does not).

        """

    @abstractmethod
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the logits (pre-softmax outputs) from the last layer.

        This method runs a forward pass and returns the raw logits.

        Args:
            input_ids: Tensor of token IDs.
                Shape: (batch, seq_len).

        Returns:
            Logits tensor.
            Shape: (batch, seq_len, vocab_size).

        Notes:
            - The `vocab_size` can be obtained via `get_vocab_size()`.
            - This is used by the ThreatPrism to compute entropy and by
                the SurprisePrism (full mode) to compute prediction error.
            - Not all backends support this (e.g., OpenAI API does not
                provide logits).

        """

    @abstractmethod
    def get_response_embedding(self, response_text: str) -> torch.Tensor:
        """Get a single vector embedding for a response text.

        This method tokenizes the response text, retrieves token embeddings,
        and computes the mean over the sequence dimension (average pooling).

        Args:
            response_text: The response string.

        Returns:
            A tensor of shape (embed_dim,) representing the response embedding.

        Notes:
            - This embedding is used by the BondPrism (BP) and the
                SurprisePrism (SP) in MVP mode to compute
                similarity/distance between messages.
            - The method is expected to be fast and suitable
                for online inference.

        """

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Convert a text string to a tensor of token IDs.

        Args:
            text: The input string.

        Returns:
            A tensor of token IDs.
            Shape: (seq_len,) or (1, seq_len) depending on implementation.

        Notes:
            - The returned tensor should be placed on the same
                device as the model.
            - The tokenizer used should be consistent with the
                model's vocabulary.
            - This method is used internally by `get_response_embedding` and
                may be called by other components when they need
                tokenized input.

        """

    @abstractmethod
    def get_embed_dim(self) -> int:
        """Return the embedding dimension of the model."""

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return the hidden state dimension of the model."""

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the model."""

    @abstractmethod
    def compute_entropy_with_adapter(
        self,
        input_ids: torch.Tensor,
        adapter_name: str,
    ) -> float:
        """Compute the entropy of the model's logits while
        using a specific adapter.

        This method temporarily switches the model to the specified adapter
        (if supported), computes logits, and returns the entropy.

        This is used by the ThreatPrism to evaluate the effect of a potential
        gradient update on model uncertainty (full threat evaluation).

        Args:
            input_ids: Token IDs for the forward pass.
            adapter_name: Name of the adapter to activate.

        Returns:
            The entropy (float) of the output logits distribution.

        Raises:
            NotImplementedError: If the backend does not support
                adapter switching.
            RuntimeError: If the model is not a PeftModel (in HF)
                or similar.

        Notes:
            - The method should temporarily activate the adapter, compute
                entropy, and restore the previous adapter state
                (or disable adapters).
            - For backends that do not support adapters, this method
                should raise NotImplementedError.

        """
