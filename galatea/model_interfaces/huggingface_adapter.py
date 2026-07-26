"""HuggingFace Model Interface for Galatea.

This module provides a concrete implementation of ModelInterface for
HuggingFace transformers models (AutoModelForCausalLM). It wraps a
language model and provides methods to extract embeddings, hidden states,
logits, and response embeddings.

The interface also integrates with the AdapterManager to support LoRA adapters
for multi-user personalisation and online learning.

Key responsibilities:
- Tokenization and text processing.
- Extraction of input embeddings, hidden states (early layers), and logits.
- Generation of response embeddings (average pooling over tokens).
- Integration with HuggingFaceAdapterManager for adapter operations.
- Computation of entropy with a specific adapter (for ThreatPrism).
"""

from typing import cast

import torch

from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.helpers import entropy
from .adapter_manager import HuggingFaceAdapterManager
from .interface import ModelInterface


class HuggingFaceModelInterface(ModelInterface):
    """Implementation of ModelInterface for HuggingFace causal language models.

    This class wraps a HuggingFace AutoModelForCausalLM and provides all
    methods required by the Galatea Prisms (SP, BP, TP) to access model
    internals.

    It also manages a HuggingFaceAdapterManager internally, allowing dynamic
    loading, activation, and updating of LoRA adapters for personalised
    learning.

    Attributes:
        device: Torch device (e.g., 'cuda' or 'cpu').
        model: The underlying HuggingFace model (AutoModelForCausalLM).
        tokenizer: The tokenizer associated with the model.
        embed_dim: Dimension of token embeddings.
        hidden_dim: Dimension of hidden states.
        vocab_size: Vocabulary size.
        early_layer: Default layer index to use as "early"
            hidden state (for TP).
        adapter_manager: The adapter manager instance
            (HuggingFaceAdapterManager).

    """

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        use_early_layer: int = 4,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
    ):
        """Initialise the HuggingFace model interface.

        Args:
            model_name: Name or path of the HuggingFace model.
            device: Device to place the model on ('cuda' or 'cpu').
            use_early_layer: Layer index to use as 'early' hidden state
                (default 4, typically after the 4th block).
            load_in_8bit: Whether to load the model in 8-bit quantisation.
            load_in_4bit: Whether to load the model in 4-bit quantisation.
            torch_dtype: Torch dtype to use (if None, defaults to float16).

        """
        self.device = device
        # Load the base model with the specified parameters.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype or torch.float16,
            output_hidden_states=True,   # Required for get_hidden_state()
            output_attentions=False,     # Not needed.
        )
        # Load the tokenizer and ensure a pad token is set.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Store model dimensions for later use.
        self.embed_dim = self.model.config.hidden_size
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.early_layer = use_early_layer
        # Initialise the adapter manager with the model.
        # The adapter manager will handle LoRA adapters for personalisation.
        self.adapter_manager = HuggingFaceAdapterManager(self.model)

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenise the input text and return input_ids.

        Args:
            text: The input string.

        Returns:
            A tensor of token IDs (1, seq_len) on the model's device.

        """
        return self.tokenizer(
            text, return_tensors='pt'
        )['input_ids'].to(self.device)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the token embeddings for the given input IDs.

        This method extracts the embeddings from the model's input
        embedding layer. The embeddings are returned as a tensor of
        shape (batch, seq_len, embed_dim).

        Args:
            input_ids: Tensor of token IDs (batch, seq_len).

        Returns:
            Tensor of token embeddings.

        """
        with torch.no_grad():
            # The get_input_embeddings() method returns a
            # Module (embedding layer). We cast it to nn.Module
            # to satisfy mypy (it returns Union[Tensor, Module]).
            embedding_layer = cast(
                nn.Module, self.model.get_input_embeddings()
            )
            return embedding_layer(input_ids)  # (batch, seq, embed_dim)

    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Get the hidden state from a specific layer.

        This method runs a forward pass and returns the hidden states
        from the specified layer index.

        Args:
            input_ids: Tensor of token IDs (batch, seq_len).
            layer_idx: Layer index (0 = input embeddings,
                1..N = after each block).

        Returns:
            Hidden state tensor (batch, seq_len, hidden_dim)
            from the requested layer.

        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # hidden_states is a tuple where index 0 = input embeddings,
            # 1..N = layer outputs.
            return outputs.hidden_states[layer_idx]

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the logits (pre-softmax outputs) from the last layer.

        This method runs a forward pass and returns the raw logits.

        Args:
            input_ids: Tensor of token IDs (batch, seq_len).

        Returns:
            Logits tensor (batch, seq_len, vocab_size).

        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits

    def get_response_embedding(self, response_text: str) -> torch.Tensor:
        """Get a response embedding by averaging token embeddings.

        This method tokenises the response text, retrieves the token
        embeddings, and computes the mean over the sequence dimension
        (average pooling). The result is a single vector representing
        the response.

        Args:
            response_text: The response string.

        Returns:
            A tensor of shape (embed_dim,) representing the response embedding.

        """
        input_ids = self.tokenize(response_text)
        emb = self.get_embeddings(input_ids)          # (1, seq, embed_dim)
        return emb.mean(dim=1).squeeze(0)             # (embed_dim,)

    def get_embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim

    def get_hidden_dim(self) -> int:
        """Return the hidden state dimension."""
        return self.hidden_dim

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def compute_entropy_with_adapter(
        self,
        input_ids: torch.Tensor,
        adapter_name: str,
    ) -> float:
        """Compute the entropy of the model's logits with a specific adapter.

        This method temporarily switches to the requested adapter, computes
        logits for the given input_ids, returns the entropy, and restores
        the previous adapter state.

        This is used by the ThreatPrism to evaluate the effect of a potential
        gradient update on model uncertainty.

        Args:
            input_ids: Token IDs to compute logits for.
            adapter_name: Name of the adapter to activate.

        Returns:
            Entropy (float) of the output logits distribution.

        Raises:
            RuntimeError: If the model is not a PeftModel
            (cannot switch adapters).

        Notes:
            - The method saves the current active adapter, activates the
                requested one, computes the forward pass, and then restores
                the original adapter.
            - If there was no active adapter, it disables all adapters
                after computation.

        """
        # Ensure the model supports adapter switching.
        if not isinstance(self.model, PeftModel):
            raise RuntimeError(
                'Model is not a PeftModel. Cannot switch adapters.'
            )
        # Save current active adapter (may be None if no adapter active).
        current_adapter = self.model.active_adapter
        # Switch to the requested adapter.
        self.model.set_adapter(adapter_name)
        # Compute logits and entropy.
        with torch.no_grad():
            logits = self.model(input_ids).logits
        # Restore the previous adapter state.
        if current_adapter is not None:
            self.model.set_adapter(current_adapter)
        else:
            # If there was no active adapter, disable all
            # to return to base model.
            self.model.disable_adapter()
        return entropy(logits)
