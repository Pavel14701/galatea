"""Model Interfaces Module.
=======================

This module provides abstractions for language model backends used
by the Galatea Prisms (Surprise, Bond, Threat). It defines a unified
interface (`ModelInterface`) for tokenisation, embedding extraction,
hidden state access, and logit computation.

Additionally, it provides a full integration with PEFT (LoRA) adapters
through the `AdapterManager` interface and its HuggingFace implementation.
The `AdapterStateStore` allows persistence of adapter parameters.

Public Components:
    - ModelInterface: Abstract base class for model wrappers.
    - HuggingFaceModelInterface: Concrete implementation for HuggingFace
        transformers models (AutoModelForCausalLM).
    - create_model_interface: Factory function to instantiate model interfaces
        based on a type string ('huggingface', 'openai', 'vllm').
    - AdapterManager: Abstract interface for managing LoRA adapters
        (load, activate, gradient step, entropy with adapter).
    - HuggingFaceAdapterManager: Implementation of AdapterManager for PEFT.
    - AdapterStateStore: Storage for adapter parameters (in-memory by default).

All components are designed to be interchangeable, allowing the Galatea
system to switch between local HuggingFace models, OpenAI API, vLLM,
or other backends without changing the Prism logic.

Usage Example:
    >>> from galatea.model_interfaces import create_model_interface
    >>> model = create_model_interface('huggingface', 'meta-llama/Llama-3-8B')
    >>> emb = model.get_response_embedding("Hello world")
    >>> manager = model.adapter_manager  # if using HuggingFaceModelInterface
    >>> manager.load_adapter('user_123')
    >>> manager.set_active_adapter('user_123')
"""

from .adapter_manager import AdapterManager, HuggingFaceAdapterManager
from .adapter_state_store import AdapterStateStore
from .factory import create_model_interface
from .huggingface_adapter import HuggingFaceModelInterface
from .interface import ModelInterface

__all__ = (
    'ModelInterface',
    'HuggingFaceModelInterface',
    'create_model_interface',
    'AdapterManager',
    'HuggingFaceAdapterManager',
    'AdapterStateStore',
)
