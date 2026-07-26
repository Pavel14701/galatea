"""Factory for Creating Model Interfaces.

This module provides a factory function to instantiate model wrappers
for different backends. Currently, only HuggingFace is supported,
as it provides full access to embeddings, hidden states, and logits
required by the Galatea prisms.

Other backends (OpenAI, vLLM) are not supported because they do not
provide the necessary low-level model internals.
"""

from .huggingface_adapter import HuggingFaceModelInterface
from .interface import ModelInterface


def create_model_interface(
    model_type: str = 'huggingface',
    model_name: str = 'meta-llama/Llama-3-8B',
    **kwargs
) -> ModelInterface:
    """Create a model interface instance based on the specified type.

    Args:
        model_type: Type of model interface. Currently only
            'huggingface' is supported.
        model_name: Name or path of the model to load
            (e.g., 'meta-llama/Llama-3-8B' for HuggingFace).
        **kwargs: Additional parameters passed to the adapter constructor
            (e.g., device, load_in_8bit, torch_dtype).

    Returns:
        An instance of ModelInterface.

    Raises:
        ValueError: If the model_type is not supported.

    Examples:
        >>> model = create_model_interface('huggingface', 'microsoft/phi-2')
        >>> emb = model.get_response_embedding("Hello")

    """
    if model_type == 'huggingface':
        return HuggingFaceModelInterface(model_name, **kwargs)
    else:
        raise ValueError(
            f'Unsupported model type: {model_type}. ',
            'Only "huggingface" is supported.'
        )
