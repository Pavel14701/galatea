"""
Абстракции над языковыми моделями для универсального доступа из Призм.
"""

from .interface import ModelInterface
from .huggingface_adapter import HuggingFaceModelInterface
from .openai_adapter import OpenAIModelInterface
from .factory import create_model_interface

__all__ = (
    "ModelInterface",
    "HuggingFaceModelInterface",
    "OpenAIModelInterface",
    "create_model_interface",
)
