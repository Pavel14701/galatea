from .interface import ModelInterface
from .huggingface_adapter import HuggingFaceModelInterface
from .openai_adapter import OpenAIModelInterface


def create_model_interface(
    model_type: str = "huggingface",
    model_name: str = "meta-llama/Llama-3-8B",
    **kwargs
) -> ModelInterface:
    """
    Фабрика для создания адаптера модели по типу.

    Аргументы:
        model_type: "huggingface", "openai", "vllm"
        model_name: имя модели (для HF – название на HuggingFace,
            для OpenAI – модель чата)
        **kwargs: дополнительные параметры, передаваемые в конструктор адаптера

    Возвращает:
        экземпляр ModelInterface
    """
    if model_type == "huggingface":
        return HuggingFaceModelInterface(model_name, **kwargs)
    elif model_type == "openai":
        # Для OpenAI model_name обычно используется как chat_model,
        # но можно передать и embedding_model отдельно.
        return OpenAIModelInterface(chat_model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
