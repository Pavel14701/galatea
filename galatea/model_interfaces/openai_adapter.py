import torch
import numpy as np
from .interface import ModelInterface

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
                "OpenAI package is not installed. Please install openai."
            )


class OpenAIModelInterface(ModelInterface):
    """
    Адаптер для моделей OpenAI (GPT-3.5, GPT-4) и их эмбеддингов.
    Использует новый клиент OpenAI (>=1.0.0).
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        chat_model: str = "gpt-4o-mini",   # или gpt-4, gpt-3.5-turbo
        embed_dim: int = 1536,             # для ada-002
        api_key: str | None = None,
        org_id: str | None = None,
    ):
        self.client = OpenAI(api_key=api_key, organization=org_id)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim  # условно, для совместимости
        self.vocab_size = 50257      # для GPT-3, можно как заглушку

    def tokenize(self, text: str) -> torch.Tensor:
        # OpenAI не предоставляет токенизацию в этом интерфейсе
        raise NotImplementedError(
            "OpenAI interface does not support tokenization."
        )

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "OpenAI does not provide token-level embeddings."
        )

    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        raise NotImplementedError("OpenAI does not provide hidden states.")

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("OpenAI does not provide logits.")

    def get_response_embedding(self, response_text: str) -> torch.Tensor:
        """
        Получает эмбеддинг текста через OpenAI Embeddings API.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=response_text,
        )
        emb = response.data[0].embedding
        return torch.tensor(emb, dtype=torch.float32)

    def get_token_probs(
        self,
        prompt: str,
        max_tokens: int = 1
    ) -> dict[str, float]:
        """
        Получает вероятности следующих токенов (для оценки энтропии).
        Использует Chat Completions API с параметром logprobs.
        """
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            # Извлекаем топ-логарифмы для первого токена
            top_logprobs = choice.logprobs.content[0].top_logprobs
            probs = {item.token: np.exp(item.logprob) for item in top_logprobs}
            return probs
        return {}

    def get_embed_dim(self) -> int:
        return self.embed_dim

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_vocab_size(self) -> int:
        return self.vocab_size
