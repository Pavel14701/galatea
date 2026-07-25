from abc import ABC, abstractmethod
import torch


class ModelInterface(ABC):
    """
    Абстрактный интерфейс к языковой модели.
    Все адаптеры должны реализовывать эти методы.
    """

    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Возвращает эмбеддинги для входных токенов.
        Аргументы:
            input_ids: (batch, seq_len) или (seq_len,)
        Возвращает:
            тензор (batch, seq_len, embed_dim)
        """
        pass

    @abstractmethod
    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Возвращает скрытое состояние на указанном слое.
        Аргументы:
            input_ids: (batch, seq_len)
            layer_idx: номер слоя
                (0 = входные эмбеддинги, 1 = после первого блока и т.д.)
        Возвращает:
            тензор (batch, seq_len, hidden_dim)
        """
        pass

    @abstractmethod
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Возвращает логиты последнего слоя.
        Аргументы:
            input_ids: (batch, seq_len)
        Возвращает:
            тензор (batch, seq_len, vocab_size)
        """
        pass

    @abstractmethod
    def get_response_embedding(self, response_text: str) -> torch.Tensor:
        """
        Возвращает эмбеддинг для текста ответа
        (например, через средний пул или отдельный энкодер).

        Аргументы:
            response_text: строка
        Возвращает:
            тензор (embed_dim,)
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Преобразует текст в тензор токенов (input_ids).
        Аргументы:
            text: строка
        Возвращает:
            тензор (seq_len,) или (1, seq_len)
        """
        pass

    @abstractmethod
    def get_embed_dim(self) -> int:
        """Возвращает размерность эмбеддингов модели."""
        pass

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Возвращает размерность скрытого состояния модели."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря."""
        pass
