import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .interface import ModelInterface


class HuggingFaceModelInterface(ModelInterface):
    """
    Адаптер для моделей, загружаемых через библиотеку
    transformers (HuggingFace). Поддерживает любые модели
    семейства Llama, Mistral, Gemma, GPT-2 и др.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_early_layer: int = 4,          # слой, используемый как "ранний" (h_early)  # noqa: E501
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype or torch.float16,
            output_hidden_states=True,
            output_attentions=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embed_dim = self.model.config.hidden_size
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.early_layer = use_early_layer

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt"
        )["input_ids"].to(self.device)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Входной слой эмбеддингов
            emb = self.model.get_input_embeddings()(input_ids)  # (batch, seq, embed_dim)  # noqa: E501
            return emb

    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:  # noqa: E501
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # hidden_states[0] – входные эмбеддинги, hidden_states[1..N] – слои
            return outputs.hidden_states[layer_idx]  # (batch, seq, hidden_dim)

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits  # (batch, seq, vocab_size)

    def get_response_embedding(self, response_text: str) -> torch.Tensor:
        # Токенизируем ответ
        input_ids = self.tokenize(response_text)
        # Получаем эмбеддинги и усредняем по токенам (средний пул)
        emb = self.get_embeddings(input_ids)  # (1, seq, embed_dim)
        return emb.mean(dim=1).squeeze(0)      # (embed_dim,)

    def get_embed_dim(self) -> int:
        return self.embed_dim

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_vocab_size(self) -> int:
        return self.vocab_size
