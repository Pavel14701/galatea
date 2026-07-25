import torch


def get_early_embedding(model, input_ids: torch.Tensor, layer_idx: int = 4) -> torch.Tensor:
    """
    Извлекает эмбеддинги из ранних слоёв Коры (после `layer_idx`-го блока).
    Используется для `h_early` в TP.
    Аргументы:
        model: экземпляр модели (например, LlamaForCausalLM)
        input_ids: тензор токенов (batch, seq_len)
        layer_idx: номер слоя (0 – эмбеддинги, 1 – после первого блока...)
    Возвращает:
        тензор (batch, seq_len, hidden_dim)
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # hidden_states[0] – входные эмбеддинги, [1..N] – слои
        return outputs.hidden_states[layer_idx]


def get_response_embedding(model, response_text: str, tokenizer) -> torch.Tensor:
    """
    Получает эмбеддинг ответа как среднее по токенам.
    Используется для BP и SP в MVP-режиме.
    Аргументы:
        model: экземпляр модели
        response_text: строка ответа
        tokenizer: токенизатор
    Возвращает:
        тензор (embed_dim,)
    """
    input_ids = tokenizer(response_text, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        emb = model.get_input_embeddings()(input_ids)  # (1, seq, embed_dim)
        return emb.mean(dim=1).squeeze(0)              # (embed_dim,)
