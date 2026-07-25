import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Вычисляет косинусное сходство между двумя тензорами (1D или 2D).
    Если тензоры многомерные, усредняет по последнему измерению.
    """
    if a.dim() > 1:
        a = a.view(-1)
    if b.dim() > 1:
        b = b.view(-1)
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return torch.dot(a, b).item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Вычисляет косинусное расстояние (1 - сходство)."""
    return 1.0 - cosine_similarity(a, b)


def entropy(logits: torch.Tensor) -> float:
    """
    Вычисляет энтропию распределения (по логитам).
    logits: тензор формы (..., vocab_size)
    Возвращает скаляр (усреднённую энтропию по батчу и последовательности).
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # Берём отрицательное среднее по всем измерениям, кроме последнего (vocab)
    ent = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq_len)
    return ent.mean().item()  # скаляр


def trend(values: list[float], window: int = 20) -> float:
    """
    Вычисляет тренд (наклон линейной регрессии) последних `window` значений.
    Если значений меньше window, возвращает 0.
    """
    if len(values) < window:
        return 0.0
    x = torch.arange(window, dtype=torch.float32)
    y = torch.tensor(values[-window:], dtype=torch.float32)
    # Ковариация и дисперсия
    cov = torch.dot(x - x.mean(), y - y.mean()) / (window - 1)
    var = torch.var(x, unbiased=True)
    if var < 1e-8:
        return 0.0
    return (cov / var).item()