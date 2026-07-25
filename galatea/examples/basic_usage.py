"""
Пример использования всех компонентов быстрых Призм.
Симулирует диалог из нескольких шагов, вычисляет surprise, bond, threat и λ_t.
"""

import sys
import time
import torch
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from galatea.model_interfaces import ModelInterface
from galatea.prisms import SurprisePrism, BondPrism, ThreatPrism, LambdaAggregator
from galatea.prisms.state_manager import InMemoryStateManager


class MockModelInterface(ModelInterface):
    """
    Заглушка модели для демонстрации.
    Генерирует случайные эмбеддинги заданной размерности.
    """

    def __init__(self, embed_dim=128, hidden_dim=128, vocab_size=1000):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def get_embeddings(self, input_ids):
        return torch.randn(1, 5, self.embed_dim)

    def get_hidden_state(self, input_ids, layer_idx):
        return torch.randn(1, 5, self.hidden_dim)

    def get_logits(self, input_ids):
        return torch.randn(1, 5, self.vocab_size)

    def get_response_embedding(self, response_text):
        return torch.randn(self.embed_dim)

    def tokenize(self, text):
        # Для демонстрации возвращаем случайные токены
        return torch.randint(0, self.vocab_size, (1, 10))

    def get_embed_dim(self):
        return self.embed_dim

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_vocab_size(self):
        return self.vocab_size


def main():
    print("=== Galatea – Быстрые Призмы: Демонстрация ===\n")

    # 1. Создаём заглушку модели
    model = MockModelInterface(embed_dim=64, hidden_dim=64)

    # 2. Инициализируем Призмы (MVP-режим для SP)
    sp = SurprisePrism(model, mode='mvp', tau=0.1)
    bp = BondPrism(model, hidden_size=32)
    tp = ThreatPrism(model, fast_mlp_hidden=16, full_mlp_hidden=16)

    # 3. Агрегатор с настройками по умолчанию
    aggr = LambdaAggregator()

    # 4. Менеджер состояний (в памяти)
    state_mgr = InMemoryStateManager()
    user_id = "demo_user"

    # 5. Симулируем диалог (список пар user/assistant)
    dialogue = [
        ("Привет!", "Здравствуйте! Чем могу помочь?"),
        ("Расскажи про ИИ.", "ИИ – это область компьютерных наук..."),
        ("А как ты обучаешься?", "Я использую пластичность и гормональную регуляцию."),
        ("Это интересно!", "Рад, что вам интересно."),
        ("А что такое Призмы?", "Призмы – это перцептивные модули, оценивающие удивление, близость и угрозу."),
    ]

    print("Начинаем диалог...\n")
    prev_user_text = None

    for step, (user_text, response_text) in enumerate(dialogue, 1):
        timestamp = time.time()

        # Получаем оценку удивления (передаём предыдущий текст для фильтра смены темы)
        surprise = sp.forward(user_text, timestamp=timestamp, prev_texts=[prev_user_text] if prev_user_text else None)

        # Получаем оценку близости
        bond = bp.forward(user_text, response_text, timestamp, surprise, threat=0.3)

        # Получаем токены запроса для TP
        input_ids = model.tokenize(user_text)

        # Получаем оценку угрозы
        threat = tp.forward(input_ids, user_text, response_text, surprise, bond)

        # Вычисляем λ_t (гормоны задаём фиксированными)
        serotonin = 0.6 + 0.1 * (step % 3)  # небольшие колебания
        orexin = 0.5
        oxytocin = 0.3
        lambda_t = aggr.compute(surprise, bond, threat, serotonin, orexin, oxytocin)

        # Обновляем SP (добавляем текущий запрос в историю)
        sp.update(user_text, lambda_t, timestamp=timestamp)

        # Сохраняем состояние BP (для следующего шага) – в реальности сохраняется в БД
        state_mgr.set_state(user_id, 'bond_state', bp.state)

        print(f"Шаг {step}:")
        print(f"  User: {user_text[:30]}...")
        print(f"  Surprise: {surprise:.3f}, Bond: {bond:.3f}, Threat: {threat:.3f}")
        print(f"  λ_t: {lambda_t:.3f}  (серотонин={serotonin:.2f})")
        print()

        prev_user_text = user_text

    # В конце покажем, что состояние BP сохранилось и может быть восстановлено
    restored_state = state_mgr.get_state(user_id, 'bond_state')
    print(f"Состояние BP сохранено: {restored_state is not None}")
    print("Демонстрация завершена.")


if __name__ == "__main__":
    main()