import time
import unittest

import torch

from galatea.model_interfaces import ModelInterface
from galatea.prisms import BondPrism


class MockModelForBond(ModelInterface):
    def __init__(self, embed_dim=8):
        self.embed_dim = embed_dim

    def get_response_embedding(self, text):
        # Для воспроизводимости используем хеш
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        vec = [int(h[i:i + 2], 16) / 255.0 for i in range(0, min(16, len(h)), 2)]
        vec += [0.0] * (self.embed_dim - len(vec))
        return torch.tensor(vec[:self.embed_dim], dtype=torch.float32)

    def get_embed_dim(self):
        return self.embed_dim

    # Заглушки остальных методов
    def get_embeddings(self, input_ids): pass
    def get_hidden_state(self, input_ids, layer_idx): pass
    def get_logits(self, input_ids): pass
    def tokenize(self, text): return torch.randint(0, 100, (1, 5))
    def get_hidden_dim(self): return self.embed_dim
    def get_vocab_size(self): return 1000


class TestBondPrism(unittest.TestCase):

    def setUp(self):
        self.model = MockModelForBond(embed_dim=8)
        self.bp = BondPrism(
            self.model,
            hidden_size=16,
            decay_after_seconds=3600,
            decay_factor=0.9,
            initial_bond=0.2,
            cold_start_steps=5,
            cold_start_coef=1.5,
            skepticism_threshold=0.5,
            repeat_cos_threshold=0.95,
            repeat_count=3,
            repeat_window=5,
        )

    def test_initial_bond(self):
        """Первый вызов возвращает bond ~0.2 (с учётом холодного старта)."""
        # Первый шаг: state инициализируется, bond = 0.2
        bond = self.bp.forward('user1', 'resp1', timestamp=time.time(), surprise=0.5, threat=0.2)
        self.assertAlmostEqual(bond, 0.2, delta=0.1)  # может немного отличаться из-за MLP

    def test_bond_grows(self):
        """При нормальном диалоге bond растёт."""
        # Делаем несколько шагов с высоким surprise и низкой threat
        t = time.time()
        for i in range(3):
            bond = self.bp.forward(f'user{i}', f'resp{i}', timestamp=t + i * 10,
                                   surprise=0.7, threat=0.2)
        # После нескольких шагов bond должен быть > 0.2
        self.assertGreater(bond, 0.25)

    def test_cold_start_acceleration(self):
        """В первые 5 шагов при surprise>0.6, threat<0.4 рост ускорен."""
        self.bp.reset()
        # Первые 5 шагов с высоким surprise и низкой threat
        t = time.time()
        bonds = []
        for i in range(5):
            b = self.bp.forward(f'u{i}', f'r{i}', timestamp=t + i * 10,
                                surprise=0.7, threat=0.2)
            bonds.append(b)
        # bond должен расти быстрее, чем без ускорения (но мы не можем точно проверить,
        # просто проверим, что после 5 шагов bond > 0.3)
        self.assertGreater(bonds[-1], 0.3)

    def test_skepticism(self):
        """Взлёт bond > 0.5 за сессию вызывает занижение."""
        self.bp.reset()
        t = time.time()
        # Имитируем быстрый рост
        # Первый шаг – базовый
        b1 = self.bp.forward('u1', 'r1', timestamp=t, surprise=0.9, threat=0.1)
        # Второй шаг – очень высокий bond (искусственно)
        # В реальности bond не может так резко вырасти, но для теста мы форсируем?
        # В защите скепсис срабатывает при взлёте >0.5 за сессию.
        # Для этого нужно, чтобы bond скакнул. Мы не можем напрямую задать bond,
        # но можем создать условия для его быстрого роста (много шагов с высоким surprise)
        # Однако проще протестировать логику отдельно: вызовем _apply_skepticism напрямую.
        bond_high = 0.8
        self.bp.session_bond_start = 0.2  # симулируем старт
        new_bond = self.bp._apply_skepticism(bond_high)
        # Если взлёт > 0.5, то bond умножается на 0.5
        expected = bond_high * 0.5
        self.assertAlmostEqual(new_bond, expected, places=3)

    def test_repeat_protection(self):
        """Повторяющиеся состояния (cos > 0.95 трижды) занижают bond на 30%."""
        # Для теста заполним историю состояний одинаковыми векторами
        self.bp.reset()
        # Создаём одинаковые состояния
        state = torch.zeros(1, 1, 16)
        for _ in range(5):
            self.bp.state = state.clone()
            self.bp._update_history()
        # Теперь применим защиту
        bond = 0.7
        new_bond = self.bp._apply_repeat_protection(bond)
        self.assertAlmostEqual(new_bond, bond * 0.7, places=3)

    def test_diversity_protection(self):
        """Низкая дисперсия surprise замедляет рост."""
        self.bp.reset()
        # Заполняем историю surprise одинаковыми значениями
        for _ in range(10):
            self.bp.surprise_history.append(0.5)
        bond = 0.6
        new_bond = self.bp._apply_diversity_protection(bond, 0.5)
        # Должен быть чуть ниже (сглаживание)
        self.assertLess(new_bond, bond)

    def test_ethics_violation(self):
        """Этическое нарушение снижает bond на 0.3."""
        self.bp.reset()
        self.bp.state = torch.zeros(1, 1, 16)  # инициализируем
        bond = self.bp.forward('u', 'r', timestamp=time.time(), surprise=0.5, threat=0.2)
        # Устанавливаем флаг нарушения
        self.bp.set_ethics_violation()
        new_bond = self.bp.forward('u2', 'r2', timestamp=time.time() + 10, surprise=0.5, threat=0.2)
        self.assertAlmostEqual(new_bond, max(0.0, bond - 0.3), delta=0.05)


if __name__ == '__main__':
    unittest.main()