import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .base import Prism
from ..utils.helpers import entropy, trend
from ..model_interfaces.interface import ModelInterface

class ThreatPrism(Prism):
    """
    Призма Угрозы (TP).
    Быстрая оценка на каждом шагу, полная – для кандидатов.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        fast_mlp_hidden: int = 32,
        full_mlp_hidden: int = 32,
        threat_boost_duration: int = 5,  # минут
        ethics_threshold: float = 0.7,
    ):
        super().__init__()
        self.model = model_interface
        self.embed_dim = model_interface.get_embed_dim()
        self.hidden_dim = model_interface.get_hidden_dim()
        self.threat_boost_duration = threat_boost_duration
        self.ethics_threshold = ethics_threshold

        # Быстрый MLP: вход = [H, trend, h_early, e_user, e_response]
        # Размер: 2 + embed_dim + 2*embed_dim = 2 + 3*embed_dim
        fast_input_dim = 2 + 3 * self.embed_dim
        self.fast_mlp = nn.Sequential(
            nn.Linear(fast_input_dim, fast_mlp_hidden),
            nn.ReLU(),
            nn.Linear(fast_mlp_hidden, 1),
            nn.Sigmoid()
        )

        # Полный MLP: добавляем ΔH
        full_input_dim = fast_input_dim + 1
        self.full_mlp = nn.Sequential(
            nn.Linear(full_input_dim, full_mlp_hidden),
            nn.ReLU(),
            nn.Linear(full_mlp_hidden, 1),
            nn.Sigmoid()
        )

        # Состояние для тренда
        self.entropy_history = []
        self.boost_counter = 0
        self.boost_active = False
        self.ethics_violation = False

        # Для полной оценки требуется доступ к адаптеру (передаётся извне)
        self.adapter = None  # будет установлен при вызове forward с копией

    def forward(
        self,
        input_ids: torch.Tensor,          # токены запроса (для получения скрытых состояний и логитов)
        user_text: str,
        response_text: str,
        surprise: float,
        bond: float,
        adapter_copy: Optional[torch.nn.Module] = None,   # для полной оценки
        gradient_step: Optional[dict] = None,             # для пробного обновления
    ) -> float:
        """
        Вычисляет threat_score.
        Быстрая оценка всегда выполняется.
        Полная – если surprise * bond > 0.5 и переданы adapter_copy и gradient_step.
        """
        # Получаем эмбеддинги и скрытые состояния
        e_user = self.model.get_response_embedding(user_text)
        e_response = self.model.get_response_embedding(response_text)
        h_early = self.model.get_hidden_state(input_ids, layer_idx=4).mean(dim=1).squeeze(0)  # усредняем по токенам

        # Энтропия и тренд
        logits = self.model.get_logits(input_ids)  # (1, seq_len, vocab_size)
        H = entropy(logits)
        self.entropy_history.append(H)
        if len(self.entropy_history) > 20:
            self.entropy_history.pop(0)
        tr = trend(self.entropy_history, window=20)

        # Быстрая оценка
        x_fast = torch.cat([
            torch.tensor([H, tr], dtype=torch.float32),
            h_early,
            e_user,
            e_response
        ])
        threat_base = self.fast_mlp(x_fast).item()

        # Этическое нарушение
        if self.ethics_violation:
            threat_base = 0.9
            self.ethics_violation = False

        # Временный буст
        if self.boost_active:
            threat_base = min(1.0, threat_base + 0.15)

        # Полная оценка (если кандидат)
        if (surprise * bond > 0.5) and (adapter_copy is not None) and (gradient_step is not None):
            threat_full = self._full_evaluation(
                adapter_copy, gradient_step, h_early, e_user, e_response, H, tr
            )
            threat = max(threat_base, threat_full)
        else:
            threat = threat_base

        # Защита от сбоев
        if np.isnan(threat) or np.isinf(threat):
            threat = 0.1

        return max(0.0, min(1.0, threat))

    def _full_evaluation(
        self,
        adapter_copy: torch.nn.Module,
        gradient_step: dict,
        h_early: torch.Tensor,
        e_user: torch.Tensor,
        e_response: torch.Tensor,
        H_before: float,
        trend_before: float,
    ) -> float:
        """
        Полная оценка: применяет пробное обновление к копии адаптера,
        вычисляет ΔH и возвращает threat.
        """
        # Здесь нужно применить gradient_step к adapter_copy и пересчитать энтропию.
        # Это сложная операция, в реальности требует вызова модели с обновлёнными весами.
        # В данной реализации – заглушка, возвращающая случайное значение.
        delta_H = np.random.normal(0, 0.05)  # искусственно
        H_after = H_before + delta_H
        delta_H_rel = (H_after - H_before) / (H_before + 1e-8)

        x_full = torch.cat([
            torch.tensor([H_before, trend_before, delta_H_rel], dtype=torch.float32),
            h_early,
            e_user,
            e_response
        ])
        threat_full = self.full_mlp(x_full).item()
        return max(0.0, min(1.0, threat_full))

    def update(self, **kwargs):
        # Для TP обновление не требуется (кроме сброса буста)
        pass

    def reset(self):
        self.entropy_history = []
        self.boost_active = False
        self.boost_counter = 0
        self.ethics_violation = False

    def set_ethics_violation(self):
        self.ethics_violation = True
        self.boost_active = True
        self.boost_counter = 0

    def tick_boost(self, minutes: int = 1):
        if self.boost_active:
            self.boost_counter += minutes
            if self.boost_counter >= self.threat_boost_duration:
                self.boost_active = False