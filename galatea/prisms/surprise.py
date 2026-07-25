import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Prism
from ..utils.helpers import cosine_similarity, cosine_distance
from ..model_interfaces.interface import ModelInterface

class SurprisePrism(Prism):
    """
    Призма Удивления (SP).
    - MVP-режим: использует средний эмбеддинг последних 5 запросов.
    - Полный режим: использует MLP f_pred для предсказания следующего эмбеддинга.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        mode: str = 'mvp',          # 'mvp' или 'full'
        tau: float = 0.1,
        topic_threshold: float = 0.7,
        timeout_minutes: float = 5.0,
        # Параметры для полной версии
        hidden_size: int = 256,
        lr_pred: float = 1e-4,
        epsilon_explore: float = 0.02,
        lambda_var: float = 0.01,
    ):
        super().__init__()
        self.model = model_interface
        self.embed_dim = model_interface.get_embed_dim()
        self.mode = mode
        self.tau = tau
        self.topic_threshold = topic_threshold
        self.timeout_minutes = timeout_minutes

        # История эмбеддингов (для MVP)
        self.history: list[torch.Tensor] = []  # список torch.Tensor размера (embed_dim,)

        # Полная версия: f_pred (MLP)
        if mode == 'full':
            # Вход: конкатенация 4-х эмбеддингов (текущий + 3 предыдущих)
            self.f_pred = nn.Sequential(
                nn.Linear(self.embed_dim * 4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.embed_dim)
            )
            self.optimizer = torch.optim.Adam(self.f_pred.parameters(), lr=lr_pred)
            self.last_pred = None
            self.lambda_var = lambda_var
            self.epsilon_explore = epsilon_explore
            self.explore_active = False
            self.low_surprise_counter = 0

        # Для таймстемпа
        self.last_timestamp = None

    def forward(
        self,
        current_text: str,                     # текст текущего запроса пользователя
        timestamp: float | None = None,
        prev_texts: list[str] | None = None,  # для полной версии – три предыдущих запроса
    ) -> float:
        """
        Вычисляет surprise_score.
        """
        # Получаем эмбеддинг текущего запроса
        embed_current = self.model.get_response_embedding(current_text)

        # Проверка таймстемпа
        if timestamp is not None and self.last_timestamp is not None:
            if (timestamp - self.last_timestamp) > (self.timeout_minutes * 60):
                return 0.5

        if self.mode == 'mvp':
            return self._forward_mvp(embed_current)
        else:  # full
            return self._forward_full(embed_current, prev_texts)

    def _forward_mvp(self, embed_current: torch.Tensor) -> float:
        if len(self.history) == 0:
            return 0.5

        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        avg_embed = torch.stack(recent).mean(dim=0)

        sim = cosine_similarity(embed_current, avg_embed)
        score = 1.0 - sim

        # Фильтр смены темы (если есть предыдущий эмбеддинг)
        if len(self.history) > 0:
            prev_embed = self.history[-1]
            dist = cosine_distance(embed_current, prev_embed)
            if dist > self.topic_threshold:
                score *= 0.3

        return max(0.0, min(1.0, score))

    def _forward_full(self, embed_current: torch.Tensor, prev_texts: list[str] | None) -> float:
        # Получаем эмбеддинги предыдущих запросов (до 3-х)
        if prev_texts is None:
            prev_embeds = []
        else:
            prev_embeds = [self.model.get_response_embedding(t) for t in prev_texts[-3:]]

        # Дополняем нулями до 3-х
        while len(prev_embeds) < 3:
            prev_embeds.append(torch.zeros(self.embed_dim))

        # Вход: текущий + 3 предыдущих (все 1D, конкатенируем)
        x = torch.cat([embed_current] + prev_embeds, dim=-1)
        e_pred = self.f_pred(x)
        self.last_pred = e_pred

        dist = cosine_distance(e_pred, embed_current)
        score = torch.sigmoid(dist / self.tau).item()

        # ε-жадное исследование
        if self.explore_active and torch.rand(1).item() < self.epsilon_explore:
            score = 0.8

        return max(0.0, min(1.0, score))

    def update(
        self,
        actual_text: str,
        lambda_t: float,
        timestamp: float | None = None
    ) -> None:
        """
        Обновляет состояние после получения реального следующего запроса.
        Для MVP: добавляет эмбеддинг в историю.
        Для full: обучает f_pred.
        """
        if timestamp is not None:
            self.last_timestamp = timestamp

        embed_actual = self.model.get_response_embedding(actual_text)

        if self.mode == 'mvp':
            self.history.append(embed_actual.detach().clone())
            if len(self.history) > 1000:
                self.history.pop(0)
        else:  # full
            if self.last_pred is not None:
                loss = F.mse_loss(self.last_pred, embed_actual)
                if self.lambda_var > 0:
                    var_loss = self.lambda_var * torch.max(
                        torch.tensor(0.0),
                        0.1 - torch.var(self.last_pred)
                    )
                    loss += var_loss
                # Масштабируем градиент на λ_t
                (loss * lambda_t).backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_pred = None

            # Мониторинг низкого удивления (заглушка – требуется сохранение истории оценок)
            # Здесь упрощённо: используем фиктивный счётчик
            self.low_surprise_counter += 1
            if self.low_surprise_counter >= 200:
                self.explore_active = True
            else:
                self.explore_active = False

    def reset(self) -> None:
        self.history = []
        self.last_pred = None
        self.last_timestamp = None
        self.low_surprise_counter = 0
        self.explore_active = False