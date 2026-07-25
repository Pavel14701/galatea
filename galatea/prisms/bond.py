import torch
import torch.nn as nn
from typing import Any
from .base import Prism
from ..utils.helpers import cosine_similarity
from ..model_interfaces.interface import ModelInterface
from .state_manager import StateManager


class BondPrism(Prism):
    """
    Призма Близости (BP) с поддержкой постоянного хранилища.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        state_manager: StateManager,
        user_id: str,
        hidden_size: int = 128,
        lr_bp: float = 1e-5,
        decay_after_seconds: float = 3600,
        decay_factor: float = 0.9,
        initial_bond: float = 0.2,
        cold_start_steps: int = 20,
        cold_start_coef: float = 1.5,
        skepticism_threshold: float = 0.5,
        repeat_cos_threshold: float = 0.95,
        repeat_count: int = 3,
        repeat_window: int = 20,
        diversity_window: int = 50,
    ):
        super().__init__()
        self.model = model_interface
        self.state_manager = state_manager
        self.user_id = user_id
        self.prism_name = "bond"  # для хранения

        self.embed_dim = model_interface.get_embed_dim()
        self.hidden_size = hidden_size
        self.decay_after_seconds = decay_after_seconds
        self.decay_factor = decay_factor
        self.initial_bond = initial_bond
        self.cold_start_steps = cold_start_steps
        self.cold_start_coef = cold_start_coef
        self.skepticism_threshold = skepticism_threshold
        self.repeat_cos_threshold = repeat_cos_threshold
        self.repeat_count = repeat_count
        self.repeat_window = repeat_window
        self.diversity_window = diversity_window

        # Модели
        input_dim = 2 * self.embed_dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=False)
        self.mlp_bond = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Оптимизатор (позже)
        self.optimizer = torch.optim.Adam(
            list(self.gru.parameters()) + list(self.mlp_bond.parameters()),
            lr=lr_bp
        )

        # Загружаем состояние из хранилища или создаём новое
        self._load_state()

    def _default_state(self) -> dict[str, Any]:
        """Создаёт начальное состояние для нового пользователя."""
        return {
            "state": torch.zeros(1, 1, self.hidden_size),
            "step_count": 0,
            "last_active": 0.0,
            "session_bond_start": self.initial_bond,
            "history_states": [],
            "surprise_history": [],
            "skepticism_triggered": False,
            "ethics_violation": False,
        }

    def _load_state(self) -> None:
        """Загружает состояние из хранилища или создаёт новое."""
        data = self.state_manager.load(self.user_id, self.prism_name)
        if data is None:
            data = self._default_state()

        # Восстанавливаем тензоры из списков
        self.state = data["state"].to(torch.float32) if isinstance(
            data["state"], torch.Tensor
        ) else torch.tensor(
            data["state"], dtype=torch.float32
        )
        self.step_count = data["step_count"]
        self.last_active = data["last_active"]
        self.session_bond_start = data["session_bond_start"]

        # Восстанавливаем списки тензоров и чисел
        self.history_states = [
            torch.tensor(
                s, dtype=torch.float32
            ) if not isinstance(s, torch.Tensor) else s
            for s in data["history_states"]
        ]
        self.surprise_history = data["surprise_history"]
        self.skepticism_triggered = data.get("skepticism_triggered", False)
        self.ethics_violation = data.get("ethics_violation", False)

        # Двойной буфер не сохраняем – он временный
        self.reserve_state: torch.Tensor | None = None
        self.pending_state: torch.Tensor | None = None

    def _save_state(self) -> None:
        """Сохраняет текущее состояние в хранилище."""
        data = {
            "state": self.state.cpu().numpy().tolist(),  # конвертируем тензор в список  # noqa: E501
            "step_count": self.step_count,
            "last_active": self.last_active,
            "session_bond_start": self.session_bond_start,
            "history_states": [
                s.cpu().numpy().tolist()
                for s in self.history_states
            ],
            "surprise_history": self.surprise_history,
            "skepticism_triggered": self.skepticism_triggered,
            "ethics_violation": self.ethics_violation,
        }
        self.state_manager.save(self.user_id, self.prism_name, data)

    def forward(
        self,
        user_text: str,
        response_text: str,
        timestamp: float,
        surprise: float,
        threat: float,
    ) -> float:
        """
        Вычисляет bond_score.
        """
        # Получаем эмбеддинги через интерфейс модели
        e_user = self.model.get_response_embedding(user_text)
        e_response = self.model.get_response_embedding(response_text)
        # Временное затухание
        if self.last_active > 0 and (
            timestamp - self.last_active
        ) > self.decay_after_seconds:
            self.state = self.state * self.decay_factor
        # Подготовка входа для GRU
        ts_norm = torch.tensor(
            [timestamp / 3600.0],
            dtype=torch.float32
        ).unsqueeze(0)  # (1,1)
        if e_user.dim() == 1:
            e_user = e_user.unsqueeze(0)
        if e_response.dim() == 1:
            e_response = e_response.unsqueeze(0)
        x = torch.cat([e_user, e_response, ts_norm], dim=-1)  # (1, input_dim)
        x = x.unsqueeze(0)  # (1, 1, input_dim)
        # Шаг GRU
        self.state, _ = self.gru(x, self.state)
        self.last_active = timestamp
        self.step_count += 1
        # Вычисляем bond
        bond = self.mlp_bond(self.state.squeeze(0)).item()
        # Холодный старт (ускорение в первые 20 шагов)
        if self.step_count <= self.cold_start_steps:
            if (surprise > 0.6) and (threat < 0.4) and not self._has_repeats():
                bond = self.initial_bond + (
                    bond - self.initial_bond
                ) * self.cold_start_coef
        bond = max(0.0, min(1.0, bond))
        # Защиты
        bond = self._apply_skepticism(bond)
        bond = self._apply_repeat_protection(bond)
        bond = self._apply_diversity_protection(bond, surprise)
        # Этическое нарушение
        if self.ethics_violation:
            bond = max(0.0, bond - 0.3)
            self.ethics_violation = False
        # Сохраняем состояние для детектора повторов
        self._update_history()
        return bond

    def _apply_skepticism(self, bond: float) -> float:
        if (bond - self.session_bond_start) > self.skepticism_threshold:
            bond *= 0.5
            self.skepticism_triggered = True
        else:
            self.skepticism_triggered = False
        return bond

    def _apply_repeat_protection(self, bond: float) -> float:
        if len(self.history_states) < self.repeat_window:
            return bond
        current = self.state.squeeze().clone().detach()
        sims = [
            cosine_similarity(current, s)
            for s in self.history_states[-self.repeat_window:]
        ]
        if sum(
            1 for s in sims if s > self.repeat_cos_threshold
        ) >= self.repeat_count:
            bond *= 0.7
        return bond

    def _apply_diversity_protection(
        self,
        bond: float,
        surprise: float
    ) -> float:
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > self.diversity_window:
            self.surprise_history.pop(0)
        if len(self.surprise_history) >= 10:
            var = torch.var(torch.tensor(self.surprise_history))
            if var < 0.05:
                bond = 0.9 * bond + 0.1 * self._get_prev_bond()
        return bond

    def _has_repeats(self) -> bool:
        if len(self.history_states) < 5:
            return False
        current = self.state.squeeze().clone().detach()
        for s in self.history_states[-5:]:
            if cosine_similarity(current, s) > self.repeat_cos_threshold:
                return True
        return False

    def _update_history(self) -> None:
        self.history_states.append(self.state.squeeze().clone().detach())
        if len(self.history_states) > self.repeat_window:
            self.history_states.pop(0)

    def _get_prev_bond(self) -> float:
        # Для простоты возвращаем текущий bond
        # (в реальности нужно хранить историю)
        return 0.5

    def update(self, **kwargs) -> None:
        """
        Обучение BP на целевых сигналах (реализуется отдельно).
        Здесь заглушка.
        """
        pass

    def reset(self) -> None:
        """Сброс состояния для нового пользователя."""
        self.state = torch.zeros(1, 1, self.hidden_size)
        self.step_count = 0
        self.last_active = 0.0
        self.session_bond_start = self.initial_bond
        self.history_states = []
        self.surprise_history = []
        self.skepticism_triggered = False
        self.reserve_state = None
        self.pending_state = None
        self.ethics_violation = False

    def set_ethics_violation(self) -> None:
        self.ethics_violation = True

    def save_reserve(self) -> None:
        self.reserve_state = self.state.clone()

    def save_pending(self) -> None:
        self.pending_state = self.state.clone()

    def rollback(self) -> None:
        if self.reserve_state is not None:
            self.state = self.reserve_state.clone()
            self.reserve_state = None
