"""Threat Prism (TP) Module.

This module implements the ThreatPrism, which evaluates the risk of a
potential gradient update destabilising the model.

The ThreatPrism operates in two modes:
    - Fast evaluation: runs on every step, using entropy, trend,
        early hidden states, and embeddings to produce a base threat score.
    - Full evaluation: only for candidate events (surprise * bond > 0.5),
        where a temporary gradient step is applied to a specific adapter
        (LoRA) and the resulting entropy change (ΔH) is used to refine
        the threat score.

The state (entropy history, boost counter, etc.) is persisted via a
StateManager to survive across sessions.

The threat score is used by the LambdaAggregator to suppress plasticity
when the model is at risk of being damaged.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..model_interfaces.interface import ModelInterface
from ..model_interfaces.adapter_manager import AdapterManager
from ..utils.helpers import entropy, trend
from .base import Prism
from .state_manager import StateManager


class ThreatPrism(Prism):
    """Threat Prism (TP) - evaluates model instability risk.

    The TP provides a fast threat estimate on every step and a more
    accurate full evaluation for high-significance events. The full
    evaluation temporarily applies a gradient step to an adapter and
    measures the resulting entropy change; if the entropy increases
    too much, the threat score is raised.

    The prism maintains a persistent state (entropy history, boost flags)
    using a StateManager.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        adapter_manager: AdapterManager,
        state_manager: StateManager,
        user_id: str,
        fast_mlp_hidden: int = 32,
        full_mlp_hidden: int = 32,
        threat_boost_duration: int = 5,      # minutes
        ethics_threshold: float = 0.7,
        learning_rate: float = 0.01,
    ):
        """Initialise the ThreatPrism.

        Args:
            model_interface: Interface to the language model.
            adapter_manager: Manager for LoRA adapters.
            state_manager: Persistent storage for prism state.
            user_id: Identifier for the current user.
            fast_mlp_hidden: Hidden size of the fast MLP.
            full_mlp_hidden: Hidden size of the full MLP (with ΔH).
            threat_boost_duration: Duration of threat boost after
                ethics violation (minutes).
            ethics_threshold: Threshold for ethics violation detection.
            learning_rate: Learning rate for the temporary gradient step
                during full evaluation.

        """
        super().__init__()
        self.model = model_interface
        self.adapter_manager = adapter_manager
        self.state_manager = state_manager
        self.user_id = user_id
        self.prism_name = 'threat'

        self.embed_dim = model_interface.get_embed_dim()
        self.hidden_dim = model_interface.get_hidden_dim()
        self.threat_boost_duration = threat_boost_duration
        self.ethics_threshold = ethics_threshold
        self.learning_rate = learning_rate

        # Fast MLP: input = [H, trend, h_early, e_user, e_response]
        # Dimension: 2 + embed_dim + 2*embed_dim = 2 + 3*embed_dim
        fast_input_dim = 2 + 3 * self.embed_dim
        self.fast_mlp = nn.Sequential(
            nn.Linear(fast_input_dim, fast_mlp_hidden),
            nn.ReLU(),
            nn.Linear(fast_mlp_hidden, 1),
            nn.Sigmoid()
        )

        # Full MLP: input adds ΔH
        full_input_dim = fast_input_dim + 1
        self.full_mlp = nn.Sequential(
            nn.Linear(full_input_dim, full_mlp_hidden),
            nn.ReLU(),
            nn.Linear(full_mlp_hidden, 1),
            nn.Sigmoid()
        )

        # Load or initialise state.
        self._load_state()

    def _default_state(self) -> dict[str, Any]:
        """Create a default state for a new user (no history)."""
        return {
            'entropy_history': [],
            'boost_counter': 0,
            'boost_active': False,
            'ethics_violation': False,
        }

    def _load_state(self) -> None:
        """Load the threat state from the persistent store.

        If no state exists, a default state is created.
        """
        data = self.state_manager.load(self.user_id, self.prism_name)
        if data is None:
            data = self._default_state()

        self.entropy_history: list[float] = data.get('entropy_history', [])
        self.boost_counter: int = data.get('boost_counter', 0)
        self.boost_active: bool = data.get('boost_active', False)
        self.ethics_violation: bool = data.get('ethics_violation', False)

    def _save_state(self) -> None:
        """Save the current threat state to the persistent store."""
        data = {
            'entropy_history': self.entropy_history,
            'boost_counter': self.boost_counter,
            'boost_active': self.boost_active,
            'ethics_violation': self.ethics_violation,
        }
        self.state_manager.save(self.user_id, self.prism_name, data)

    def forward(
        self,
        input_ids: torch.Tensor,
        user_text: str,
        response_text: str,
        surprise: float,
        bond: float,
        adapter_name: str | None = None,
        gradient_step: dict[str, torch.Tensor] | None = None,
    ) -> float:
        """Compute the threat score for the current interaction.

        The method:
            1. Retrieves embeddings and early hidden state.
            2. Computes entropy and trend (over last 20 steps).
            3. Runs the fast MLP to get a base threat score.
            4. Applies ethics violation override (threat = 0.9)
                and temporary boost.
            5. If the event is a candidate (surprise * bond > 0.5)
                and an adapter name and gradient step are provided,
                performs a full evaluation with a temporary gradient step.
            6. Returns the final threat score (clamped to [0,1]).

        Args:
            input_ids: Token IDs of the user query.
            user_text: User message text.
            response_text: Galatea's response text.
            surprise: Surprise score from SP (0-1).
            bond: Bond score from BP (0-1).
            adapter_name: Name of the adapter to test (for full evaluation).
            gradient_step: Gradients to apply for the temporary update.

        Returns:
            Threat score (float, 0-1).

        """
        # Get embeddings and early hidden state.
        e_user = self.model.get_response_embedding(user_text)
        e_response = self.model.get_response_embedding(response_text)
        h_early = self.model.get_hidden_state(
            input_ids, layer_idx=4
        ).mean(dim=1).squeeze(0)
        # Compute entropy and trend.
        logits = self.model.get_logits(input_ids)
        h = entropy(logits)
        self.entropy_history.append(h)
        if len(self.entropy_history) > 20:
            self.entropy_history.pop(0)
        tr = trend(self.entropy_history, window=20)

        # Fast evaluation.
        x_fast = torch.cat([
            torch.tensor([h, tr], dtype=torch.float32),
            h_early,
            e_user,
            e_response
        ])
        threat_base = self.fast_mlp(x_fast).item()
        # Override for ethics violation.
        if self.ethics_violation:
            threat_base = 0.9
            self.ethics_violation = False
        # Apply temporary boost (from recent ethics violation).
        if self.boost_active:
            threat_base = min(1.0, threat_base + 0.15)
        # Full evaluation (if candidate and adapter data provided).
        if (
            surprise * bond > 0.5
        ) and (
            adapter_name is not None
        ) and (
            gradient_step is not None
        ):
            threat_full = self._full_evaluation(
                input_ids, adapter_name, gradient_step, h_early, e_user, e_response, h, tr  # noqa: E501
            )
            threat = max(threat_base, threat_full)
        else:
            threat = threat_base
        # Safety fallback for NaN/inf.
        if np.isnan(threat) or np.isinf(threat):
            threat = 0.1
        # Save state after each forward pass.
        self._save_state()
        return max(0.0, min(1.0, threat))

    def _full_evaluation(
        self,
        input_ids: torch.Tensor,
        adapter_name: str,
        gradient_step: dict[str, torch.Tensor],
        h_early: torch.Tensor,
        e_user: torch.Tensor,
        e_response: torch.Tensor,
        h_before: float,
        trend_before: float,
    ) -> float:
        """Perform full evaluation with a temporary gradient step.

        This method:
            1. Applies the gradient step to the specified adapter
                using the AdapterManager.
            2. Computes the entropy of the model with the updated adapter
                (via AdapterManager.compute_entropy_with_adapter).
            3. Computes the relative entropy change ΔH.
            4. Feeds ΔH into the full MLP to get a refined threat score.

        Args:
            input_ids: Token IDs for the forward pass.
            adapter_name: Name of the adapter to test.
            gradient_step: Gradients to apply.
            h_early: Early hidden state.
            e_user: User embedding.
            e_response: Response embedding.
            h_before: Entropy before the gradient step.
            trend_before: Trend before the gradient step.

        Returns:
            Threat score from full evaluation (float, 0-1).

        """
        # Apply the gradient step to the adapter.
        self.adapter_manager.apply_gradient_step(
            adapter_name, gradient_step, self.learning_rate
        )
        # Compute entropy after the update.
        h_after = self.adapter_manager.compute_entropy_with_adapter(
            input_ids, adapter_name
        )
        # Compute relative entropy change.
        delta_h_rel = (h_after - h_before) / (h_before + 1e-8)
        # Feed into full MLP.
        x_full = torch.cat([
            torch.tensor(
                [h_before, trend_before, delta_h_rel],
                dtype=torch.float32
            ),
            h_early,
            e_user,
            e_response
        ])
        threat_full = self.full_mlp(x_full).item()
        return max(0.0, min(1.0, threat_full))

    def update(self, *args, **kwargs) -> None:
        """Training step (placeholder)."""

    def reset(self) -> None:
        """Reset the threat state for a new user or session."""
        self.entropy_history = []
        self.boost_counter = 0
        self.boost_active = False
        self.ethics_violation = False
        self.state_manager.delete(self.user_id, self.prism_name)

    def set_ethics_violation(self) -> None:
        """Trigger an ethics violation, setting threat to 0.9 and
        activating a boost.
        """
        self.ethics_violation = True
        self.boost_active = True
        self.boost_counter = 0

    def tick_boost(self, minutes: int = 1) -> None:
        """Decrease the boost counter; deactivate boost
        when duration is exceeded.
        """
        if self.boost_active:
            self.boost_counter += minutes
            if self.boost_counter >= self.threat_boost_duration:
                self.boost_active = False
