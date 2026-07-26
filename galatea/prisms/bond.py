"""Bond Prism (BP) Module.

This module implements the BondPrism, which measures the depth, non-randomness,
and emotional significance of the relationship with a user.

The BondPrism uses a GRU-based neural network that accumulates interaction
history and produces a `bond_score` (0-1) representing the current
bond strength. It also includes multiple protection mechanisms against
manipulation, temporal decay, and ethical violation handling.

The state (GRU hidden state, history of states, etc.) is persisted via a
StateManager to survive across sessions and users.
"""

from typing import Any

import torch
import torch.nn as nn

from ..model_interfaces.interface import ModelInterface
from ..utils.helpers import cosine_similarity
from .base import Prism
from .state_manager import StateManager


class BondPrism(Prism):
    """Bond Prism (BP) - evaluates relationship closeness.

    The BP maintains a GRU-based recurrent state that evolves with each
    interaction (user message + Galatea response). The state is then fed
    through an MLP with a sigmoid output to produce a scalar `bond_score`.

    Key features:
        - Temporal decay: bond fades if inactive for >1 hour.
        - Cold start: new users start with bond=0.2; accelerated growth
            in first 20 steps.
        - Skepticism mode: detects rapid bond inflation (>0.5 in a session)
            and halves it.
        - Repeat protection: detects repeated identical GRU states and
            penalises bond by 30%.
        - Diversity protection: if surprise is too flat (low variance),
            bond growth slows.
        - Ethics violation handling: can be externally triggered to reduce
            bond by 0.3.
        - Persistent state: loads/saves all relevant data via StateManager.

    The state includes:
        - GRU hidden state (torch.Tensor)
        - Step count
        - Last active timestamp
        - History of recent GRU states (for repeat detection)
        - History of recent surprise values (for diversity protection)
        - Flags for skepticism and ethics violation
        - Previous bond value (for smoothing in diversity protection)

    The bond score is used by the LambdaAggregator to influence the plasticity
    coefficient λ_t, and by other components (Adrenaline, Imprinting)
    as a signal of user importance.
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
        """Initialise the BondPrism.

        Args:
            model_interface: Interface to the language model (for embeddings).
            state_manager: Persistent storage for bond state.
            user_id: Identifier for the current user.
            hidden_size: Dimension of GRU hidden state (default 128).
            lr_bp: Learning rate for BP training (default 1e-5).
            decay_after_seconds: Inactivity period after which bond
                decays (default 1 hour).
            decay_factor: Multiplier applied during decay (default 0.9).
            initial_bond: Starting bond score for new users (default 0.2).
            cold_start_steps: Number of initial steps with accelerated
                growth (default 20).
            cold_start_coef: Acceleration multiplier (default 1.5).
            skepticism_threshold: Bond jump threshold triggering
                skepticism (default 0.5).
            repeat_cos_threshold: Cosine similarity threshold for
                repeat detection (default 0.95).
            repeat_count: Number of consecutive repeats to trigger
                protection (default 3).
            repeat_window: Window size for repeat detection (default 20).
            diversity_window: Window size for diversity
                protection (default 50).

        """
        super().__init__()
        self.model = model_interface
        self.state_manager = state_manager
        self.user_id = user_id
        self.prism_name = 'bond'  # Key used in storage.
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
        # Neural network components:
        # GRU takes concatenated
        # [user_embedding, response_embedding, timestamp_normalised]
        input_dim = 2 * self.embed_dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=False)
        self.mlp_bond = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Optimiser for online learning
        # (not used in current version, but kept for future).
        self.optimizer = torch.optim.Adam(
            list(self.gru.parameters()) + list(self.mlp_bond.parameters()),
            lr=lr_bp
        )
        # Previous bond value for smoothing in diversity protection.
        self.prev_bond: float = self.initial_bond
        # Load or initialise state.
        self._load_state()

    def _default_state(self) -> dict[str, Any]:
        """Create a default state for a new user (no history)."""
        return {
            'state': torch.zeros(1, 1, self.hidden_size),  # GRU hidden state
            'step_count': 0,
            'last_active': 0.0,
            'session_bond_start': self.initial_bond,
            # List of past GRU states for repeat detection
            'history_states': [],
            # List of recent surprise scores
            'surprise_history': [],
            'skepticism_triggered': False,
            'ethics_violation': False,
            'prev_bond': self.initial_bond,
        }

    def _load_state(self) -> None:
        """Load the bond state from the persistent store.

        If no state exists for this user, a default state is created.
        The state includes tensors, scalars, and lists that are restored
        from their serialised representation.
        """
        data = self.state_manager.load(self.user_id, self.prism_name)
        if data is None:
            data = self._default_state()
        # Restore GRU state tensor.
        self.state = data['state'].to(torch.float32) if isinstance(
            data['state'], torch.Tensor
        ) else torch.tensor(data['state'], dtype=torch.float32)
        # Restore scalar fields.
        self.step_count = data['step_count']
        self.last_active = data['last_active']
        self.session_bond_start = data['session_bond_start']
        self.prev_bond = data.get('prev_bond', self.initial_bond)
        # Restore history lists (tensors may be stored as lists).
        self.history_states = [
            torch.tensor(
                s, dtype=torch.float32
            ) if not isinstance(s, torch.Tensor) else s
            for s in data['history_states']
        ]
        self.surprise_history = data['surprise_history']
        self.skepticism_triggered = data.get('skepticism_triggered', False)
        self.ethics_violation = data.get('ethics_violation', False)
        # Temporary buffers (not persisted).
        self.reserve_state: torch.Tensor | None = None
        self.pending_state: torch.Tensor | None = None

    def _save_state(self) -> None:
        """Save the current bond state to the persistent store.

        Tensors are converted to lists for serialisation.
        The state is stored under the user_id and prism_name ('bond').
        """
        data = {
            'state': self.state.cpu().numpy().tolist(),
            'step_count': self.step_count,
            'last_active': self.last_active,
            'session_bond_start': self.session_bond_start,
            'history_states': [
                s.cpu().numpy().tolist()
                for s in self.history_states
            ],
            'surprise_history': self.surprise_history,
            'skepticism_triggered': self.skepticism_triggered,
            'ethics_violation': self.ethics_violation,
            'prev_bond': self.prev_bond,
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
        """Compute the bond score for the current interaction.

        The method:
            1. Retrieves embeddings for user and response.
            2. Applies temporal decay if inactive for >1 hour.
            3. Updates the GRU state with the new interaction.
            4. Computes the raw bond score from the MLP.
            5. Saves the current bond as previous for smoothing.
            6. Applies cold-start acceleration (first 20 steps).
            7. Applies protection mechanisms (skepticism, repeat, diversity).
            8. Handles ethics violation (if flagged).
            9. Updates history for repeat detection.
            10. Saves the state.

        Args:
            user_text: The user's message.
            response_text: Galatea's response.
            timestamp: Current time (seconds since epoch).
            surprise: Surprise score from SP (0-1).
            threat: Threat score from TP (0-1).

        Returns:
            The bond score (float, 0-1).

        """
        # Get embeddings via the model interface.
        e_user = self.model.get_response_embedding(user_text)
        e_response = self.model.get_response_embedding(response_text)
        # Temporal decay: if inactive for longer than decay_after_seconds,
        # the GRU state is multiplied by decay_factor.
        if self.last_active > 0 and (timestamp - self.last_active) > self.decay_after_seconds:  # noqa: E501
            self.state = self.state * self.decay_factor
        # Prepare GRU input: concatenate
        # [e_user, e_response, timestamp_normalised]
        ts_norm = torch.tensor(
            [timestamp / 3600.0], dtype=torch.float32
        ).unsqueeze(0)  # (1,1)
        if e_user.dim() == 1:
            e_user = e_user.unsqueeze(0)
        if e_response.dim() == 1:
            e_response = e_response.unsqueeze(0)
        x = torch.cat([e_user, e_response, ts_norm], dim=-1)  # (1, input_dim)
        x = x.unsqueeze(0)  # (1, 1, input_dim)
        # GRU step: update hidden state.
        self.state, _ = self.gru(x, self.state)
        self.last_active = timestamp
        self.step_count += 1
        # Compute bond score from the MLP.
        bond = self.mlp_bond(self.state.squeeze(0)).item()
        # Save current bond as previous for smoothing.
        self.prev_bond = bond
        # Cold start: accelerated growth during the
        # first `cold_start_steps` steps if conditions
        # (high surprise, low threat, no repeats) are met.
        if self.step_count <= self.cold_start_steps:
            if (surprise > 0.6) and (threat < 0.4) and not self._has_repeats():
                bond = self.initial_bond + (bond - self.initial_bond) * self.cold_start_coef  # noqa: E501
        bond = max(0.0, min(1.0, bond))
        # Apply protection mechanisms.
        bond = self._apply_skepticism(bond)
        bond = self._apply_repeat_protection(bond)
        bond = self._apply_diversity_protection(bond, surprise)
        # Handle ethics violation (external trigger).
        if self.ethics_violation:
            bond = max(0.0, bond - 0.3)
            self.ethics_violation = False
        # Append current state to history for repeat detection.
        self._update_history()
        # Automatically save state after each forward pass.
        self._save_state()
        return bond

    def _apply_skepticism(self, bond: float) -> float:
        """Skepticism protection: if bond rises more than
        `skepticism_threshold` in a single session,
        halve the bond and set a flag.

        This prevents rapid inflation of bond via flattery or manipulation.
        """
        if (bond - self.session_bond_start) > self.skepticism_threshold:
            bond *= 0.5
            self.skepticism_triggered = True
        else:
            self.skepticism_triggered = False
        return bond

    def _apply_repeat_protection(self, bond: float) -> float:
        """Repeat protection: if the last `repeat_window`
        states are too similar (cosine similarity > repeat_cos_threshold)
        for `repeat_count` times, reduce bond by 30%.

        This prevents mechanical repetition from inflating bond.
        """
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
        """Diversity protection: if the variance of recent surprise scores
        is very low (<0.05), slow down bond growth (smoothing).

        This encourages diverse interactions and discourages
        boring/static dialogue.
        """
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > self.diversity_window:
            self.surprise_history.pop(0)
        if len(self.surprise_history) >= 10:
            var = torch.var(torch.tensor(self.surprise_history))
            if var < 0.05:
                # Use the real previous bond value for smoothing.
                bond = 0.9 * bond + 0.1 * self.prev_bond
        return bond

    def _has_repeats(self) -> bool:
        """Quick check for repeats in the last 5 steps.
        Used during cold start acceleration to avoid granting
        extra growth when the user is repeating themselves.
        """
        if len(self.history_states) < 5:
            return False
        current = self.state.squeeze().clone().detach()
        for s in self.history_states[-5:]:
            if cosine_similarity(current, s) > self.repeat_cos_threshold:
                return True
        return False

    def _update_history(self) -> None:
        """Append the current GRU state to the history,
        keeping only the last `repeat_window` states.
        """
        self.history_states.append(self.state.squeeze().clone().detach())
        if len(self.history_states) > self.repeat_window:
            self.history_states.pop(0)

    def update(self, *args, **kwargs) -> None:
        """Training step (placeholder).

        Training is handled by the separate BondTrainer class.
        This method exists only to satisfy the abstract Prism interface.
        """

    def reset(self) -> None:
        """Reset the bond state to initial values (for a new user/session)."""
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
        self.prev_bond = self.initial_bond

    def set_ethics_violation(self) -> None:
        """Trigger an ethics violation, which will reduce
        bond on the next forward pass.
        """
        self.ethics_violation = True

    # Double-buffer methods for safe rollback
    # (used by ThreatPrism or Ethics classifier).

    def save_reserve(self) -> None:
        """Save a copy of the current state as a reserve (for rollback)."""
        self.reserve_state = self.state.clone()

    def save_pending(self) -> None:
        """Save a copy of the current state as a pending update."""
        self.pending_state = self.state.clone()

    def rollback(self) -> None:
        """Revert to the reserve state if available."""
        if self.reserve_state is not None:
            self.state = self.reserve_state.clone()
            self.reserve_state = None
