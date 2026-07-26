"""Surprise Prism (SP) Module.

This module implements the SurprisePrism, which evaluates how unexpected
the user's next query is given the conversation history.

The SurprisePrism operates in two modes:
    - MVP mode: uses average embedding of the last 5 queries
        as a baseline.
    - Full mode: uses an MLP predictor `f_pred` to predict
        the next embedding and computes surprise as the distance
        between predicted and actual embeddings.

The state (history of embeddings, last prediction, counters, etc.)
is persisted via a StateManager to survive across sessions and users.

The surprise score is used by the LambdaAggregator to influence the
plasticity coefficient λ_t, and by the Adrenaline Prism for activation.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ..model_interfaces.interface import ModelInterface
from ..utils.helpers import cosine_distance, cosine_similarity
from .base import Prism
from .state_manager import StateManager


class SurprisePrism(Prism):
    """Surprise Prism (SP) - evaluates novelty of user input.

    The SP measures how surprising the current user query is relative
    to the recent interaction history. High surprise indicates that the
    user has introduced new information or shifted the conversation,
    which may trigger increased plasticity.

    Two operational modes:
        - 'mvp': simple baseline using average of last 5 embeddings.
        - 'full': learned predictor `f_pred` that forecasts the next
            embedding; surprise is derived from the prediction error.

    The prism maintains a persistent state (history, last prediction,
    exploration flag, etc.) using a StateManager.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        state_manager: StateManager,
        user_id: str,
        mode: str = 'mvp',
        tau: float = 0.1,
        topic_threshold: float = 0.7,
        timeout_minutes: float = 5.0,
        hidden_size: int = 256,
        lr_pred: float = 1e-4,
        epsilon_explore: float = 0.02,
        lambda_var: float = 0.01,
    ):
        """Initialise the SurprisePrism.

        Args:
            model_interface: Interface to the language model.
            state_manager: Persistent storage for prism state.
            user_id: Identifier for the current user.
            mode: 'mvp' or 'full' (default 'mvp').
            tau: Temperature for sigmoid scaling in full mode (default 0.1).
            topic_threshold: Cosine distance threshold for topic shift
                filtering (default 0.7).
            timeout_minutes: Inactivity period after which surprise
                is set to neutral (0.5) (default 5.0).
            hidden_size: Hidden size for the predictor MLP (default 256).
            lr_pred: Learning rate for the predictor (default 1e-4).
            epsilon_explore: Exploration probability for ε-greedy
                (default 0.02).
            lambda_var: Regularisation weight for variance penalty
                (default 0.01).

        """
        super().__init__()
        self.model = model_interface
        self.state_manager = state_manager
        self.user_id = user_id
        self.prism_name = 'surprise'
        self.embed_dim = model_interface.get_embed_dim()
        self.mode = mode
        self.tau = tau
        self.topic_threshold = topic_threshold
        self.timeout_minutes = timeout_minutes
        # Fields for full mode.
        self.f_pred: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lambda_var = lambda_var
        self.epsilon_explore = epsilon_explore
        if mode == 'full':
            self.f_pred = nn.Sequential(
                nn.Linear(self.embed_dim * 4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.embed_dim)
            )
            self.optimizer = torch.optim.Adam(
                self.f_pred.parameters(),
                lr=lr_pred
            )
        # Load or initialise state.
        self._load_state()

    def _default_state(self) -> dict[str, Any]:
        """Create a default state for a new user (no history)."""
        return {
            'history': [],              # List of past embeddings
            'last_pred': None,          # Last prediction (for training)
            'last_timestamp': None,     # Timestamp of last update
            'low_surprise_counter': 0,  # Counter for low-surprise detection
            'explore_active': False,    # Exploration flag
        }

    def _load_state(self) -> None:
        """Load the surprise state from the persistent store.

        If no state exists, a default state is created.
        Tensors are restored from their serialised representations.
        """
        data = self.state_manager.load(self.user_id, self.prism_name)
        if data is None:
            data = self._default_state()
        # Restore history of embeddings.
        self.history: list[torch.Tensor] = []
        for h in data.get('history', []):
            if isinstance(h, list):
                self.history.append(torch.tensor(h, dtype=torch.float32))
            elif isinstance(h, torch.Tensor):
                self.history.append(h)
            else:
                self.history.append(torch.tensor(h, dtype=torch.float32))
        # Restore last prediction (if any).
        self.last_pred: torch.Tensor | None = None
        lp = data.get('last_pred')
        if lp is not None:
            if isinstance(lp, list):
                self.last_pred = torch.tensor(lp, dtype=torch.float32)
            elif isinstance(lp, torch.Tensor):
                self.last_pred = lp
            else:
                self.last_pred = torch.tensor(lp, dtype=torch.float32)
        self.last_timestamp: float | None = data.get('last_timestamp')
        self.low_surprise_counter: int = data.get('low_surprise_counter', 0)
        self.explore_active: bool = data.get('explore_active', False)

    def _save_state(self) -> None:
        """Save the current surprise state to the persistent store.

        Tensors are converted to lists for serialisation.
        """
        data = {
            'history': [h.cpu().numpy().tolist() for h in self.history],
            'last_pred': (
                self.last_pred.cpu().numpy().tolist()
                if self.last_pred is not None
                else None
            ),
            'last_timestamp': self.last_timestamp,
            'low_surprise_counter': self.low_surprise_counter,
            'explore_active': self.explore_active,
        }
        self.state_manager.save(self.user_id, self.prism_name, data)

    def forward(
        self,
        current_text: str,
        timestamp: float | None = None,
        prev_texts: list[str] | None = None,
    ) -> float:
        """Compute the surprise score for the current user query.

        The method:
            1. Retrieves the embedding for the current query.
            2. If a timeout has occurred (> timeout_minutes),
                returns neutral (0.5).
            3. Computes surprise either in MVP or full mode.
            4. Saves the state after the computation.

        Args:
            current_text: The user's query text.
            timestamp: Timestamp of the query (optional).
            prev_texts: Previous query texts (for full mode, only last 3 used).

        Returns:
            Surprise score (float, 0-1).

        """
        embed_current = self.model.get_response_embedding(current_text)
        # Timeout: if the user has been inactive for too long,
        # return neutral surprise.
        if timestamp is not None and self.last_timestamp is not None:
            if (timestamp - self.last_timestamp) > (self.timeout_minutes * 60):
                return 0.5
        score: float
        if self.mode == 'mvp':
            score = self._forward_mvp(embed_current)
        else:
            score = self._forward_full(embed_current, prev_texts)
        # Save state after each forward pass.
        self._save_state()
        return score

    def _forward_mvp(self, embed_current: torch.Tensor) -> float:
        """Compute surprise using the MVP baseline
        (average of last 5 embeddings).

        Args:
            embed_current: Embedding of the current query.

        Returns:
            Surprise score (float, 0-1).

        """
        if len(self.history) == 0:
            return 0.5
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        avg_embed = torch.stack(recent).mean(dim=0)
        sim = cosine_similarity(embed_current, avg_embed)
        score = 1.0 - sim
        # Topic shift filter: if the query is very different from
        # the previous one, reduce surprise to avoid false
        # positives on topic changes.
        if len(self.history) > 0:
            prev_embed = self.history[-1]
            dist = cosine_distance(embed_current, prev_embed)
            if dist > self.topic_threshold:
                score *= 0.3
        return max(0.0, min(1.0, score))

    def _forward_full(
        self,
        embed_current: torch.Tensor,
        prev_texts: list[str] | None
    ) -> float:
        """Compute surprise using the learned predictor `f_pred`.

        Args:
            embed_current: Embedding of the current query.
            prev_texts: Previous query texts (only the last 3 are used).

        Returns:
            Surprise score (float, 0-1).

        Raises:
            RuntimeError: If `f_pred` is not initialised (mode != 'full').

        """
        if self.f_pred is None:
            raise RuntimeError("f_pred not initialised. Use mode='full'.")
        # Get embeddings of the last 3 previous queries (if available).
        if prev_texts is None:
            prev_embeds = []
        else:
            prev_embeds = [
                self.model.get_response_embedding(t)
                for t in prev_texts[-3:]
            ]
        # Pad with zeros if fewer than 3 previous queries.
        while len(prev_embeds) < 3:
            prev_embeds.append(torch.zeros(self.embed_dim))
        # Concatenate current and previous embeddings as input.
        x = torch.cat([embed_current] + prev_embeds, dim=-1)
        e_pred = self.f_pred(x)
        self.last_pred = e_pred
        # Compute distance and apply sigmoid with temperature.
        dist = cosine_distance(e_pred, embed_current)
        score = 1.0 / (1.0 + math.exp(-dist / (self.tau + 1e-8)))
        # ε-greedy exploration to avoid self-reinforcing predictability.
        if self.explore_active and torch.rand(1).item() < self.epsilon_explore:
            score = 0.8
        return max(0.0, min(1.0, score))

    def update(
        self,
        actual_text: str,
        lambda_t: float,
        timestamp: float | None = None
    ) -> None:
        """Update the surprise state with the actual next query.

        This method:
            1. Updates the timestamp.
            2. Adds the actual query embedding to the history (MVP mode).
            3. For full mode, trains the predictor using the prediction error
                (MSE loss) scaled by λ_t, and applies the variance penalty.
            4. Updates the low-surprise counter to activate exploration
                if surprise has been consistently low.

        Args:
            actual_text: The actual next user query (after prediction).
            lambda_t: Plasticity coefficient from aggregator
                (scales the update).
            timestamp: Timestamp of the actual query (optional).

        """
        if timestamp is not None:
            self.last_timestamp = timestamp
        embed_actual = self.model.get_response_embedding(actual_text)
        if self.mode == 'mvp':
            self.history.append(embed_actual.detach().clone())
            # Limit history size to avoid unbounded growth.
            if len(self.history) > 1000:
                self.history.pop(0)
        else:  # full mode
            if self.last_pred is not None:
                if self.optimizer is None:
                    raise RuntimeError(
                        'Optimizer not initialised for full mode.'
                    )
                loss = F.mse_loss(self.last_pred, embed_actual)
                # Variance penalty to prevent overconfidence.
                if self.lambda_var > 0:
                    var_loss = self.lambda_var * torch.max(
                        torch.tensor(0.0),
                        0.1 - torch.var(self.last_pred)
                    )
                    loss += var_loss
                # Scale loss by λ_t and backpropagate.
                (loss * lambda_t).backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_pred = None
            # Monitor low-surprise periods to activate exploration.
            self.low_surprise_counter += 1
            if self.low_surprise_counter >= 200:
                self.explore_active = True
            else:
                self.explore_active = False
        # Save state after update.
        self._save_state()

    def reset(self) -> None:
        """Reset the surprise state for a new user or session."""
        self.history = []
        self.last_pred = None
        self.last_timestamp = None
        self.low_surprise_counter = 0
        self.explore_active = False
        # Remove state from storage.
        self.state_manager.delete(self.user_id, self.prism_name)
