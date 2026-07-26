"""Bond Prism Trainer Module.

This module provides the BondTrainer class, which implements online learning
for the BondPrism using three training signals:
    1. User return prediction (binary classification)
    2. Contrastive learning (real vs shuffled interaction chains)
    3. Soft signal from other prisms (e.g., from SP or TP)

The trainer holds the optimizer and provides methods for each training signal.
It will be used by the Hippocampus or an external orchestration layer during
online learning (Stage 1+).
"""

import torch

from torch import optim

from .bond import BondPrism


class BondTrainer:
    """Trainer for online learning of BondPrism.

    The trainer manages the optimizer and implements three training signals.
    It operates on the BondPrism's GRU and MLP components.

    The trainer is designed to be called from the online learning pipeline,
    where batches of interactions are collected and used for updates.

    Attributes:
        bond_prism: The BondPrism instance to train.
        optimizer: PyTorch optimizer (Adam) for GRU and MLP parameters.
        device: Device to use for computations (cuda or cpu).

    """

    def __init__(
        self,
        bond_prism: BondPrism,
        lr: float = 1e-5,
        device: str = 'cuda',
    ):
        """Initialise the BondTrainer.

        Args:
            bond_prism: The BondPrism instance to train.
            lr: Learning rate for the optimizer.
            device: Device to use for computations ('cuda' or 'cpu').

        """
        self.bond_prism = bond_prism
        self.device = device
        self.optimizer = optim.Adam(
            list(
                bond_prism.gru.parameters()
            ) + list(
                bond_prism.mlp_bond.parameters()
            ),
            lr=lr,
        )

    def train_on_return_prediction(
        self,
        user_emb: torch.Tensor,
        response_emb: torch.Tensor,
        timestamp: float,
        returned: bool,
    ) -> float:
        """Train BP on the user return prediction task (binary classification).

        This signal trains the BP to predict whether the user will return
        after the current interaction. The target is `returned` (True/False).
        The loss is binary cross-entropy.

        Args:
            user_emb: User embedding (tensor).
            response_emb: Response embedding (tensor).
            timestamp: Timestamp of the interaction.
            returned: Whether the user returned (True/False).

        Returns:
            The loss value (float) for logging.

        """
        # Implementation will be added in Stage 1.
        # This is a placeholder.
        raise NotImplementedError(
            'train_on_return_prediction will be implemented in Stage 1'
        )

    def train_contrastive(
        self,
        positive_chain: list[tuple[torch.Tensor, torch.Tensor, float]],
        negative_chain: list[tuple[torch.Tensor, torch.Tensor, float]],
    ) -> float:
        """Train BP using contrastive learning between real and
        shuffled chains.

        This signal encourages the BP to differentiate between real interaction
        sequences (positive) and shuffled ones (negative). The loss is a
        contrastive loss (e.g., InfoNCE or triplet margin).

        Args:
            positive_chain: List of (user_emb, response_emb, timestamp)
                from a real chain.
            negative_chain: List of (user_emb, response_emb, timestamp) from a
                shuffled chain.

        Returns:
            The loss value (float) for logging.

        """
        raise NotImplementedError(
            'train_contrastive will be implemented in Stage 1'
        )

    def train_with_soft_signal(
        self,
        user_emb: torch.Tensor,
        response_emb: torch.Tensor,
        timestamp: float,
        soft_target: float,
    ) -> float:
        """Train BP using a soft signal from other prisms
        (e.g., from SP or TP).

        This signal allows the BP to incorporate information from other prisms,
        such as high surprise or low threat, to adjust its bond estimate.

        Args:
            user_emb: User embedding (tensor).
            response_emb: Response embedding (tensor).
            timestamp: Timestamp of the interaction.
            soft_target: A target value (0-1) derived from other prisms.

        Returns:
            The loss value (float) for logging.

        """
        raise NotImplementedError(
            'train_with_soft_signal will be implemented in Stage 1'
        )

    def step(self) -> None:
        """Perform a single optimisation step (gradient update)."""  # noqa: D402, E501
        self.optimizer.step()
        self.optimizer.zero_grad()
