"""Prism Base Module.

This module defines the abstract base class `Prism` that all Galatea prisms
must implement. A prism is a perceptual unit that processes a single step
of interaction and produces a scalar score in the range [0, 1].

Concrete prisms are:
    - SurprisePrism (SP): novelty of user input.
    - BondPrism (BP): emotional closeness and relationship depth.
    - ThreatPrism (TP): risk of model instability.

Each prism maintains its own internal state (if any) that is updated
after each step. The `forward` method computes the score, `update` refreshes
the state based on the new interaction, and `reset` clears the state for
a new user or session.

All prisms are designed to be used by the LambdaAggregator, which combines
their outputs into the plasticity coefficient λ_t.

Usage Example:
    >>> class MyPrism(Prism):
    ...     def forward(self, *args, **kwargs) -> float:
    ...         return 0.5
    ...     def update(self, *args, **kwargs) -> None:
    ...         pass
    ...     def reset(self) -> None:
    ...         pass
"""

from abc import ABC, abstractmethod


class Prism(ABC):
    """Abstract base class for all Galatea prisms.

    A prism provides a single scalar evaluation of the current interaction
    step, typically in the range [0, 1]. It may maintain an internal state
    that evolves over time (e.g., GRU hidden state for BP, history of
    embeddings for SP).

    All subclasses must implement the `forward`, `update`, and `reset`
    methods.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> float:
        """Compute the prism's score for the current step.

        Concrete implementations define their own arguments (e.g., embeddings,
        timestamps, text, input_ids). The return value must be a float
        between 0 and 1 inclusive.

        Returns:
            A float score representing the prism's evaluation.

        Raises:
            ValueError: If the input arguments are invalid or missing.
            RuntimeError: If the prism is not properly initialised.

        """

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the prism's internal state based on the new interaction.

        This method is called after `forward` to incorporate new information
        (e.g., the actual next query for SP, the response embedding for BP).
        It may also perform online learning (e.g., training the predictor
        in SP full mode).

        The concrete implementation may receive additional arguments such
        as the current query, response, timestamp, or plasticity coefficient.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the prism's internal state to its initial values.

        This method should be called when starting a new user session or
        when clearing the state for any reason. After reset, the prism
        should behave as if no interaction has occurred yet.

        It may also delete persistent state from the storage backend
        if the prism uses a StateManager.
        """
