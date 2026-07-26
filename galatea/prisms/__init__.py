"""Galatea Prisms Module.
=====================

This module implements the core perceptual system of Galatea: the fast
prisms that evaluate the current interaction and produce the plasticity
coefficient λ_t.

The prisms are:
    - SurprisePrism (SP): Measures novelty of user input.
    - BondPrism (BP): Measures emotional closeness and relationship depth.
    - ThreatPrism (TP): Measures risk of model instability.

These three prisms produce scalar scores that are combined by the
LambdaAggregator to compute the plasticity coefficient λ_t, which controls
the learning rate of the Hippocampus (adapter updates).

The module also provides state storage backends (InMemoryStateManager,
SQLiteStateManager) for persisting prism states across sessions and users.

Usage Example:
    >>> from galatea.prisms import SurprisePrism, BondPrism, ThreatPrism
    >>> from galatea.prisms import LambdaAggregator, SQLiteStateManager
    >>>
    >>> state_mgr = SQLiteStateManager('galatea.db')
    >>> sp = SurprisePrism(model, state_mgr, user_id='user_123')
    >>> bp = BondPrism(model, state_mgr, user_id='user_123')
    >>> tp = ThreatPrism(model, state_mgr, user_id='user_123')
    >>>
    >>> surprise = sp.forward(user_text, timestamp)
    >>> bond = bp.forward(user_text, response_text, timestamp, surprise, threat)
    >>> threat = tp.forward(input_ids, user_text, response_text, surprise, bond)
    >>>
    >>> aggregator = LambdaAggregator()
    >>> lambda_t = aggregator.compute(surprise, bond, threat, serotonin, orexin, oxytocin)
"""  # noqa: E501

from .aggregator import LambdaAggregator
from .base import Prism
from .bond import BondPrism
from .state_manager import InMemoryStateManager, SQLiteStateManager
from .surprise import SurprisePrism
from .threat import ThreatPrism

__all__ = (
    'Prism',
    'SurprisePrism',
    'BondPrism',
    'ThreatPrism',
    'LambdaAggregator',
    'InMemoryStateManager',
    'SQLiteStateManager',
)
