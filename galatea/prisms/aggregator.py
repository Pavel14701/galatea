"""Lambda Aggregator Module.

This module implements the LambdaAggregator class, which computes
the plasticity coefficient λ_t (lambda) based on the assessments from the
fast Prisms (SP, BP, TP) and the current hormonal state
(serotonin, orexin, oxytocin).

The λ_t value is the core signal that controls
the learning rate of the Hippocampus:
- Higher λ_t means more plasticity (faster adaptation).
- Lower λ_t means less plasticity (slower or blocked learning).

The computation accounts for:
- Surprise (novelty) and Bond (relationship strength) as positive drivers.
- Threat as a negative modulator (high threat suppresses learning).
- Hormonal modulation (orexin amplifies surprise, oxytocin reduces threat).
- Adrenaline (AP) as a rare, high-impact event that temporarily boosts λ_t.
- Normalisation when multiple adapters are active.

The aggregator also enforces limits on AP activations per user and globally.
"""

from typing import TypedDict


class APThresholds(TypedDict):
    """Type definition for adrenaline (AP) thresholds."""

    surprise: float
    bond: float
    threat: float


class LambdaAggregator:
    """Computes the plasticity coefficient λ_t from Prism outputs and hormones.

    The core formula is:
        λ_t = min( (surprise_eff * bond) / (threat_eff + ε), λ_max_base )

    Where:
        - surprise_eff = surprise * orexin_factor
        - threat_eff = max(min_threat, threat - oxytocin_effect * oxytocin)
        - orexin_factor = 1.0 + 0.3 * (orexin - 0.5)
        - λ_max_base = 4.0 + 2.0 * serotonin (range: 4.0-6.0)

    If Adrenaline (AP) conditions are met:
        λ_t = min(λ_t * ap_multiplier, ap_lambda_max)
        (typically 2.0x boost, capped at 10.0)

    The result is then normalised by the number of active adapters.

    AP is triggered only if:
        - surprise_eff > threshold (default 0.9, 0.95 for young users)
        - bond > threshold (default 0.8, 0.85 for young users)
        - threat_eff < threshold (default 0.5, 0.4 for young users)
        - per-user and global counters are not exhausted.

    Attributes:
        epsilon: Small constant to avoid division by zero.
        lambda_max_base_min: Minimum base λ_max (when serotonin = 0).
        lambda_max_base_max: Maximum base λ_max (when serotonin = 1).
        ap_surprise_threshold: Surprise threshold for AP (default users).
        ap_bond_threshold: Bond threshold for AP (default users).
        ap_threat_threshold: Threat threshold for AP (default users).
        ap_lambda_max: Maximum λ_t allowed during AP.
        ap_multiplier: Multiplier applied during AP.
        oxytocin_effect: How much oxytocin reduces threat (per unit).
        min_threat_after_oxytocin: Minimum threat after oxytocin reduction.
        young_ap_*: Stricter thresholds for young (new) users.
        ap_user_counter: Count of AP activations for the current user in
            the last 10 cycles.
        ap_global_counter: Count of AP activations globally in the
            current cycle.

    """

    def __init__(
        self,
        epsilon: float = 0.01,
        lambda_max_base_min: float = 4.0,
        lambda_max_base_max: float = 6.0,
        ap_surprise_threshold: float = 0.9,
        ap_bond_threshold: float = 0.8,
        ap_threat_threshold: float = 0.5,
        ap_lambda_max: float = 10.0,
        ap_multiplier: float = 2.0,
        oxytocin_effect: float = 0.05,
        min_threat_after_oxytocin: float = 0.05,
        # Stricter thresholds for young users (new users with limited history)
        young_ap_surprise_threshold: float | None = 0.95,
        young_ap_bond_threshold: float | None = 0.85,
        young_ap_threat_threshold: float | None = 0.4,
    ):
        """Initialise the LambdaAggregator with configurable parameters.

        Args:
            epsilon: Small epsilon to avoid division by zero.
            lambda_max_base_min: Lower bound of base λ_max (when serotonin=0).
            lambda_max_base_max: Upper bound of base λ_max (when serotonin=1).
            ap_surprise_threshold: Surprise threshold for AP (normal users).
            ap_bond_threshold: Bond threshold for AP (normal users).
            ap_threat_threshold: Threat threshold for AP (normal users).
            ap_lambda_max: Upper limit for λ_t during AP.
            ap_multiplier: Factor to multiply λ_t when AP is triggered.
            oxytocin_effect: Reduction of threat per unit of oxytocin.
            min_threat_after_oxytocin: Floor for threat after
                oxytocin reduction.
            young_ap_surprise_threshold: Surprise threshold for young users.
            young_ap_bond_threshold: Bond threshold for young users.
            young_ap_threat_threshold: Threat threshold for young users.

        """
        self.epsilon = epsilon
        self.lambda_max_base_min = lambda_max_base_min
        self.lambda_max_base_max = lambda_max_base_max
        self.ap_surprise_threshold = ap_surprise_threshold
        self.ap_bond_threshold = ap_bond_threshold
        self.ap_threat_threshold = ap_threat_threshold
        self.ap_lambda_max = ap_lambda_max
        self.ap_multiplier = ap_multiplier
        self.oxytocin_effect = oxytocin_effect
        self.min_threat_after_oxytocin = min_threat_after_oxytocin
        # If young thresholds are not provided, fall back to the normal ones.
        self.young_ap_surprise_threshold = young_ap_surprise_threshold or ap_surprise_threshold  # noqa: E501
        self.young_ap_bond_threshold = young_ap_bond_threshold or ap_bond_threshold  # noqa: E501
        self.young_ap_threat_threshold = young_ap_threat_threshold or ap_threat_threshold  # noqa: E501
        # Counters to enforce AP limits (prevent overuse).
        # AP count for current user in the last 10 cycles.
        self.ap_user_counter: int = 0
        # AP count across all users in the current cycle.
        self.ap_global_counter: int = 0

    def compute(
        self,
        surprise: float,
        bond: float,
        threat: float,
        serotonin: float,
        orexin: float,
        oxytocin: float,
        num_adapters: int = 1,
        is_young_user: bool = False,
        ap_available: bool = True,
        max_ap_per_user_cycles: int = 10,
        max_ap_per_global_cycle: int = 3,
    ) -> float:
        """Compute the plasticity coefficient λ_t for the current step.

        Args:
            surprise: Surprise score from SP (0-1).
            bond: Bond score from BP (0-1).
            threat: Threat score from TP (0-1).
            serotonin: Serotonin level (0-1), influences base λ_max.
            orexin: Orexin level (0-1), amplifies surprise.
            oxytocin: Oxytocin level (0-1), reduces threat.
            num_adapters: Number of active adapters (for normalisation).
            is_young_user: If True, use stricter AP thresholds.
            ap_available: Whether AP is allowed in this step.
            max_ap_per_user_cycles: Max AP triggers per user per 10 cycles.
            max_ap_per_global_cycle: Max AP triggers globally per cycle.

        Returns:
            The computed λ_t value (capped between 0 and 10).

        """
        # 1. Orexin modulation: boosts surprise when orexin is high.
        orexin_factor = 1.0 + 0.3 * (orexin - 0.5)  # ranges 0.85-1.15
        surprise_eff = max(0.0, min(1.0, surprise * orexin_factor))
        # 2. Oxytocin reduces perceived threat (but not below a floor).
        threat_eff = max(
            self.min_threat_after_oxytocin,
            threat - self.oxytocin_effect * oxytocin
        )
        # 3. Compute base λ_max from serotonin (linear interpolation).
        lambda_max_base = self.lambda_max_base_min + (
            self.lambda_max_base_max - self.lambda_max_base_min
        ) * serotonin
        lambda_max_base = max(
            self.lambda_max_base_min,
            min(self.lambda_max_base_max, lambda_max_base)
        )
        # 4. Base formula: surprise * bond divided by (threat + ε)
        numerator = surprise_eff * bond
        denominator = threat_eff + self.epsilon
        lambda_t = numerator / denominator
        lambda_t = min(lambda_t, lambda_max_base)
        # 5. Check for Adrenaline (AP) conditions.
        if ap_available:
            thresholds = self._get_ap_thresholds(is_young_user)
            if (
                surprise_eff > thresholds['surprise']
                and bond > thresholds['bond']
                and threat_eff < thresholds['threat']
            ):
                # AP allowed only if counters are not exhausted.
                if (
                    self.ap_user_counter < max_ap_per_user_cycles
                    and self.ap_global_counter < max_ap_per_global_cycle
                ):
                    # Apply AP boost: multiply and cap at ap_lambda_max.
                    lambda_t = min(
                        lambda_t * self.ap_multiplier,
                        self.ap_lambda_max
                    )
                    self.ap_user_counter += 1
                    self.ap_global_counter += 1
        # 6. Normalise if multiple adapters are active.
        if num_adapters > 1:
            lambda_t /= num_adapters
        # Final safety clamp.
        return max(0.0, min(10.0, lambda_t))

    def _get_ap_thresholds(self, is_young: bool) -> APThresholds:
        """Return the appropriate AP thresholds based on user age.

        Args:
            is_young: True for new/young users, False for established users.

        Returns:
            A dictionary with keys 'surprise', 'bond', 'threat'.

        """
        if is_young:
            return {
                'surprise': self.young_ap_surprise_threshold,
                'bond': self.young_ap_bond_threshold,
                'threat': self.young_ap_threat_threshold,
            }
        else:
            return {
                'surprise': self.ap_surprise_threshold,
                'bond': self.ap_bond_threshold,
                'threat': self.ap_threat_threshold,
            }

    def reset_ap_counters(self, global_cycle_reset: bool = False) -> None:
        """Reset the AP counters.

        This should be called at the beginning of each sleep cycle or when
        a new user session starts.

        Args:
            global_cycle_reset: If True, also reset the global AP counter.
                                Otherwise, only reset the per-user counter.

        """
        self.ap_user_counter = 0
        if global_cycle_reset:
            self.ap_global_counter = 0
