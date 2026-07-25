class LambdaAggregator:
    """
    Вычисляет коэффициент пластичности λ_t на основе оценок Призм и гормонов.
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
        # Пороги для молодых пользователей
        young_ap_surprise_threshold: float | None = 0.95,
        young_ap_bond_threshold: float | None = 0.85,
        young_ap_threat_threshold: float | None = 0.4,
    ):
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

        self.young_ap_surprise_threshold = young_ap_surprise_threshold or ap_surprise_threshold  # noqa: E501
        self.young_ap_bond_threshold = young_ap_bond_threshold or ap_bond_threshold  # noqa: E501
        self.young_ap_threat_threshold = young_ap_threat_threshold or ap_threat_threshold  # noqa: E501

        # Счётчики AP (лимиты)
        self.ap_user_counter = 0
        self.ap_global_counter = 0

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
        # 1. Орексин
        orexin_factor = 1.0 + 0.3 * (orexin - 0.5)
        surprise_eff = max(0.0, min(1.0, surprise * orexin_factor))

        # 2. Окситоцин → снижение угрозы
        threat_eff = max(
            self.min_threat_after_oxytocin, threat - self.oxytocin_effect
            * oxytocin
        )

        # 3. Базовый λ_max
        lambda_max_base = self.lambda_max_base_min + (
            self.lambda_max_base_max - self.lambda_max_base_min
        ) * serotonin
        lambda_max_base = max(
            self.lambda_max_base_min, min(
                self.lambda_max_base_max, lambda_max_base
            )
        )

        # 4. Базовая формула
        numerator = surprise_eff * bond
        denominator = threat_eff + self.epsilon
        lambda_t = numerator / denominator
        lambda_t = min(lambda_t, lambda_max_base)

        # 5. Проверка AP
        if ap_available:
            thresholds = self._get_ap_thresholds(is_young_user)
            if (
                surprise_eff > thresholds['surprise']
            ) and (
                bond > thresholds['bond']
            ) and (
                threat_eff < thresholds['threat']
            ):
                if (
                    self.ap_user_counter < max_ap_per_user_cycles
                ) and (
                    self.ap_global_counter < max_ap_per_global_cycle
                ):
                    lambda_t = min(
                        lambda_t * self.ap_multiplier, self.ap_lambda_max
                    )
                    self.ap_user_counter += 1
                    self.ap_global_counter += 1

        # 6. Нормировка
        if num_adapters > 1:
            lambda_t /= num_adapters

        return max(0.0, min(10.0, lambda_t))

    def _get_ap_thresholds(self, is_young: bool) -> dict:
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

    def reset_ap_counters(self, global_cycle_reset: bool = False):
        self.ap_user_counter = 0
        if global_cycle_reset:
            self.ap_global_counter = 0
