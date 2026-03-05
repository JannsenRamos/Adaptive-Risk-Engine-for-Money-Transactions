import numpy as np
from typing import Dict

class FrictionSlider:
    """
    MLE Role: Weighted Decision Matrix for real-time inference.
    DS Role: Sigmoid-based probability squashing to normalize multi-vector risk.
    
    Improvements:
    - Config-driven weights (loaded from weights_config.json)
    - Amount risk signal added as 5th input
    - Returns full signal breakdown for explainability
    """

    def __init__(self, base_threshold: float = 0.5, config: dict = None):
        self.base_threshold = base_threshold

        cfg = (config or {}).get("friction_weights", {})
        self.w_model    = cfg.get("w_model",    0.25)
        self.w_drift    = cfg.get("w_drift",    0.30)
        self.w_preamble = cfg.get("w_preamble", 0.20)
        self.w_velocity = cfg.get("w_velocity", 0.15)
        self.w_amount   = cfg.get("w_amount",   0.10)
        # Sharpness > 1 steepens the sigmoid so scores spread across the
        # full [0.2, 0.9] output range, giving clear separation between
        # normal (low) and attack (high) friction values.
        self.sigmoid_sharpness = cfg.get("sigmoid_sharpness", 3.0)

    def _squash(self, x: float) -> float:
        """
        Maps any risk score to [0.2, 0.9] via sigmoid.
        sigmoid_sharpness controls steepness:
          sharpness=1 → very flat (original, attacks ~0.61 — too low)
          sharpness=3 → steep  (attacks ~0.70+, normals ~0.35)
        """
        return 0.2 + (0.7 / (1 + np.exp(-self.sigmoid_sharpness * x)))

    def calculate_friction(self,
                           p_model: float,
                           drift: float,
                           preamble_score: float,
                           vel_penalty: float,
                           amount_risk: float = 0.0) -> float:
        friction, _ = self.calculate_friction_with_breakdown(
            p_model, drift, preamble_score, vel_penalty, amount_risk
        )
        return friction

    def calculate_friction_with_breakdown(self,
                                          p_model: float,
                                          drift: float,
                                          preamble_score: float,
                                          vel_penalty: float,
                                          amount_risk: float = 0.0) -> tuple:
        """
        Returns (friction_score, contributions_dict).
        contributions_dict maps each signal name to its weighted contribution.
        """
        # Normalise velocity penalty to [0,1] for the linear combination
        vel_norm = min((vel_penalty - 1.0) / 2.0, 1.0)  # 1.0→0, 3.0→1

        contributions = {
            "ML Model":       p_model      * self.w_model,
            "Purpose Drift":  drift        * self.w_drift,
            "Preamble":       preamble_score * self.w_preamble,
            "Velocity":       vel_norm     * self.w_velocity,
            "Amount Risk":    amount_risk  * self.w_amount,
        }

        raw_risk = sum(contributions.values())
        risk_input = raw_risk - self.base_threshold
        friction_score = round(float(self._squash(risk_input)), 2)

        return friction_score, contributions

    def get_weights(self) -> Dict[str, float]:
        return {
            "w_model":    self.w_model,
            "w_drift":    self.w_drift,
            "w_preamble": self.w_preamble,
            "w_velocity": self.w_velocity,
            "w_amount":   self.w_amount,
        }