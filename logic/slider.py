import numpy as np
from typing import Dict

class FrictionSlider:
    """
    MLE Role: Implementation of a Weighted Decision Matrix for real-time inference.
    DS Role: Applying Sigmoid-based probability squashing to normalize multi-vector risk.
    """
    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold
        # Weights (Hyperparameters) - In a real DS role, these are optimized via 
        # Grid Search or Bayesian Optimization to minimize Churn.
        self.w_drift = 0.4    # Purpose Drift Weight
        self.w_preamble = 0.3 # Human Preamble Weight
        self.w_velocity = 0.3 # Velocity Penalty Weight

    def _squash(self, x: float) -> float:
        """
        A custom Sigmoid function to map any risk score to the [0.2, 0.9] range.
        Formula: 0.2 + (0.7 / (1 + e^-x))
        """
        return 0.2 + (0.7 / (1 + np.exp(-x)))

    def calculate_friction(self, 
                           p_model: float, 
                           drift: float, 
                           preamble_score: float, 
                           vel_penalty: float) -> float:
        """
        Combines all SAVE signals into a final Friction Score.
        """
        # 1. Calculate the Raw Risk Signal (Linear Combination)
        # We start with the base model probability (p_model)
        raw_risk = (p_model * 0.5) + \
                   (drift * self.w_drift) + \
                   (preamble_score * self.w_preamble) + \
                   (vel_penalty * self.w_velocity)
        
        # 2. Adjust based on the persona's default sensitivity
        # Shift the raw risk based on the user's base threshold
        risk_input = raw_risk - self.base_threshold
        
        # 3. Squash to [0.2, 0.9]
        friction_score = self._squash(risk_input)
        
        return round(float(friction_score), 2)