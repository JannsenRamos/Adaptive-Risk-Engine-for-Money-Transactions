import numpy as np
import pandas as pd
from logic.context import ContextRetriever
from logic.velocity import VelocityEngine
from logic.slider import FrictionSlider


class SAVESimulator:
    """
    MLE Role: Benchmarking high-throughput inference across synthetic populations.
    DS Role: Evaluating the Cost-Benefit Frontier of dynamic thresholding.
    """
    def __init__(self, persona_data, config: dict = None):
        self.persona = persona_data
        self.config = config or {}
        self.retriever = ContextRetriever(persona_data)
        self.velocity = VelocityEngine(
            persona_data['persona_id'],
            persona_data['behavioral_traits']['avg_daily_velocity'],
            avg_daily_spend=persona_data['behavioral_traits'].get('avg_daily_spend', 200),
            config=self.config
        )
        self.slider = FrictionSlider(
            base_threshold=persona_data['default_threshold'],
            config=self.config
        )

    def run_trial(self, is_attack: bool, tx_purpose: str, amount: float = 500.0):
        """Simulates a single transaction event."""
        preamble_events = ["app_nav", "check_balance"] if not is_attack else []
        preamble_score = self.retriever.validate_human_preamble(preamble_events)
        drift = self.retriever.calculate_purpose_drift(tx_purpose)
        amount_risk = self.retriever.calculate_amount_risk(amount)
        _, vel_penalty = self.velocity.check_velocity(amount)

        p_model = np.random.uniform(0.7, 0.95) if is_attack else np.random.uniform(0.01, 0.2)

        friction, contributions = self.slider.calculate_friction_with_breakdown(
            p_model, drift, preamble_score, vel_penalty, amount_risk
        )

        cfg_thresh = self.config.get("thresholds", {})
        hard_block = cfg_thresh.get("hard_block", 0.70)

        return {
            "is_attack": is_attack,
            "friction": friction,
            "blocked": friction > hard_block,
            "drift": drift,
            "preamble_score": preamble_score,
            "vel_penalty": vel_penalty,
            "amount_risk": amount_risk,
            **{f"contrib_{k.lower().replace(' ', '_')}": v
               for k, v in contributions.items()}
        }


def run_benchmark(persona, config: dict = None, n_normal: int = 1000, n_attack: int = 100):
    sim = SAVESimulator(persona, config=config)
    results = []

    purposes = list(persona['behavioral_traits']['purpose_weights'].keys())

    for _ in range(n_normal):
        purp = np.random.choice(purposes)
        amount = float(np.random.normal(
            persona['behavioral_traits'].get('avg_daily_spend', 200), 150
        ))
        results.append(sim.run_trial(is_attack=False, tx_purpose=purp, amount=max(10, amount)))

    for _ in range(n_attack):
        amount = float(np.random.uniform(5000, 30000))
        results.append(sim.run_trial(is_attack=True, tx_purpose="p2p_transfer", amount=amount))

    return pd.DataFrame(results)