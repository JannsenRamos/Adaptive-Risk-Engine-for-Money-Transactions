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
    def __init__(self, persona_data):
        self.persona = persona_data
        self.retriever = ContextRetriever(persona_data)
        self.velocity = VelocityEngine(persona_data['persona_id'], 
                                       persona_data['behavioral_traits']['avg_daily_velocity'])
        self.slider = FrictionSlider(base_threshold=persona_data['default_threshold'])

    def run_trial(self, is_attack: bool, tx_purpose: str):
        """Simulates a single transaction event."""
        # 1. Generate Preamble (Attackers bypass this)
        preamble_events = ["app_nav", "check_balance"] if not is_attack else []
        preamble_score = self.retriever.validate_human_preamble(preamble_events)
        
        # 2. Get Purpose Drift
        drift = self.retriever.calculate_purpose_drift(tx_purpose)
        
        # 3. Check Velocity
        _, vel_penalty = self.velocity.check_velocity()
        
        # 4. Base ML Score (Simulated: Attacks are 'obvious' to the base model)
        p_model = np.random.uniform(0.7, 0.95) if is_attack else np.random.uniform(0.01, 0.2)
        
        # 5. Get SAVE Friction Score
        friction = self.slider.calculate_friction(p_model, drift, preamble_score, vel_penalty)
        
        return {
            "is_attack": is_attack,
            "friction": friction,
            "blocked": friction > 0.6  # Our 'Hard Block/Step-Up' Threshold
        }

# --- Execution Logic ---
def run_benchmark(persona):
    sim = SAVESimulator(persona)
    results = []
    
    # Simulate 1000 Normal Transactions
    for _ in range(1000):
        # Students mostly buy food
        purp = "food_bev" if np.random.random() < 0.8 else "p2p_transfer"
        results.append(sim.run_trial(is_attack=False, tx_purpose=purp))
        
    # Simulate 100 Attack Transactions (The 'Account Drain')
    for _ in range(100):
        results.append(sim.run_trial(is_attack=True, tx_purpose="p2p_transfer"))
        
    return pd.DataFrame(results)