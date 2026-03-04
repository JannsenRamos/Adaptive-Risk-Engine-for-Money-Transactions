import json
import numpy as np

class PersonaAgent:
    """
    MLE Role: Encapsulating behavioral states for population-scale simulation.
    DS Role: Stochastic generation of transactions based on persona priors.
    """
    def __init__(self, persona_id, personas_path="data/personas.json"):
        with open(personas_path, 'r') as f:
            all_personas = json.load(f)
            # Find the specific persona in the list
            self.data = next(p for p in all_personas if p['persona_id'] == persona_id)
        
        self.id = persona_id
        self.frustration_score = 0
        self.is_active = True

    def simulate_behavior(self, is_attack=False):
        """Generates a transaction event based on the JSON weights."""
        weights = self.data['behavioral_traits']['purpose_weights']
        
        if is_attack:
            return "p2p_transfer", np.random.uniform(5000, 50000)
        
        # Weighted choice for normal behavior
        purposes = list(weights.keys())
        probabilities = list(weights.values())
        # Normalize probabilities to sum to 1
        probabilities = [p/sum(probabilities) for p in probabilities]
        
        chosen_purpose = np.random.choice(purposes, p=probabilities)
        amount = np.random.normal(1000, 500) # Simplified for MVP
        
        return chosen_purpose, max(10, amount)