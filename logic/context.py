from typing import Dict

class ContextRetriever:

    def __init__(self, persona_data: Dict):
        self.persona = persona_data
        self.weights = persona_data['behavioral_traits']['purpose_weights']

    def calculate_purpose_drift(self, transaction_purpose: str) -> float:
       
        # Retrieve the historical weight of this purpose for the persona
        # If the purpose is unknown (e.g., 'Crypto Buy'), weight defaults to 0.0
        historical_affinity = self.weights.get(transaction_purpose, 0.0)
        
        # Purpose Drift is the inverse of historical affinity
        drift_score = 1.0 - historical_affinity
        return drift_score

    def validate_human_preamble(self, session_events: list) -> float:
        if not session_events or len(session_events) < 2:
            return 0.9  # High Friction for 'Flash Transactions'
        
        passive_events = ['check_balance', 'view_history', 'app_nav']
        has_preamble = any(event in session_events for event in passive_events)
        
        return 0.1 if has_preamble else 0.7