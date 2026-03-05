from typing import Dict

class ContextRetriever:
    """
    Computes context-based risk signals:
    - Purpose Drift: how unusual is this transaction type for the persona?
    - Human Preamble: did the session look like a real human navigating the app?
    - Amount Risk: is the amount anomalously high vs. this persona's typical spend?
    """

    def __init__(self, persona_data: Dict):
        self.persona = persona_data
        self.weights = persona_data['behavioral_traits']['purpose_weights']
        self.avg_daily_spend = persona_data['behavioral_traits'].get('avg_daily_spend', 200)

    def calculate_purpose_drift(self, transaction_purpose: str) -> float:
        """
        Purpose Drift is the inverse of historical affinity.
        Unknown categories (e.g., 'crypto') are treated as maximum drift.
        """
        historical_affinity = self.weights.get(transaction_purpose, 0.0)
        drift_score = 1.0 - historical_affinity
        return round(drift_score, 3)

    def validate_human_preamble(self, session_events: list) -> float:
        """
        FR2: Human Preamble Gate.
        A real user navigates the app before transacting.
        A bot / hijacked session jumps straight to payment.
        """
        if not session_events or len(session_events) < 2:
            return 0.9  # High Friction for 'Flash Transactions'

        passive_events = ['check_balance', 'view_history', 'app_nav']
        has_preamble = any(event in session_events for event in passive_events)
        return 0.1 if has_preamble else 0.7

    def calculate_amount_risk(self, amount: float) -> float:
        """
        Amount Risk Signal: flags transactions that are disproportionately
        large relative to the persona's average daily spend.
        Score is clipped to [0, 1]. Anything 3x the daily spend = max risk.
        """
        ratio = amount / max(self.avg_daily_spend, 1.0)
        # Normalize: 0x = 0.0 risk, 3x+ = 1.0 risk
        risk = min(ratio / 3.0, 1.0)
        return round(risk, 3)