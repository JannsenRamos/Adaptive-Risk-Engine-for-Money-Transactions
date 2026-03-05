import time
from datetime import datetime
from typing import Dict, Tuple

class VelocityEngine:
    """
    MLE Role: Token Bucket Algorithm for real-time rate limiting.
    DS Role: Parameterizing 'Burstable Trust' based on persona daily velocity.
    
    Improvements:
    - Time-of-day awareness: off-hours transactions incur extra penalty
    - Cumulative daily spend cap: tracks total ₱ spent today
    """

    def __init__(self, persona_id: str, avg_daily_vel: int,
                 avg_daily_spend: float = 200.0, config: dict = None):
        cfg = config or {}
        vel_cfg = cfg.get("velocity", {})

        self.capacity = vel_cfg.get("burst_capacity", 3.0)
        self.refill_rate = avg_daily_vel / 24.0   # tokens per hour
        self.tokens = self.capacity
        self.last_update = time.time()

        # Time-of-day config
        self.off_hours_start = vel_cfg.get("off_hours_start", 23)
        self.off_hours_end   = vel_cfg.get("off_hours_end", 6)
        self.off_hours_penalty = vel_cfg.get("off_hours_penalty", 1.5)

        # Cumulative daily spend
        self.daily_spend_limit = avg_daily_spend * vel_cfg.get("daily_spend_multiplier", 3.0)
        self.cumulative_spend_today = 0.0
        self._spend_date = datetime.now().date()

    # ------------------------------------------------------------------
    def _refill(self):
        """Refills tokens based on elapsed time."""
        now = time.time()
        elapsed = (now - self.last_update) / 3600  # hours
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def _reset_daily_spend_if_new_day(self):
        today = datetime.now().date()
        if today != self._spend_date:
            self.cumulative_spend_today = 0.0
            self._spend_date = today

    def is_off_hours(self) -> bool:
        hour = datetime.now().hour
        if self.off_hours_start > self.off_hours_end:
            return hour >= self.off_hours_start or hour < self.off_hours_end
        return self.off_hours_start <= hour < self.off_hours_end

    # ------------------------------------------------------------------
    def check_velocity(self, amount: float = 0.0) -> Tuple[bool, float]:
        """
        Returns (is_allowed, velocity_penalty).
        Penalty > 1.0 means elevated friction.
        """
        self._refill()
        self._reset_daily_spend_if_new_day()

        penalty = 1.0

        # 1. Token bucket — frequency check
        if self.tokens >= 1.0:
            self.tokens -= 1.0
        else:
            penalty *= 2.0  # Transacting too fast

        # 2. Time-of-day check
        if self.is_off_hours():
            penalty *= self.off_hours_penalty

        # 3. Cumulative daily spend cap
        self.cumulative_spend_today += amount
        if self.cumulative_spend_today > self.daily_spend_limit:
            overage_ratio = self.cumulative_spend_today / max(self.daily_spend_limit, 1)
            penalty *= min(overage_ratio, 3.0)  # cap at 3x

        return True, round(penalty, 2)

    def get_spend_info(self) -> dict:
        """Returns current daily spend stats for display."""
        self._reset_daily_spend_if_new_day()
        return {
            "spent_today": round(self.cumulative_spend_today, 2),
            "daily_limit": round(self.daily_spend_limit, 2),
            "pct_used": round(min(self.cumulative_spend_today / max(self.daily_spend_limit, 1), 1.0), 3),
            "is_off_hours": self.is_off_hours(),
            "tokens_remaining": round(self.tokens, 2)
        }