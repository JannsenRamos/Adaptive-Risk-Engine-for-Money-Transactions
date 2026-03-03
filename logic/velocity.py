import time
from typing import Dict, Tuple

class VelocityEngine:
    """
    MLE Role: Implementation of the Token Bucket Algorithm for real-time rate limiting.
    DS Role: Parameterizing 'Burstable Trust' based on Persona daily velocity.
    """
    def __init__(self, persona_id: str, avg_daily_vel: int):
        # We translate 'Daily Velocity' into an hourly 'Refill Rate'
        # Example: Student (5/day) gets ~0.2 tokens/hour. 
        # But we allow a 'Burst' of 3 to account for lunch/transport spikes.
        self.capacity = 3.0 
        self.refill_rate = avg_daily_vel / 24.0
        self.tokens = self.capacity
        self.last_update = time.time()

    def _refill(self):
        """Refills tokens based on elapsed time since the last transaction."""
        now = time.time()
        elapsed = (now - self.last_update) / 3600  # Convert seconds to hours
        new_tokens = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    def check_velocity(self) -> Tuple[bool, float]:
        """
        FR3: The Temporal Velocity Cap.
        Returns (is_allowed, current_friction_multiplier).
        """
        self._refill()
        
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 1.0  # Normal friction
        else:
            # If bucket is empty, we don't necessarily 'Block' yet; 
            # we apply a 'Velocity Penalty' to the Friction Slider.
            penalty = 2.0  # Double the friction if they are moving too fast
            return True, penalty