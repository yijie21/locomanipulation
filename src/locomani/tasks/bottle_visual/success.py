"""Scoring only: SONIC's >6 cm lift held for >2 seconds, checked on sim time.

No phase, grip, tilt or hand-speed gate. Task truth here is never a VLA input.
"""

from dataclasses import dataclass

PROTOCOL = "sonic_lift6cm_hold2s_v1"


@dataclass
class LiftSuccess:
    lifted_at: float | None = None

    def update(self, t: float, rise_m: float, valid: bool = True) -> bool:
        if not valid or not rise_m > 0.06:
            self.lifted_at = None
            return False
        if self.lifted_at is None:
            self.lifted_at = t
        # Avoid a floating-point rounding error accepting exactly two seconds.
        return t - self.lifted_at > 2.0 + 1e-9
