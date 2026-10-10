"""Wall-time scheduling without changing the deterministic simulator timestep."""

import math


class FixedStepClock:
    def __init__(self, dt=1 / 60, suspend_after=0.5):
        self.dt = dt
        self.suspend_after = suspend_after
        self.pending = 0.0
        self.suspended = False

    def advance(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and nonnegative")
        self.suspended = elapsed > self.suspend_after
        if self.suspended:
            # Window dragging/debugging stalls pause play, never fast-forward unseen input.
            self.pending = 0.0
            return 0
        self.pending += elapsed
        count = int((self.pending + 1e-10) / self.dt)
        self.pending -= count * self.dt
        return count

    def reset(self):
        self.pending = 0.0
        self.suspended = False
