"""Keyboard adapter only. Policies still send unassisted environment actions."""

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class KeyboardSettings:
    version: int = 1
    assisted: bool = True
    rise_seconds: float = 0.65
    return_seconds: float = 0.24
    lateral_acceleration: float = 13.0


class KeyboardController:
    def __init__(self, assisted=True):
        self.settings = KeyboardSettings(assisted=assisted)
        self.reset()

    def reset(self):
        self.steer = 0.0
        self.action = np.zeros(3, np.float32)

    def metadata(self):
        return {"device": "keyboard", **asdict(self.settings)}

    def update(self, target, speed, dt):
        s = self.settings
        rate = 1 / (s.return_seconds if target[0] == 0 else s.rise_seconds)
        self.steer += float(np.clip(target[0] - self.steer, -rate * dt, rate * dt))
        # Geometric steering envelope; does not read the road or auto-correct yaw.
        limit = min(0.45, math.atan(3.4 * s.lateral_acceleration / max(speed * speed, 1)))
        self.action[0] = self.steer * (limit / 0.45 if s.assisted else 1.0)
        self.action[1:] += np.clip(np.asarray(target[1:]) - self.action[1:], -5 * dt, 3 * dt)
        return self.action.copy()
