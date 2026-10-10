"""Input-device adapters. Policy actions bypass keyboard shaping."""

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
    def __init__(self, assisted=True, lateral_acceleration=13.0):
        if not np.isfinite(lateral_acceleration) or not 3 <= lateral_acceleration <= 30:
            raise ValueError("Keyboard steering limit must be between 3 and 30 m/s²")
        self.settings = KeyboardSettings(
            assisted=assisted, lateral_acceleration=lateral_acceleration
        )
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


class GamepadController:
    """Configurable joystick axes; default mapping follows common Xbox devices."""

    def __init__(self, index=0, steer_axis=0, throttle_axis=5, brake_axis=2):
        import pygame

        pygame.joystick.init()
        if not 0 <= index < pygame.joystick.get_count():
            raise ValueError("Gamepad not found; run racing-lab controllers to list devices")
        self.device = pygame.joystick.Joystick(index)
        self.device.init()
        self.axes = (steer_axis, throttle_axis, brake_axis)
        if any(not 0 <= i < self.device.get_numaxes() for i in self.axes):
            raise ValueError("Axis mapping does not match this gamepad")
        self.previous = set()

    def reset(self):
        self.previous = set()

    def metadata(self):
        return {
            "device": "gamepad",
            "name": self.device.get_name(),
            "axes": self.axes,
            "trigger_range": [-1, 1],
            "steering_deadzone": 0.06,
            "steering_exponent": 1.7,
            "buttons": {"shift_up": 5, "shift_down": 4, "reverse": 3, "recover": 6},
        }

    def sample(self, env):
        if not self.device.get_init():
            raise ValueError("Gamepad disconnected")
        steer = float(self.device.get_axis(self.axes[0]))
        steer = np.sign(steer) * (max(0.0, abs(steer) - 0.06) / 0.94) ** 1.7
        pedals = [np.clip((self.device.get_axis(i) + 1) / 2, 0, 1) for i in self.axes[1:]]
        pressed = {i for i in range(self.device.get_numbuttons()) if self.device.get_button(i)}
        edges = pressed - self.previous
        self.previous = pressed
        commands = {}
        if 5 in edges:
            commands["shift"] = 1
        if 4 in edges:
            commands["shift"] = -1
        if 3 in edges:
            commands["direction"] = 1 if env.vehicle.gear < 0 else -1
        if 6 in edges:
            commands["recover"] = True
        return np.array([steer, *pedals], dtype=np.float32), commands
