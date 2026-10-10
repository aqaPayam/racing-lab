"""Deterministic fixed-step bicycle model; intentionally not an F1 engineering simulator."""

import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from .track import Track
from .vehicle import road_envelope, SIMULATOR_VERSION
from .dynamics import advance

DT = 1 / 60
LOOKAHEAD = np.array([10, 25, 50, 90, 150, 230], dtype=float)
OBS_NAMES = [
    "speed/85",
    "lateral_speed/20",
    "yaw_rate/3",
    "steering/0.45",
    "lateral_offset/half_width",
    "sin_heading_error",
    "cos_heading_error",
    "half_width/20",
] + [f"preview_{int(d)}m_{axis}/230" for d in LOOKAHEAD for axis in ("forward", "right")]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        track="monza",
        render_mode=None,
        max_seconds=300,
        reward_weights=None,
        view="2d",
        renderer="auto",
    ):
        if view not in ("2d", "3d"):
            raise ValueError("view must be 2d or 3d")
        self.view = view
        if renderer not in ("auto", "gpu", "software"):
            raise ValueError("renderer must be auto, gpu, or software")
        self.renderer = renderer
        self.track = Track(track)
        self.render_mode = render_mode
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, human, or rgb_array")
        if not np.isfinite(max_seconds):
            raise ValueError("max_seconds must be finite")
        self.max_steps = int(max_seconds / DT)
        if self.max_steps < 1:
            raise ValueError("max_seconds must allow at least one simulation step")
        self.weights = {"progress": 1.0, "time": 0.1, "offtrack": 2.0, "smoothness": 0.02}
        self.weights.update(reward_weights or {})
        if set(self.weights) != {"progress", "time", "offtrack", "smoothness"}:
            raise ValueError("Unknown reward feature")
        if not all(np.isfinite(v) for v in self.weights.values()):
            raise ValueError("Reward weights must be finite")
        self.action_space = spaces.Box(np.array([-1, 0, 0], np.float32), np.ones(3, np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBS_NAMES),), np.float32)
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.track.sample(0).copy()
        _, _, self.heading = self.track.project(self.pos)
        jitter = (options or {}).get("lateral_jitter", 0.0)
        if not np.isfinite(jitter) or jitter < 0 or jitter > self.track.width / 2 - 1:
            raise ValueError("lateral_jitter must keep the initial car within the road")
        self.reset_seed = seed
        self.reset_options = deepcopy(options or {})
        self.pos += self.np_random.uniform(-jitter, jitter) * np.array(
            [-np.sin(self.heading), np.cos(self.heading)]
        )
        self.speed = self.lateral_speed = self.yaw_rate = self.steering = 0.0
        self.telemetry = advance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)[-1]
        self.steps = self.offtrack_steps = 0
        self.progress = self.last_s = self.smoothness = 0.0
        self.distance = 0.0
        self.last_s = self.track.project(self.pos)[0]
        self.next_checkpoint = 1
        self.action = np.zeros(3, np.float32)
        self.done = False
        return self.observe(), self.info()

    def state(self):
        """Return physical telemetry only; use snapshot() for complete branching state."""
        return np.array(
            [*self.pos, self.heading, self.speed, self.lateral_speed, self.yaw_rate, self.steering],
            dtype=np.float64,
        )

    def snapshot(self):
        """Copy all simulation/RNG state for local counterfactual rollouts (no rendering)."""
        fields = (
            "pos",
            "heading",
            "speed",
            "lateral_speed",
            "yaw_rate",
            "steering",
            "steps",
            "offtrack_steps",
            "progress",
            "last_s",
            "smoothness",
            "next_checkpoint",
            "action",
            "done",
            "reset_seed",
            "reset_options",
            "telemetry",
            "distance",
        )
        return {
            "simulator_version": SIMULATOR_VERSION,
            "track_sha256": self.track.digest,
            "max_steps": self.max_steps,
            "reward_weights": self.weights.copy(),
            "state": {k: deepcopy(getattr(self, k)) for k in fields},
            "rng_state": deepcopy(self.np_random.bit_generator.state),
        }

    def restore(self, snapshot):
        """Restore a trusted snapshot created by this version and configuration."""
        if (
            snapshot["simulator_version"] != SIMULATOR_VERSION
            or snapshot["track_sha256"] != self.track.digest
            or snapshot["max_steps"] != self.max_steps
            or snapshot["reward_weights"] != self.weights
        ):
            raise ValueError(
                "Snapshot belongs to a different simulator, track, or experiment configuration"
            )
        for key, value in snapshot["state"].items():
            setattr(self, key, deepcopy(value))
        self.np_random.bit_generator.state = deepcopy(snapshot["rng_state"])
        if self.viewer:
            self.viewer.trail = []
        return self.observe(), self.info()

    def observe(self):
        s, lateral, tangent = self.track.project(self.pos)
        h = self.heading
        delta = self.track.sample(s + LOOKAHEAD) - self.pos
        local = delta @ np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        return np.array(
            [
                self.speed / 85,
                self.lateral_speed / 20,
                self.yaw_rate / 3,
                self.steering / 0.45,
                lateral / (self.track.width / 2),
                np.sin(h - tangent),
                np.cos(h - tangent),
                self.track.width / 40,
                *(local / 230).ravel(),
            ],
            dtype=np.float32,
        )

    def info(self):
        _, lateral, tangent = self.track.project(self.pos)
        success = self.progress >= self.track.length and self.next_checkpoint >= 41
        return {
            "track": self.track.name,
            "elapsed": self.steps * DT,
            "progress": self.progress / self.track.length,
            "offtrack": bool(
                abs(lateral) + road_envelope(self.heading - tangent) > self.track.width / 2
            ),
            "offtrack_fraction": self.offtrack_steps / max(1, self.steps),
            "success": bool(success),
            "lap_time": self.steps * DT if success else None,
            "smoothness": self.smoothness / max(1, self.steps),
            "checkpoint": self.next_checkpoint - 1,
            "telemetry": self.telemetry.copy(),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended: call reset() before step().")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("Action must be three finite values: steering, throttle, brake")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        change = float(np.sum((a - self.action) ** 2))
        self.smoothness += change
        self.action = a.copy()
        self.steering += np.clip(a[0] * 0.45 - self.steering, -1.5 * DT, 1.5 * DT)
        off = self.info()["offtrack"]
        # Integrate axle forces at 240 Hz while retaining the 60 Hz research API.
        for _ in range(4):
            self.speed, self.lateral_speed, self.yaw_rate, self.telemetry = advance(
                self.speed,
                self.lateral_speed,
                self.yaw_rate,
                self.steering,
                float(a[1]),
                float(a[2]),
                off,
                DT / 4,
            )
            self.heading = wrap(self.heading + self.yaw_rate * DT / 4)
            self.distance += float(np.hypot(self.speed, self.lateral_speed)) * DT / 4
            forward = np.array([np.cos(self.heading), np.sin(self.heading)])
            self.pos += (
                DT
                / 4
                * (forward * self.speed + np.array([-forward[1], forward[0]]) * self.lateral_speed)
            )
        s, lateral, _ = self.track.project(self.pos)
        delta = (
            s - self.last_s + self.track.length / 2
        ) % self.track.length - self.track.length / 2
        self.last_s = s
        off = self.info()["offtrack"]
        # Ignore projection jumps; require every gate in order on the road.
        valid_delta = delta if abs(delta) < max(5, self.speed * DT * 3) else 0.0
        self.progress += valid_delta
        gate = self.next_checkpoint * self.track.length / 40
        if not off and gate <= self.progress <= gate + self.track.width:
            self.next_checkpoint += 1
        self.steps += 1
        self.offtrack_steps += int(off)
        features = {
            "progress": valid_delta,
            "time": -DT,
            "offtrack": -float(off) * DT,
            "smoothness": -change,
        }
        reward = sum(self.weights[k] * v for k, v in features.items())
        info = self.info()
        info["reward_features"] = features
        terminated = info["success"]
        truncated = self.steps >= self.max_steps
        self.done = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self.observe(), float(reward), terminated, truncated, info

    def render(self):
        from .viewer import Viewer

        if self.viewer is None:
            self.viewer = Viewer(self)
        return self.viewer.draw(rgb=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
