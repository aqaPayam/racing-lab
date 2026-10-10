"""Deterministic fixed-step four-wheel model; intentionally not an F1 engineering simulator."""

import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
from .track import Track
from .vehicle import road_envelope, SIMULATOR_VERSION
from dataclasses import asdict
from .vehicle_model import VehicleState, DrivingConfig, advance_vehicle
from .collisions import resolve


DT = 1 / 60
LOOKAHEAD = np.array([10, 25, 50, 90, 150, 230], dtype=float)
FEATURE_NAMES = ["progress", "time", "offtrack", "smoothness", "collision", "recovery"]
OBS_NAMES = (
    [
        "speed/85",
        "lateral_speed/20",
        "yaw_rate/3",
        "steering/0.45",
        "lateral_offset/half_width",
        "sin_heading_error",
        "cos_heading_error",
        "half_width/20",
    ]
    + [f"preview_{int(d)}m_{axis}/230" for d in LOOKAHEAD for axis in ("forward", "right")]
    + [
        "gear/8",
        "rpm/15000",
        *[f"wheel_{w}_speed/300" for w in ("fl", "fr", "rl", "rr")],
        *[f"wheel_{w}_slip" for w in ("fl", "fr", "rl", "rr")],
        "pitch",
        "roll",
        "pitch_rate",
        "roll_rate",
        "grade",
        "bank",
        "shift_timer/.1",
        "reverse",
        "barrier_contact",
    ]
)


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
        driving=None,
    ):
        if view not in ("2d", "3d"):
            raise ValueError("view must be 2d or 3d")
        self.driving = DrivingConfig() if driving is None else driving
        if not isinstance(self.driving, DrivingConfig):
            raise ValueError("driving must be a DrivingConfig")
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
        self.weights = {
            "progress": 1.0,
            "time": 0.1,
            "offtrack": 2.0,
            "smoothness": 0.02,
            "collision": 1.0,
            "recovery": 20.0,
        }
        self.weights.update(reward_weights or {})
        if set(self.weights) != set(FEATURE_NAMES):
            raise ValueError("Unknown reward feature")
        if not all(np.isfinite(v) for v in self.weights.values()):
            raise ValueError("Reward weights must be finite")
        self.action_space = spaces.Box(np.array([-1, 0, 0], np.float32), np.ones(3, np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBS_NAMES),), np.float32)
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = float((options or {}).get("start_distance", 0))
        initial_speed = float((options or {}).get("speed", 0))
        if (
            not np.isfinite(start)
            or not np.isfinite(initial_speed)
            or not -6 <= initial_speed <= 90
        ):
            raise ValueError("Reset distance must be finite and speed between -6 and 90 m/s")
        self.pos = self.track.sample(start).copy()
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
        self.speed = initial_speed
        self.vehicle = VehicleState.rolling(initial_speed)
        self.telemetry = advance_vehicle(
            initial_speed, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, self.vehicle, self.driving
        )[-1]
        self.collisions = self.recoveries = 0
        self.contact = False
        self.impact = 0.0
        self.last_commands = np.zeros(3, dtype=np.int8)
        self.sectors = []
        self.invalid_lap = False
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
            "vehicle",
            "collisions",
            "recoveries",
            "contact",
            "impact",
            "last_commands",
            "sectors",
            "invalid_lap",
        )
        return {
            "simulator_version": SIMULATOR_VERSION,
            "track_sha256": self.track.digest,
            "max_steps": self.max_steps,
            "reward_weights": self.weights.copy(),
            "driving": asdict(self.driving),
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
            or snapshot["driving"] != asdict(self.driving)
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
                self.vehicle.gear / 8,
                self.vehicle.rpm / 15000,
                *(self.vehicle.omega / 300),
                *self.vehicle.slip,
                self.vehicle.pitch,
                self.vehicle.roll,
                self.vehicle.pitch_rate,
                self.vehicle.roll_rate,
                float(self.track.profile(s)[1]),
                float(self.track.profile(s)[2]),
                self.vehicle.shift_timer / 0.1,
                float(self.vehicle.gear < 0),
                float(self.contact),
            ],
            dtype=np.float32,
        )

    def info(self):
        _, lateral, tangent = self.track.project(self.pos)
        finished = self.progress >= self.track.length and self.next_checkpoint >= 41
        success = finished and not self.invalid_lap
        return {
            "track": self.track.name,
            "elapsed": self.steps * DT,
            "progress": self.progress / self.track.length,
            "offtrack": bool(
                abs(lateral) + road_envelope(self.heading - tangent) > self.track.width / 2
            ),
            "offtrack_fraction": self.offtrack_steps / max(1, self.steps),
            "success": bool(success),
            "finished": bool(finished),
            "invalid_lap": self.invalid_lap,
            "collisions": self.collisions,
            "impact_mps": self.impact,
            "barrier_contact": self.contact,
            "recoveries": self.recoveries,
            "sector_times": self.sectors.copy(),
            "lap_time": self.steps * DT if success else None,
            "smoothness": self.smoothness / max(1, self.steps),
            "checkpoint": self.next_checkpoint - 1,
            "telemetry": self.telemetry.copy(),
        }

    def step(self, action, *, commands=None):
        if self.done:
            raise RuntimeError("Episode ended: call reset() before step().")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("Action must be three finite values: steering, throttle, brake")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        commands = commands or {}
        if set(commands) - {"shift", "direction", "recover"}:
            raise ValueError("Unknown driving command")
        shift, direction, recover = (commands.get(k, 0) for k in ("shift", "direction", "recover"))
        if (
            shift not in (-1, 0, 1)
            or direction not in (-1, 0, 1)
            or recover not in (0, 1, False, True)
        ):
            raise ValueError("shift/direction must be -1, 0 or 1; recover must be boolean")
        change = float(np.sum((a - self.action) ** 2))
        self.smoothness += change
        self.action = a.copy()
        self.steering += np.clip(a[0] * 0.45 - self.steering, -1.5 * DT, 1.5 * DT)
        self.last_commands = np.array([shift, direction, recover], dtype=np.int8)
        if recover:
            self.pos = self.track.sample(self.last_s)
            _, _, self.heading = self.track.project(self.pos)
            self.speed = self.lateral_speed = self.yaw_rate = self.steering = 0.0
            self.vehicle = VehicleState.rolling()
            self.recoveries += 1
            self.invalid_lap = True
        if direction and abs(self.speed) < 1.5:
            self.vehicle.gear = -1 if direction < 0 else 1
        if shift and not self.driving.automatic:
            self.vehicle.shift(shift)
        self.impact = 0.0
        touched = False
        was_contact = self.contact
        hint = self.last_s
        # Resolve contacts after each 1/240-second integration step.
        for _ in range(4):
            c, sn = np.cos(self.heading), np.sin(self.heading)
            rotation = np.array([[c, sn], [-sn, c]])
            wheel_positions = (
                np.array([[1.8, -0.75], [1.8, 0.75], [-1.6, -0.75], [-1.6, 0.75]]) @ rotation
                + self.pos
            )
            _, offsets, _ = self.track.project_many(wheel_positions, hint=hint)
            offroad = np.abs(offsets) > self.track.width / 2
            _, _, tangent = self.track.project(self.pos)
            _, grade, bank = self.track.profile(hint)
            error = self.heading - tangent
            slope = float(grade * np.cos(error) + np.tan(bank) * np.sin(error))
            cross_slope = float(np.arctan(np.tan(bank) * np.cos(error) - grade * np.sin(error)))
            self.speed, self.lateral_speed, self.yaw_rate, self.telemetry = advance_vehicle(
                self.speed,
                self.lateral_speed,
                self.yaw_rate,
                self.steering,
                float(a[1]),
                float(a[2]),
                offroad,
                DT / 4,
                self.vehicle,
                self.driving,
                slope,
                cross_slope,
            )
            self.heading = wrap(self.heading + self.yaw_rate * DT / 4)
            forward = np.array([np.cos(self.heading), np.sin(self.heading)])
            before = self.pos.copy()
            self.pos += (
                DT
                / 4
                * (forward * self.speed + np.array([-forward[1], forward[0]]) * self.lateral_speed)
            )
            self.pos, self.speed, self.lateral_speed, self.yaw_rate, contact, impact = resolve(
                self.track,
                self.pos,
                self.heading,
                self.speed,
                self.lateral_speed,
                self.yaw_rate,
                hint,
            )
            touched |= contact
            self.impact = max(self.impact, impact)
            self.distance += float(np.linalg.norm(self.pos - before))
            hint = float(self.track.project_many(self.pos[None], hint=hint)[0][0])
        self.contact = bool(touched)
        if self.contact and not was_contact and self.impact > 0.25:
            self.collisions += 1
        s, lateral, _ = self.track.project(self.pos)
        delta = (
            s - self.last_s + self.track.length / 2
        ) % self.track.length - self.track.length / 2
        self.last_s = s
        off = self.info()["offtrack"]
        # Ignore projection jumps; require every gate in order on the road.
        valid_delta = delta if abs(delta) < max(5, abs(self.speed) * DT * 3) else 0.0
        previous_progress = self.progress
        self.progress += valid_delta
        if self.next_checkpoint >= 1:
            for sector in range(len(self.sectors) + 1, 4):
                boundary = sector * self.track.length / 3
                if previous_progress < boundary <= self.progress and self.next_checkpoint >= int(
                    sector * 40 / 3
                ):
                    fraction = (boundary - previous_progress) / max(valid_delta, 1e-10)
                    absolute = (self.steps + fraction) * DT
                    self.sectors.append(float(absolute - sum(self.sectors)))
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
            "collision": -self.impact,
            "recovery": -float(recover),
        }
        reward = sum(self.weights[k] * v for k, v in features.items())
        info = self.info()
        info["reward_features"] = features
        terminated = info["finished"]
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
