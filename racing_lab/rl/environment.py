"""Author-style controls/history/reward using existing numerical Racing Lab sensors.

Adapted from Remonda's MIT-licensed ac_env.py; see THIRD_PARTY_NOTICES.md.
No trajectory, expert controller, dataset loader or image observations are used.
"""

from collections import deque
from dataclasses import asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..env import OBS_NAMES, RacingEnv
from ..vehicle_model import DrivingConfig, WHEEL_X, WHEEL_Y
from .config import ADAPTER_VERSION, RacingTaskConfig

CONTROL_NAMES = ("steering", "throttle", "brake")
SAC_OBSERVATION_NAMES = (
    tuple(f"current/{name}" for name in OBS_NAMES)
    + tuple(f"past_{i}/{name}" for name in CONTROL_NAMES for i in range(3, 0, -1))
    + tuple(f"applied/{name}" for name in CONTROL_NAMES)
    + tuple(f"past_{i}/{name}" for i in range(3, 0, -1) for name in OBS_NAMES)
)


class SACRacingEnv(gym.Wrapper):
    """Fixed-rate, multi-lap online task; all actions are normalized control increments."""

    def __init__(self, track="monza", config=None, *, render_mode=None, view="3d"):
        self.config = config or RacingTaskConfig()
        dt = 1 / self.config.decision_hz
        super().__init__(
            RacingEnv(
                track,
                dt=dt,
                laps=1_000_000,
                max_seconds=self.config.max_seconds,
                driving=DrivingConfig(
                    automatic=True,
                    traction_control=self.config.traction_control,
                    abs=self.config.abs,
                ),
                render_mode=render_mode,
                view=view,
            )
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (len(SAC_OBSERVATION_NAMES),), np.float32
        )
        # Source: 600/180 normalized steering units/s, 12 pedal units/s.
        self.delta_limits = np.array([600 / 180, 12, 12], np.float32) * dt
        self.metadata = dict(self.env.metadata, render_fps=self.config.decision_hz)
        self.history = deque(maxlen=3)
        self.control_history = deque(maxlen=3)

    def _feedback(self):
        return np.array([self.env.steering / 0.45, *self.env.action[1:]], np.float32)

    def _observe(self, current):
        # Source pads the entire history until three preceding samples are available.
        if len(self.history) < 3:
            past_obs = np.zeros(3 * len(OBS_NAMES), np.float32)
            past_controls = np.zeros(9, np.float32)
        else:
            past_obs = np.concatenate(self.history)
            past_controls = np.asarray(self.control_history).T.ravel()
        return np.concatenate([current, past_controls, self._feedback(), past_obs]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        current, info = self.env.reset(seed=seed, options=options)
        self.action_space.seed(seed)
        self.targets = np.array([0, -1, -1], np.float32)
        self.history.clear()
        self.control_history.clear()
        self.current = current.copy()
        self.low_speed_steps = self.completed_laps = self.valid_laps = 0
        self.lap_times = []
        self.last_lap_elapsed = 0.0
        self.lap_start_collisions = self.lap_start_offtrack = 0
        self.ended = False
        return self._observe(current), self._info(info, "running")

    def _info(self, info, reason):
        result = dict(info)
        result.update(
            adapter_version=ADAPTER_VERSION,
            completed_laps=self.completed_laps,
            valid_laps=self.valid_laps,
            best_valid_lap=min(self.lap_times) if self.lap_times else None,
            valid_lap_times=list(self.lap_times),
            success=self.valid_laps > 0,
            termination_reason=reason,
            requested_controls=self.targets.tolist(),
            applied_controls=self._feedback().tolist(),
            task_config=asdict(self.config),
        )
        return result

    def step(self, action):
        if self.ended:
            raise RuntimeError("Episode ended: call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("SAC action must contain three finite control increments")
        self.history.append(self.current.copy())
        self.control_history.append(self._feedback())
        self.targets = np.clip(self.targets + np.clip(action, -1, 1) * self.delta_limits, -1, 1)
        absolute = np.array(
            [self.targets[0], (self.targets[1] + 1) / 2, (self.targets[2] + 1) / 2], np.float32
        )
        current, _, _, truncated, info = self.env.step(absolute)
        self.current = current
        distance, gap, _ = self.env.track.project(self.env.pos)
        # Source 'speed' is overall speed; substitute planar speed magnitude.
        speed = float(np.hypot(self.env.speed, self.env.lateral_speed))
        reward = 3.6 * speed * (1 - abs(gap) / 12) / 300
        self.low_speed_steps = self.low_speed_steps + 1 if speed < self.config.low_speed_mps else 0
        c, s = np.cos(self.env.heading), np.sin(self.env.heading)
        wheels = np.column_stack([WHEEL_X, WHEEL_Y]) @ np.array([[c, s], [-s, c]])
        offsets = self.env.track.project_many(wheels + self.env.pos, hint=distance)[1]
        offroad = int(np.count_nonzero(np.abs(offsets) > self.env.track.width / 2)) >= 3
        stalled = self.low_speed_steps > self.config.low_speed_seconds * self.config.decision_hz
        terminated = offroad or stalled
        reason = "offroad" if offroad else "low_speed" if stalled else "running"
        next_lap = self.completed_laps + 1
        if (
            self.env.progress >= next_lap * self.env.track.length
            and self.env.next_checkpoint >= 40 * next_lap + 1
        ):
            self.completed_laps += 1
            valid = (
                self.env.collisions == self.lap_start_collisions
                and self.env.offtrack_steps == self.lap_start_offtrack
                and self.env.recoveries == 0
            )
            if valid:
                self.valid_laps += 1
                self.lap_times.append(info["elapsed"] - self.last_lap_elapsed)
            self.last_lap_elapsed = info["elapsed"]
            self.lap_start_collisions = self.env.collisions
            self.lap_start_offtrack = self.env.offtrack_steps
        if self.config.max_laps and self.completed_laps >= self.config.max_laps:
            truncated = True  # Evaluation cutoff, not an absorbing task failure.
            reason = "lap_limit" if not terminated else reason
        elif truncated and not terminated:
            reason = "time_limit"
        self.ended = bool(terminated or truncated)
        self.env.done = self.ended
        return (
            self._observe(current),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(info, reason),
        )
