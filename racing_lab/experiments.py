"""Reusable, headless evaluation for any callable observation -> action policy."""

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable

import numpy as np

from .env import RacingEnv, DT
from .vehicle import SIMULATOR_VERSION


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "simulator": SIMULATOR_VERSION,
    }
    for name in ("racing-lab", "numpy", "gymnasium", "scikit-learn", "pygame-ce"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions


@dataclass(frozen=True)
class EvaluationConfig:
    tracks: tuple[str, ...] = ("monza",)
    episodes: int = 3
    seed: int = 0
    max_seconds: float = 300.0
    lateral_jitter: float = 1.0

    def __post_init__(self):
        if not self.tracks or isinstance(self.tracks, str):
            raise ValueError("tracks must be a nonempty sequence of track IDs or JSON paths")
        if not isinstance(self.episodes, int) or self.episodes < 1:
            raise ValueError("episodes must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.max_seconds) or self.max_seconds < DT:
            raise ValueError("max_seconds must allow at least one step")
        if not np.isfinite(self.lateral_jitter) or self.lateral_jitter < 0:
            raise ValueError("lateral_jitter must be finite and nonnegative")


def summarize(rows):
    """Aggregate per track. Never include failed laps in lap-time statistics."""
    summaries = {}
    for track in sorted({r["track_id"] for r in rows}):
        group = [r for r in rows if r["track_id"] == track]
        times = [r["lap_time"] for r in group if r["success"]]
        summaries[track] = {
            "episodes": len(group),
            "successes": len(times),
            "success_rate": len(times) / len(group),
            "lap_time_mean": float(np.mean(times)) if times else None,
            "lap_time_std": float(np.std(times)) if times else None,
            "offtrack_fraction_mean": float(np.mean([r["offtrack_fraction"] for r in group])),
            "action_change_mean": float(np.mean([r["smoothness"] for r in group])),
        }
    return summaries


def evaluate_policy(
    policy: Callable | None,
    config: EvaluationConfig | None = None,
    *,
    policy_name="custom",
    reward_weights=None,
    render=False,
    view="2d",
    on_episode=None,
):
    """Evaluate an observation-only callable; None selects the privileged reference driver.

    A policy with a reset() method is reset at each episode. The evaluator cannot
    seed randomness inside arbitrary callables: initialize that in your algorithm.
    """
    from .learning import reference_driver

    config = config or EvaluationConfig()
    rows = []
    interrupted = False
    clock = None
    if render:
        import pygame

        clock = pygame.time.Clock()
    for track in config.tracks:
        env = RacingEnv(
            track, max_seconds=config.max_seconds, reward_weights=reward_weights, view=view
        )
        try:
            for episode in range(config.episodes):
                seed = config.seed + episode
                obs, _ = env.reset(seed=seed, options={"lateral_jitter": config.lateral_jitter})
                if policy is not None and callable(getattr(policy, "reset", None)):
                    policy.reset()
                total = 0.0
                if env.viewer:
                    env.viewer.trail = []
                while True:
                    action = reference_driver(env) if policy is None else policy(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    if render:
                        env.render()
                        env.viewer.note = "AUTONOMOUS EVALUATION | ESC exits"
                        events = pygame.event.get()
                        for event in events:
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                                env.viewer.cycle_camera()
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                                env.viewer.mode = "2d" if env.viewer.mode == "3d" else "3d"
                        if any(
                            e.type == pygame.QUIT
                            or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                            for e in events
                        ):
                            interrupted = True
                            break
                        clock.tick(60)
                    if terminated or truncated:
                        break
                if interrupted:
                    break
                rows.append(
                    {
                        "track_id": str(track),
                        "track_sha256": env.track.digest,
                        "seed": seed,
                        "return": total,
                        "terminated": terminated,
                        "truncated": truncated,
                        **{k: v for k, v in info.items() if k != "reward_features"},
                    }
                )
                if on_episode:
                    on_episode(rows[-1])
            if interrupted:
                break
        finally:
            env.close()
    weights = {"progress": 1.0, "time": 0.1, "offtrack": 2.0, "smoothness": 0.02}
    weights.update(reward_weights or {})
    return {
        "report_version": 1,
        "policy": "reference (privileged)" if policy is None else policy_name,
        "config": asdict(config),
        "reward_weights": weights,
        "dt": DT,
        "runtime": runtime_versions(),
        "interrupted": interrupted,
        "episodes": rows,
        "summary": summarize(rows),
    }


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
