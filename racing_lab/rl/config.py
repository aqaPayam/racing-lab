"""Published SAC settings and explicitly versioned Racing Lab task substitutions."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

SOURCE_REVISION = "e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4"
PAPER_URL = "https://arxiv.org/abs/2407.16680"
ADAPTER_VERSION = 1


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.992
    n_step: int = 3
    hidden_units: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 128
    replay_capacity: int = 10_000_000
    start_steps: int = 2_000
    policy_lr: float = 0.0003
    q_lr: float = 0.0003
    entropy_lr: float = 0.0003
    tau: float = 0.005
    eval_interval: int = 100_000
    eval_episodes: int = 1
    checkpoint_interval: int = 200_000

    def __post_init__(self):
        if self.n_step != 3:
            raise ValueError("This paper baseline requires n_step=3")
        for name in ("batch_size", "replay_capacity", "n_step", "eval_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("start_steps", "eval_interval", "checkpoint_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.hidden_units or any(type(v) is not int or v < 1 for v in self.hidden_units):
            raise ValueError("hidden_units must contain positive integers")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if not math.isfinite(self.tau) or not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        for name in ("policy_lr", "q_lr", "entropy_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RacingTaskConfig:
    decision_hz: int = 25
    max_seconds: float = 600.0
    low_speed_mps: float = 5 / 3.6
    low_speed_seconds: float = 2.0
    max_laps: int | None = None
    traction_control: bool = True
    abs: bool = True

    def __post_init__(self):
        if type(self.decision_hz) is not int or not 10 <= self.decision_hz <= 240:
            raise ValueError("decision_hz must be an integer between 10 and 240")
        for name in ("max_seconds", "low_speed_seconds", "low_speed_mps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_seconds < 1 / self.decision_hz:
            raise ValueError("max_seconds must allow at least one decision")
        if self.max_laps is not None and (type(self.max_laps) is not int or self.max_laps < 1):
            raise ValueError("max_laps must be None or a positive integer")
        if type(self.traction_control) is not bool or type(self.abs) is not bool:
            raise ValueError("Driving aids must be booleans")


def config_hash(config):
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
