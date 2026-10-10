"""Portable NPZ trajectories, with JSON metadata and no pickle objects."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
from .env import DT, OBS_NAMES, FEATURE_NAMES
from .vehicle_model import VEHICLE_STATE_NAMES, DrivingConfig
from dataclasses import asdict
from .vehicle import SIMULATOR_VERSION, CAR_LENGTH, CAR_WIDTH, WHEELBASE
from .dynamics import parameter_dict


TELEMETRY_NAMES = [
    "longitudinal_g",
    "lateral_g",
    "downforce_n",
    "front_slip_deg",
    "rear_slip_deg",
    "rpm",
    "gear",
    "pitch",
    "roll",
    "grade",
    "bank",
    "tc_active",
    "abs_active",
]


class Recorder:
    def __init__(self, env, source="human", seed=None, control_settings=None):
        if env.steps != 0:
            raise ValueError("Create Recorder immediately after reset(), before stepping")
        if seed is not None and seed != env.reset_seed:
            raise ValueError("Recorder seed must match the environment reset seed")
        self.env = env
        from copy import deepcopy

        self.control_settings = deepcopy(control_settings or {"device": "direct_actions"})
        self.track_digest = env.track.digest
        self.reset_options = dict(env.reset_options)
        self.observations = [env.observe()]
        self.states = [env.state()]
        self.vehicle_states = [env.vehicle.vector()]
        self.commands, self.contacts = [], []
        self.actions, self.rewards, self.features = [], [], []
        self.terminated, self.truncated = [], []
        self.telemetry = []
        self.source, self.seed = source, env.reset_seed

    def append(self, action, transition):
        if self.env.steps != len(self.actions) + 1 or self.env.track.digest != self.track_digest:
            raise ValueError(
                "Record exactly once after every step; do not reset or switch tracks during a recording"
            )
        applied = np.clip(np.asarray(action), self.env.action_space.low, self.env.action_space.high)
        if applied.shape != (3,) or not np.allclose(applied, self.env.action):
            raise ValueError("Recorded action does not match the most recent environment action")
        obs, reward, terminated, truncated, info = transition
        self.actions.append(self.env.action.copy())
        self.observations.append(obs.copy())
        self.states.append(self.env.state())
        self.vehicle_states.append(self.env.vehicle.vector())
        self.commands.append(self.env.last_commands.copy())
        self.contacts.append(
            [self.env.contact, self.env.impact, self.env.collisions, self.env.recoveries]
        )
        self.rewards.append(reward)
        self.features.append([info["reward_features"][k] for k in FEATURE_NAMES])
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.telemetry.append([info["telemetry"][k] for k in TELEMETRY_NAMES])

    def save(self, directory="data", label="unlabeled", reason="completed"):
        if not self.actions:
            return None
        if label not in ("good", "bad", "unlabeled"):
            raise ValueError("Unknown label")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"{stamp}_{uuid4().hex[:8]}.npz"
        env = self.env
        meta = {
            "schema_version": 2,
            "control_settings": self.control_settings,
            "simulator_version": SIMULATOR_VERSION,
            "dynamics_parameters": parameter_dict(),
            "telemetry_names": TELEMETRY_NAMES,
            "vehicle_state_names": VEHICLE_STATE_NAMES,
            "command_names": ["shift", "direction", "recover"],
            "contact_names": ["contact", "impact_mps", "collisions", "recoveries"],
            "driving_config": asdict(env.driving),
            "track": env.track.name,
            "vehicle": {"length_m": CAR_LENGTH, "width_m": CAR_WIDTH, "wheelbase_m": WHEELBASE},
            "track_version": env.track.version,
            "track_sha256": env.track.digest,
            "dt": DT,
            "seed": self.seed,
            "label": label,
            "source": self.source,
            "reason": reason,
            "reset_options": self.reset_options,
            "max_steps": env.max_steps,
            "observation_names": OBS_NAMES,
            "state_names": ["x", "y", "heading", "speed", "lateral_speed", "yaw_rate", "steering"],
            "action_names": ["steering", "throttle", "brake"],
            "feature_names": FEATURE_NAMES,
            "reward_weights": env.weights,
            "result": env.info(),
        }
        np.savez_compressed(
            path,
            observations=np.asarray(self.observations, np.float32),
            states=np.asarray(self.states),
            actions=np.asarray(self.actions, np.float32),
            rewards=np.asarray(self.rewards),
            reward_features=np.asarray(self.features),
            terminated=np.asarray(self.terminated),
            truncated=np.asarray(self.truncated),
            timestamps=np.arange(len(self.states)) * DT,
            telemetry=np.asarray(self.telemetry),
            vehicle_states=np.asarray(self.vehicle_states),
            commands=np.asarray(self.commands),
            contacts=np.asarray(self.contacts),
            metadata=json.dumps(meta),
        )
        return path


def load(path):
    """Load and validate a trajectory. Never deserialize Python pickle objects."""
    try:
        with np.load(path, allow_pickle=False) as z:
            result = {k: z[k].copy() for k in z.files if k != "metadata"}
            result["metadata"] = json.loads(str(z["metadata"]))
        validate(result)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid trajectory {path}: {exc}") from exc
    return result


def validate(data):
    meta = data["metadata"]
    if meta["schema_version"] not in (1, 2):
        raise ValueError("Unsupported dataset schema")
    if meta["label"] not in ("good", "bad", "unlabeled"):
        raise ValueError("Invalid label")
    for key in (
        "simulator_version",
        "source",
        "track_sha256",
        "result",
        "observation_names",
        "action_names",
        "state_names",
        "feature_names",
        "reward_weights",
    ):
        if key not in meta:
            raise ValueError(f"Missing metadata: {key}")
    if not np.isfinite(meta["dt"]) or meta["dt"] <= 0:
        raise ValueError("Invalid timestep")
    n = len(data["actions"])
    if "telemetry" in data:
        if (
            data["telemetry"].shape != (n, len(meta.get("telemetry_names", [])))
            or not np.isfinite(data["telemetry"]).all()
        ):
            raise ValueError("telemetry shape must match the named fields")
    if n < 1:
        raise ValueError("Empty trajectory")
    shapes = {
        "actions": (n, 3),
        "observations": (n + 1, len(meta["observation_names"])),
        "states": (n + 1, 7),
        "rewards": (n,),
        "reward_features": (n, len(meta["feature_names"])),
        "terminated": (n,),
        "truncated": (n,),
        "timestamps": (n + 1,),
    }
    if meta["schema_version"] == 2:
        shapes.update(
            vehicle_states=(n + 1, len(VEHICLE_STATE_NAMES)), commands=(n, 3), contacts=(n, 4)
        )
        if meta["vehicle_state_names"] != VEHICLE_STATE_NAMES:
            raise ValueError("Unsupported vehicle-state ordering")
        DrivingConfig(**meta["driving_config"])
        if meta["command_names"] != ["shift", "direction", "recover"]:
            raise ValueError("Unsupported command ordering")
        if meta["contact_names"] != ["contact", "impact_mps", "collisions", "recoveries"]:
            raise ValueError("Unsupported contact ordering")
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"{key}: expected finite array of shape {shape}")
    if meta["schema_version"] == 2:
        if (
            np.any(np.abs(data["commands"][:, :2]) > 1)
            or not np.isin(data["commands"], [-1, 0, 1]).all()
            or np.any(data["commands"][:, 2] < 0)
        ):
            raise ValueError("Invalid driving commands")
    if np.any(data["actions"] < np.array([-1, 0, 0])) or np.any(data["actions"] > 1):
        raise ValueError("Actions outside supported bounds")
    for key in ("terminated", "truncated"):
        if data[key].dtype != np.bool_ or data[key][:-1].any():
            raise ValueError(
                f"{key}: flags must be boolean and only the last transition may end the run"
            )
    if not np.allclose(data["timestamps"], np.arange(n + 1) * meta["dt"], atol=1e-8, rtol=1e-7):
        raise ValueError("Timestamps do not match the simulation timestep")
    if meta["feature_names"] != (
        FEATURE_NAMES if meta["schema_version"] == 2 else FEATURE_NAMES[:4]
    ):
        raise ValueError("Unsupported feature ordering")
    weights = np.array([meta["reward_weights"][k] for k in meta["feature_names"]])
    if not np.isfinite(weights).all() or not np.allclose(
        data["rewards"], data["reward_features"] @ weights
    ):
        raise ValueError("Rewards do not match recorded features and weights")


def transitions(trajectory):
    """Return aligned (observation, action, next_observation) arrays for IO/IRL/SL."""
    validate(trajectory)
    return trajectory["observations"][:-1], trajectory["actions"], trajectory["observations"][1:]


def feature_sum(trajectory, gamma=1.0):
    """Discounted feature sum in metadata ordering, useful for linear IO/IRL objectives."""
    validate(trajectory)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    return np.sum(
        trajectory["reward_features"] * (gamma ** np.arange(len(trajectory["actions"])))[:, None],
        axis=0,
    )
