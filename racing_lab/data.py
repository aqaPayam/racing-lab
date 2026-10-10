"""Portable NPZ trajectories, with JSON metadata and no pickle objects."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
from .env import DT, OBS_NAMES


class Recorder:
    def __init__(self, env, source="human", seed=0):
        self.env = env
        self.observations = [env.observe()]
        self.states = [env.state()]
        self.actions, self.rewards, self.features = [], [], []
        self.terminated, self.truncated = [], []
        self.source, self.seed = source, seed

    def append(self, action, transition):
        obs, reward, terminated, truncated, info = transition
        self.actions.append(np.asarray(action).copy())
        self.observations.append(obs.copy())
        self.states.append(self.env.state())
        self.rewards.append(reward)
        self.features.append(list(info["reward_features"].values()))
        self.terminated.append(terminated)
        self.truncated.append(truncated)

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
        meta = {"schema_version": 1, "simulator_version": "0.1.0", "track": env.track.name,
                "track_version": env.track.version, "track_sha256": env.track.digest,
                "dt": DT, "seed": self.seed, "label": label, "source": self.source, "reason": reason,
                "observation_names": OBS_NAMES, "state_names": ["x","y","heading","speed","lateral_speed","yaw_rate","steering"],
                "action_names": ["steering", "throttle", "brake"],
                "feature_names": ["progress","time","offtrack","smoothness"],
                "reward_weights": env.weights, "result": env.info()}
        np.savez_compressed(path, observations=np.asarray(self.observations, np.float32),
                            states=np.asarray(self.states), actions=np.asarray(self.actions,np.float32),
                            rewards=np.asarray(self.rewards), reward_features=np.asarray(self.features),
                            terminated=np.asarray(self.terminated), truncated=np.asarray(self.truncated),
                            timestamps=np.arange(len(self.states))*DT, metadata=json.dumps(meta))
        return path


def load(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k:z[k].copy() for k in z.files if k!="metadata"}
        result["metadata"] = json.loads(str(z["metadata"]))
    if result["metadata"]["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema")
    return result
