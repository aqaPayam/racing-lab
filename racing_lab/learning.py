"""Behavior cloning baseline. Split by entire run, never adjacent frames."""

import json
from pathlib import Path
import numpy as np
from .data import load
from .env import OBS_NAMES
from .vehicle import WHEELBASE, SIMULATOR_VERSION
from .experiments import file_digest, runtime_versions
from .track import Track


def train(directory, output, seed=0, epochs=100, *, tracks=None, completed_only=False):
    """Train behavior cloning on good human runs, splitting whole unique trajectories."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if Path(output).resolve().parent == Path(directory).resolve():
        raise ValueError("Save models outside the dataset directory")
    allowed = {Track(n).digest for n in tracks} if tracks else None
    runs = []
    digests = {}
    for path in sorted(Path(directory).glob("*.npz")):
        run = load(path)
        if allowed is not None and run["metadata"]["track_sha256"] not in allowed:
            continue
        if completed_only and not run["metadata"]["result"]["success"]:
            continue
        if run["metadata"]["label"] == "good" and run["metadata"]["source"] == "human":
            if run["metadata"]["simulator_version"] != SIMULATOR_VERSION:
                raise ValueError(
                    f"{path}: recording uses an older simulator; keep it for replay and collect new training runs"
                )
            if run["metadata"]["observation_names"] != OBS_NAMES:
                raise ValueError(f"Incompatible observations: {path}")
            if not run["metadata"]["driving_config"]["automatic"] or np.any(run["commands"]):
                raise ValueError(
                    f"{path}: the three-output BC baseline requires automatic forward driving without supplemental commands"
                )
            # Copies of one run must not appear on opposite sides of a train/validation split.
            import hashlib

            trajectory_hash = hashlib.sha256(
                run["observations"].tobytes() + run["actions"].tobytes()
            ).hexdigest()
            if trajectory_hash in digests:
                raise ValueError(f"Duplicate demonstration: {path} and {digests[trajectory_hash]}")
            digests[trajectory_hash] = str(path)
            runs.append((path, run))
    if len(runs) < 2:
        raise ValueError("Record at least two human runs labeled good (five or more recommended).")
    driving_config = runs[0][1]["metadata"]["driving_config"]
    if any(run["metadata"]["driving_config"] != driving_config for _, run in runs):
        raise ValueError(
            "Do not mix different vehicle aid/gearbox configurations in one BC experiment"
        )
    order = np.random.default_rng(seed).permutation(len(runs))
    n_val = max(1, round(len(runs) * 0.2))
    val_ids, train_ids = order[:n_val], order[n_val:]

    def arrays(ids):
        return (
            np.concatenate([runs[i][1]["observations"][:-1] for i in ids]),
            np.concatenate([runs[i][1]["actions"] for i in ids]),
        )

    x, y = arrays(train_ids)
    vx, vy = arrays(val_ids)
    scaler = StandardScaler().fit(x)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        max_iter=epochs,
        random_state=seed,
        batch_size=min(256, len(x)),
        learning_rate_init=0.001,
    )
    model.fit(scaler.transform(x), y)
    prediction = np.clip(model.predict(scaler.transform(vx)), [-1, 0, 0], [1, 1, 1])
    report = {
        "simulator_version": SIMULATOR_VERSION,
        "driving_config": driving_config,
        "seed": seed,
        "validation_action_mse": float(np.mean((prediction - vy) ** 2)),
        "runtime": runtime_versions(),
        "epochs": epochs,
        "tracks": tracks,
        "completed_only": completed_only,
        "run_sha256": {p.name: file_digest(p) for p, _ in runs},
        "train_runs": [runs[i][0].name for i in train_ids],
        "validation_runs": [runs[i][0].name for i in val_ids],
        "train_samples": len(x),
        "validation_samples": len(vx),
        "observation_names": OBS_NAMES,
        "warning": "Action prediction error is not evidence of successful autonomous driving.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = {f"w{i}": w for i, w in enumerate(model.coefs_)}
    biases = {f"b{i}": b for i, b in enumerate(model.intercepts_)}
    with output.open("wb") as f:
        np.savez_compressed(
            f,
            mean=scaler.mean_,
            scale=scaler.scale_,
            **weights,
            **biases,
            metadata=json.dumps(report),
        )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class Policy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            self.driving_config = metadata.get("driving_config", {})
            if metadata.get("simulator_version") != SIMULATOR_VERSION:
                raise ValueError(
                    "Model uses an older simulator; retrain on current-version demonstrations"
                )
            self.mean, self.scale = z["mean"].copy(), z["scale"].copy()
            self.weights = [(z[f"w{i}"].copy(), z[f"b{i}"].copy()) for i in range(3)]
        if metadata.get("observation_names") != OBS_NAMES:
            raise ValueError("Model observation ordering is incompatible")
        if self.mean.shape != (len(OBS_NAMES),) or self.scale.shape != self.mean.shape:
            raise ValueError("Invalid model normalization shapes")
        if (
            not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Model normalization must be finite with positive scales")
        incoming = len(OBS_NAMES)
        for w, b in self.weights:
            if w.ndim != 2 or w.shape[0] != incoming or b.shape != (w.shape[1],):
                raise ValueError("Invalid model layer shapes")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("Model weights must be finite")
            incoming = w.shape[1]
        if incoming != 3:
            raise ValueError("Model must output three actions")

    def __call__(self, observation):
        observation = np.asarray(observation)
        if observation.shape != self.mean.shape or not np.isfinite(observation).all():
            raise ValueError("Policy requires one finite observation vector")
        x = (observation - self.mean) / self.scale
        for i, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if i < 2:
                x = np.maximum(x, 0)
        return np.clip(x, [-1, 0, 0], [1, 1, 1]).astype(np.float32)


def reference_driver(env):
    """Privileged geometric controller for smoke tests; not an expert dataset."""
    s, _, _ = env.track.project(env.pos)
    look = 10 + env.speed * 0.65
    target = env.track.sample(s + look) - env.pos
    alpha = np.arctan2(target[1], target[0]) - env.heading
    steer = np.arctan2(2 * WHEELBASE * np.sin(alpha), look) / 0.45
    if env.track.length > 4000:
        preview = env.track.sample(s + np.arange(0, 161, 20))
        directions = np.diff(preview, axis=0)
        angles = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
        curvature = max(0.002, float(np.max(np.abs(np.diff(angles)))) / 20)
        target_speed = min(32.0, np.sqrt(6.0 / curvature))
    else:
        target_speed = 14.0
    return np.array(
        [
            np.clip(steer, -1, 1),
            np.clip((target_speed - env.speed) * 0.5, 0, 1),
            np.clip((env.speed - target_speed) * 0.3, 0, 1),
        ],
        np.float32,
    )
