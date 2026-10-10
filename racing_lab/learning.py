"""Behavior cloning baseline. Split by entire run, never adjacent frames."""
import json
from pathlib import Path
import numpy as np
from .data import load
from .env import OBS_NAMES


def train(directory, output, seed=0, epochs=100):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    runs = []
    for path in sorted(Path(directory).glob("*.npz")):
        run = load(path)
        if run["metadata"]["label"] == "good" and run["metadata"]["source"] == "human":
            if run["metadata"]["observation_names"] != OBS_NAMES:
                raise ValueError(f"Incompatible observations: {path}")
            runs.append((path, run))
    if len(runs)<2:
        raise ValueError("Record at least two human runs labeled good (five or more recommended).")
    order = np.random.default_rng(seed).permutation(len(runs))
    n_val = max(1, round(len(runs)*.2))
    val_ids, train_ids = order[:n_val], order[n_val:]
    def arrays(ids):
        return (np.concatenate([runs[i][1]["observations"][:-1] for i in ids]),
                np.concatenate([runs[i][1]["actions"] for i in ids]))
    x,y = arrays(train_ids)
    vx,vy = arrays(val_ids)
    scaler = StandardScaler().fit(x)
    model = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", max_iter=epochs,
                         random_state=seed, batch_size=min(256,len(x)), learning_rate_init=.001)
    model.fit(scaler.transform(x),y)
    prediction = np.clip(model.predict(scaler.transform(vx)), [-1,0,0],[1,1,1])
    report = {"seed":seed, "validation_action_mse":float(np.mean((prediction-vy)**2)),
              "train_runs":[runs[i][0].name for i in train_ids], "validation_runs":[runs[i][0].name for i in val_ids],
              "train_samples":len(x), "validation_samples":len(vx), "observation_names":OBS_NAMES,
              "warning":"Action prediction error is not evidence of successful autonomous driving."}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    weights = {f"w{i}":w for i,w in enumerate(model.coefs_)}
    biases = {f"b{i}":b for i,b in enumerate(model.intercepts_)}
    with output.open("wb") as f:
        np.savez_compressed(f, mean=scaler.mean_, scale=scaler.scale_, **weights, **biases, metadata=json.dumps(report))
    output.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


class Policy:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.mean,self.scale = z["mean"].copy(),z["scale"].copy()
            self.weights = [(z[f"w{i}"].copy(),z[f"b{i}"].copy()) for i in range(3)]

    def __call__(self, observation):
        x = (observation-self.mean)/self.scale
        for i,(w,b) in enumerate(self.weights):
            x=x@w+b
            if i<2:
                x=np.maximum(x,0)
        return np.clip(x,[-1,0,0],[1,1,1]).astype(np.float32)


def reference_driver(env):
    """Privileged geometric controller for smoke tests; not an expert dataset."""
    s, _, _ = env.track.project(env.pos)
    look = 10+env.speed*.65
    target = env.track.sample(s+look)-env.pos
    alpha = np.arctan2(target[1],target[0])-env.heading
    steer = np.arctan2(2*3.6*np.sin(alpha),look)/.45
    target_speed = 14.
    return np.array([np.clip(steer,-1,1),np.clip((target_speed-env.speed)*.5,0,1),
                     np.clip((env.speed-target_speed)*.3,0,1)],np.float32)
