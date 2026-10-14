"""Standalone Python 3.9 bridge to the pinned authors' Assetto Corsa environment.

This entry point evaluates a released policy; it never creates a replay buffer,
downloads demonstrations, or takes optimizer steps. Run with --help for usage.
"""

from pathlib import Path
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
AUTHOR_COMMIT = "ef0fa502bdd257fc2affad9361c02b3dbdfe1632"
POLICY_SHA256 = "3f30cbb1b88ea221bf686c28b74d531ababe843da2e9529dc89a8a47aa157e84"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def management(command):
    """Use the local authors' management protocol without executing reply text."""
    with socket.create_connection(("127.0.0.1", 2347), timeout=3) as connection:
        connection.sendall(command.encode())
        connection.settimeout(5)
        message = connection.recv(2**20).decode()
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return ast.literal_eval(message)


def load_config(author):
    from omegaconf import OmegaConf

    config = OmegaConf.load(author / "config.yml")
    config.disable_wandb = True
    config.load_offline_data = False
    config.pre_train = False
    config.Agent.use_offline_buffer = False
    config.AssettoCorsa.track = "monza"
    config.AssettoCorsa.car = "bmw_z4_gt3"
    config.AssettoCorsa.remote_machine_ip = "127.0.0.1"
    config.AssettoCorsa.screen_capture_enable = False
    return config


def author_imports(author):
    revision = subprocess.check_output(
        ["git", "-C", str(author), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AUTHOR_COMMIT:
        raise ValueError("Author source revision differs from this verified integration")
    changes = subprocess.check_output(
        ["git", "-C", str(author), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if changes:
        raise ValueError("Tracked author source has local changes; preserve or audit them first")
    sys.path.insert(0, str(author / "assetto_corsa_gym"))
    sys.path.insert(0, str(author / "algorithm/discor"))


@contextmanager
def racing_session(author_root, output):
    """Importable legacy Gym session for one's own numerical policy, with cleanup.

    Call reset() -> observation, step(action) -> observation, reward, done, info.
    Actions are the authors' three relative controls in [-1, 1], at 25 Hz.
    The game must already be in Drive on Monza with the BMW Z4 GT3.
    """
    author = Path(author_root)
    author_imports(author)
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(author)
    env = make_ac_env(config, str(output))
    try:
        yield env
    finally:
        try:
            env.close()
        finally:
            env.client.controls.local_controls.close()


def load_policy(author, checkpoint):
    import torch
    from discor.network import GaussianPolicy

    if sha256(checkpoint / "policy_net.pth") != POLICY_SHA256:
        raise ValueError("Checkpoint differs from the verified released Monza GT3 policy")
    weights = torch.load(checkpoint / "policy_net.pth", map_location="cpu")
    observation_size = weights["net.0.weight"].shape[1]
    action_size = weights["net.6.weight"].shape[0] // 2
    config = load_config(author)
    policy = GaussianPolicy(observation_size, action_size, list(config.SAC.policy_hidden_units))
    policy.load_state_dict(weights, strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy, observation_size, action_size


def check_policy(args):
    import numpy as np
    import torch
    from discor.algorithm import SAC

    policy, obs_size, action_size = load_policy(args.author, args.checkpoint)
    config = load_config(args.author)
    from omegaconf import OmegaConf

    algorithm = SAC(
        obs_size,
        action_size,
        torch.device("cpu"),
        seed=config.seed,
        **OmegaConf.to_container(config.SAC),
    )
    algorithm._policy_net.load_state_dict(policy.state_dict())
    observations = np.random.default_rng(20261014).normal(size=(100, obs_size)).astype(np.float32)
    errors = []
    for observation in observations:
        with torch.no_grad():
            output = policy(torch.tensor(observation[None]))[2].numpy()[0]
        reference, _ = algorithm.exploit(observation)
        errors.append(float(np.max(np.abs(output - reference))))
    if max(errors) != 0:
        raise AssertionError("Deterministic actions differ from the author SAC exploit path")
    # The native environment must produce the checkpoint's exact observation format.
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    environment = make_ac_env(config, work_dir=str(args.output))
    if environment.observation_space.shape != (obs_size,):
        raise ValueError("Checkpoint and native observation shapes differ")
    environment.client.controls.local_controls.close()
    result = {
        "author_commit": AUTHOR_COMMIT,
        "observation_size": obs_size,
        "action_size": action_size,
        "exact_author_action_checks": len(errors),
        "maximum_action_error": max(errors),
        "training_decisions": 0,
        "optimizer_steps": 0,
        "human_data_loaded": False,
        "policy_sha256": POLICY_SHA256,
        "inference_device": "cpu",
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "gym", "pandas", "numba", "scipy")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def capture(args):
    """Passive numerical telemetry: no reset and no joystick commands."""
    rows = []
    start = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(5)
        connection.sendto(b"connect", ("127.0.0.1", 2345))
        try:
            while time.perf_counter() - start < args.seconds:
                data, peer = connection.recvfrom(2**18)
                if peer[0] != "127.0.0.1" or peer[1] != 2345:
                    raise ValueError("Unexpected telemetry sender")
                message = data.decode()
                if message == "identified":
                    continue
                if message == "disconnect":
                    break
                state = json.loads(message)
                rows.append(state)
        finally:
            connection.sendto(b"disconnect", ("127.0.0.1", 2345))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "telemetry.json").write_text(json.dumps(rows))
    result = {
        "samples": len(rows),
        "wall_seconds": time.perf_counter() - start,
        "observed_telemetry_hz": (len(rows) - 1)
        / (rows[-1]["timestamp_ac"] - rows[0]["timestamp_ac"])
        if len(rows) > 1 and rows[-1]["timestamp_ac"] > rows[0]["timestamp_ac"]
        else None,
        "channels": sorted(rows[-1]) if rows else [],
        "control_actions_sent": 0,
        "training_decisions": 0,
        "human_data_used_for_training": False,
    }
    (args.output / "telemetry_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def evaluate(args):
    import numpy as np
    import torch
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from omegaconf import OmegaConf

    static = management("get_static_info")
    if static["TrackName"] != "monza" or static["CarName"] != "bmw_z4_gt3":
        raise ValueError("Start a BMW Z4 GT3 session on Monza first")
    policy, observation_size, _ = load_policy(args.author, args.checkpoint)
    config = load_config(args.author)
    args.output.mkdir(parents=True, exist_ok=False)
    config.work_dir = str(args.output)
    (args.output / "config.yaml").write_text(OmegaConf.to_yaml(config))
    env = make_ac_env(config, str(args.output))
    env.set_eval_mode()
    decisions = 0
    start = time.perf_counter()
    status = "initializing"
    laps = []
    previous_lap = None
    lap_dirty = False
    # Bound the author's infinite socket-retry loops if the game closes or pauses.
    import _thread

    watchdog = threading.Timer(args.seconds + 30, _thread.interrupt_main)
    watchdog.daemon = True
    watchdog.start()
    try:
        observation = env.reset()
        if observation.shape != (observation_size,) or not np.all(np.isfinite(observation)):
            raise ValueError("Invalid native observation")
        status = "running"
        while time.perf_counter() - start < args.seconds:
            with torch.no_grad():
                action = policy(torch.tensor(observation[None].copy(), dtype=torch.float32))[
                    2
                ].numpy()[0]
            observation, reward, done, info = env.step(action)
            decisions += 1
            state = env.states[-1]
            current_lap = state["LapCount"]
            if previous_lap is not None and current_lap != previous_lap:
                laps.append(
                    {
                        "lap_count": previous_lap,
                        "seconds": state["iLastTime"] / 1000,
                        "observed_invalid": lap_dirty,
                        "full_lap_observed": len(laps) > 0,
                    }
                )
                print("Lap: " + json.dumps(laps[-1]), flush=True)
                lap_dirty = False
            lap_dirty |= (
                bool(state.get("LapInvalidated", 0)) or state.get("numberOfTyresOut", 0) > 2
            )
            previous_lap = current_lap
            if decisions % 250 == 0:
                print(
                    "Driving: "
                    + json.dumps(
                        {
                            "decisions": decisions,
                            "speed_kmh": round(state["speed"] * 3.6, 1),
                            "lap_count": current_lap,
                        }
                    ),
                    flush=True,
                )
            if done:
                status = "episode_finished"
                break
        else:
            status = "time_limit"
    except BaseException:
        status = "interrupted_or_failed"
        raise
    finally:
        watchdog.cancel()
        cleanup_error = None
        stats = {}
        try:
            stats = env.close()
        except Exception as error:
            cleanup_error = repr(error)
        finally:
            env.client.controls.local_controls.close()
        result = {
            "status": status,
            "author_commit": AUTHOR_COMMIT,
            "policy_sha256": POLICY_SHA256,
            "pretrained_reference": True,
            "human_data_loaded": False,
            "training_decisions": 0,
            "optimizer_steps": 0,
            "evaluation_decisions": decisions,
            "wall_seconds": time.perf_counter() - start,
            "inference_device": "cpu",
            "laps": laps,
            "author_episode_statistics": stats,
            "cleanup_error": cleanup_error,
            "note": "Initial partial out-lap excluded; native observed validity, not our custom-game envelope rule.",
        }
        (args.output / "evaluation.json").write_text(
            json.dumps(result, indent=2, default=lambda x: x.item())
        )
        print(json.dumps(result, indent=2, default=lambda x: x.item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-policy", "capture", "evaluate"))
    parser.add_argument("--author", type=Path, default=ROOT / ".tmp/assetto-corsa-author-20261014")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/assetto-corsa/monza-gt3-author-step05400000",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 5 if args.mode == "capture" else 600
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.output is None:
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = ROOT / "results/assetto-corsa" / (args.mode + "-" + suffix)
    author_imports(args.author)
    import torch

    torch.set_num_threads(1)
    {"check-policy": check_policy, "capture": capture, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
