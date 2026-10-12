"""Passive, frozen-policy lap references and human practice comparisons.

PyTorch is imported only when preparing a SAC reference. No training is performed.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import math
import numpy as np

from .vehicle import SIMULATOR_VERSION


class AgentGhost:
    """One clean standing-start lap, replayed by simulation time without physics."""

    def __init__(self, states, times, progress, actions, identity):
        self.states = np.array(states, dtype=float, copy=True)
        self.times = np.array(times, dtype=float, copy=True)
        self.progress = np.array(progress, dtype=float, copy=True)
        self.actions = np.array(actions, dtype=float, copy=True)
        self.identity = dict(identity)
        n = len(self.times)
        if (
            n < 2
            or self.states.shape != (n, 7)
            or self.progress.shape != (n,)
            or self.actions.shape != (n - 1, 3)
            or not all(
                np.isfinite(a).all() for a in (self.states, self.times, self.progress, self.actions)
            )
            or self.times[0] != 0
            or np.any(np.diff(self.times) <= 0)
            or not np.allclose(self.times, np.arange(n) * identity["dt"])
            or np.any(self.actions < [-1, 0, 0])
            or np.any(self.actions > 1)
            or identity["simulator_version"] != SIMULATOR_VERSION
            or identity["selection"] != "first_clean_standing_start_lap"
        ):
            raise ValueError("Invalid or incompatible SAC ghost samples")
        for array in (self.states, self.times, self.progress, self.actions):
            array.flags.writeable = False
        self.best_time = float(self.times[-1])
        self.label = "SAC / standing start"
        maximum = np.maximum.accumulate(self.progress)
        self._first_progress, indices = np.unique(maximum, return_index=True)
        self._first_times = self.times[indices]

    def ghost(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            return None
        if elapsed >= self.best_time:
            return self.states[-1].copy()
        index = min(len(self.times) - 2, max(0, int(np.searchsorted(self.times, elapsed) - 1)))
        fraction = (elapsed - self.times[index]) / (self.times[index + 1] - self.times[index])
        a, b = self.states[index : index + 2]
        pose = a + (b - a) * fraction
        pose[2] = a[2] + ((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi) * fraction
        return pose

    def delta(self, progress, elapsed):
        """Human minus SAC first-passage time at equal progress; never extrapolate."""
        if not all(math.isfinite(v) for v in (progress, elapsed)):
            return None
        if not self._first_progress[0] <= progress <= self._first_progress[-1]:
            return None
        return float(elapsed - np.interp(progress, self._first_progress, self._first_times))

    def validate_player(self, env):
        if (
            env.track.digest != self.identity["track_sha256"]
            or env.dt != self.identity["dt"]
            or asdict(env.driving) != self.identity["driving_config"]
            or env.steps != 0
            or not np.array_equal(env.state(), self.states[0])
        ):
            raise ValueError("SAC ghost requires the same track, timestep, aids and standing start")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                states=self.states,
                times=self.times,
                progress=self.progress,
                actions=self.actions,
                identity=json.dumps(self.identity, sort_keys=True),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path, identity):
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(data["identity"]))
            if stored != identity:
                raise ValueError("SAC ghost cache belongs to a different checkpoint or task")
            return cls(data["states"], data["times"], data["progress"], data["actions"], stored)


def prepare_sac_ghost(model, track="monza", directory="results/sac-ghost/cache"):
    """Cache the first deterministic, strictly clean lap; never select a flying best."""
    import torch
    from .rl.config import RacingTaskConfig
    from .rl.environment import SACRacingEnv, SAC_OBSERVATION_NAMES
    from .rl.sac import SACAgent

    torch.set_num_threads(1)
    model = Path(model)
    checkpoint_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    agent, metadata = SACAgent.load(model, device="cpu")
    if (
        metadata["versions"]["simulator"] != SIMULATOR_VERSION
        or metadata["expert_dataset_used"]
        or metadata["pretrained_initialization"]
        or metadata["data_source"] != "online_self_generated"
        or metadata["observation_names"] != list(SAC_OBSERVATION_NAMES)
    ):
        raise ValueError("Ghost requires a compatible, online-only SAC checkpoint")
    task = replace(RacingTaskConfig(**metadata["task"]), max_laps=1)
    env = SACRacingEnv(track, task)
    try:
        if env.env.track.digest != metadata["track_sha256"]:
            raise ValueError("Select the checkpoint's training layout for this ghost comparison")
        observation, _ = env.reset(seed=10000)
        source = Path(__file__).parent
        physics_files = [
            "env.py",
            "vehicle_model.py",
            "dynamics.py",
            "collisions.py",
            "track.py",
            "rl/environment.py",
            "rl/sac.py",
            "rl/config.py",
        ]
        physics_hash = hashlib.sha256(
            b"".join((source / p).read_bytes() for p in physics_files)
        ).hexdigest()
        identity = dict(
            format_version=1,
            algorithm="SAC-3step",
            checkpoint_sha256=checkpoint_hash,
            track_sha256=env.env.track.digest,
            track_argument=track,
            simulator_version=SIMULATOR_VERSION,
            physics_code_sha256=physics_hash,
            dt=env.env.dt,
            driving_config=asdict(env.env.driving),
            seed=10000,
            initial_state=env.env.state().tolist(),
            selection="first_clean_standing_start_lap",
            task=asdict(task),
            expert_dataset_used=False,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
        )
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        path = Path(directory) / f"{key}.npz"
        if path.exists():
            reference = AgentGhost.load(path, identity)
            print(f"Loaded SAC ghost: {reference.best_time:.2f}s standing-start lap", flush=True)
            return reference
        print("Preparing frozen SAC ghost (first standing-start lap; no training)...", flush=True)
        states, times, progress, actions = [env.env.state()], [0.0], [0.0], []
        while True:
            observation, _, terminated, truncated, info = env.step(
                agent.act(observation, deterministic=True)
            )
            states.append(env.env.state())
            times.append(info["elapsed"])
            progress.append(env.env.progress)
            actions.append(env.env.action.copy())
            if terminated or truncated:
                break
        if info["completed_laps"] != 1 or info["valid_laps"] != 1:
            raise ValueError(
                f"SAC's first standing-start lap was not clean ({info['termination_reason']}); no ghost created"
            )
        reference = AgentGhost(states, times, progress, actions, identity)
        reference.save(path)
        print(f"SAC ghost ready: {reference.best_time:.2f}s; cached at {path}", flush=True)
        return reference
    finally:
        env.close()


class GhostRaceStart:
    """Countdown that advances neither car; both share the player's simulation clock."""

    def __init__(self):
        self.phase = "ready"
        self.remaining = 3.0

    def start(self):
        self.phase = "countdown"
        self.remaining = 3.0

    def advance(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Countdown elapsed time must be finite and nonnegative")
        if self.phase != "countdown" or elapsed > 0.5:
            return False
        self.remaining = max(0.0, self.remaining - elapsed)
        if self.remaining <= 1e-9:
            self.phase = "racing"
            return True
        return False


def strict_clean(env):
    return not (env.collisions or env.offtrack_steps or env.recoveries)


def comparison_report(recorder, reference, recording_path, reason):
    """Practice result; failed laps have no invented lap time or finish delta."""
    env = recorder.env
    info = env.info()
    clean = bool(info["finished"] and strict_clean(env))
    elapsed = info["elapsed"]
    human_time = elapsed if clean else None
    progress = np.r_[0, np.cumsum(np.asarray(recorder.features)[:, 0])]
    return dict(
        first="You (orange)",
        second="Frozen SAC standing-start lap (teal)",
        recording=str(recording_path),
        reference=reference.identity,
        lap_comparison=dict(
            human_valid=clean,
            human_lap_seconds=human_time,
            human_elapsed_seconds=elapsed,
            sac_valid=True,
            sac_lap_seconds=reference.best_time,
            human_minus_sac_seconds=human_time - reference.best_time if clean else None,
            reason=reason,
            collisions=env.collisions,
            offtrack_steps=env.offtrack_steps,
            recoveries=env.recoveries,
            human_control_settings=recorder.control_settings,
            description="One standing-start practice lap; human input shaping differs from SAC controls. No training or independent-trial claim.",
        ),
        runs=[
            dict(
                progress_m=progress.tolist(),
                time_s=(np.arange(len(recorder.states)) * env.dt).tolist(),
                speed_kmh=(np.asarray(recorder.states)[:, 3] * 3.6).tolist(),
                brake=np.r_[0, np.asarray(recorder.actions)[:, 2]].tolist(),
                trajectory_xy=np.asarray(recorder.states)[:, :2].tolist(),
                simulator_version=SIMULATOR_VERSION,
            ),
            dict(
                progress_m=reference.progress.tolist(),
                time_s=reference.times.tolist(),
                speed_kmh=(reference.states[:, 3] * 3.6).tolist(),
                brake=np.r_[0, reference.actions[:, 2]].tolist(),
                trajectory_xy=reference.states[:, :2].tolist(),
                simulator_version=SIMULATOR_VERSION,
            ),
        ],
    )
