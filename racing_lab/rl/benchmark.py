"""Frozen online benchmark manifests, isolated workers and complete run accounting."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from importlib.metadata import version
import subprocess
import sys
import time

from .config import RacingTaskConfig, SACConfig, config_hash
from .resources import hardware
from ..track import Track, names


@dataclass(frozen=True)
class BenchmarkConfig:
    track: str = "monza"
    steps: int = 200_000
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    workers: int = 1
    evaluation_jitter: float = 1.0
    varied_start_episodes: int = 5
    held_out_tracks: tuple[str, ...] = ("silverstone", "spa", "suzuka")
    device: str = "cpu"
    torch_threads: int = 1

    def __post_init__(self):
        if self.track not in names():
            raise ValueError("Unknown training track")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be positive")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(v) is not int or v < 0 for v in self.seeds)
        ):
            raise ValueError("Training seeds must be distinct nonnegative integers")
        if type(self.workers) is not int or not 1 <= self.workers <= len(self.seeds) * 2:
            raise ValueError("Invalid worker count")
        if self.device not in ("cpu", "cuda") or (self.device == "cuda" and self.workers != 1):
            raise ValueError("Use CPU workers or one CUDA worker")
        if type(self.torch_threads) is not int or self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        if type(self.varied_start_episodes) is not int or self.varied_start_episodes < 1:
            raise ValueError("varied_start_episodes must be positive")
        if not 0 < self.evaluation_jitter <= 1:
            raise ValueError("Evaluation jitter must be in (0,1] metres")
        family = self.track.removesuffix("_long")
        if len(set(self.held_out_tracks)) != len(self.held_out_tracks) or any(
            track not in names() or track.removesuffix("_long") == family
            for track in self.held_out_tracks
        ):
            raise ValueError("Held-out tracks must belong to different circuit families")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((root / "racing_lab").rglob("*.py"))
    }


def runtime_versions():
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        **{name: version(name) for name in ("torch", "numpy", "gymnasium", "racing-lab")},
    )


def make_manifest(config):
    from .td3 import TD3Config, TD3_PAPER_URL, TD3_SOURCE_REVISION
    from .config import PAPER_URL, SOURCE_REVISION

    runs = []
    for index, seed in enumerate(config.seeds):
        # Alternate order within each pair; worker counts/hardware remain recorded.
        for algorithm in ("sac", "td3") if index % 2 == 0 else ("td3", "sac"):
            runs.append(dict(algorithm=algorithm, seed=seed, directory=f"{algorithm}-seed{seed}"))
    task = RacingTaskConfig()
    manifest = dict(
        schema_version=1,
        config=asdict(config),
        task=asdict(task),
        algorithms={"sac": asdict(SACConfig()), "td3": asdict(TD3Config())},
        sources={
            "sac": dict(paper=PAPER_URL, revision=SOURCE_REVISION),
            "td3": dict(paper=TD3_PAPER_URL, revision=TD3_SOURCE_REVISION),
        },
        code_files_sha256=source_hashes(),
        hardware=hardware(),
        runtime_versions=runtime_versions(),
        runs=runs,
        tracks={track: Track(track).digest for track in (config.track, *config.held_out_tracks)},
        evaluation=dict(
            fixed_start=dict(seed=10000, episodes=1, max_laps=4),
            varied_start=dict(
                seed=20000,
                episodes=config.varied_start_episodes,
                max_laps=1,
                lateral_jitter=config.evaluation_jitter,
            ),
            held_out=dict(seed=30000, episodes=1, max_laps=1, lateral_jitter=0),
            checkpoint_selection="final budget, never best evaluation checkpoint",
            held_out_policy="zero-shot only; never used to tune or select a checkpoint",
            validity="all gates; zero off-road steps, counted collisions or recovery",
        ),
        comparison="shared observations/history, controls, reward, reset, timing, termination and aids",
        budget_meaning="equal collected training transitions, not equal optimizer work or compute",
        assistance=dict(expert_data=False, pretrained_policy=False, launch_controller=False),
        timing_condition="isolated sequential workers"
        if config.workers == 1
        else "concurrent CPU workers; wall time includes contention and is not isolated algorithm speed",
        resource_limits=dict(
            torch_threads=config.torch_threads,
            workers=config.workers,
            blas_threads=1,
            rendering=False,
        ),
        full_original_paper_experiments_reproduced=False,
    )
    manifest["benchmark_hash"] = config_hash(manifest)
    return manifest


def validate_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "benchmark_hash"}
    if config_hash(payload) != manifest["benchmark_hash"]:
        raise ValueError("Benchmark manifest hash mismatch")
    if source_hashes() != manifest["code_files_sha256"]:
        raise ValueError("Code changed after protocol freeze; create a new benchmark")
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("Runtime changed after protocol freeze")
    for track, expected in manifest["tracks"].items():
        if Track(track).digest != expected:
            raise ValueError("Track changed after protocol freeze")


def run_benchmark(output, config=None, *, continue_queue=False):
    """Launch each run in a fresh process. Continuing skips completed runs only.

    Interrupted training is never resumed with a missing replay/RNG state. A failed
    directory is preserved, and the queue requires a new output root to retry it.
    """
    from .training import write_json

    output = Path(output).resolve()
    config = config or BenchmarkConfig()
    if continue_queue:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
        loaded = dict(manifest["config"])
        loaded["seeds"], loaded["held_out_tracks"] = (
            tuple(loaded["seeds"]),
            tuple(loaded["held_out_tracks"]),
        )
        config = BenchmarkConfig(**loaded)
    else:
        if output.exists():
            raise ValueError("Benchmark output exists; choose a new directory or --continue-queue")
        manifest = make_manifest(config)
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest)
    validate_manifest(manifest)
    pending, completed = [], []
    for index, run in enumerate(manifest["runs"]):
        directory = output / run["directory"]
        if directory.exists():
            marker = directory / "worker_result.json"
            if not marker.exists():
                raise ValueError(
                    f"Incomplete run preserved at {directory}; exact training resume is unsupported"
                )
            result = json.loads(marker.read_text(encoding="utf-8"))
            if (
                result.get("status") != "completed"
                or result.get("benchmark_hash") != manifest["benchmark_hash"]
            ):
                raise ValueError("Cannot reuse an incomplete or incompatible run")
            completed.append(run["directory"])
        else:
            pending.append(index)
    started, active, failures = time.perf_counter(), {}, []
    status_path = output / "queue.json"
    try:
        while pending or active:
            while pending and len(active) < config.workers and not failures:
                index = pending.pop(0)
                env = dict(
                    os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "racing_lab.rl.benchmark",
                    "--worker",
                    str(output / "manifest.json"),
                    "--index",
                    str(index),
                ]
                process = subprocess.Popen(command, env=env)
                active[process.pid] = (process, index)
            for pid, (process, index) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                run = manifest["runs"][index]
                marker = output / run["directory"] / "worker_result.json"
                if code == 0 and marker.exists():
                    completed.append(run["directory"])
                else:
                    failures.append(dict(run=run["directory"], exit_code=code))
                del active[pid]
            write_json(
                status_path,
                dict(
                    status="failed"
                    if failures
                    else "running"
                    if pending or active
                    else "completed",
                    benchmark_hash=manifest["benchmark_hash"],
                    completed=completed,
                    active=[manifest["runs"][index]["directory"] for _, index in active.values()],
                    pending=[manifest["runs"][index]["directory"] for index in pending],
                    failures=failures,
                    queue_wall_seconds=time.perf_counter() - started,
                    timing_scope="this invocation, excludes any earlier queue invocations",
                ),
            )
            if failures and not active:
                raise RuntimeError(f"Benchmark worker failed: {failures}")
            if pending or active:
                time.sleep(1)
        from .benchmark_report import generate_report

        return generate_report(output)
    finally:
        # Interrupt only children launched by this queue, never unrelated game sessions.
        for process, _ in active.values():
            process.terminate()
            process.wait()


def run_worker(manifest_path, index):
    from .online import evaluate_online, train_online
    from .resources import ResourceMeter
    from .sac import SACAgent
    from .td3 import TD3Agent, TD3Config
    from .training import write_json

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    spec = manifest["runs"][index]
    root, settings = manifest_path.parent, manifest["config"]
    output = root / spec["directory"]
    meter = ResourceMeter(settings["device"])
    worker_started = time.time()
    config_values = dict(manifest["algorithms"][spec["algorithm"]])
    config_values["hidden_units"] = tuple(config_values["hidden_units"])
    config = (SACConfig if spec["algorithm"] == "sac" else TD3Config)(**config_values)
    task = RacingTaskConfig(**manifest["task"])
    result = train_online(
        output,
        algorithm=spec["algorithm"],
        track=settings["track"],
        steps=settings["steps"],
        seed=spec["seed"],
        config=config,
        task=task,
        device=settings["device"],
        torch_threads=settings["torch_threads"],
        benchmark_hash=manifest["benchmark_hash"],
    )
    if result["status"] != "completed":
        raise RuntimeError("Training did not complete")
    agent, _ = (SACAgent if spec["algorithm"] == "sac" else TD3Agent).load(
        output / "final.pt", device=settings["device"]
    )
    evaluation = manifest["evaluation"]
    checkpoint_before = hashlib.sha256((output / "final.pt").read_bytes()).hexdigest()
    with meter.measure("post_training_evaluation"):
        for label, tracks, key in (
            ("robustness", (settings["track"],), "varied_start"),
            ("held_out", tuple(settings["held_out_tracks"]), "held_out"),
        ):
            if not tracks:
                continue
            condition = evaluation[key]
            report = evaluate_online(
                agent,
                tracks,
                replace(task, max_laps=condition["max_laps"]),
                episodes=condition["episodes"],
                seed=condition["seed"],
                lateral_jitter=condition["lateral_jitter"],
                trajectory_output=output / "trajectories" / label,
            )
            write_json(output / f"evaluation_{label}.json", report)
    if hashlib.sha256((output / "final.pt").read_bytes()).hexdigest() != checkpoint_before:
        raise RuntimeError("Evaluation changed the frozen checkpoint")
    write_json(
        output / "worker_result.json",
        dict(
            status="completed",
            benchmark_hash=manifest["benchmark_hash"],
            pid=os.getpid(),
            started_unix=worker_started,
            finished_unix=time.time(),
            resources=meter.report(),
            checkpoint_sha256=checkpoint_before,
            timing_scope="fresh worker after framework imports; includes training and all evaluations",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.worker, args.index)
