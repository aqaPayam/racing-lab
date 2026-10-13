"""Frozen one-factor TD3 diagnostics, separate from the SAC--TD3 benchmark."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from .benchmark import BenchmarkConfig, make_manifest, validate_manifest
from .config import RacingTaskConfig, config_hash

CONDITIONS = {
    "baseline": dict(
        reward_profile="common", uniform_warmup_decisions=0, random_training_starts=False
    ),
    "uniform_warmup": dict(
        reward_profile="common", uniform_warmup_decisions=2000, random_training_starts=False
    ),
    "random_starts": dict(
        reward_profile="common", uniform_warmup_decisions=0, random_training_starts=True
    ),
    "cth_reward": dict(
        reward_profile="cth", uniform_warmup_decisions=0, random_training_starts=False
    ),
}


@dataclass(frozen=True)
class DiagnosticConfig(BenchmarkConfig):
    steps: int = 50_000
    seeds: tuple[int, ...] = (5, 6, 7)
    held_out_tracks: tuple[str, ...] = ()
    eval_interval: int = 10_000

    def __post_init__(self):
        super().__post_init__()
        if self.track != "monza" or self.held_out_tracks:
            raise ValueError("This diagnostic uses Monza only; no held-out track tuning")
        if type(self.eval_interval) is not int or self.eval_interval < 1:
            raise ValueError("eval_interval must be positive")


def make_diagnostic_manifest(config):
    from .td3 import TD3Config

    manifest = make_manifest(config)
    manifest.update(
        kind="td3_diagnostic",
        config=asdict(config),
        conditions={label: dict(profile) for label, profile in CONDITIONS.items()},
        algorithms={
            "td3": asdict(
                replace(
                    TD3Config(),
                    eval_interval=config.eval_interval,
                    checkpoint_interval=config.steps,
                )
            )
        },
        sources={"td3": manifest["sources"]["td3"]},
        comparison="one-factor diagnostic: identical learner and evaluation; vary behavior warmup, training reset or reward separately",
        research_question="Does exploration, training coverage or reward explain the first-chicane failure? Short pilot, not a final algorithm ranking.",
        adaptations=dict(
            warmup="Behavior only: uniform actions for 2000 decisions; learning begins after 100 transitions in every condition.",
            resets="Uniform longitudinal centreline distance from numpy.default_rng(seed+episode); zero speed, no lateral jitter or launch assistance.",
            reward="Existing CTH-style Racing Lab reward: lap completion +1, counted collision -1, otherwise max(v_forward/85*cos(heading_error)-abs(lateral_gap_m),0). Same termination as baseline.",
            evaluation="Common reward and standing starts for every condition; every 10000 decisions; final-budget checkpoint only; five paired lateral-jitter Monza starts at final budget.",
            coverage="64 centreline-distance bins of post-step training poses; random spawning improves exposure, not necessarily driving ability.",
            timing="Same 25 Hz and gamma=0.99; original TD3 training frequency and absolute target-speed controller are not restored in this pilot.",
        ),
    )
    manifest["runs"] = []
    labels = tuple(CONDITIONS)
    for index, seed in enumerate(config.seeds):
        order = labels[index % len(labels) :] + labels[: index % len(labels)]
        manifest["runs"].extend(
            dict(algorithm="td3", condition=label, seed=seed, directory=f"{label}-seed{seed}")
            for label in order
        )
    manifest["benchmark_hash"] = config_hash(
        {k: v for k, v in manifest.items() if k != "benchmark_hash"}
    )
    return manifest


@contextmanager
def controller_lock(root):
    """OS lock prevents two live controllers; released even after a process crash."""
    with (root / "controller.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("A diagnostic controller already holds this queue") from error

            def unlock():
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(lock, fcntl.LOCK_UN)

        try:
            yield
        finally:
            lock.seek(0)
            unlock()


def _source_snapshot(root, *, workspace=None):
    workspace = Path(workspace) if workspace is not None else Path(__file__).resolve().parents[2]
    if (workspace / ".git").exists():
        files = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=workspace)
            .decode("utf-8")
            .split("\0")
        )
        scope = "Exact working-tree bytes of tracked files, including original line endings"
    else:
        files = [
            str(path.relative_to(workspace))
            for path in sorted((workspace / "racing_lab").rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        ]
        scope = "Installed package source and assets; no Git checkout required"
    path = root / "source_snapshot.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            if name and (workspace / name).is_file():
                archive.write(workspace / name, name)
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope=scope,
    )


def run_diagnostic(output, config=None, *, continue_queue=False):
    from .training import write_json
    from .diagnostic_report import generate_diagnostic_report

    output = Path(output).resolve()
    if continue_queue:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        values = dict(manifest["config"])
        values["seeds"], values["held_out_tracks"] = (
            tuple(values["seeds"]),
            tuple(values["held_out_tracks"]),
        )
        config = DiagnosticConfig(**values)
    else:
        if output.exists():
            raise ValueError("Output exists; use a fresh root or --continue-queue")
        config = config or DiagnosticConfig()
        manifest = make_diagnostic_manifest(config)
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest)
        write_json(output / "source_snapshot.json", _source_snapshot(output))
    validate_manifest(manifest)
    if manifest.get("kind") != "td3_diagnostic":
        raise ValueError("Not a TD3 diagnostic manifest")
    with controller_lock(output):
        pending, completed = [], []
        for index, spec in enumerate(manifest["runs"]):
            directory = output / spec["directory"]
            if not directory.exists():
                pending.append(index)
                continue
            marker = directory / "worker_result.json"
            if not marker.exists():
                raise ValueError(
                    f"Incomplete run preserved at {directory}; exact resume unsupported"
                )
            worker = json.loads(marker.read_text(encoding="utf-8"))
            if (
                worker["status"] != "completed"
                or worker["benchmark_hash"] != manifest["benchmark_hash"]
            ):
                raise ValueError("Cannot reuse an incompatible run")
            completed.append(spec["directory"])
        active, failures = {}, []
        started, cpu_started = time.perf_counter(), time.process_time()
        started_unix = time.time()
        prevent_sleep = None
        if os.name == "nt":
            import ctypes

            prevent_sleep = ctypes.windll.kernel32.SetThreadExecutionState
            prevent_sleep(0x80000001)
        try:
            while pending or active:
                while pending and len(active) < config.workers and not failures:
                    index = pending.pop(0)
                    env = dict(
                        os.environ,
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "racing_lab.rl.diagnostic",
                            "--worker",
                            str(output / "manifest.json"),
                            "--index",
                            str(index),
                        ],
                        env=env,
                    )
                    active[process.pid] = (process, index)
                for pid, (process, index) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    spec = manifest["runs"][index]
                    marker = output / spec["directory"] / "worker_result.json"
                    if code == 0 and marker.exists():
                        completed.append(spec["directory"])
                    else:
                        failures.append(dict(run=spec["directory"], exit_code=code))
                    del active[pid]
                write_json(
                    output / "queue.json",
                    dict(
                        status="failed"
                        if failures
                        else "running"
                        if pending or active
                        else "completed",
                        benchmark_hash=manifest["benchmark_hash"],
                        controller_pid=os.getpid(),
                        started_unix=started_unix,
                        completed=completed,
                        active=[
                            dict(pid=pid, directory=manifest["runs"][index]["directory"])
                            for pid, (_, index) in active.items()
                        ],
                        pending=[manifest["runs"][index]["directory"] for index in pending],
                        failures=failures,
                        queue_wall_seconds=time.perf_counter() - started,
                        timing_scope="this queue invocation only; fresh worker imports included in queue wall time",
                    ),
                )
                if failures and not active:
                    raise RuntimeError(f"Diagnostic worker failed: {failures}")
                if pending or active:
                    time.sleep(1)
            summary = generate_diagnostic_report(output)
            write_json(
                output / "completion.json",
                dict(
                    status="completed",
                    benchmark_hash=manifest["benchmark_hash"],
                    completed_runs=len(completed),
                    completed_unix=time.time(),
                    queue_and_reporting_wall_seconds=time.perf_counter() - started,
                    controller_cpu_seconds=time.process_time() - cpu_started,
                    report_sha256=hashlib.sha256(
                        (output / "report/report.html").read_bytes()
                    ).hexdigest(),
                    source_snapshot=json.loads(
                        (output / "source_snapshot.json").read_text(encoding="utf-8")
                    ),
                ),
            )
            return summary
        except BaseException as error:
            write_json(
                output / "driver_failure.json",
                dict(error=str(error), traceback=traceback.format_exc(), time_unix=time.time()),
            )
            raise
        finally:
            for process, _ in active.values():
                if process.poll() is None:
                    process.terminate()
                process.wait()
            if prevent_sleep:
                prevent_sleep(0x80000000)


def run_diagnostic_worker(manifest_path, index):
    from .online import evaluate_online, train_online
    from .resources import ResourceMeter
    from .td3 import TD3Agent, TD3Config
    from .training import write_json

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    spec, settings = manifest["runs"][index], manifest["config"]
    profile = manifest["conditions"][spec["condition"]]
    values = dict(manifest["algorithms"]["td3"])
    values["hidden_units"] = tuple(values["hidden_units"])
    task, config = RacingTaskConfig(**manifest["task"]), TD3Config(**values)
    output = manifest_path.parent / spec["directory"]
    meter, started = ResourceMeter(settings["device"]), time.time()
    result = train_online(
        output,
        algorithm="td3",
        track=settings["track"],
        steps=settings["steps"],
        seed=spec["seed"],
        config=config,
        task=task,
        device=settings["device"],
        torch_threads=settings["torch_threads"],
        benchmark_hash=manifest["benchmark_hash"],
        capture_diagnostics=True,
        **profile,
    )
    if result["status"] != "completed":
        raise RuntimeError("Training incomplete")
    checkpoint = output / "final.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, _ = TD3Agent.load(checkpoint, device=settings["device"])
    evaluation = manifest["evaluation"]["varied_start"]
    with meter.measure("post_training_evaluation"):
        report = evaluate_online(
            agent,
            (settings["track"],),
            replace(task, max_laps=evaluation["max_laps"]),
            episodes=evaluation["episodes"],
            seed=evaluation["seed"],
            lateral_jitter=evaluation["lateral_jitter"],
            reward_profile=profile["reward_profile"],
            trajectory_output=output / "trajectories/robustness",
        )
        write_json(output / "evaluation_robustness.json", report)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Evaluation changed checkpoint")
    write_json(
        output / "worker_result.json",
        dict(
            status="completed",
            condition=spec["condition"],
            benchmark_hash=manifest["benchmark_hash"],
            checkpoint_sha256=checkpoint_hash,
            pid=os.getpid(),
            started_unix=started,
            finished_unix=time.time(),
            resources=meter.report(),
            timing_scope="worker after framework imports; includes training and all evaluation",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    run_diagnostic_worker(args.worker, args.index)
