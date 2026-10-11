"""Scratch-only SAC runner, frozen-policy evaluation, and reproducible local artifacts."""

import csv
import hashlib
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
import torch

from ..vehicle import SIMULATOR_VERSION
from .config import (
    ADAPTER_VERSION,
    PAPER_URL,
    SOURCE_REVISION,
    SACConfig,
    RacingTaskConfig,
    config_hash,
)
from .environment import SACRacingEnv, SAC_OBSERVATION_NAMES
from .replay import NStepReplay
from .sac import SACAgent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _versions():
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "simulator": SIMULATOR_VERSION,
    }
    for package in ("torch", "numpy", "gymnasium", "racing-lab"):
        result[package] = version(package)
    source_root = Path(__file__).resolve().parents[2]
    result["code_sha256"] = config_hash(
        {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source_root / "racing_lab").rglob("*.py")
        }
    )
    result["git_commit"] = None
    try:
        if not (source_root / ".git").exists():
            return result
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=source_root
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
                cwd=source_root,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def evaluate_sac(
    agent,
    tracks=("monza",),
    task=None,
    *,
    episodes=1,
    seed=10000,
    render=False,
    view="3d",
    lateral_jitter=0.0,
):
    """Evaluate deterministic mean actions; no updates, expert queries or replay collection."""
    if isinstance(tracks, str) or not tracks or type(episodes) is not int or episodes < 1:
        raise ValueError("Evaluation requires track IDs and a positive episode count")
    task = task or RacingTaskConfig(max_laps=4)
    rows = []
    for track in tracks:
        env = SACRacingEnv(track, task, view=view)
        try:
            for episode in range(episodes):
                observation, info = env.reset(
                    seed=seed + episode, options={"lateral_jitter": lateral_jitter}
                )
                reset_hash = hashlib.sha256(env.env.state().tobytes()).hexdigest()
                total, decisions = 0.0, 0
                epoch = time.perf_counter()
                while True:
                    action = agent.act(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    decisions += 1
                    total += reward
                    interrupted = False
                    if render:
                        import pygame

                        env.render()
                        events = pygame.event.get()
                        interrupted = any(
                            e.type == pygame.QUIT
                            or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                            for e in events
                        )
                        remaining = epoch + info["elapsed"] - time.perf_counter()
                        if remaining > 0:
                            time.sleep(min(remaining, env.env.dt))
                    if terminated or truncated or interrupted:
                        break
                rows.append(
                    dict(
                        track=env.env.track.name,
                        track_sha256=env.env.track.digest,
                        seed=seed + episode,
                        episode=episode,
                        reset_state_sha256=reset_hash,
                        return_sum=total,
                        decisions=decisions,
                        simulated_seconds=info["elapsed"],
                        completed_laps=info["completed_laps"],
                        valid_laps=info["valid_laps"],
                        valid_lap_times=info["valid_lap_times"],
                        best_valid_lap=info["best_valid_lap"],
                        progress_laps=info["progress"],
                        collisions=info["collisions"],
                        offtrack_fraction=info["offtrack_fraction"],
                        termination_reason="interrupted"
                        if interrupted
                        else info["termination_reason"],
                        success=bool(info["valid_laps"] > 0 and not interrupted),
                    )
                )
                if interrupted:
                    break
        finally:
            env.close()
    times = [t for row in rows for t in row["valid_lap_times"]]
    return dict(
        algorithm="SAC-3step",
        deterministic=True,
        reset_variation={"lateral_jitter": lateral_jitter},
        trials="deterministic repetitions" if lateral_jitter == 0 else "varied starts",
        gradient_updates=0,
        task=asdict(task),
        task_hash=config_hash(task),
        rows=rows,
        summary=dict(
            episodes=len(rows),
            successful_episodes=sum(r["success"] for r in rows),
            success_rate=sum(r["success"] for r in rows) / len(rows),
            valid_laps=len(times),
            mean_valid_lap=float(np.mean(times)) if times else None,
            best_valid_lap=min(times) if times else None,
        ),
    )


def train_sac(
    output,
    *,
    track="monza",
    steps=10_000,
    seed=0,
    config=None,
    task=None,
    device="cpu",
    torch_threads=1,
    progress=True,
):
    """Train fresh networks with an empty online buffer. There is no data-loading argument.

    Steps are policy decisions. The default budget is a pilot, not a full-paper run.
    Existing output directories are rejected so experiment artifacts cannot be mixed.
    """
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("steps must be positive and seed must be nonnegative integers")
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    config, task = config or SACConfig(), task or RacingTaskConfig()
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; use a new directory for each training run")
    torch.set_num_threads(torch_threads)
    env = SACRacingEnv(track, task)
    try:
        agent = SACAgent(env.observation_space.shape[0], config=config, seed=seed, device=device)
        replay = NStepReplay(
            config.replay_capacity,
            agent.observation_dim,
            gamma=config.gamma,
            n_step=config.n_step,
            seed=seed,
        )
        observation, _ = env.reset(seed=seed)
        metadata = dict(
            algorithm="SAC-3step",
            adapter_version=ADAPTER_VERSION,
            source_revision=SOURCE_REVISION,
            paper=PAPER_URL,
            data_source="online_self_generated",
            expert_dataset_used=False,
            pretrained_initialization=False,
            replay_initial_size=0,
            config=asdict(config),
            task=asdict(task),
            seed=seed,
            track=env.env.track.name,
            track_sha256=env.env.track.digest,
            observation_names=list(SAC_OBSERVATION_NAMES),
            requested_steps=steps,
            full_paper_budget_completed=False,
            budget_kind="pilot" if steps < 8_000_000 else "step_budget_adaptation",
            device=str(agent.device),
            torch_threads=torch_threads,
            versions=_versions(),
        )
        metadata["configuration_hash"] = config_hash(
            dict(
                config=metadata["config"],
                task=metadata["task"],
                seed=seed,
                track_sha256=metadata["track_sha256"],
                observation_names=metadata["observation_names"],
                requested_steps=steps,
                device=str(agent.device),
                adapter_version=ADAPTER_VERSION,
            )
        )
        output.mkdir(parents=True)
        write_json(output / "run.json", metadata)
        started = time.perf_counter()
        try:
            initial_eval = evaluate_sac(
                agent,
                (track,),
                replace(task, max_laps=4),
                episodes=config.eval_episodes,
                seed=seed + 10000,
            )
        except BaseException as exc:
            write_json(
                output / "result.json",
                dict(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    phase="initial_evaluation",
                    decisions=0,
                    updates=0,
                    expert_dataset_used=False,
                ),
            )
            raise
        write_json(output / "evaluation_initial.json", initial_eval)
        episode, episode_return = 0, 0.0
        stats = {}
        eval_decisions = sum(r["decisions"] for r in initial_eval["rows"])
        collected_steps = 0
        status = "completed"
        with (
            (output / "learning.csv").open("w", newline="", encoding="utf-8") as log,
            (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_log,
        ):
            writer = csv.DictWriter(
                log,
                fieldnames=[
                    "decision",
                    "updates",
                    "replay_size",
                    "elapsed_seconds",
                    "policy_loss",
                    "q_loss",
                    "entropy_loss",
                    "alpha",
                ],
            )
            episodes = csv.DictWriter(
                episode_log,
                fieldnames=[
                    "episode",
                    "decision",
                    "return_sum",
                    "progress_laps",
                    "valid_laps",
                    "best_valid_lap",
                    "reason",
                ],
            )
            writer.writeheader()
            episodes.writeheader()
            decision = 0
            try:
                for decision in range(1, steps + 1):
                    collected = decision - 1
                    action = (
                        env.action_space.sample()
                        if collected < config.start_steps
                        else agent.act(observation)
                    )
                    # Match source: learn from already collected data, before appending this step.
                    if collected >= config.start_steps and len(replay) >= config.batch_size:
                        stats = agent.update(replay.sample(config.batch_size))
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    replay.append(
                        observation, action, reward, next_observation, terminated, truncated
                    )
                    collected_steps += 1
                    observation = next_observation
                    episode_return += reward
                    if terminated or truncated:
                        episodes.writerow(
                            dict(
                                episode=episode,
                                decision=decision,
                                return_sum=episode_return,
                                progress_laps=info["progress"],
                                valid_laps=info["valid_laps"],
                                best_valid_lap=info["best_valid_lap"],
                                reason=info["termination_reason"],
                            )
                        )
                        episode_log.flush()
                        episode += 1
                        episode_return = 0.0
                        observation, _ = env.reset(seed=seed + episode)
                    if decision % 100 == 0 or decision == steps:
                        writer.writerow(
                            dict(
                                decision=decision,
                                updates=agent.updates,
                                replay_size=len(replay),
                                elapsed_seconds=time.perf_counter() - started,
                                **stats,
                            )
                        )
                        log.flush()
                    if progress and (decision % 1000 == 0 or decision == steps):
                        print(
                            f"SAC {decision}/{steps} decisions | {agent.updates} updates | "
                            f"{time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                    if config.checkpoint_interval and decision % config.checkpoint_interval == 0:
                        agent.save(
                            output / f"step_{decision:09d}.pt", dict(metadata, decisions=decision)
                        )
                    if config.eval_interval and decision % config.eval_interval == 0:
                        report = evaluate_sac(
                            agent,
                            (track,),
                            replace(task, max_laps=4),
                            episodes=config.eval_episodes,
                            seed=seed + 10000,
                        )
                        eval_decisions += sum(row["decisions"] for row in report["rows"])
                        write_json(output / f"evaluation_{decision:09d}.json", report)
            except KeyboardInterrupt:
                status = "interrupted"
            except Exception:
                status = "failed"
                raise
            finally:
                decision = collected_steps
                replay.flush()
                result = dict(
                    status="training_completed" if status == "completed" else status,
                    phase="final_evaluation_pending" if status == "completed" else "training",
                    decisions=decision,
                    updates=agent.updates,
                    replay_size=len(replay),
                    episodes_finished=episode,
                    simulated_seconds=decision / task.decision_hz,
                    physics_substeps=decision * int(np.ceil(env.env.dt / (1 / 240))),
                    random_action_decisions=min(decision, config.start_steps),
                    evaluation_decisions=eval_decisions,
                    wall_seconds=time.perf_counter() - started,
                    last_update=stats,
                    expert_dataset_used=False,
                )
                agent.save(output / "final.pt", dict(metadata, result=result))
                write_json(output / "result.json", result)
        if status == "completed":
            try:
                final_eval = evaluate_sac(
                    agent,
                    (track,),
                    replace(task, max_laps=4),
                    episodes=config.eval_episodes,
                    seed=seed + 10000,
                )
            except BaseException as exc:
                result.update(
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    phase="final_evaluation",
                )
                write_json(output / "result.json", result)
                agent.save(output / "final.pt", dict(metadata, result=result))
                raise
            result.update(
                status="completed", phase="complete", wall_seconds=time.perf_counter() - started
            )
            write_json(output / "evaluation_final.json", final_eval)
            result["evaluation_decisions"] += sum(r["decisions"] for r in final_eval["rows"])
            result["evaluation_summary"] = final_eval["summary"]
            write_json(output / "result.json", result)
            agent.save(output / "final.pt", dict(metadata, result=result))
        return result
    finally:
        env.close()
