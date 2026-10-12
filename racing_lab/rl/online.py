"""Matched-task online trainers with explicit interaction and compute accounting."""

import csv
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .config import (
    ADAPTER_VERSION,
    PAPER_URL,
    SOURCE_REVISION,
    RacingTaskConfig,
    SACConfig,
    config_hash,
)
from .environment import SACRacingEnv, SAC_OBSERVATION_NAMES
from .replay import NStepReplay
from .resources import ResourceMeter, hardware
from .sac import SACAgent
from .td3 import TD3Agent, TD3Config, TD3_PAPER_URL, TD3_SOURCE_REVISION
from .training import _versions, evaluate_sac, write_json


class RewardVariantEnv(SACRacingEnv):
    """Paper reward ablations on the common control task, not full F1TENTH reproduction."""

    def __init__(self, *args, reward_profile="common", **kwargs):
        if reward_profile not in ("common", "progress", "cth"):
            raise ValueError("Unknown reward profile")
        self.reward_profile = reward_profile
        super().__init__(*args, **kwargs)

    def step(self, action):
        previous_progress, previous_laps = self.env.progress, self.completed_laps
        previous_collisions = self.env.collisions
        observation, reward, terminated, truncated, info = super().step(action)
        if self.reward_profile != "common":
            if self.completed_laps > previous_laps:
                reward = 1.0
            elif self.env.collisions > previous_collisions:
                reward = -1.0
            elif self.reward_profile == "progress":
                reward = 100 * (self.env.progress - previous_progress) / self.env.track.length
            else:
                _, gap, heading = self.env.track.project(self.env.pos)
                reward = max(self.env.speed / 85 * np.cos(heading - self.env.heading) - abs(gap), 0)
        return observation, float(reward), terminated, truncated, info


def evaluate_online(agent, tracks, task, *, reward_profile="common", **kwargs):
    # Main benchmark always evaluates the common task. Reward variants are explicitly
    # scored with the common reward too, so lap/progress outcomes stay comparable.
    report = evaluate_sac(agent, tracks, task, **kwargs)
    report["algorithm"] = getattr(agent, "algorithm", "SAC-3step")
    report["training_reward_profile"] = reward_profile
    report["evaluation_reward_profile"] = "common"
    return report


def train_online(
    output,
    *,
    algorithm="sac",
    track="monza",
    steps=10_000,
    seed=0,
    config=None,
    task=None,
    device="cpu",
    torch_threads=1,
    progress=True,
    reward_profile="common",
    benchmark_hash=None,
):
    """Fresh initialization only. No external data, resume, controller or model input.

    SAC keeps its existing learning rule and update-before-append order. TD3 keeps
    the author's two-batch call and local actor-delay order. Counts refer to optimizer
    steps, not individual critic networks; each critic step trains both Q networks.
    """
    if algorithm not in ("sac", "td3"):
        raise ValueError("algorithm must be sac or td3")
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("steps must be positive and seed a nonnegative integer")
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    config = config or (SACConfig() if algorithm == "sac" else TD3Config())
    if not isinstance(config, SACConfig if algorithm == "sac" else TD3Config):
        raise ValueError("Configuration does not match algorithm")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; use a fresh run directory")
    task = task or RacingTaskConfig()
    torch.set_num_threads(torch_threads)
    meter = ResourceMeter(device)
    env = RewardVariantEnv(track, task, reward_profile=reward_profile)
    output.mkdir(parents=True)
    collected, episodes, eval_decisions, episode_return = 0, 0, 0, 0.0
    training_laps = training_valid_laps = training_collisions = training_offtrack_steps = 0
    training_distance = 0.0
    replay, agent, info, stats = None, None, None, {}
    phase, status = "initialization", "failed"
    try:
        with meter.measure("initialization"):
            agent_class = SACAgent if algorithm == "sac" else TD3Agent
            agent = agent_class(168, config=config, seed=seed, device=device)
            replay = NStepReplay(
                config.replay_capacity, 168, gamma=config.gamma, n_step=config.n_step, seed=seed
            )
            observation, info = env.reset(seed=seed)
            metadata = dict(
                algorithm=getattr(agent, "algorithm", "SAC-3step"),
                adapter_version=ADAPTER_VERSION,
                source_revision=SOURCE_REVISION if algorithm == "sac" else TD3_SOURCE_REVISION,
                paper=PAPER_URL if algorithm == "sac" else TD3_PAPER_URL,
                data_source="online_self_generated",
                expert_dataset_used=False,
                pretrained_initialization=False,
                expert_controller_queries=0,
                replay_initial_size=0,
                config=asdict(config),
                task=asdict(task),
                seed=seed,
                track=env.env.track.name,
                track_id=track,
                track_sha256=env.env.track.digest,
                observation_names=list(SAC_OBSERVATION_NAMES),
                requested_steps=steps,
                reward_profile=reward_profile,
                benchmark_hash=benchmark_hash,
                device=str(agent.device),
                torch_threads=torch_threads,
                versions=_versions(),
                hardware=hardware(),
                full_paper_budget_completed=False,
                budget_kind="Racing Lab matched interaction budget",
                accounting_schema=1,
                checkpoint_resume_supported=False,
                exploration=f"uniform first {config.start_steps} decisions then Gaussian SAC"
                if algorithm == "sac"
                else "Gaussian noisy actor from decision one; no uniform warmup or launch assistance",
            )
            metadata["configuration_hash"] = config_hash(metadata)
            write_json(output / "run.json", metadata)
        phase = "initial_evaluation"
        with meter.measure("evaluation"):
            initial = evaluate_online(
                agent,
                (track,),
                replace(task, max_laps=4),
                seed=10000,
                reward_profile=reward_profile,
            )
            write_json(output / "evaluation_initial.json", initial)
            eval_decisions += sum(row["decisions"] for row in initial["rows"])
        phase = "training"
        fields = [
            "decision",
            "updates",
            "actor_updates",
            "replay_size",
            "replay_allocated_bytes",
            "elapsed_seconds",
            "cpu_seconds",
            "peak_rss_bytes",
            "sample_presentations",
            "policy_loss",
            "q_loss",
            "entropy_loss",
            "alpha",
        ]
        episode_fields = [
            "episode",
            "decision",
            "decisions",
            "return_sum",
            "progress_laps",
            "completed_laps",
            "valid_laps",
            "best_valid_lap",
            "collisions",
            "offtrack_steps",
            "reason",
        ]
        episode_start = 0
        with (
            (output / "learning.csv").open("w", newline="", encoding="utf-8") as learning,
            (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
        ):
            writer, episode_writer = (
                csv.DictWriter(learning, fields),
                csv.DictWriter(episode_file, episode_fields),
            )
            writer.writeheader()
            episode_writer.writeheader()
            for decision in range(1, steps + 1):
                with meter.measure("action_selection"):
                    action = (
                        env.action_space.sample()
                        if algorithm == "sac" and collected < config.start_steps
                        else agent.act(observation)
                    )
                if collected >= config.start_steps and len(replay) >= config.batch_size:
                    with meter.measure("learning_and_sampling"):

                        def sampler():
                            return replay.sample(config.batch_size)

                        stats = agent.update(sampler() if algorithm == "sac" else sampler)
                with meter.measure("training_simulation"):
                    before_laps, before_valid = env.completed_laps, env.valid_laps
                    before_collisions, before_offtrack = env.env.collisions, env.env.offtrack_steps
                    before_distance = env.env.distance
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    training_laps += info["completed_laps"] - before_laps
                    training_valid_laps += info["valid_laps"] - before_valid
                    training_collisions += env.env.collisions - before_collisions
                    training_offtrack_steps += env.env.offtrack_steps - before_offtrack
                    training_distance += env.env.distance - before_distance
                with meter.measure("replay_storage"):
                    replay.append(
                        observation, action, reward, next_observation, terminated, truncated
                    )
                collected += 1
                observation = next_observation
                episode_return += reward
                if terminated or truncated:
                    with meter.measure("logging_and_resets"):
                        episode_writer.writerow(
                            dict(
                                episode=episodes,
                                decision=decision,
                                decisions=decision - episode_start,
                                return_sum=episode_return,
                                progress_laps=info["progress"],
                                completed_laps=info["completed_laps"],
                                valid_laps=info["valid_laps"],
                                best_valid_lap=info["best_valid_lap"],
                                collisions=info["collisions"],
                                offtrack_steps=env.env.offtrack_steps,
                                reason=info["termination_reason"],
                            )
                        )
                        episode_file.flush()
                        episodes += 1
                        episode_start, episode_return = decision, 0.0
                        observation, info = env.reset(seed=seed + episodes)
                if decision % 1000 == 0 or decision == steps:
                    with meter.measure("logging_and_resets"):
                        resources = meter.report()
                        writer.writerow(
                            dict(
                                decision=decision,
                                updates=agent.updates,
                                actor_updates=getattr(agent, "actor_updates", agent.updates),
                                replay_size=len(replay),
                                replay_allocated_bytes=replay_bytes(replay),
                                elapsed_seconds=resources["wall_seconds"],
                                cpu_seconds=resources["process_cpu_seconds"],
                                peak_rss_bytes=resources["memory"]["peak_rss_bytes"],
                                sample_presentations=agent.updates * config.batch_size,
                                **stats,
                            )
                        )
                        learning.flush()
                    if progress:
                        print(
                            f"{algorithm.upper()} seed {seed}: {decision}/{steps} decisions | {agent.updates} critic updates | {resources['wall_seconds']:.1f}s",
                            flush=True,
                        )
                if config.eval_interval and decision % config.eval_interval == 0:
                    phase = "periodic_evaluation"
                    with meter.measure("evaluation"):
                        report = evaluate_online(
                            agent,
                            (track,),
                            replace(task, max_laps=4),
                            seed=10000,
                            reward_profile=reward_profile,
                            trajectory_output=output / "trajectories" / f"step_{decision:09d}",
                        )
                        eval_decisions += sum(row["decisions"] for row in report["rows"])
                        write_json(output / f"evaluation_{decision:09d}.json", report)
                    phase = "training"
                if config.checkpoint_interval and decision % config.checkpoint_interval == 0:
                    with meter.measure("checkpoint_io"):
                        agent.save(
                            output / f"step_{decision:09d}.pt", dict(metadata, decisions=decision)
                        )
            phase = "final_evaluation"
            # Avoid repeating an identical final-budget evaluation, but retain its file.
            periodic_path = output / f"evaluation_{collected:09d}.json"
            if periodic_path.exists():
                import json

                final = json.loads(periodic_path.read_text(encoding="utf-8"))
            else:
                with meter.measure("evaluation"):
                    final = evaluate_online(
                        agent,
                        (track,),
                        replace(task, max_laps=4),
                        seed=10000,
                        reward_profile=reward_profile,
                        trajectory_output=output / "trajectories" / "final",
                    )
                    eval_decisions += sum(row["decisions"] for row in final["rows"])
            write_json(output / "evaluation_final.json", final)
            with meter.measure("checkpoint_io"):
                replay.flush()
                agent.save(output / "final.pt", dict(metadata, decisions=collected))
            phase, status = "complete", "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if replay is not None:
            replay.flush()
        result = dict(
            status=status,
            phase=phase,
            decisions=collected,
            episodes_finished=episodes,
            training_completed_laps=training_laps,
            training_valid_laps=training_valid_laps,
            training_collisions=training_collisions,
            training_offtrack_steps=training_offtrack_steps,
            training_distance_metres=training_distance,
            unfinished_episode_decisions=env.env.steps if info else None,
            unfinished_episode_return=episode_return,
            simulated_seconds=collected / task.decision_hz,
            physics_substeps=collected * int(np.ceil(env.env.dt / (1 / 240))),
            evaluation_decisions=eval_decisions,
            evaluation_simulated_seconds=eval_decisions / task.decision_hz,
            updates=agent.updates if agent else 0,
            actor_updates=getattr(agent, "actor_updates", agent.updates) if agent else 0,
            entropy_updates=agent.updates if agent and algorithm == "sac" else 0,
            training_calls=(agent.updates if algorithm == "sac" else agent.training_calls)
            if agent
            else 0,
            sampled_transition_presentations=(agent.updates * config.batch_size) if agent else 0,
            sample_count_definition="critic minibatch rows, with replacement; not unique driving data",
            unique_training_transitions=collected,
            replay_lifetime_insertions=collected,
            replay_size=len(replay) if replay else 0,
            replay_capacity=config.replay_capacity,
            replay_allocated_bytes=replay_bytes(replay) if replay else 0,
            random_action_decisions=min(collected, config.start_steps) if algorithm == "sac" else 0,
            external_dataset_transitions=0,
            expert_dataset_used=False,
            expert_controller_queries=0,
            resources=meter.report(),
            last_update=stats,
        )
        if status == "completed":
            result["evaluation_summary"] = final["summary"]
            result["final_evaluation_reused_periodic"] = periodic_path.exists()
        elif agent:
            replay.flush()
            agent.save(output / "interrupted.pt", dict(metadata, result=result))
        write_json(output / "result.json", result)
        env.close()
    return result


def replay_bytes(replay):
    return sum(array.nbytes for chunk in replay.chunks.values() for array in chunk)
