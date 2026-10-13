"""Validated per-seed tables, scientific figures and a standalone benchmark report."""

import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .config import config_hash
from .training import write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def descriptive(values):
    values = np.asarray([v for v in values if v is not None], float)
    return dict(
        n=len(values),
        mean=float(values.mean()) if len(values) else None,
        std=float(values.std(ddof=1)) if len(values) > 1 else None,
        minimum=float(values.min()) if len(values) else None,
        maximum=float(values.max()) if len(values) else None,
    )


def _evaluation_files(directory, result):
    files = [directory / "evaluation_initial.json", *sorted(directory.glob("evaluation_0*.json"))]
    final_path = directory / "evaluation_final.json"
    if result["final_evaluation_reused_periodic"]:
        matching = directory / f"evaluation_{result['decisions']:09d}.json"
        if _read(matching) != _read(final_path):
            raise ValueError("Reused final evaluation differs from its periodic measurement")
    else:
        files.append(final_path)
    files.extend(
        path
        for path in (
            directory / "evaluation_robustness.json",
            directory / "evaluation_held_out.json",
        )
        if path.exists()
    )
    return files


def _validate_evaluation(report, path, manifest, *, training_reward_profile="common"):
    from dataclasses import replace
    from .config import RacingTaskConfig
    from .environment import SACRacingEnv

    settings = manifest["config"]
    if path.name == "evaluation_robustness.json":
        condition = manifest["evaluation"]["varied_start"]
        tracks = (settings["track"],)
    elif path.name == "evaluation_held_out.json":
        condition = manifest["evaluation"]["held_out"]
        tracks = tuple(settings["held_out_tracks"])
    else:
        condition = dict(manifest["evaluation"]["fixed_start"], lateral_jitter=0.0)
        tracks = (settings["track"],)
    task = replace(RacingTaskConfig(**manifest["task"]), max_laps=condition["max_laps"])
    if report["task"] != vars(task) or report["reset_variation"] != {
        "lateral_jitter": condition["lateral_jitter"]
    }:
        raise ValueError("Evaluation task or reset variation differs from the protocol")
    if (
        report.get("training_reward_profile") != training_reward_profile
        or report.get("evaluation_reward_profile") != "common"
    ):
        raise ValueError("Reward ablation cannot enter the common benchmark")
    expected = {
        (manifest["tracks"][track], condition["seed"] + episode)
        for track in tracks
        for episode in range(condition["episodes"])
    }
    actual = [(row["track_sha256"], row["seed"]) for row in report["rows"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Evaluation is missing required tracks/starts or uses different seeds")
    for track in tracks:
        env = SACRacingEnv(track, task)
        try:
            for episode in range(condition["episodes"]):
                seed = condition["seed"] + episode
                env.reset(seed=seed, options={"lateral_jitter": condition["lateral_jitter"]})
                expected_hash = hashlib.sha256(env.env.state().tobytes()).hexdigest()
                row = next(
                    row
                    for row in report["rows"]
                    if row["track_sha256"] == manifest["tracks"][track] and row["seed"] == seed
                )
                if row["reset_state_sha256"] != expected_hash:
                    raise ValueError("Actual evaluation reset does not match the declared start")
                if len(row["valid_lap_times"]) != row["valid_laps"]:
                    raise ValueError("Evaluation valid-lap count/time mismatch")
                if any(time <= 0 or not np.isfinite(time) for time in row["valid_lap_times"]):
                    raise ValueError("Invalid lap time")
        finally:
            env.close()


def collect_runs(root, *, diagnostic=False):
    root = Path(root)
    manifest = _read(root / "manifest.json")
    if bool(manifest.get("kind") == "td3_diagnostic") != diagnostic:
        raise ValueError("Diagnostic conditions require their separate report")
    if (
        config_hash({k: v for k, v in manifest.items() if k != "benchmark_hash"})
        != manifest["benchmark_hash"]
    ):
        raise ValueError("Manifest was modified after freezing")
    records, reset_hashes, artifacts = [], {}, {}
    for spec in manifest["runs"]:
        directory = root / spec["directory"]
        if not (directory / "worker_result.json").exists():
            continue
        worker, metadata, result = (
            _read(directory / name) for name in ("worker_result.json", "run.json", "result.json")
        )
        algorithm, config = spec["algorithm"], manifest["algorithms"][spec["algorithm"]]
        if worker["status"] != result["status"] or worker["status"] != "completed":
            raise ValueError("An incomplete worker cannot be scored")
        if (
            worker["benchmark_hash"] != manifest["benchmark_hash"]
            or metadata["benchmark_hash"] != manifest["benchmark_hash"]
        ):
            raise ValueError("Run belongs to a different protocol")
        if metadata["config"] != config or metadata["task"] != manifest["task"]:
            raise ValueError("Run settings do not match the frozen manifest")
        if metadata["track_sha256"] != manifest["tracks"][manifest["config"]["track"]]:
            raise ValueError("Training track differs from the frozen protocol")
        if metadata["source_revision"] != manifest["sources"][algorithm]["revision"]:
            raise ValueError("Learning source revision mismatch")
        if any(
            metadata["versions"].get(key) != value
            for key, value in manifest["runtime_versions"].items()
        ):
            raise ValueError("Run runtime differs from the frozen versions")
        if (
            config_hash(
                {key: value for key, value in metadata.items() if key != "configuration_hash"}
            )
            != metadata["configuration_hash"]
        ):
            raise ValueError("Run specification hash mismatch")
        if metadata["seed"] != spec["seed"] or result["decisions"] != manifest["config"]["steps"]:
            raise ValueError("Seed or training budget mismatch")
        if (
            metadata["expert_dataset_used"]
            or metadata["pretrained_initialization"]
            or result["external_dataset_transitions"]
        ):
            raise ValueError("External data is forbidden in this benchmark")
        checkpoint_hash = hashlib.sha256((directory / "final.pt").read_bytes()).hexdigest()
        if checkpoint_hash != worker["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        calls = max(0, result["decisions"] - config["start_steps"])
        expected_critic = calls * (1 if algorithm == "sac" else config["critic_steps_per_call"])
        expected_actor = (
            calls
            if algorithm == "sac"
            else calls * len(range(0, config["critic_steps_per_call"], config["policy_delay"]))
        )
        if result["updates"] != expected_critic or result["actor_updates"] != expected_actor:
            raise ValueError("Optimizer counts disagree with the declared learning schedule")
        files = _evaluation_files(directory, result)
        interval = config["eval_interval"]
        required = {
            "evaluation_initial.json",
            "evaluation_final.json",
            "evaluation_robustness.json",
        }
        if manifest["config"]["held_out_tracks"]:
            required.add("evaluation_held_out.json")
        if interval:
            required.update(
                f"evaluation_{step:09d}.json"
                for step in range(interval, result["decisions"] + 1, interval)
            )
        if any(not (directory / name).is_file() for name in required):
            raise ValueError("Completed worker is missing required evaluation artifacts")
        evaluations = [(path, _read(path)) for path in files]
        for path, report in evaluations:
            expected_reward = (
                manifest["conditions"][spec["condition"]]["reward_profile"]
                if diagnostic
                else "common"
            )
            _validate_evaluation(report, path, manifest, training_reward_profile=expected_reward)
            if report["gradient_updates"] != 0 or not report["deterministic"]:
                raise ValueError("Evaluation must be frozen and deterministic")
            for row in report["rows"]:
                key = (path.name, row["track_sha256"], row["seed"])
                if key in reset_hashes and reset_hashes[key] != row["reset_state_sha256"]:
                    raise ValueError("Methods were evaluated from different reset states")
                reset_hashes[key] = row["reset_state_sha256"]
        fixed = _read(directory / "evaluation_final.json")
        fixed_row = fixed["rows"][0]
        extra = [
            report
            for path, report in evaluations
            if path.name in ("evaluation_robustness.json", "evaluation_held_out.json")
        ]
        eval_decisions = sum(
            row["decisions"] for _, report in evaluations for row in report["rows"]
        )
        train_evaluation_decisions = sum(
            row["decisions"]
            for path, report in evaluations
            if path.name not in ("evaluation_robustness.json", "evaluation_held_out.json")
            for row in report["rows"]
        )
        if train_evaluation_decisions != result["evaluation_decisions"]:
            raise ValueError("Training evaluator interaction count mismatch")
        all_extra_rows = [row for report in extra for row in report["rows"]]
        robustness = [
            row for report in extra if report["trials"] == "varied starts" for row in report["rows"]
        ]
        held_out = [
            row for report in extra if report["trials"] != "varied starts" for row in report["rows"]
        ]
        resource = worker["resources"]
        components = result["resources"]["component_seconds"]
        records.append(
            dict(
                algorithm=algorithm,
                seed=spec["seed"],
                directory=spec["directory"],
                training_transitions=result["unique_training_transitions"],
                training_episodes_finished=result["episodes_finished"],
                unfinished_episode_decisions=result["unfinished_episode_decisions"],
                training_completed_laps=result["training_completed_laps"],
                training_valid_laps=result["training_valid_laps"],
                training_collisions=result["training_collisions"],
                training_offtrack_steps=result["training_offtrack_steps"],
                training_distance_metres=result["training_distance_metres"],
                evaluation_transitions=eval_decisions,
                total_transitions=result["decisions"] + eval_decisions,
                training_simulated_seconds=result["simulated_seconds"],
                evaluation_simulated_seconds=eval_decisions / manifest["task"]["decision_hz"],
                training_physics_substeps=result["physics_substeps"],
                evaluation_physics_substeps=sum(
                    report["evaluation_physics_substeps"] for _, report in evaluations
                ),
                twin_critic_optimizer_steps=result["updates"],
                actor_optimizer_steps=result["actor_updates"],
                entropy_optimizer_steps=result["entropy_updates"],
                critic_minibatch_row_presentations=result["sampled_transition_presentations"],
                actor_minibatch_row_presentations=result["actor_updates"] * config["batch_size"],
                entropy_minibatch_row_presentations=result["entropy_updates"]
                * config["batch_size"],
                replay_occupancy=result["replay_size"],
                replay_lifetime_insertions=result["replay_lifetime_insertions"],
                replay_capacity=result["replay_capacity"],
                replay_allocated_bytes=result["replay_allocated_bytes"],
                worker_wall_seconds=resource["wall_seconds"],
                trainer_wall_seconds=result["resources"]["wall_seconds"],
                training_simulation_wall_seconds=components.get("training_simulation", 0),
                learning_and_sampling_wall_seconds=components.get("learning_and_sampling", 0),
                action_selection_wall_seconds=components.get("action_selection", 0),
                initialization_wall_seconds=components.get("initialization", 0),
                replay_storage_wall_seconds=components.get("replay_storage", 0),
                logging_and_resets_wall_seconds=components.get("logging_and_resets", 0),
                checkpoint_io_wall_seconds=components.get("checkpoint_io", 0),
                training_evaluator_wall_seconds=components.get("evaluation", 0),
                post_training_evaluator_wall_seconds=resource["component_seconds"].get(
                    "post_training_evaluation", 0
                ),
                trainer_unassigned_wall_seconds=result["resources"]["unassigned_seconds"],
                process_cpu_seconds=resource["process_cpu_seconds"],
                peak_rss_bytes=resource["memory"]["peak_rss_bytes"],
                external_dataset_transitions=0,
                expert_controller_queries=0,
                fixed_valid_laps=fixed_row["valid_laps"],
                fixed_completed_laps=fixed_row["completed_laps"],
                fixed_success=int(fixed_row["success"]),
                fixed_clean_four_laps=int(fixed_row["clean_four_lap_completion"]),
                standing_start_valid_seconds=fixed_row["standing_start_valid_lap"],
                mean_valid_lap_seconds=fixed["summary"]["mean_valid_lap"],
                best_valid_lap_seconds=fixed["summary"]["best_valid_lap"],
                fixed_progress_laps=fixed_row["progress_laps"],
                fixed_simulated_seconds=fixed_row["simulated_seconds"],
                fixed_distance_metres=fixed_row["distance_metres"],
                fixed_average_speed_kmh=fixed_row["average_speed_kmh"],
                fixed_maximum_speed_kmh=fixed_row["maximum_speed_kmh"],
                flying_valid_lap_mean_seconds=fixed_row["flying_valid_lap_mean"],
                fixed_termination=fixed_row["termination_reason"],
                fixed_collisions=fixed_row["collisions"],
                fixed_offtrack_steps=fixed_row["offtrack_steps"],
                inference_mean_ms=fixed_row["inference_mean_ms"],
                inference_p95_ms=fixed_row["inference_p95_ms"],
                varied_start_success_rate=sum(row["success"] for row in robustness)
                / len(robustness)
                if robustness
                else None,
                held_out_success_rate=sum(row["success"] for row in held_out) / len(held_out)
                if held_out
                else None,
                extra_evaluation_episodes=len(all_extra_rows),
            )
        )
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv", ".pt", ".npz"):
                artifacts[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return manifest, records, artifacts


def generate_report(root):
    root = Path(root)
    manifest, records, artifacts = collect_runs(root)
    if not records:
        raise ValueError("No completed benchmark runs to report")
    report_directory = root / "report"
    report_directory.mkdir(exist_ok=True)
    with (report_directory / "per_seed.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = [
        key
        for key in records[0]
        if key not in ("algorithm", "seed", "directory", "fixed_termination")
    ]
    aggregate = {
        algorithm: {
            key: descriptive([row[key] for row in records if row["algorithm"] == algorithm])
            for key in metrics
        }
        for algorithm in ("sac", "td3")
    }
    complete = len(records) == len(manifest["runs"])
    summary = dict(
        status="completed" if complete else "partial",
        benchmark_hash=manifest["benchmark_hash"],
        completed_runs=len(records),
        planned_runs=len(manifest["runs"]),
        per_seed=records,
        aggregate=aggregate,
        artifacts_sha256=artifacts,
        reporting_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics="descriptive mean, sample SD and range across independently trained policies; no lap/reset pseudoreplication",
        conditional_lap_time="only successful policies; valid n reported; compare reliability first",
        timing_condition=manifest["timing_condition"],
        original_paper_scores_comparable=False,
    )
    write_json(report_directory / "summary.json", summary)
    _figures(root, report_directory, records)
    _html(report_directory, manifest, summary)
    return summary


def _figures(root, directory, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sac": "#087f8c", "td3": "#db6a28"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row in records:
        run = root / row["directory"]
        with (run / "episodes.csv").open(encoding="utf-8") as file:
            episodes = list(csv.DictReader(file))
        if episodes:
            x = [int(item["decision"]) for item in episodes]
            for axis, key in zip(axes[0], ("return_sum", "progress_laps")):
                values = np.array([float(item[key]) for item in episodes])
                smooth = np.array(
                    [values[max(0, i - 19) : i + 1].mean() for i in range(len(values))]
                )
                axis.plot(x, smooth, color=colors[row["algorithm"]], alpha=0.55, linewidth=1)
        paths = [run / "evaluation_initial.json", *sorted(run.glob("evaluation_0*.json"))]
        if not (run / f"evaluation_{row['training_transitions']:09d}.json").exists():
            paths.append(run / "evaluation_final.json")
        x, valid, progress = [], [], []
        for path in paths:
            report = _read(path)
            x.append(
                0
                if path.stem == "evaluation_initial"
                else row["training_transitions"]
                if path.stem == "evaluation_final"
                else int(path.stem.split("_")[1])
            )
            valid.append(report["rows"][0]["valid_laps"])
            progress.append(report["rows"][0]["progress_laps"])
        axes[1, 0].plot(x, valid, marker="o", color=colors[row["algorithm"]], alpha=0.55)
        axes[1, 1].plot(x, progress, marker="o", color=colors[row["algorithm"]], alpha=0.55)
    for axis, title, ylabel in zip(
        axes.flat,
        (
            "Training return: trailing 20 completed episodes",
            "Training net progress: trailing 20 episodes",
            "Frozen fixed-start valid laps (four-lap cap)",
            "Frozen fixed-start net progress",
        ),
        ("Return", "Track lengths", "Strictly valid laps", "Track lengths"),
    ):
        axis.set(title=title, xlabel="Collected training decisions", ylabel=ylabel)
        axis.grid(alpha=0.2)
    from matplotlib.lines import Line2D

    figure.legend(
        [Line2D([], [], color=colors[key]) for key in colors],
        ["SAC-3step", "TD3"],
        loc="outside upper center",
        ncol=2,
    )
    figure.savefig(directory / "learning_curves.png", dpi=160)
    figure.savefig(directory / "learning_curves.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, key, title, divisor in (
        (axes[0], "fixed_clean_four_laps", "Clean four-lap completion", 1),
        (axes[1], "worker_wall_seconds", "Training + all evaluation", 3600),
        (axes[2], "peak_rss_bytes", "Peak resident memory", 1024**3),
    ):
        for index, algorithm in enumerate(colors):
            values = [
                row[key] / divisor
                for row in records
                if row["algorithm"] == algorithm and row[key] is not None
            ]
            positions = (
                index + np.linspace(-0.06, 0.06, len(values))
                if len(values) > 1
                else np.full(len(values), index)
            )
            axis.scatter(positions, values, color=colors[algorithm], s=35)
        axis.set(xticks=[0, 1], xticklabels=["SAC", "TD3"], title=title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set(ylim=(-0.05, 1.05), yticks=[0, 1], ylabel="One trained policy per point")
    axes[1].set_ylabel("Worker wall hours")
    axes[2].set_ylabel("GiB, process lifetime")
    figure.savefig(directory / "outcomes_and_cost.png", dpi=160)
    figure.savefig(directory / "outcomes_and_cost.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for row in records:
        with (root / row["directory"] / "learning.csv").open(encoding="utf-8") as file:
            logs = list(csv.DictReader(file))
        for axis, key in zip(axes, ("policy_loss", "q_loss", "alpha")):
            selected = [item for item in logs if item.get(key)]
            if not selected:
                continue
            values = np.asarray([float(item[key]) for item in selected])
            smooth = [values[max(0, index - 9) : index + 1].mean() for index in range(len(values))]
            axis.plot(
                [int(item["decision"]) for item in selected],
                smooth,
                color=colors[row["algorithm"]],
                alpha=0.55,
            )
    for axis, title in zip(axes, ("Actor loss", "Twin-critic loss", "SAC entropy temperature")):
        axis.set(title=title + "\nTrailing 10 logged values", xlabel="Training decisions")
        axis.grid(alpha=0.2)
    figure.savefig(directory / "optimization_curves.png", dpi=160)
    figure.savefig(directory / "optimization_curves.svg")
    plt.close(figure)


def _html(directory, manifest, summary):
    aggregate_metrics = (
        "fixed_success",
        "fixed_clean_four_laps",
        "standing_start_valid_seconds",
        "flying_valid_lap_mean_seconds",
        "varied_start_success_rate",
        "held_out_success_rate",
        "training_transitions",
        "evaluation_transitions",
        "twin_critic_optimizer_steps",
        "actor_optimizer_steps",
        "critic_minibatch_row_presentations",
        "worker_wall_seconds",
        "process_cpu_seconds",
        "peak_rss_bytes",
        "inference_mean_ms",
    )

    def format_stats(stats):
        if not stats["n"]:
            return "— (n=0)"
        deviation = "—" if stats["std"] is None else f"{stats['std']:.3f}"
        return f"{stats['mean']:.3f} ± {deviation}; [{stats['minimum']:.3f}, {stats['maximum']:.3f}]; n={stats['n']}"

    aggregate_rows = "".join(
        f"<tr><th>{html.escape(metric.replace('_', ' '))}</th>"
        + "".join(
            f"<td>{format_stats(summary['aggregate'][algorithm][metric])}</td>"
            for algorithm in ("sac", "td3")
        )
        + "</tr>"
        for metric in aggregate_metrics
    )
    columns = [
        "algorithm",
        "seed",
        "training_transitions",
        "evaluation_transitions",
        "twin_critic_optimizer_steps",
        "actor_optimizer_steps",
        "worker_wall_seconds",
        "process_cpu_seconds",
        "peak_rss_bytes",
        "fixed_valid_laps",
        "fixed_clean_four_laps",
        "standing_start_valid_seconds",
        "best_valid_lap_seconds",
        "varied_start_success_rate",
        "held_out_success_rate",
    ]
    header = "".join(f"<th>{html.escape(key.replace('_', ' '))}</th>" for key in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape('—' if row[key] is None else f'{row[key]:.3f}' if isinstance(row[key], float) else str(row[key]))}</td>"
            for key in columns
        )
        + "</tr>"
        for row in summary["per_seed"]
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Racing Lab online benchmark</title><style>body{{font:16px system-ui;margin:32px;color:#172b34;background:#f5f8fa}}main{{max-width:1250px;margin:auto}}img{{width:100%;background:white}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px;border:1px solid #ccd7dd;text-align:right}}th{{min-width:100px}}pre{{white-space:pre-wrap;background:white;padding:20px}}</style><main>
<h1>SAC–TD3 online benchmark</h1><p>Status: <b>{summary["status"]}</b>; {summary["completed_runs"]}/{summary["planned_runs"]} fresh runs. Training circuit: {html.escape(manifest["config"]["track"])}.</p>
<p>Every policy starts from fresh networks and an empty replay buffer. No expert dataset, pretrained policy or launch controller. Main comparisons share the numerical inputs, controls, reward, clock, driving aids and episode rules.</p>
<p>{html.escape(manifest["timing_condition"])}. Equal interaction budget does not mean equal compute. Peak RSS includes framework startup; worker wall/CPU timing starts after framework imports. A twin-critic optimizer step trains both Q networks.</p>
<img src="learning_curves.png" alt="Per-seed training and frozen evaluation learning curves">
<img src="outcomes_and_cost.png" alt="Per-seed clean completion and resource costs">
<img src="optimization_curves.png" alt="Per-seed losses and SAC temperature diagnostics">
<h2>Every training seed</h2><p>Seconds and bytes are explicit in column names. Missing or failed lap times are shown as —. A valid lap requires all gates and zero off-road steps, counted collisions or recovery. Standing and flying laps are separate.</p>
<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>
<p><a href="per_seed.csv">All per-seed metrics (CSV)</a> · <a href="summary.json">Aggregates, counts and artifact hashes (JSON)</a> · <a href="../manifest.json">Frozen protocol</a></p>
<h2>Variation across training seeds</h2><p>Mean ± sample standard deviation; [minimum, maximum]; n counts trained policies with an available measurement. Time means are conditional on successful policies.</p>
<div class="scroll"><table><thead><tr><th>Metric</th><th>SAC</th><th>TD3</th></tr></thead><tbody>{aggregate_rows}</tbody></table></div>
<h2>Statistical interpretation</h2><p>{html.escape(summary["statistics"])}. Mean lap time is conditional on successful policies, so compare reliability first. Five training seeds still provide limited precision. Fixed-start repeats and multiple laps from one policy are not independent trained agents. Unseen circuits are zero-shot tests, and compact/long variants belong to the same family.</p>
<h2>Reproduction limits</h2><p>These are source-based learning rules adapted to Racing Lab. Our generic vehicle and inspired circuits differ from the original simulators. The budget is 200,000 decisions per run by default; it is shorter than the SAC racing paper's approximately eight million decisions. Paper-reward Progress/CTH ablations are excluded from this main ranking.</p>
<h2>Frozen settings and source attribution</h2><pre>{html.escape(json.dumps(manifest, indent=2))}</pre></main></html>"""
    (directory / "report.html").write_text(content, encoding="utf-8")
