"""Validated diagnostic outcomes, exposure, braking, paired seeds and cost figures."""

import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np

from .benchmark_report import collect_runs, descriptive
from .config import RacingTaskConfig
from .diagnostic import CONDITIONS
from .environment import SACRacingEnv
from .training import write_json

LABELS = dict(
    baseline="Baseline",
    uniform_warmup="Random warmup",
    random_starts="Random starts",
    cth_reward="CTH reward",
)
COLORS = dict(
    baseline="#747b89", uniform_warmup="#087f8c", random_starts="#a35ec7", cth_reward="#db6a28"
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_resets(root, spec, manifest, metadata, result):
    profile = manifest["conditions"][spec["condition"]]
    if any(metadata.get(key) != value for key, value in profile.items()) or not metadata.get(
        "capture_diagnostics"
    ):
        raise ValueError("Diagnostic condition differs from manifest")
    expected_random = min(result["decisions"], profile["uniform_warmup_decisions"])
    if result["random_action_decisions"] != expected_random:
        raise ValueError("Uniform behavior count mismatch")
    diagnostics = result["diagnostics"]
    counts = np.asarray(diagnostics["training_position_counts_64_bins"])
    if (
        counts.shape != (64,)
        or np.any(counts < 0)
        or counts.sum() != result["decisions"]
        or np.count_nonzero(counts) != diagnostics["visited_training_bins"]
    ):
        raise ValueError("Training coverage count mismatch")
    resets = _read(root / spec["directory"] / "training_resets.json")
    if len(resets) != result["episodes_finished"] + 1:
        raise ValueError("Training reset count mismatch")
    with (root / spec["directory"] / "episodes.csv").open(encoding="utf-8") as file:
        episodes = list(csv.DictReader(file))
    if len(episodes) != result["episodes_finished"]:
        raise ValueError("Episode log count mismatch")
    if (
        sum(int(row["decisions"]) for row in episodes) + result["unfinished_episode_decisions"]
        != result["decisions"]
    ):
        raise ValueError("Episode decision counts do not cover training budget")
    env = SACRacingEnv(manifest["config"]["track"], RacingTaskConfig(**manifest["task"]))
    try:
        for episode, row in enumerate(resets):
            seed = spec["seed"] + episode
            start = (
                float(np.random.default_rng(seed).uniform(0, env.env.track.length))
                if profile["random_training_starts"]
                else 0.0
            )
            first = 1 if episode == 0 else int(episodes[episode - 1]["decision"]) + 1
            if (
                row["episode"] != episode
                or row["seed"] != seed
                or row["first_decision"] != first
                or row["start_distance_metres"] != start
            ):
                raise ValueError("Training start differs from frozen sampling rule")
            env.reset(
                seed=seed,
                options={"start_distance": start} if profile["random_training_starts"] else None,
            )
            if hashlib.sha256(env.env.state().tobytes()).hexdigest() != row["reset_state_sha256"]:
                raise ValueError("Training reset physical state mismatch")
    finally:
        env.close()


def _trace_metrics(directory, fixed, result):
    folder = (
        f"step_{result['decisions']:09d}" if result["final_evaluation_reused_periodic"] else "final"
    )
    path = directory / "trajectories" / folder / fixed["trajectory_file"]
    with np.load(path, allow_pickle=False) as data:
        n = fixed["decisions"]
        if (
            data["states"].shape != (n + 1, 7)
            or data["observations"].shape != (n + 1, 168)
            or data["actions"].shape != (n, 3)
            or data["policy_actions"].shape != (n, 3)
        ):
            raise ValueError("Trajectory dimensions mismatch")
        if any(not np.isfinite(data[key]).all() for key in data.files):
            raise ValueError("Non-finite evaluation trajectory")
        if np.any(np.abs(data["policy_actions"]) > 1.000001):
            raise ValueError("Invalid policy action range")
        after = data["timestamps"][1:] > 3.0
        braking = after & (data["actions"][:, 2] > 0.1)
        return dict(
            fixed_braking_decisions_after_3s=int(np.count_nonzero(braking)),
            fixed_braking_fraction_after_3s=float(
                np.count_nonzero(braking) / np.count_nonzero(after)
            )
            if np.any(after)
            else None,
            fixed_first_brake_after_3s_seconds=float(data["timestamps"][1:][braking][0])
            if np.any(braking)
            else None,
            fixed_brake_increment_saturation_fraction=float(
                np.mean(np.abs(data["policy_actions"][:, 2]) > 0.99)
            ),
        )


def collect_diagnostic(root):
    root = Path(root)
    manifest, records, artifacts = collect_runs(root, diagnostic=True)
    if manifest["conditions"] != CONDITIONS:
        raise ValueError("Unrecognized diagnostic condition definitions")
    for row in records:
        spec = next(spec for spec in manifest["runs"] if spec["directory"] == row["directory"])
        directory = root / row["directory"]
        metadata, result = _read(directory / "run.json"), _read(directory / "result.json")
        _validate_resets(root, spec, manifest, metadata, result)
        if (
            result["expert_controller_queries"]
            or metadata["expert_controller_queries"]
            or metadata["replay_initial_size"]
            or result["entropy_updates"]
        ):
            raise ValueError("External assistance or wrong learner used")
        fixed = _read(directory / "evaluation_final.json")["rows"][0]
        row.update(
            condition=spec["condition"],
            process_lifetime_cpu_seconds=_read(directory / "worker_result.json")[
                "process_lifetime_cpu_seconds"
            ],
            random_action_decisions=result["random_action_decisions"],
            visited_training_bins=result["diagnostics"]["visited_training_bins"],
            training_coverage_fraction=result["diagnostics"]["visited_training_bins"] / 64,
            maximum_training_episode_progress_laps=result["diagnostics"][
                "maximum_episode_progress_laps"
            ],
            **_trace_metrics(directory, fixed, result),
        )
        with (directory / "learning.csv").open(encoding="utf-8") as file:
            logs = list(csv.DictReader(file))
        if not logs or int(logs[-1]["decision"]) != result["decisions"]:
            raise ValueError("Learning log does not cover the budget")
        if any(
            not np.isfinite(float(log[key]))
            for log in logs
            for key in ("q_loss", "policy_loss")
            if log[key]
        ):
            raise ValueError("Non-finite optimizer loss")
        for path in directory.glob("trajectories/*/*.npz"):
            with np.load(path, allow_pickle=False) as trace:
                n = len(trace["actions"])
                if (
                    trace["observations"].shape != (n + 1, 168)
                    or trace["states"].shape != (n + 1, 7)
                    or trace["policy_actions"].shape != (n, 3)
                    or trace["terminated_truncated"].shape != (n, 2)
                ):
                    raise ValueError("Trajectory shape mismatch")
                if any(not np.isfinite(trace[key]).all() for key in trace.files):
                    raise ValueError("Non-finite trajectory")
                if not np.array_equal(
                    trace["timestamps"], np.arange(n + 1) / manifest["task"]["decision_hz"]
                ):
                    raise ValueError("Trajectory decision clock mismatch")
    return manifest, records, artifacts


def generate_diagnostic_report(root):
    root = Path(root)
    manifest, records, artifacts = collect_diagnostic(root)
    report = root / "report"
    report.mkdir(exist_ok=True)
    if not records:
        raise ValueError("No completed diagnostic runs")
    with (report / "per_seed.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    numeric = [
        key for key, value in records[0].items() if value is None or isinstance(value, (int, float))
    ]
    aggregate = {
        condition: {
            key: descriptive([row[key] for row in records if row["condition"] == condition])
            for key in numeric
        }
        for condition in CONDITIONS
    }
    baseline = {row["seed"]: row for row in records if row["condition"] == "baseline"}
    paired = {
        condition: {
            key: descriptive(
                [
                    row[key] - baseline[row["seed"]][key]
                    for row in records
                    if row["condition"] == condition
                    and row["seed"] in baseline
                    and row[key] is not None
                    and baseline[row["seed"]][key] is not None
                ]
            )
            for key in (
                "fixed_success",
                "fixed_progress_laps",
                "varied_start_success_rate",
                "training_coverage_fraction",
                "worker_wall_seconds",
            )
        }
        for condition in CONDITIONS
        if condition != "baseline"
    }
    summary = dict(
        status="completed" if len(records) == len(manifest["runs"]) else "partial",
        benchmark_hash=manifest["benchmark_hash"],
        completed_runs=len(records),
        planned_runs=len(manifest["runs"]),
        total_training_decisions=sum(row["training_transitions"] for row in records),
        total_evaluation_decisions=sum(row["evaluation_transitions"] for row in records),
        per_seed=records,
        aggregate=aggregate,
        paired_difference_from_baseline=paired,
        artifacts_sha256=artifacts,
        statistics="Descriptive mean, sample SD and range across three trained policies; repeated starts/laps are not independent seeds",
        interpretation="Short one-factor diagnostic, not a converged algorithm ranking. Random-start exposure and reward changes have different training objectives; evaluation is identical.",
        reporting_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(report / "summary.json", summary)
    _figures(root, report, records)
    _html(report, manifest, summary)
    capacity = []
    from .td3 import TD3Agent

    for row in records:
        agent, _ = TD3Agent.load(root / row["directory"] / "final.pt")
        result = _read(root / row["directory"] / "result.json")
        fixed = _read(root / row["directory"] / "evaluation_final.json")["rows"][0]
        folder = (
            f"step_{result['decisions']:09d}"
            if result["final_evaluation_reused_periodic"]
            else "final"
        )
        with np.load(
            root / row["directory"] / "trajectories" / folder / fixed["trajectory_file"],
            allow_pickle=False,
        ) as trace:
            reproduced = np.asarray(
                [
                    agent.act(observation, deterministic=True)
                    for observation in trace["observations"][:-1]
                ]
            )
            if not np.array_equal(reproduced, trace["policy_actions"]):
                raise ValueError("Final checkpoint does not reproduce saved evaluation actions")
        capacity.append(
            dict(
                condition=row["condition"],
                seed=row["seed"],
                trainable_parameters=sum(
                    parameter.numel()
                    for module in (agent.actor, agent.critic_1, agent.critic_2)
                    for parameter in module.parameters()
                ),
                checkpoint_bytes=(root / row["directory"] / "final.pt").stat().st_size,
            )
        )
    write_json(
        report / "costs.json",
        dict(
            training_decisions=summary["total_training_decisions"],
            evaluation_decisions=summary["total_evaluation_decisions"],
            worker_cpu_seconds=sum(row["process_cpu_seconds"] for row in records),
            worker_lifetime_cpu_seconds=sum(row["process_lifetime_cpu_seconds"] for row in records),
            sum_worker_wall_seconds=sum(row["worker_wall_seconds"] for row in records),
            queue_wall_seconds=_read(root / "queue.json")["queue_wall_seconds"]
            if (root / "queue.json").exists()
            else None,
            timing_note="Four concurrent CPU workers in the main pilot. Sum of worker wall times is not elapsed queue duration. Workers measured after framework imports; process lifetime peak RSS includes imports. Final completion.json includes report generation and controller CPU.",
            model_capacity=capacity,
            per_seed=records,
        ),
    )
    snapshot = (
        _read(root / "source_snapshot.json") if (root / "source_snapshot.json").exists() else None
    )
    if snapshot:
        path = root / snapshot["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["sha256"]:
            raise ValueError("Source archive changed")
        with zipfile.ZipFile(path) as archive:
            for name, expected in manifest["code_files_sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError("Source archive does not match frozen code bytes")
            for track, expected in manifest["tracks"].items():
                if (
                    hashlib.sha256(archive.read(f"racing_lab/tracks/{track}.json")).hexdigest()
                    != expected
                ):
                    raise ValueError("Source archive does not match frozen track bytes")
    write_json(
        report / "validation.json",
        dict(
            status="passed",
            benchmark_hash=manifest["benchmark_hash"],
            completed_runs=len(records),
            raw_artifacts_verified=len(artifacts),
            checks=[
                "manifest/runtime/track hashes",
                "per-run conditions/seeds/budgets",
                "no external data or expert/controller queries",
                "uniform behavior and unchanged optimizer counts",
                "deterministic paired evaluation reset reconstruction",
                "training reset rule and physical-state reconstruction",
                "final checkpoint hashes",
                "finite losses, every trajectory shape and exact decision clock",
                "final checkpoint exactly reproduces recorded deterministic actions",
                "final evaluation alias counted once",
                "coverage counts include exactly the training decisions",
                "exact source archive code and track bytes",
            ],
            source_snapshot=snapshot,
        ),
    )
    return summary


def _figures(root, report, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        if not rows:
            continue
        histories, logs = [], []
        for row in rows:
            directory = root / row["directory"]
            files = [(0, directory / "evaluation_initial.json")] + [
                (int(path.stem.split("_")[-1]), path)
                for path in sorted(directory.glob("evaluation_0*.json"))
            ]
            if not any(step == row["training_transitions"] for step, _ in files):
                files.append((row["training_transitions"], directory / "evaluation_final.json"))
            histories.append([(step, _read(path)["rows"][0]) for step, path in files])
            with (directory / "learning.csv").open(encoding="utf-8") as file:
                logs.append(list(csv.DictReader(file)))
        for ax, metric in zip(axes[0], ("progress_laps", "valid_laps")):
            x = [step for step, _ in histories[0]]
            values = np.array([[row[metric] for _, row in history] for history in histories])
            mean, std = (
                values.mean(axis=0),
                values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(x)),
            )
            ax.plot(x, mean, marker="o", color=COLORS[condition], label=LABELS[condition])
            ax.fill_between(x, mean - std, mean + std, color=COLORS[condition], alpha=0.13)
            for value in values:
                ax.plot(x, value, color=COLORS[condition], alpha=0.2, linewidth=0.8)
        for ax, key in zip(axes[1], ("visited_training_bins", "maximum_episode_progress_laps")):
            x = [int(log["decision"]) for log in logs[0]]
            values = np.array([[float(log[key]) for log in sequence] for sequence in logs])
            if key == "visited_training_bins":
                values = values / 64 * 100
            ax.plot(x, values.mean(axis=0), color=COLORS[condition], label=LABELS[condition])
            ax.fill_between(
                x, values.min(axis=0), values.max(axis=0), color=COLORS[condition], alpha=0.13
            )
    for ax, title, ylabel in zip(
        axes.ravel(),
        (
            "Frozen evaluation: travelled lap fraction",
            "Frozen evaluation: strictly clean laps",
            "Training exposure: visited track bins",
            "Training: greatest episode progress so far",
        ),
        ("Lap progress", "Clean laps", "Visited bins (%)", "Lap progress"),
    ):
        ax.set(title=title, xlabel="Training decisions", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle(
        "TD3 diagnostic: three seeds per condition; upper bands = sample SD, lower bands = seed range"
    )
    for suffix in ("png", "svg"):
        figure.savefig(report / f"learning_curves.{suffix}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = [LABELS[condition] for condition in CONDITIONS]
    metrics = (
        "fixed_success",
        "varied_start_success_rate",
        "worker_wall_seconds",
        "fixed_braking_fraction_after_3s",
    )
    titles = (
        "Policies with a clean fixed-start lap",
        "Clean success on five paired Monza starts",
        "Per-worker wall time including evaluations",
        "Final applied brake >0.1 after first 3 seconds",
    )
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        for index, condition in enumerate(CONDITIONS):
            values = [
                row[metric]
                for row in records
                if row["condition"] == condition and row[metric] is not None
            ]
            if metric == "worker_wall_seconds":
                values = [value / 60 for value in values]
            if values:
                ax.bar(index, np.mean(values), color=COLORS[condition], alpha=0.55)
                ax.scatter(
                    np.full(len(values), index) + np.linspace(-0.1, 0.1, len(values)),
                    values,
                    color=COLORS[condition],
                    edgecolor="white",
                    zorder=3,
                )
        ax.set_xticks(range(4), labels, rotation=12)
        ax.set_title(title)
        ax.set_ylabel("Minutes" if metric == "worker_wall_seconds" else "Fraction")
        ax.grid(axis="y", alpha=0.2)
        if metric != "worker_wall_seconds":
            ax.set_ylim(0, 1.08)
    figure.suptitle("Final-budget outcomes and costs; dots represent individual trained policies")
    for suffix in ("png", "svg"):
        figure.savefig(report / f"outcomes_and_cost.{suffix}", dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    counts = [
        _read(root / row["directory"] / "result.json")["diagnostics"][
            "training_position_counts_64_bins"
        ]
        for row in records
    ]
    matrix = np.asarray(counts)
    image = ax.imshow(
        np.log10(1 + matrix),
        aspect="auto",
        cmap="viridis",
        extent=(0, 100, len(records) - 0.5, -0.5),
    )
    ax.set_yticks(
        range(len(records)), [f"{LABELS[row['condition']]} / seed {row['seed']}" for row in records]
    )
    ax.set(
        xlabel="Position along Monza centreline (%)",
        title="Self-generated training exposure; 64 bins (random spawning is not proof of lap-driving ability)",
    )
    figure.colorbar(image, ax=ax, label="log10(1 + post-step visits)")
    for suffix in ("png", "svg"):
        figure.savefig(report / f"training_coverage.{suffix}", dpi=160)
    plt.close(figure)


def _html(report, manifest, summary):
    rows = summary["per_seed"]

    def number(value, precision=2):
        return "—" if value is None else f"{value:.{precision}f}"

    table = "".join(
        f"<tr><td>{LABELS[row['condition']]}</td><td>{row['seed']}</td><td>{row['fixed_valid_laps']}/{row['fixed_completed_laps']}</td><td>{number(row['standing_start_valid_seconds'])}</td><td>{number(row['fixed_progress_laps'] * 100, 1)}%</td><td>{number(row['varied_start_success_rate'] * 100, 0)}%</td><td>{number(row['training_coverage_fraction'] * 100, 1)}%</td><td>{row['fixed_braking_decisions_after_3s']}</td><td>{number(row['worker_wall_seconds'] / 60, 1)}</td><td>{number(row['process_cpu_seconds'] / 60, 1)}</td><td>{number(row['peak_rss_bytes'] / 1048576, 1)}</td></tr>"
        for row in rows
    )
    protocol = html.escape(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "config",
                    "task",
                    "algorithms",
                    "conditions",
                    "adaptations",
                    "evaluation",
                    "runtime_versions",
                    "hardware",
                )
            },
            indent=2,
        )
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>TD3 diagnostic pilot</title>
<style>body{{margin:0;background:#f4f6fa;color:#18232e;font:16px/1.6 system-ui}}main{{max-width:1200px;margin:32px auto;padding:28px;background:white;border-radius:16px}}h1,h2{{line-height:1.25}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce1e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;font-size:12px}}a{{color:#087f8c}}</style>
<main><h1>TD3 diagnostic pilot on Monza</h1><p>Status: <strong>{summary["status"]}</strong>. {len(rows)} completed runs; {summary["total_training_decisions"]:,} training and {summary["total_evaluation_decisions"]:,} separately counted evaluation decisions. All policies start from fresh random initialization and empty replay. No expert dataset, pretrained model, controller queries, labels or trajectory-aided reward.</p>
<p>Four one-factor conditions, seeds 5–7, 50,000 decisions per run. Learning starts after 100 transitions in every condition. Random warmup changes behavior for 2,000 decisions; random starts remain stationary. Final-budget selection and all evaluations use identical standing-start Monza conditions and the common reward. Five additional paired lateral-jitter starts per policy are repeated tests of that policy, not independent seeds.</p>
<p>The learning recipe follows <a href="https://arxiv.org/abs/2402.18558">Evans et al., Unifying F1TENTH Autonomous Racing</a> and <a href="https://github.com/BDEvan5/f1tenth_benchmarks/tree/c9730789dd0fa85fc18dd4b600c5246e134ebe22">pinned author code</a>. Observation/control/simulator substitutions remain from the original common benchmark. CTH is the existing adapted code profile, with nonnegative clipping and 85 m/s scaling; no original-paper score reproduction is claimed.</p>
<h2>Learning and exposure</h2><img src="learning_curves.png" alt="Frozen evaluation and training exposure curves"><p>Upper panels: mean and sample SD across trained policies, with individual seed traces. Lower panels: mean and seed range. Training exposure counts projected post-step positions, including off-road poses; broad random-start exposure does not establish successful driving.</p>
<img src="training_coverage.png" alt="Track exposure per seed"><h2>Final outcomes and costs</h2><img src="outcomes_and_cost.png" alt="Reliability, costs and braking">
<div class="scroll"><table><tr><th>Condition</th><th>Seed</th><th>Clean/completed laps</th><th>Standing clean seconds</th><th>Fixed progress</th><th>Varied-start success</th><th>Training coverage</th><th>Brake decisions after 3s</th><th>Worker min</th><th>CPU min</th><th>Peak MiB</th></tr>{table}</table></div>
<p>Validity requires every gate and zero vehicle-envelope off-road decision steps, counted collisions or recovery. Missing lap times remain missing. Reliability must be assessed before conditional lap times. Three seeds provide limited precision; this pilot does not isolate interactions or establish algorithm superiority.</p>
<p>Costs include training and evaluations. Concurrent worker timings include contention; their wall times must not be added and called elapsed queue duration. RSS is process lifetime peak including imports. Framework startup is included in queue duration but excluded from per-worker meters. Reporting/controller cost is stored in completion.json. Sample presentations count replay reuse, not new data.</p>
<p><a href="per_seed.csv">All per-seed metrics</a> · <a href="summary.json">Aggregates and paired differences</a> · <a href="costs.json">Compute, data and capacity ledger</a> · <a href="validation.json">Validation</a> · <a href="../source_snapshot.zip">Frozen source</a></p>
<h2>Frozen protocol</h2><p>Manifest hash: <code>{summary["benchmark_hash"]}</code></p><pre>{protocol}</pre></main></html>"""
    (report / "report.html").write_text(content, encoding="utf-8")
