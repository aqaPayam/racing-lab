"""One-factor isolation, unchanged learning during warmup and honest scoring."""

# ruff: noqa: E402
from dataclasses import asdict, replace
import json
import zipfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
torch.set_num_threads(1)

from racing_lab.rl.config import RacingTaskConfig, config_hash
from racing_lab.rl.diagnostic import (
    DiagnosticConfig,
    controller_lock,
    make_diagnostic_manifest,
    run_diagnostic_worker,
    _source_snapshot,
)
from racing_lab.rl.diagnostic_report import collect_diagnostic, generate_diagnostic_report
from racing_lab.rl.benchmark_report import collect_runs
from racing_lab.rl.online import train_online
from racing_lab.rl.td3 import TD3Config
from racing_lab.rl.training import write_json


def small_config():
    return replace(
        TD3Config(),
        hidden_units=(8, 8),
        batch_size=4,
        replay_capacity=8,
        start_steps=4,
        eval_interval=0,
        checkpoint_interval=0,
    )


def test_conditions_change_only_one_factor_and_do_not_alias_global_definitions():
    manifest = make_diagnostic_manifest(DiagnosticConfig())
    baseline = manifest["conditions"]["baseline"]
    assert len(manifest["runs"]) == 12
    for label, profile in manifest["conditions"].items():
        assert sum(profile[key] != baseline[key] for key in baseline) == (label != "baseline")
    manifest["conditions"]["uniform_warmup"]["uniform_warmup_decisions"] = 5
    assert (
        make_diagnostic_manifest(DiagnosticConfig())["conditions"]["uniform_warmup"][
            "uniform_warmup_decisions"
        ]
        == 2000
    )
    with pytest.raises(ValueError, match="Monza only"):
        DiagnosticConfig(held_out_tracks=("spa",))


def test_behavior_warmup_learns_at_unchanged_threshold_and_counts_random_actions(tmp_path):
    result = train_online(
        tmp_path / "warmup",
        algorithm="td3",
        steps=20,
        seed=5,
        config=small_config(),
        task=RacingTaskConfig(max_seconds=0.08),
        progress=False,
        uniform_warmup_decisions=12,
        capture_diagnostics=True,
    )
    assert result["random_action_decisions"] == 12
    assert result["updates"] == 32 and result["actor_updates"] == 16
    assert (
        result["diagnostics"]["training_position_counts_64_bins"]
        and sum(result["diagnostics"]["training_position_counts_64_bins"]) == 20
    )
    metadata = json.loads((tmp_path / "warmup/run.json").read_text())
    assert metadata["config"]["start_steps"] == 4


def test_diagnostic_observation_logging_preserves_baseline_learning(tmp_path):
    arguments = dict(
        algorithm="td3",
        steps=20,
        seed=5,
        config=small_config(),
        task=RacingTaskConfig(max_seconds=0.08),
        progress=False,
    )
    train_online(tmp_path / "plain", **arguments)
    train_online(tmp_path / "diagnostic", capture_diagnostics=True, **arguments)
    first = torch.load(tmp_path / "plain/final.pt", weights_only=True)
    second = torch.load(tmp_path / "diagnostic/final.pt", weights_only=True)
    for name in (
        "actor",
        "critic_1",
        "critic_2",
        "actor_target",
        "critic_target_1",
        "critic_target_2",
    ):
        for key in first[name]:
            torch.testing.assert_close(first[name][key], second[name][key], rtol=0, atol=0)


def test_all_conditions_validated_and_corruption_rejected(tmp_path):
    manifest = make_diagnostic_manifest(
        DiagnosticConfig(steps=8, seeds=(5,), eval_interval=4, varied_start_episodes=2)
    )
    manifest["task"] = asdict(RacingTaskConfig(max_seconds=0.08))
    manifest["algorithms"]["td3"] = asdict(
        replace(small_config(), eval_interval=4, checkpoint_interval=8)
    )
    manifest["benchmark_hash"] = config_hash(
        {key: value for key, value in manifest.items() if key != "benchmark_hash"}
    )
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    for index in range(4):
        run_diagnostic_worker(path, index)
    summary = generate_diagnostic_report(tmp_path)
    assert summary["status"] == "completed" and summary["total_training_decisions"] == 32
    for row in summary["per_seed"]:
        assert row["evaluation_transitions"] == 10  # initial2 + periodic4 + two robustness starts4
        assert row["twin_critic_optimizer_steps"] == 8
        assert row["standing_start_valid_seconds"] is None
    with pytest.raises(ValueError, match="separate report"):
        collect_runs(tmp_path)
    directory = tmp_path / "random_starts-seed5"
    resets = json.loads((directory / "training_resets.json").read_text())
    assert len({row["start_distance_metres"] for row in resets}) > 1
    original = (directory / "training_resets.json").read_bytes()
    resets[0]["start_distance_metres"] += 1
    write_json(directory / "training_resets.json", resets)
    with pytest.raises(ValueError, match="sampling rule"):
        collect_diagnostic(tmp_path)
    (directory / "training_resets.json").write_bytes(original)
    result = json.loads((directory / "result.json").read_text())
    result["diagnostics"]["training_position_counts_64_bins"][0] += 1
    write_json(directory / "result.json", result)
    with pytest.raises(ValueError, match="coverage count"):
        collect_diagnostic(tmp_path)


def test_second_controller_cannot_acquire_live_queue(tmp_path):
    with controller_lock(tmp_path):
        with pytest.raises((RuntimeError, BlockingIOError)):
            with controller_lock(tmp_path):
                pytest.fail("Two live controllers acquired the same queue")
    with controller_lock(tmp_path):
        pass


def test_installed_package_can_save_exact_source_without_git(tmp_path):
    package = tmp_path / "installed/racing_lab"
    package.mkdir(parents=True)
    (package / "module.py").write_bytes(b"example = 1\r\n")
    (package / "tracks").mkdir()
    (package / "tracks/monza.json").write_bytes(b"{}\r\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/module.pyc").write_bytes(b"ignored")
    output = tmp_path / "report"
    output.mkdir()
    snapshot = _source_snapshot(output, workspace=package.parent)
    with zipfile.ZipFile(output / snapshot["file"]) as archive:
        assert archive.read("racing_lab/module.py") == b"example = 1\r\n"
        assert archive.read("racing_lab/tracks/monza.json") == b"{}\r\n"
        assert len(archive.namelist()) == 2


@pytest.mark.parametrize(
    "extra", [dict(uniform_warmup_decisions=-1), dict(random_training_starts=1)]
)
def test_invalid_behavior_flags_fail_before_creating_run(tmp_path, extra):
    with pytest.raises(ValueError):
        train_online(tmp_path / "invalid", algorithm="td3", **extra)
    assert not (tmp_path / "invalid").exists()
