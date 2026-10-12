"""Frozen protocols, reporting integrity and independent-seed aggregation."""

# ruff: noqa: E402
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from racing_lab.rl.benchmark import BenchmarkConfig, make_manifest, run_worker, validate_manifest
from racing_lab.rl.benchmark_report import descriptive, generate_report
from racing_lab.rl.config import RacingTaskConfig, config_hash
from racing_lab.rl.resources import ResourceMeter, hardware
from racing_lab.rl.training import write_json


def tiny_manifest():
    manifest = make_manifest(
        BenchmarkConfig(steps=4, seeds=(0,), held_out_tracks=(), varied_start_episodes=2)
    )
    manifest["task"] = vars(RacingTaskConfig(max_seconds=0.08))
    manifest["benchmark_hash"] = config_hash(
        {k: v for k, v in manifest.items() if k != "benchmark_hash"}
    )
    return manifest


def test_manifest_is_frozen_and_family_split_prevents_leakage():
    manifest = make_manifest(BenchmarkConfig())
    validate_manifest(manifest)
    manifest["config"]["steps"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match="families"):
        BenchmarkConfig(held_out_tracks=("monza_long",))
    with pytest.raises(ValueError, match="distinct"):
        BenchmarkConfig(seeds=(0, 0))


def test_two_fresh_workers_report_counts_and_detect_modified_artifacts(tmp_path):
    manifest = tiny_manifest()
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    for index in range(2):
        run_worker(path, index)
    summary = generate_report(tmp_path)
    assert summary["status"] == "completed"
    assert summary["aggregate"]["sac"]["fixed_success"]["n"] == 1
    for row in summary["per_seed"]:
        assert row["training_transitions"] == 4
        assert row["evaluation_transitions"] == 8  # initial/final2 + two varied starts2
        assert row["total_transitions"] == 12
        assert row["standing_start_valid_seconds"] is None
    assert (tmp_path / "report" / "learning_curves.png").is_file()
    robustness = tmp_path / manifest["runs"][0]["directory"] / "evaluation_robustness.json"
    original = robustness.read_bytes()
    modified = json.loads(original)
    modified["rows"][0]["seed"] += 50
    write_json(robustness, modified)
    with pytest.raises(ValueError, match="different seeds"):
        generate_report(tmp_path)
    robustness.write_bytes(original)
    robustness.rename(robustness.with_suffix(".missing"))
    with pytest.raises(ValueError, match="missing required"):
        generate_report(tmp_path)
    robustness.with_suffix(".missing").rename(robustness)
    checkpoint = tmp_path / manifest["runs"][0]["directory"] / "final.pt"
    with checkpoint.open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="checkpoint changed"):
        generate_report(tmp_path)


def test_seed_statistics_do_not_turn_missing_laps_into_times():
    stats = descriptive([None, 50.0, 52.0, None])
    assert stats["n"] == 2 and stats["mean"] == 51.0
    assert stats["std"] == pytest.approx(np.sqrt(2))
    assert descriptive([None])["mean"] is None


def test_resource_scopes_are_nonnegative_and_hardware_is_explicit():
    meter = ResourceMeter()
    with meter.measure("work"):
        sum(range(100))
    report = meter.report()
    assert report["wall_seconds"] >= report["component_seconds"]["work"] >= 0
    assert report["process_cpu_seconds"] >= 0
    assert report["memory"]["peak_rss_bytes"] is None or report["memory"]["peak_rss_bytes"] > 0
    assert "physical_memory_bytes" in hardware()
