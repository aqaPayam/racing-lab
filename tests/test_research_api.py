import copy
import json

import numpy as np
import pytest

from racing_lab import (
    RacingEnv,
    Recorder,
    load_trajectory,
    transitions,
    EvaluationConfig,
    evaluate_policy,
    save_report,
)
from racing_lab.data import validate
from racing_lab import feature_sum
from racing_lab.learning import train


def test_snapshot_branches_are_identical_and_independent():
    env = RacingEnv()
    env.reset(seed=17, options={"lateral_jitter": 0.5})
    for _ in range(30):
        env.step([0.1, 0.5, 0])
    snapshot = env.snapshot()
    first = env.step([-0.3, 0.8, 0])
    final = env.snapshot()
    env.restore(snapshot)
    second = env.step([-0.3, 0.8, 0])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert env.progress == final["state"]["progress"]
    env.pos[0] += 50
    assert not np.array_equal(env.pos, snapshot["state"]["pos"])
    other = RacingEnv("spa")
    other.reset()
    with pytest.raises(ValueError):
        other.restore(snapshot)


def test_record_actual_action_and_validate_corruption(tmp_path):
    env = RacingEnv()
    env.reset(seed=4)
    rec = Recorder(env, source="test")
    action = [0, 2, -1]
    rec.append(action, env.step(action))
    path = rec.save(tmp_path, reason="test")
    data = load_trajectory(path)
    assert data["telemetry"].shape == (1, 5)
    assert data["metadata"]["dynamics_parameters"]["mass"] == 800
    np.testing.assert_array_equal(data["actions"], [env.action])
    obs, act, nxt = transitions(data)
    assert obs.shape == nxt.shape == (1, 20)
    assert data["metadata"]["seed"] == 4
    weights = np.array(list(env.weights.values()))
    assert feature_sum(data) @ weights == pytest.approx(data["rewards"].sum())
    broken = copy.deepcopy(data)
    broken["observations"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate(broken)
    broken = copy.deepcopy(data)
    broken["rewards"][0] += 100
    with pytest.raises(ValueError):
        validate(broken)
    with pytest.raises(ValueError):
        rec.append(action, (obs, 0, False, False, {}))


def test_evaluation_keeps_failed_laps_out_of_time_summary(tmp_path):
    class Stationary:
        resets = 0

        def reset(self):
            self.resets += 1

        def __call__(self, obs):
            return [0, 0, 0]

    policy = Stationary()
    config = EvaluationConfig(episodes=2, max_seconds=0.1, seed=3)
    report = evaluate_policy(policy, config, policy_name="stationary")
    assert policy.resets == 2
    assert report["summary"]["monza"]["success_rate"] == 0
    assert report["summary"]["monza"]["lap_time_mean"] is None
    assert [r["seed"] for r in report["episodes"]] == [3, 4]
    assert all(r["truncated"] for r in report["episodes"])
    save_report(report, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["report_version"] == 1


def test_duplicate_demonstrations_rejected(tmp_path):
    for _ in range(2):
        env = RacingEnv()
        env.reset(seed=0)
        rec = Recorder(env)
        for _ in range(3):
            rec.append([0, 1, 0], env.step([0, 1, 0]))
        rec.save(tmp_path, "good")
    with pytest.raises(ValueError, match="Duplicate"):
        train(tmp_path, tmp_path / "models" / "bc.npz", epochs=1)


@pytest.mark.parametrize(
    "kwargs",
    [{"episodes": 0}, {"tracks": ()}, {"lateral_jitter": float("nan")}, {"max_seconds": 0}],
)
def test_invalid_evaluation_configuration(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)
