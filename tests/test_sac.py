"""Numerical source parity, episode-safe returns, and scratch-only integration."""

# Optional dependency must be checked before importing modules that use PyTorch.
# ruff: noqa: E402

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from racing_lab import RacingEnv
from racing_lab.data import Recorder, load
from racing_lab.rl import RacingTaskConfig, SACConfig, SACRacingEnv
from racing_lab.rl.environment import SAC_OBSERVATION_NAMES
from racing_lab.rl.replay import NStepReplay
from racing_lab.rl.sac import SACAgent
from racing_lab.rl.training import evaluate_sac, train_sac

FIXTURE = Path(__file__).parent / "fixtures" / "sac_author"


def test_author_source_snapshot_hashes():
    manifest = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / "discor" / name).read_bytes()).hexdigest() == expected


def test_network_and_three_updates_match_pinned_author_source(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURE))
    from discor.algorithm.sac import SAC as OriginalSAC

    class Writer:
        def add_scalar(self, *args):
            pass

    config = SACConfig()
    ours = SACAgent(5, config=config, seed=32)
    original = OriginalSAC(
        5,
        3,
        torch.device("cpu"),
        gamma=config.gamma,
        nstep=3,
        policy_hidden_units=list(config.hidden_units),
        q_hidden_units=list(config.hidden_units),
        seed=32,
    )
    modules = [
        (ours.actor, original._policy_net),
        (ours.critic, original._online_q_net),
        (ours.target_critic, original._target_q_net),
    ]
    for ours_module, original_module in modules:
        for a, b in zip(ours_module.parameters(), original_module.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    rng = np.random.default_rng(91)
    states = rng.normal(size=(128, 5)).astype(np.float32)
    actions = rng.uniform(-1, 1, (128, 3)).astype(np.float32)
    rewards = rng.normal(size=(128, 1)).astype(np.float32)
    next_states = rng.normal(size=(128, 5)).astype(np.float32)
    dones = rng.integers(2, size=(128, 1)).astype(np.float32)
    for i in range(3):
        torch.manual_seed(900 + i)
        ours.update((states, actions, rewards, next_states, (1 - dones) * config.gamma**3))
        torch.manual_seed(900 + i)
        original.update_online_networks(
            tuple(torch.tensor(x) for x in (states, actions, rewards, next_states, dones)), Writer()
        )
        original.update_target_networks()
        for ours_module, original_module in modules:
            for a, b in zip(ours_module.parameters(), original_module.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(ours.log_alpha, original._log_alpha, rtol=1e-6, atol=1e-7)


def test_full_and_terminal_returns_match_author_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURE))
    from discor.replay_buffer import ReplayBuffer

    ours = NStepReplay(100, 1)
    original = ReplayBuffer(100, (1,), (3,), gamma=0.992, nstep=3)
    for i in range(5):
        values = (np.array([i]), np.array([0, 0.1, 0.2]), float(i + 1), np.array([i + 1]))
        terminated = i == 4
        ours.append(*values, terminated)
        original.append(*values, terminated, episode_done=terminated)
    states, actions, rewards, next_states, discounts = ours.get(np.arange(5))
    np.testing.assert_array_equal(states, original._states[:5])
    np.testing.assert_array_equal(actions, original._actions[:5])
    np.testing.assert_allclose(rewards, original._rewards[:5])
    np.testing.assert_array_equal(next_states, original._next_states[:5])
    np.testing.assert_allclose(discounts, (1 - original._dones[:5]) * 0.992**3)


def test_truncated_returns_bootstrap_at_correct_horizon_and_do_not_cross_reset():
    replay = NStepReplay(8, 1, gamma=0.9, n_step=3)
    for i, reward in enumerate([1, 2]):
        replay.append([i], [0, 0, 0], reward, [i + 1], False, truncated=i == 1)
    replay.append([100], [0, 0, 0], 10, [101], True)
    states, _, rewards, next_states, discounts = replay.get([0, 1, 2])
    np.testing.assert_array_equal(states[:, 0], [0, 1, 100])
    np.testing.assert_allclose(rewards[:, 0], [2.8, 2, 10])
    np.testing.assert_array_equal(next_states[:, 0], [2, 2, 101])
    np.testing.assert_allclose(discounts[:, 0], [0.81, 0.9, 0])
    assert not replay.pending


def test_lazy_replay_capacity_wraparound_retains_last_transitions():
    replay = NStepReplay(5, 1, n_step=1, chunk_size=2)
    assert replay.chunks == {}
    for i in range(12):
        replay.append([i], [0, 0, 0], 0, [i + 1], False)
    assert len(replay) == 5
    assert set(replay.get(np.arange(5))[0][:, 0]) == set(range(7, 12))
    assert len(replay.chunks) == 3


def test_adapter_controls_reward_history_and_exact_clock():
    env = SACRacingEnv(config=RacingTaskConfig(max_seconds=1))
    obs, _ = env.reset(seed=3)
    assert obs.shape == (len(SAC_OBSERVATION_NAMES),) == (168,)
    assert np.all(obs[39:] == 0)
    previous = env.current.copy()
    obs, reward, _, _, info = env.step([1, 1, 0])
    assert env.env.action[0] == pytest.approx((600 / 180) / 25)
    assert env.env.action[1] == pytest.approx(0.24)
    assert env.env.action[2] == 0
    assert info["elapsed"] == pytest.approx(0.04)
    gap = abs(env.env.track.project(env.env.pos)[1])
    assert reward == pytest.approx(
        3.6 * np.hypot(env.env.speed, env.env.lateral_speed) * (1 - gap / 12) / 300
    )
    assert np.all(obs[39:48] == 0)
    env.step([0, 1, 0])
    obs, *_ = env.step([0, 1, 0])
    np.testing.assert_array_equal(obs[51:90], previous)
    env.reset(seed=3)
    assert len(env.history) == 0
    np.testing.assert_array_equal(env.targets, [0, -1, -1])


def test_paper_failure_and_time_limit_are_distinct():
    env = SACRacingEnv(config=RacingTaskConfig(max_seconds=3))
    env.reset()
    for _ in range(51):
        _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated
    assert info["termination_reason"] == "low_speed"
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0])
    env = SACRacingEnv(config=RacingTaskConfig(max_seconds=0.08))
    env.reset()
    env.step([0, 0, 0])
    assert env.step([0, 0, 0])[2:4] == (False, True)


def test_multilap_adapter_keeps_running_and_does_not_count_contact_lap_as_valid():
    env = SACRacingEnv(config=RacingTaskConfig(max_seconds=10))
    env.reset()
    env.env.progress = env.env.track.length + 1
    env.env.next_checkpoint = 41
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert info["completed_laps"] == info["valid_laps"] == 1
    assert not terminated and not truncated
    env.env.progress = 2 * env.env.track.length + 1
    env.env.next_checkpoint = 81
    env.env.collisions = 1
    assert env.step([0, 0, 0])[4]["valid_laps"] == 1


def test_variable_clock_snapshot_and_recording_roundtrip(tmp_path):
    env = RacingEnv(dt=0.04, laps=2)
    env.reset(seed=7)
    rec = Recorder(env)
    snapshot = env.snapshot()
    first = env.step([0, 0.1, 0])
    rec.append([0, 0.1, 0], first)
    recording = load(rec.save(tmp_path))
    assert recording["metadata"]["dt"] == 0.04
    np.testing.assert_allclose(recording["timestamps"], [0, 0.04])
    env.restore(snapshot)
    np.testing.assert_array_equal(env.step([0, 0.1, 0])[0], first[0])
    other = RacingEnv(dt=0.02, max_seconds=150, laps=2)
    other.reset()
    with pytest.raises(ValueError):
        other.restore(snapshot)


def test_checkpoint_inference_and_frozen_evaluation(tmp_path):
    config = replace(SACConfig(), hidden_units=(8, 8, 8))
    env = SACRacingEnv()
    observation, _ = env.reset()
    agent = SACAgent(len(observation), config=config, seed=9)
    path = tmp_path / "policy.pt"
    agent.save(path, {"task": {}})
    restored, _ = SACAgent.load(path)
    np.testing.assert_array_equal(
        restored.act(observation, deterministic=True), agent.act(observation, deterministic=True)
    )
    before = [p.detach().clone() for p in agent.actor.parameters()]
    rng = torch.get_rng_state().clone()
    report = evaluate_sac(agent, task=RacingTaskConfig(max_seconds=0.12))
    assert report["gradient_updates"] == agent.updates == 0
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for previous, current in zip(before, agent.actor.parameters()):
        torch.testing.assert_close(previous, current, rtol=0, atol=0)
    assert report["rows"][0]["decisions"] == 3


def test_scratch_training_artifacts_updates_and_reproducibility(tmp_path, monkeypatch):
    import racing_lab.data

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not load demonstrations")

    monkeypatch.setattr(racing_lab.data, "load", forbidden)
    config = replace(
        SACConfig(),
        hidden_units=(8, 8, 8),
        batch_size=4,
        start_steps=5,
        replay_capacity=32,
        eval_interval=10,
        checkpoint_interval=10,
    )
    task = RacingTaskConfig(max_seconds=0.16)
    results = []
    for name in ("first", "second"):
        output = tmp_path / name
        result = train_sac(output, steps=20, seed=2, config=config, task=task, progress=False)
        assert result["decisions"] == 20
        assert result["updates"] > 0
        assert result["random_action_decisions"] == 5
        assert not result["expert_dataset_used"]
        assert result["simulated_seconds"] == 0.8
        assert (output / "step_000000010.pt").is_file()
        assert (output / "evaluation_000000010.json").is_file()
        manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert manifest["replay_initial_size"] == 0
        assert manifest["data_source"] == "online_self_generated"
        results.append(SACAgent.load(output / "final.pt")[0])
    for a, b in zip(results[0].actor.parameters(), results[1].actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="already exists"):
        train_sac(tmp_path / "first", steps=1)


def test_base_import_and_cli_help_do_not_import_torch():
    code = (
        "import sys; import racing_lab; from racing_lab import rl_commands; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "failing_call, phase", [(1, "initial_evaluation"), (2, "final_evaluation")]
)
def test_evaluation_failure_cannot_claim_completed_training(
    tmp_path, monkeypatch, failing_call, phase
):
    import racing_lab.rl.training as training

    evaluate = training.evaluate_sac
    calls = 0

    def sometimes_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise RuntimeError("Evaluation failed")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate_sac", sometimes_fails)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        train_sac(
            output,
            steps=2,
            config=replace(SACConfig(), hidden_units=(8, 8, 8)),
            task=RacingTaskConfig(max_seconds=0.08),
            progress=False,
        )
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["phase"] == phase
    assert result["decisions"] == (0 if failing_call == 1 else 2)
    if failing_call == 2:
        _, metadata = SACAgent.load(output / "final.pt")
        assert metadata["result"]["status"] == "failed"


def test_evaluation_reports_fixed_repetitions_and_real_reset_variation():
    agent = SACAgent(168, config=replace(SACConfig(), hidden_units=(8, 8, 8)))
    task = RacingTaskConfig(max_seconds=0.04)
    fixed = evaluate_sac(agent, task=task, episodes=2)
    assert fixed["trials"] == "deterministic repetitions"
    assert len({row["reset_state_sha256"] for row in fixed["rows"]}) == 1
    varied = evaluate_sac(agent, task=task, episodes=2, lateral_jitter=1)
    assert varied["trials"] == "varied starts"
    assert len({row["reset_state_sha256"] for row in varied["rows"]}) == 2


def test_code_provenance_does_not_depend_on_callers_working_directory(tmp_path, monkeypatch):
    from racing_lab.rl.training import _versions

    expected = _versions()
    monkeypatch.chdir(tmp_path)
    actual = _versions()
    assert actual["code_sha256"] == expected["code_sha256"]
    assert actual["git_commit"] == expected["git_commit"]


@pytest.mark.parametrize(
    "changes",
    [
        {"n_step": 2},
        {"gamma": 1},
        {"tau": float("nan")},
        {"replay_capacity": 1},
        {"start_steps": -1},
    ],
)
def test_invalid_learning_configuration(changes):
    with pytest.raises(ValueError):
        replace(SACConfig(), **changes)
