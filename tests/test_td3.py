"""Numerical parity with licensed author code and no-expert online integration."""

# ruff: noqa: E402
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from racing_lab.rl.config import RacingTaskConfig, SACConfig
from racing_lab.rl.online import RewardVariantEnv, train_online
from racing_lab.rl.sac import SACAgent
from racing_lab.rl.td3 import TD3Agent, TD3Config
from racing_lab.rl.training import evaluate_sac, train_sac

FIXTURE = Path(__file__).parent / "fixtures" / "td3_author"


def author_module(monkeypatch):
    for name in ("f1tenth_benchmarks", "f1tenth_benchmarks.drl_racing"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("training_utils", "td3"):
        qualified = "f1tenth_benchmarks.drl_racing." + name
        spec = importlib.util.spec_from_file_location(qualified, FIXTURE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
    return module


def test_author_fixture_hashes():
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest


def test_three_successive_calls_match_author_initialization_and_updates(monkeypatch):
    original_module = author_module(monkeypatch)
    ours = TD3Agent(7, 3, seed=42)
    torch.manual_seed(42)
    original = original_module.TrainTD3(7, 3)
    pairs = [
        (getattr(ours, name), getattr(original, name))
        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
        )
    ]
    for current, source in pairs:
        for a, b in zip(current.parameters(), source.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    rng = np.random.default_rng(8)
    batches = []
    for _ in range(6):
        states = rng.normal(size=(100, 7)).astype(np.float32)
        actions = rng.uniform(-1, 1, (100, 3)).astype(np.float32)
        rewards = rng.normal(size=(100, 1)).astype(np.float32)
        successors = rng.normal(size=(100, 7)).astype(np.float32)
        alive = rng.integers(0, 2, (100, 1)).astype(np.float32)
        batches.append((states, actions, rewards, successors, alive * np.float32(0.99)))
    for call in range(3):
        selected = iter(batches[call * 2 : call * 2 + 2])
        torch_state = torch.get_rng_state().clone()
        ours.update(lambda: next(selected))
        original_batches = iter(batches[call * 2 : call * 2 + 2])

        class Buffer:
            def size(self):
                return 100

            def sample(self, count):
                state, action, reward, next_state, discount = next(original_batches)
                return tuple(
                    torch.as_tensor(x)
                    for x in (state, action, next_state, reward, discount / np.float32(0.99))
                )

        original.replay_buffer = Buffer()
        torch.set_rng_state(torch_state)
        original.train()
        for current, source in pairs:
            for a, b in zip(current.parameters(), source.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert ours.updates == 6 and ours.actor_updates == 3 and ours.training_calls == 3


def test_deterministic_checkpoint_and_evaluation_are_frozen(tmp_path):
    agent = TD3Agent(168, seed=4)
    path = tmp_path / "model.pt"
    agent.save(path)
    restored, _ = TD3Agent.load(path)
    state = np.zeros(168, np.float32)
    before = torch.get_rng_state().clone()
    exploration_before = json.dumps(agent.rng.bit_generator.state)
    np.testing.assert_array_equal(
        agent.act(state, deterministic=True), restored.act(state, deterministic=True)
    )
    report = evaluate_sac(agent, task=RacingTaskConfig(max_seconds=0.08))
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    assert json.dumps(agent.rng.bit_generator.state) == exploration_before
    assert report["gradient_updates"] == agent.updates == 0
    assert report["evaluation_decisions"] == 2


def test_online_accounting_and_sac_runner_learning_parity(tmp_path, monkeypatch):
    import racing_lab.data

    monkeypatch.setattr(
        racing_lab.data, "load", lambda *args: pytest.fail("Expert data was accessed")
    )
    task = RacingTaskConfig(max_seconds=0.08)
    sac = replace(
        SACConfig(),
        hidden_units=(8, 8, 8),
        batch_size=4,
        start_steps=5,
        replay_capacity=32,
        eval_interval=10,
        checkpoint_interval=10,
    )
    old, new = tmp_path / "old", tmp_path / "new"
    train_sac(old, steps=20, seed=2, config=sac, task=task, progress=False)
    result = train_online(new, steps=20, seed=2, config=sac, task=task, progress=False)
    first, _ = SACAgent.load(old / "final.pt")
    second, _ = SACAgent.load(new / "final.pt")
    for a, b in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert result["unique_training_transitions"] == result["decisions"] == 20
    assert result["updates"] == result["actor_updates"] == result["entropy_updates"] == 15
    assert result["sampled_transition_presentations"] == 60
    assert result["evaluation_decisions"] == 6  # initial, 10, 20; final reuses 20
    assert result["resources"]["process_cpu_seconds"] > 0
    td3 = replace(
        TD3Config(),
        hidden_units=(8, 8),
        batch_size=4,
        start_steps=4,
        replay_capacity=8,
        eval_interval=0,
    )
    result = train_online(
        tmp_path / "td3", algorithm="td3", steps=20, seed=2, config=td3, task=task, progress=False
    )
    assert result["updates"] == 32 and result["actor_updates"] == 16
    assert result["replay_size"] == 8
    assert result["random_action_decisions"] == result["entropy_updates"] == 0
    assert result["expert_controller_queries"] == result["external_dataset_transitions"] == 0


def test_progress_reward_units_and_cth_geometry():
    task = RacingTaskConfig(max_seconds=0.08)
    env = RewardVariantEnv(config=task, reward_profile="progress")
    env.reset(seed=0)
    before = env.env.progress
    _, reward, *_ = env.step([0, 1, 0])
    assert reward == pytest.approx(100 * (env.env.progress - before) / env.env.track.length)
    env = RewardVariantEnv(config=task, reward_profile="cth")
    env.reset(seed=0)
    _, reward, *_ = env.step([0, 1, 0])
    assert reward >= 0 and np.isfinite(reward)


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": float("nan")},
        {"start_steps": 0},
        {"replay_capacity": 1},
        {"n_step": 3},
        {"exploration_noise": -1},
    ],
)
def test_invalid_td3_configuration(changes):
    with pytest.raises(ValueError):
        replace(TD3Config(), **changes)
