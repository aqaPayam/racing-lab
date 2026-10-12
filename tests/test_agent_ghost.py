import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import asdict
import numpy as np
import pytest
from racing_lab import RacingEnv, Recorder
from racing_lab.agent_ghost import AgentGhost, GhostRaceStart, comparison_report
from racing_lab.practice import comparison_html, PracticeBook
from racing_lab.vehicle import SIMULATOR_VERSION


def reference_for(env):
    initial = env.state()
    states = np.array([initial, initial, initial])
    states[1, 2], states[2, 2] = np.pi - 0.1, -np.pi + 0.1
    states[1:, :2] = env.track.sample([1, 2])
    identity = dict(
        dt=env.dt,
        track_sha256=env.track.digest,
        simulator_version=SIMULATOR_VERSION,
        driving_config=asdict(env.driving),
        selection="first_clean_standing_start_lap",
        checkpoint_sha256="test-fixture",
    )
    return AgentGhost(states, np.arange(3) * env.dt, [0, 1, 2], np.zeros((2, 3)), identity)


def test_time_based_replay_heading_finish_and_cache(tmp_path):
    env = RacingEnv(dt=0.04)
    env.reset(seed=0)
    ghost = reference_for(env)
    ghost.validate_player(env)
    assert ghost.ghost(-1) is None
    pose = ghost.ghost(0.06)
    assert abs(pose[2]) == pytest.approx(np.pi)
    np.testing.assert_array_equal(ghost.ghost(10), ghost.states[-1])
    pose[0] = 999
    assert ghost.states[1, 0] != 999
    path = tmp_path / "reference.npz"
    ghost.save(path)
    restored = AgentGhost.load(path, ghost.identity)
    np.testing.assert_array_equal(restored.states, ghost.states)
    with pytest.raises(ValueError, match="checkpoint"):
        AgentGhost.load(path, dict(ghost.identity, checkpoint_sha256="other"))
    with pytest.raises(ValueError, match="same track"):
        other = RacingEnv(dt=1 / 60)
        other.reset()
        ghost.validate_player(other)


def test_equal_progress_delta_uses_first_passage_and_does_not_extrapolate():
    env = RacingEnv(dt=0.04)
    env.reset()
    ghost = reference_for(env)
    assert ghost.delta(1, 0.06) == pytest.approx(0.02)
    assert ghost.delta(1, 0.02) == pytest.approx(-0.02)
    assert ghost.delta(3, 0.2) is None
    assert ghost.delta(-1, 0) is None
    states = np.repeat(env.state()[None], 5, axis=0)
    reverse = AgentGhost(
        states, np.arange(5) * env.dt, [0, 1, 0.5, 1, 2], np.zeros((4, 3)), ghost.identity
    )
    assert reverse.delta(1, 0.06) == pytest.approx(0.02)


def test_countdown_stalls_and_shared_clock():
    race = GhostRaceStart()
    assert race.phase == "ready"
    race.start()
    assert not race.advance(10)  # A window/debugger stall advances neither car.
    assert race.remaining == 3
    for _ in range(179):
        assert not race.advance(1 / 60)
    assert race.advance(1 / 60)
    assert race.phase == "racing"
    assert not race.advance(1 / 60)
    race.start()
    assert race.remaining == 3


def test_dirty_incomplete_laps_have_no_finish_time_or_delta(tmp_path):
    env = RacingEnv(dt=0.04)
    env.reset()
    ghost = reference_for(env)
    rec = Recorder(env, source="synthetic-test")
    action = [0, 0.5, 0]
    rec.append(action, env.step(action))
    report = comparison_report(rec, ghost, tmp_path / "run.npz", "user_stop")
    assert report["lap_comparison"]["human_lap_seconds"] is None
    assert report["lap_comparison"]["human_minus_sac_seconds"] is None
    env.progress = env.track.length
    env.next_checkpoint = 41
    assert comparison_report(rec, ghost, "test", "completed")["lap_comparison"]["human_valid"]
    for field in ("collisions", "offtrack_steps", "recoveries"):
        setattr(env, field, 1)
        dirty = comparison_report(rec, ghost, "test", "completed")
        assert not dirty["lap_comparison"]["human_valid"]
        assert dirty["lap_comparison"]["human_lap_seconds"] is None
        assert "No valid completed lap" in comparison_html(dirty)
        setattr(env, field, 0)


def test_overlays_are_passive_and_timestep_separates_personal_bests(tmp_path):
    env = RacingEnv(dt=0.04, render_mode="rgb_array", view="2d")
    env.reset()
    ghost = reference_for(env)
    before = env.snapshot()
    normal = env.render()
    env.viewer.agent_ghost = ghost
    assert env.viewer.ghost_pose() is None  # Coincident meshes stay readable.
    saved_position = env.pos.copy()
    env.pos += [10, 0]
    assert env.viewer.ghost_pose() is not None
    env.pos = saved_position
    env.viewer.human_line = [env.state(), ghost.states[1]]
    with_ghost = env.render()
    assert not np.array_equal(normal, with_ghost)
    for key in ("pos", "steering", "steps", "progress"):
        np.testing.assert_array_equal(getattr(env, key), before["state"][key])
    env.viewer.show_ghost = env.viewer.show_lines = False
    env.viewer.mode = "3d"
    env.renderer = "software"
    env.render()
    np.testing.assert_array_equal(env.state(), ghost.states[0])
    ordinary = RacingEnv()
    ordinary.reset()
    assert PracticeBook(env, tmp_path).path != PracticeBook(ordinary, tmp_path).path
    env.close()


def test_cli_countdown_restart_and_saved_comparison(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from collections import defaultdict
    import pygame
    from racing_lab import cli
    from racing_lab import agent_ghost
    from racing_lab.data import load

    template = RacingEnv(dt=0.04)
    template.reset(seed=0)
    reference = reference_for(template)
    monkeypatch.setattr(agent_ghost, "prepare_sac_ghost", lambda *args: reference)
    captured = []

    def create_env(*args, **kwargs):
        env = RacingEnv(*args, **kwargs)
        captured.append(env)
        return env

    monkeypatch.setattr(cli, "RacingEnv", create_env)
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=lambda _: 100))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(int, {pygame.K_w: 1}))
    frame = 0

    def events():
        nonlocal frame
        frame += 1
        # Two attempts, each with a countdown; the first stops before completing.
        key = {1: pygame.K_SPACE, 34: pygame.K_r, 35: pygame.K_SPACE, 69: pygame.K_ESCAPE}.get(
            frame
        )
        if frame in (2, 20, 36, 50):
            assert captured[0].steps == 0
            np.testing.assert_array_equal(captured[0].state(), reference.states[0])
        return [pygame.event.Event(pygame.KEYDOWN, key=key)] if key is not None else []

    monkeypatch.setattr(pygame.event, "get", events)
    args = SimpleNamespace(
        track="monza",
        seconds=20,
        view="2d",
        renderer="software",
        seed=0,
        sac_ghost="synthetic-fixture",
        ghost_cache=str(tmp_path / "cache"),
        ghost_results=str(tmp_path / "sessions"),
        data=str(tmp_path / "ordinary-data"),
        practice=str(tmp_path / "practice"),
        mute=True,
    )
    cli.play(args)
    records = list((tmp_path / "sessions" / "recordings").glob("*.npz"))
    assert len(records) == 2
    assert len(list((tmp_path / "sessions").glob("*_comparison.html"))) == 2
    assert not (tmp_path / "ordinary-data").exists()
    for path in records:
        data = load(path)
        assert data["metadata"]["source"] == "human_ghost_comparison"
        assert data["metadata"]["dt"] == 0.04
        np.testing.assert_array_equal(data["states"][0], reference.states[0])
