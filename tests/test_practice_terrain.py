import numpy as np
import pytest
from racing_lab import RacingEnv, DrivingConfig
from racing_lab.track import Track, names
from racing_lab.terrain import pose_frame
from racing_lab.vehicle_model import VehicleState, advance_vehicle
from racing_lab.data import Recorder, load, validate
from racing_lab.practice import PracticeBook, compare_runs, comparison_html


@pytest.mark.parametrize("name", names())
def test_profile_surface_and_frame_agree(name):
    track = Track(name)
    s = track.length * 0.17
    np.testing.assert_allclose(track.profile(s), track.profile(s + track.length), atol=1e-12)
    h, grade, bank = track.profile(s)
    derivative = (track.profile(s + 0.01)[0] - track.profile(s - 0.01)[0]) / 0.02
    assert derivative == pytest.approx(grade, abs=1e-7)
    env = RacingEnv(name)
    env.reset(options={"start_distance": s})
    origin, frame = pose_frame(env)
    assert origin[2] == pytest.approx(h, abs=1e-8)
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    assert track.surface(env.pos[None])[0] == pytest.approx(origin[2], abs=1e-8)
    if name.endswith("_long"):
        assert 5000 < track.length < 7000
        assert np.ptp(track.profile_values[:, 0]) > 0


def test_grade_and_banking_change_actual_forces():
    def force(grade, bank):
        return advance_vehicle(
            20,
            0,
            0,
            0,
            0,
            0,
            False,
            1 / 240,
            VehicleState.rolling(20),
            DrivingConfig(),
            grade,
            bank,
        )[-1]

    flat, uphill, banked = force(0, 0), force(0.1, 0), force(0, 0.1)
    assert uphill["longitudinal_g"] < flat["longitudinal_g"] - 0.09
    assert banked["lateral_g"] < -0.09


def test_practice_ghost_filters_laps_and_comparison_is_portable(tmp_path):
    env = RacingEnv()
    env.reset()
    book = PracticeBook(env, tmp_path / "practice")
    rec = Recorder(env, source="synthetic-test")
    for _ in range(10):
        a = [0, 0.5, 0]
        rec.append(a, env.step(a))
    assert not book.consider(rec)
    # Controlled bookkeeping fixture: test persistence without fabricating a demo label.
    env.progress = env.track.length
    env.next_checkpoint = 41
    env.collisions = 1
    assert not book.consider(rec)
    env.collisions = 0
    env.recoveries = 1
    assert not book.consider(rec)
    env.recoveries = 0
    env.offtrack_steps = 1
    assert not book.consider(rec)
    env.offtrack_steps = 0
    assert book.consider(rec)
    other = RacingEnv()
    other.reset()
    restored = PracticeBook(other, tmp_path / "practice")
    np.testing.assert_array_equal(restored.ghost(0), rec.states[0])
    assert restored.ghost(20) is None
    assert restored.best_time == env.steps / 60
    assert not book.consider(rec)
    p = rec.save(tmp_path / "runs")
    report = compare_runs(p, p)
    html = comparison_html(report)
    assert "<canvas" in html and "https://" not in html
    assert len(report["runs"]) == 2
    data = load(p)
    data["commands"] = np.zeros(3)
    with pytest.raises(ValueError, match="commands"):
        validate(data)


def test_invalid_commands_do_not_mutate_state():
    env = RacingEnv()
    env.reset()
    before = env.snapshot()
    with pytest.raises(ValueError):
        env.step([1, 1, 1], commands={"shift": 2})
    np.testing.assert_array_equal(env.action, before["state"]["action"])
    assert env.steering == before["state"]["steering"]
    assert env.smoothness == before["state"]["smoothness"]


def test_gamepad_mapping_edges_and_audio_are_presentation_only(monkeypatch):
    import pygame
    from racing_lab.controls import GamepadController
    from racing_lab.audio import DrivingAudio

    class Device:
        pressed = set()

        def init(self):
            pass

        def get_init(self):
            return True

        def get_numaxes(self):
            return 6

        def get_axis(self, i):
            return {0: 0.5, 2: -1, 5: 1}[i]

        def get_numbuttons(self):
            return 7

        def get_button(self, i):
            return i in self.pressed

        def get_name(self):
            return "test-device"

    device = Device()
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 1)
    monkeypatch.setattr(pygame.joystick, "Joystick", lambda _: device)
    controller = GamepadController()
    env = RacingEnv()
    env.reset()
    device.pressed = {5}
    a, c = controller.sample(env)
    assert 0 < a[0] < 0.5 and a[1] == 1 and a[2] == 0 and c == {"shift": 1}
    assert controller.sample(env)[1] == {}
    before = env.vehicle.vector()
    sound = DrivingAudio(enabled=False)
    sound.enabled = True
    sound.update(env)
    sound.close()
    np.testing.assert_array_equal(before, env.vehicle.vector())


def test_legacy_schema_one_loads_and_replays(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    import pygame
    from racing_lab.cli import replay

    env = RacingEnv()
    env.reset()
    rec = Recorder(env, source="synthetic-test")
    a = [0, 0.2, 0]
    rec.append(a, env.step(a))
    data = load(rec.save(tmp_path / "current"))
    meta = data.pop("metadata")
    meta["schema_version"] = 1
    meta["simulator_version"] = "0.3.0"
    meta["observation_names"] = meta["observation_names"][:20]
    meta["feature_names"] = meta["feature_names"][:4]
    for key in (
        "telemetry_names",
        "vehicle_state_names",
        "command_names",
        "contact_names",
        "driving_config",
    ):
        meta.pop(key, None)
    for key in ("telemetry", "vehicle_states", "commands", "contacts"):
        data.pop(key, None)
    data["observations"] = data["observations"][:, :20]
    data["reward_features"] = data["reward_features"][:, :4]
    data["metadata"] = json.dumps(meta)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **data)
    assert load(path)["observations"].shape == (2, 20)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    replay(Namespace(file=path, view="2d", renderer="software"))
