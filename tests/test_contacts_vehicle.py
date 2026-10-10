import numpy as np
import pytest
from racing_lab import RacingEnv, DrivingConfig
from racing_lab.collisions import CORNERS
from racing_lab.data import Recorder, load
from racing_lab.vehicle_model import VehicleState, advance_vehicle


def assert_inside(env):
    c, s = np.cos(env.heading), np.sin(env.heading)
    corners = env.pos + CORNERS @ np.array([[c, s], [-s, c]])
    _, offsets, _ = env.track.project_many(corners, hint=env.last_s)
    assert max(abs(offsets)) <= env.track.barrier_offset + 0.015


@pytest.mark.parametrize("speed", [15, 55, 90])
@pytest.mark.parametrize("angle", [0.15, 0.8, 1.57])
def test_wall_does_not_tunnel_and_impact_dissipates(speed, angle):
    env = RacingEnv()
    env.reset(options={"start_distance": 100, "speed": speed})
    env.pos += np.array([-np.sin(env.heading), np.cos(env.heading)]) * (
        env.track.barrier_offset - 3
    )
    env.heading += angle
    touched = False
    for _ in range(90):
        _, _, _, _, info = env.step([0, 0, 0])
        touched |= info["barrier_contact"]
        assert_inside(env)
        assert np.isfinite(env.state()).all()
    assert touched
    assert abs(env.speed) < speed


def test_recovery_and_manual_reverse_are_explicit_reproducible_commands(tmp_path):
    env = RacingEnv(driving=DrivingConfig(automatic=False))
    env.reset(seed=3)
    rec = Recorder(env)
    for i in range(120):
        cmd = {"direction": -1} if i == 0 else ({"recover": True} if i == 90 else {})
        a = [0, 0.5, 0]
        rec.append(a, env.step(a, commands=cmd))
        if i == 80:
            assert env.speed < -0.5
    data = load(rec.save(tmp_path))
    other = RacingEnv(driving=env.driving)
    other.reset(seed=3)
    for a, c in zip(data["actions"], data["commands"]):
        other.step(a, commands=dict(zip(("shift", "direction", "recover"), map(int, c))))
    np.testing.assert_array_equal(other.state(), env.state())
    np.testing.assert_array_equal(other.vehicle.vector(), env.vehicle.vector())
    assert other.info()["invalid_lap"] and other.recoveries == 1


def test_wheels_have_spin_lockup_load_transfer_and_shared_grip():
    state = VehicleState.rolling(25)
    u, v, r = 25.0, 0.0, 0.0
    config = DrivingConfig(traction_control=False, abs=False)
    for _ in range(150):
        u, v, r, t = advance_vehicle(u, v, r, 0.01, 1, 0, False, 1 / 240, state, config)
        assert np.all(
            np.linalg.norm(t["wheel_force_n"], axis=1) <= np.array(t["wheel_capacity_n"]) + 1e-6
        )
    assert max(state.slip[2:]) > 0.1
    for _ in range(80):
        u, v, r, t = advance_vehicle(u, v, r, 0.01, 0, 1, False, 1 / 240, state, config)
    assert min(state.slip) < -0.3
    assert t["wheel_load_n"][0] != t["wheel_load_n"][1]
    assert state.pitch != 0 and state.roll != 0


def test_automatic_gearbox_shifts_and_engine_braking_slows_car():
    state = VehicleState.rolling(28)
    state.gear = 1
    state.rpm = 14000
    u, v, r, t = advance_vehicle(28, 0, 0, 0, 0.8, 0, False, 1 / 240, state, DrivingConfig())
    assert state.gear == 2 and state.shift_timer > 0
    previous = u
    for _ in range(240):
        u, v, r, t = advance_vehicle(u, v, r, 0, 0, 0, False, 1 / 240, state, DrivingConfig())
    assert u < previous


@pytest.mark.parametrize("name", __import__("racing_lab.track", fromlist=["names"]).names())
def test_barriers_hold_at_bends_on_every_track(name):
    env = RacingEnv(name)
    # Highest sampled curvature, approached at high speed toward each road side.
    angles = np.unwrap(np.arctan2(env.track.tangents[:, 1], env.track.tangents[:, 0]))
    idx = int(np.argmax(np.abs(np.diff(angles))))
    for side in (-1, 1):
        env.reset(options={"start_distance": float(env.track.s[idx]), "speed": 80})
        env.pos += (
            np.array([-np.sin(env.heading), np.cos(env.heading)])
            * side
            * (env.track.barrier_offset - 3)
        )
        env.heading += side * 0.7
        for _ in range(45):
            env.step([0, 0, 0])
            assert_inside(env)
