import math
import numpy as np
import pytest
from racing_lab.controls import KeyboardController
from racing_lab.timing import FixedStepClock
from racing_lab.vehicle_model import VehicleState, DrivingConfig, advance_vehicle
from racing_lab import RacingEnv
from racing_lab.data import Recorder, load


@pytest.mark.parametrize("fps", [20, 30, 60, 144])
def test_render_rate_does_not_change_simulated_seconds(fps):
    clock = FixedStepClock()
    assert sum(clock.advance(1 / fps) for _ in range(fps * 10)) == 600
    assert abs(clock.pending) < 1e-8


def test_stalls_pause_instead_of_applying_unseen_seconds_of_input():
    clock = FixedStepClock()
    assert clock.advance(0.1) == 6
    assert clock.advance(2) == 0 and clock.suspended
    assert clock.advance(1 / 60) == 1 and not clock.suspended


def test_200_kmh_means_about_55_metres_in_one_wall_second_at_20fps():
    env = RacingEnv()
    env.reset(options={"start_distance": 100, "speed": 200 / 3.6})
    clock = FixedStepClock()
    start = env.pos.copy()
    for _ in range(20):
        for _ in range(clock.advance(0.05)):
            env.step([0, 0.27, 0])
    assert env.steps == 60
    assert 54 < np.linalg.norm(env.pos - start) < 57
    assert env.distance == pytest.approx(np.linalg.norm(env.pos - start), abs=0.01)


def test_short_keyboard_pulse_is_controllable_at_200_kmh():
    controller = KeyboardController()
    u, v, r = 200 / 3.6, 0, 0
    heading = displacement = 0.0
    vehicle = VehicleState.rolling(u)
    for i in range(180):
        action = controller.update([1 if i < 9 else 0, 0.3, 0], u, 1 / 60)
        for _ in range(4):
            u, v, r, _ = advance_vehicle(
                u,
                v,
                r,
                float(action[0]) * 0.45,
                float(action[1]),
                0,
                False,
                1 / 240,
                vehicle,
                DrivingConfig(),
            )
            heading += r / 240
            displacement += (u * math.sin(heading) + v * math.cos(heading)) / 240
    assert 0.2 < displacement < 1.5  # 150 ms tap, followed by release; no instant spin.
    assert abs(heading) < math.radians(1)
    assert abs(r) < 0.001
    assert controller.action[0] == 0


def test_assist_is_keyboard_only_and_recorded(tmp_path):
    assisted, raw = KeyboardController(), KeyboardController(assisted=False)
    for _ in range(60):
        a = assisted.update([1, 0, 0], 200 / 3.6, 1 / 60)
        b = raw.update([1, 0, 0], 200 / 3.6, 1 / 60)
    assert a[0] < 0.04 < b[0]  # finite 1.3-g geometric envelope vs full 25.8-degree lock
    env = RacingEnv()
    env.reset()
    recorder = Recorder(env, control_settings=assisted.metadata())
    recorder.append([1, 0, 0], env.step([1, 0, 0]))
    assert env.action[0] == 1  # direct algorithm actions never get keyboard assistance
    recording = load(recorder.save(tmp_path))
    assert recording["metadata"]["control_settings"]["assisted"]
    assert recording["actions"][0, 0] == 1


@pytest.mark.parametrize("steering", [0.01, 0.02, 0.03])
def test_cornering_then_braking_remains_stable(steering):
    # Sub-limit turns (up to about 0.9 g), then firm braking. Combining maximum
    # cornering with maximum braking can exceed grip and is not required to be stable.
    u, v, r = 40, 0, 0
    vehicle = VehicleState.rolling(u)
    for _ in range(1200):
        u, v, r, t = advance_vehicle(
            u, v, r, steering, 0.1, 0, False, 1 / 240, vehicle, DrivingConfig()
        )
    assert 0 < r < 0.8
    assert 0 < t["lateral_g"] < 3
    initial = u
    for _ in range(720):
        u, v, r, _ = advance_vehicle(
            u, v, r, steering, 0, 0.7, False, 1 / 240, vehicle, DrivingConfig()
        )
        assert abs(v) < 5 and abs(r) < 1
    assert u < initial * 0.05


def test_stationary_steered_car_has_no_artificial_cornering_g():
    u, v, r, telemetry = advance_vehicle(
        0, 0, 0, 0.45, 0, 1, False, 1 / 240, VehicleState(), DrivingConfig()
    )
    assert (u, v, r) == (0, 0, 0)
    assert telemetry["lateral_g"] == telemetry["longitudinal_g"] == 0


def test_low_speed_circle_matches_geometry():
    u, v, r, _ = advance_vehicle(
        1.5, 0, 0, 0.15, 0, 0, False, 1 / 240, VehicleState.rolling(1.5), DrivingConfig()
    )
    assert u / r == pytest.approx(3.4 / math.tan(0.15))


def test_mirrored_lane_change_has_mirrored_response():
    def run(sign):
        u, v, r, h, y = 45, 0, 0, 0, 0
        vehicle = VehicleState.rolling(u)
        for i in range(720):
            steer = sign * 0.015 * math.sin(2 * math.pi * i / 480) if i < 480 else 0
            u, v, r, _ = advance_vehicle(
                u, v, r, steer, 0.2, 0, False, 1 / 240, vehicle, DrivingConfig()
            )
            h += r / 240
            y += (u * math.sin(h) + v * math.cos(h)) / 240
        return np.array([u, v, r, h, y])

    left, right = run(-1), run(1)
    np.testing.assert_allclose(left, right * [1, -1, -1, -1, -1], atol=1e-10)
    assert abs(left[2]) < 0.02
