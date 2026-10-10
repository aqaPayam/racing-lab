from dataclasses import replace
import math
import numpy as np
from racing_lab.dynamics import PARAMETERS
from racing_lab.vehicle_model import advance_vehicle, DrivingConfig, VehicleState


def run(speed, seconds=0.5, rate=240, steer=0, throttle=0, brake=0, offroad=False, p=PARAMETERS):
    state = VehicleState.rolling(speed)
    u, v, r = speed, 0.0, 0.0
    for _ in range(max(1, int(seconds * rate))):
        u, v, r, t = advance_vehicle(
            u, v, r, steer, throttle, brake, offroad, 1 / rate, state, DrivingConfig(), p=p
        )
    return np.array([u, v, r]), state, t


def test_combined_forces_stay_inside_each_wheel_friction_circle():
    for speed in (5, 25, 60, 90):
        for throttle, brake in ((1, 0), (0, 1), (0.5, 0.5), (0, 0)):
            for offroad in (False, True):
                _, _, t = run(
                    speed, 0.05, steer=0.12, throttle=throttle, brake=brake, offroad=offroad
                )
                assert np.all(
                    np.linalg.norm(t["wheel_force_n"], axis=1)
                    <= np.array(t["wheel_capacity_n"]) + 1e-8
                )


def test_load_transfer_aero_and_surface_grip():
    _, coast, _ = run(40)
    _, brake, t = run(40, brake=1)
    _, power, _ = run(40, throttle=1)
    assert brake.pitch < coast.pitch < power.pitch
    assert t["front_load_n"] > t["rear_load_n"]
    _, _, aero = run(80, seconds=0)
    _, _, no_aero = run(80, seconds=0, p=replace(PARAMETERS, downforce_area=0))
    assert sum(aero["wheel_capacity_n"]) > sum(no_aero["wheel_capacity_n"])
    _, _, grass = run(80, seconds=0, offroad=True)
    assert sum(grass["wheel_capacity_n"]) < sum(aero["wheel_capacity_n"])
    _, turned, t = run(30, steer=0.08, throttle=0.2)
    assert abs(turned.roll) > 0.001
    assert t["wheel_load_n"][0] != t["wheel_load_n"][1]


def test_standstill_coasting_and_extreme_inputs_are_finite():
    np.testing.assert_array_equal(run(0, steer=0.45)[0], [0, 0, 0])
    state = VehicleState.rolling(70)
    u, v, r = 70.0, 0.0, 0.0
    for i in range(2400):
        u, v, r, _ = advance_vehicle(
            u,
            v,
            r,
            0.4 * math.sin(i / 100),
            float(i < 1200),
            float(i >= 1200),
            False,
            1 / 240,
            state,
            DrivingConfig(),
        )
        assert np.isfinite([u, v, r]).all()
        assert abs(v) < 100 and abs(r) < 20
    assert run(50)[0][0] < 50


def test_substep_refinement_agrees_for_a_moderate_turn():
    a = run(30, 3, 240, steer=0.025, throttle=0.25)[0]
    b = run(30, 3, 480, steer=0.025, throttle=0.25)[0]
    np.testing.assert_allclose(a, b, rtol=0.01, atol=0.015)
