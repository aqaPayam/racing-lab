from dataclasses import replace
import math
import numpy as np
from racing_lab.dynamics import advance, PARAMETERS


def test_combined_forces_stay_inside_axle_friction_circles():
    for speed in (5, 25, 60, 90):
        for throttle, brake in ((1, 0), (0, 1), (0.5, 0.5), (0, 0)):
            for offroad in (False, True):
                *_, t = advance(speed, 2.0, 0.2, 0.12, throttle, brake, offroad, 1 / 240)
                for axle in ("front", "rear"):
                    assert math.hypot(*t[f"{axle}_force_n"]) <= t[f"{axle}_capacity_n"] + 1e-8


def test_load_transfer_aero_and_combined_braking():
    coast = advance(40, 0, 0, 0.08, 0, 0, False, 1 / 240)[-1]
    brake = advance(40, 0, 0, 0.08, 0, 1, False, 1 / 240)[-1]
    power = advance(40, 0, 0, 0.08, 1, 0, False, 1 / 240)[-1]
    assert brake["front_load_n"] > coast["front_load_n"] > power["front_load_n"]
    assert abs(brake["front_force_n"][1]) < abs(coast["front_force_n"][1])
    no_aero = advance(80, 0, 0, 0.08, 0, 0, False, 1 / 240, replace(PARAMETERS, downforce_area=0))[
        -1
    ]
    aero = advance(80, 0, 0, 0.08, 0, 0, False, 1 / 240)[-1]
    assert aero["front_capacity_n"] > no_aero["front_capacity_n"]
    grass = advance(40, 0, 0, 0.08, 0, 0, True, 1 / 240)[-1]
    assert grass["front_capacity_n"] < coast["front_capacity_n"]


def test_standstill_coasting_and_extreme_inputs_are_finite():
    assert advance(0, 0, 0, 0.45, 0, 0, False, 1 / 240)[:3] == (0.0, 0.0, 0.0)
    u, v, r = 70.0, 0.0, 0.0
    for i in range(2400):
        u, v, r, _ = advance(
            u,
            v,
            r,
            0.4 * math.sin(i / 100),
            1.0 if i < 1200 else 0.0,
            0.0 if i < 1200 else 1.0,
            False,
            1 / 240,
        )
        assert np.isfinite([u, v, r]).all()
        assert abs(v) < 100 and abs(r) < 20
    coast = advance(50, 0, 0, 0, 0, 0, False, 1 / 240)
    assert coast[0] < 50


def test_substep_refinement_agrees_for_a_moderate_turn():
    def run(rate):
        u, v, r = 30.0, 0.0, 0.0
        for _ in range(rate * 3):
            u, v, r, _ = advance(u, v, r, 0.025, 0.25, 0, False, 1 / rate)
        return [u, v, r]

    np.testing.assert_allclose(run(240), run(480), rtol=0.01, atol=0.015)
