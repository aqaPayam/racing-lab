# Force-based vehicle model

Simulator **0.4.0**, introduced in package **0.6.0**, is a planar dynamic single-track model. It advances longitudinal velocity, lateral velocity and yaw rate using front/rear axle forces. Four 1/240-second force steps occur inside each 1/60-second public environment step. The numerical observation/action dimensions remain unchanged.

## What changed

Front/rear tire slip angles come from axle velocities and steering. Pure lateral demand is `Fy = mu * Fz * tanh(C_alpha * slip_angle / (mu * Fz))`. Longitudinal requests are bounded by axle capacity. When `hypot(Fx, Fy) > mu * Fz`, both components are scaled by the same factor to stay within the friction circle. This replaces the previous longitudinal-priority rule, which could erase all rear cornering force at full throttle.

The default front/rear cornering stiffnesses are 95/150 kN per radian, providing an understeer-biased response. Front brake allocation is 64%. Longitudinal load transfer is estimated with five fixed-point iterations using **delivered body force**, rather than assuming the pedal request can always be realized. These are inspectable generic tuning choices, not fitted real-car values.

Drive force is limited by both a low-speed force ceiling and power divided by speed. Braking is divided between axles. Normal loads include static weight, speed-squared downforce, and a quasi-static longitudinal transfer estimate. Drag grows with squared speed; asphalt and grass use different friction/rolling resistance. Axle force moments accelerate yaw rather than setting yaw rate directly. Between 2 and 6 m/s, a smooth transition to low-speed kinematics avoids the slip-angle singularity; the vehicle cannot reverse.

## Inspectable parameters

All constants are in `racing_lab/dynamics.py`, in the immutable `VehicleParameters` dataclass:

| Parameter | Default |
| --- | --- |
| Mass | 800 kg |
| Wheelbase | 3.4 m |
| Yaw inertia | 1,100 kg mÂ² |
| CG height | 0.30 m |
| Maximum drive power | 500 kW |
| Maximum drive-force request | 14.5 kN |
| Maximum brake-force request | 32 kN |
| Front brake fraction | 64% |
| Front / rear cornering stiffness | 95 / 150 kN/rad |
| Asphalt / grass friction coefficient | 1.65 / 0.48 |
| Drag / downforce area coefficient | 1.1 / 4.2 mÂ² |

These are **generic open-wheel simulation choices**, not measured specifications of an F1 team. They are copied into new recordings and evaluation reports. If you change the model or its parameters in a fork, change `SIMULATOR_VERSION` before generating datasets; the standard environment currently uses one fixed parameter set.

## Verification and measured behavior

Isolated full-throttle straight-line integration gives approximately **2.7 seconds from 0 to 100 km/h**, **324 km/h terminal speed**, and **21 metres to stop from 100 km/h** under full braking on asphalt. These are outputs of the model, not comparisons against instrumented real-car tests. Lap geometry, grip and brake points matter; a short game straight may not permit reaching terminal speed.

Regression tests include a 200 km/h short keyboard pulse, mirrored lane changes, low-speed constant-radius geometry, sub-limit cornering followed by firm braking, fixed-step timing at 20/30/60/144 rendered frames per second, and finite behavior under aggressive inputs, force-circle bounds, load-transfer direction, combined braking/cornering, downforce and grass effects, acceleration/braking envelopes, deterministic replay, and successful reference-controller laps on all four circuits. These establish implementation consistency, not physical calibration.

`info['telemetry']` exposes longitudinal/lateral acceleration in g, downforce, front/rear slip angles, normal loads and axle forces/capacities. New NPZ recordings add a `T Ã— 5` telemetry array with names in metadata. Telemetry is diagnostic; the default policy still receives the existing 20 observations. Legacy visual replays show no force telemetry when it was not recorded.

## Research compatibility

The simulator version changes from 0.3.0 to 0.4.0. Old models and old good demonstrations are rejected by the current trainer/policy loader rather than silently reused with changed dynamics. Existing files are untouched and remain visually replayable. Use a separate directory, for example `play --data data-v04`, for new recordings; keep old data for your historical experiments.

## Boundaries of realism

There is no measured tyre data or real-car parameter identification. Axles are lumped into a single track, so lateral left/right load transfer, individual wheel speeds, actual lockups/wheelspin, tyre temperature/wear, suspension, bumps, bank/elevation, gear shifts, damage and barrier collisions are not modeled. Longitudinal tyre clipping is an idealized traction/braking limiter. The quasi-static load-transfer estimate uses delivered longitudinal force; it is not a sprung-mass suspension model. Rendered suspension links do not add suspension dynamics. Wheel animation uses traveled distance and assumes rolling without slip; it is not a wheelspin measurement.

The four compact circuits remain original approximations, including a non-crossing Suzuka adaptation. This is a more physically structured research game, not a validated F1 engineering simulator.

## Background references

- [MathWorks Bicycle Model](https://www.mathworks.com/help/vdynblks/ref/bicyclemodel.html): reference for the single-track vehicle-model family and longitudinal/lateral/yaw motion.
- [Formula 1 explanation of downforce](https://www.formula1.com/en/latest/article/f1-explains-downforce-and-why-f1-cars-have-wings-with-mclaren-aero.10xj8CJPz8s7CmwfoyWmCU): background on the tradeoff between cornering downforce and straight-line drag.

The implementation and numerical choices here are original; these references do not validate its parameters.
