# Vehicle physics

Package **0.7.0**, simulator **0.5.0** uses four independent wheel speeds, planar chassis motion, and reduced-order pitch/roll dynamics. Four 1/240-second integration substeps run inside every 1/60-second Gymnasium step. Rendering, camera effects, and sound never advance physics.

## Tires, drivetrain, and suspension

Wheel contact velocities include body longitudinal/lateral motion and yaw. Front contacts rotate with steering. Lateral demand is a smooth saturating function of slip angle; longitudinal force comes from compliant slip between each wheel's angular speed and contact velocity. A semi-implicit update avoids numerical instability from stiff longitudinal tire forces near standstill. Each tire shares one combined force budget: if `hypot(Fx,Fy)` exceeds its load-dependent capacity, both components scale together. Braking therefore consumes cornering grip.

Each wheel has angular inertia and its own spin angle, so driven wheels can spin and braking wheels can lock. The graphics use these angles. Grip decreases on grass/shoulders and its coefficient decreases slightly with increasing normal load. A damped pitch mode drives front/rear load transfer; a damped roll mode drives left/right transfer. These modes animate the body and change tire loads, but do not implement four independent suspension travel systems, chassis heave, or wheel hop.

The rear-wheel drivetrain has eight forward ratios, reverse, an RPM-dependent torque curve capped at 500 kW, a simplified launch clutch, engine braking, and a short torque interruption on shifts. Automatic shifting is the default; manual shifts reject over-revving downshifts. Reverse is limited to low speed. Optional traction control reduces drive torque with excess rear slip; optional ABS reduces brake torque with excessive braking slip. Both are **game/research aids**, not claims about real F1 equipment. They default on for accessible driving and are explicitly labeled and recorded. Use `DrivingConfig(automatic=False, traction_control=False, abs=False)` or the CLI flags to disable them.

Aerodynamic drag and downforce grow with speed squared. Height grade and banking affect gravity and normal loading. Rendering uses the same profile. The model remains constrained to the road surface: it cannot jump or roll over. Below 5 m/s a smooth blend to geometric steering avoids the slip-angle singularity.

## Scale and parameters

The car is **5.5 m long, 1.9 m wide, with a 3.4 m wheelbase**. Tracks are 14 or 16 m wide, with a separate 2 m shoulder on each side before the barrier. These lengths are shared between views and physics; the HUD converts m/s to km/h.

| Parameter | Default |
| --- | --- |
| Mass / yaw inertia | 800 kg / 1,100 kg m² |
| CG height | 0.30 m |
| Wheel radius / angular inertia | 0.36 m / 1.5 kg m² |
| Engine power ceiling | 500 kW |
| Brake-force request / front allocation | 32 kN / 64% |
| Front / rear lateral stiffness | 95 / 150 kN/rad per axle |
| Longitudinal stiffness | 80 kN per unit slip per wheel |
| Asphalt / grass nominal friction | 1.65 / 0.48 |
| Load-sensitivity exponent | -0.08 |
| Drag / downforce area coefficient | 1.1 / 4.2 m² |
| Pitch / roll natural frequency | 11 / 13 rad/s |
| Damping ratio | 0.85 |

See `dynamics.py` and `vehicle_model.py` for the complete original equations and constants. Metadata captures the parameter dictionary and simulator version. Change `SIMULATOR_VERSION` if changing the model in a fork.

## Barriers, impacts, and recovery

Both road sides have solid barriers outside the shoulder. Every integration substep checks all four corners of the car against the local road corridor and resolves penetration. Normal impulses remove closing velocity; tangential friction dissipates sliding speed; off-center impacts change yaw. There is no artificial bounce or permanent damage. Collision tests cover glancing and head-on impacts up to 324 km/h. This is a substepped local constraint solver, not general continuous collision detection for arbitrary intersecting custom tracks.

Press X near standstill to select reverse/forward, then use throttle. H explicitly recovers to the nearby centerline and invalidates the lap. Recovery and direction/shift requests are saved as commands, separate from the three continuous actions. Impacts and recoveries have separate reward features and counters. Walls prevent leaving the corridor; the shoulder is still off-track, and cutting a curb can invalidate a clean personal best.

## Diagnostics and compatibility

`info['telemetry']` includes acceleration, downforce, slip angles, each wheel's loads/forces/capacities, RPM, gear, pitch/roll, road grade/bank, and active aid flags. Schema 2 stores named telemetry, all wheel/drivetrain state, commands, and contacts. The default observation now has 39 numerical values. See [research guide](research-guide.md).

Simulator 0.4 and earlier demonstrations remain visually replayable with matching track files, but are incompatible with current training. Collect new data (for example `data-v05`) and retrain. Tests and measured local results are listed in [validation](validation.md).

## Limits and references

This is an inspectable generic open-wheel research game, **not a calibrated F1 simulator**. There is no measured tire data, temperature/wear, tire deformation, detailed suspension geometry, ERS, active aerodynamics, damage, traffic, weather, or pit strategy. Track profiles are illustrative. Successful regression tests establish numerical behavior and reproducibility, not agreement with instrumented real-car measurements.

Background on the model families: [MathWorks wheel and brake models](https://www.mathworks.com/help/vdynblks/brakes-and-wheels.html), [longitudinal wheel dynamics](https://www.mathworks.com/help/vdynblks/ref/longitudinalwheel.html), and [vehicle body forces](https://www.mathworks.com/help/sdl/ref/vehiclebody.html). These references do not validate this project's parameter choices.
