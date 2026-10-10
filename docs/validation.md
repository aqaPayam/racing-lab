# Improvement checklist and validation

Release: package 0.7.0 / simulator 0.5.0 / dataset schema 2.

## Implemented scope

1. **Road boundaries:** continuous visible barriers outside the shoulder; four-corner collision resolution each physics substep.
2. **Impact and recovery:** dissipative normal/tangential impulses, yaw response, reverse, explicit recorded recovery that invalidates a lap.
3. **2D presentation:** smoothed forward-looking camera, overview/trail, curb/shoulder/barrier layers, detailed steering car, grass/trees, shared dashboard and minimap.
4. **Drivetrain:** eight forward gears, automatic/manual shifts, RPM torque curve, engine braking, reverse, shift interruption and over-rev interlock.
5. **Wheel/tire dynamics:** independent angular speeds, slip/spin/locking, combined friction budgets, surface grip, physical wheel animation.
6. **Sprung-body dynamics:** damped pitch/roll, longitudinal/lateral load transfer and load-sensitive tire grip. This is a reduced-order model, not detailed four-corner suspension.
7. **Input and aids:** adjustable keyboard shaping, configurable gamepad axes, separate recorded input/vehicle aid settings, manual gears and switchable TC/ABS.
8. **Tracks:** four preserved compact layouts plus four longer layouts, with illustrative elevation/banking shared by physics and graphics.
9. **Driving feedback:** synthetic RPM/tire/shift/impact audio, gear/RPM LEDs, pedal/g-force readouts, optional cockpit motion, mute/volume controls.
10. **Practice:** three sectors, configuration-specific clean personal bests, passive 2D/3D ghosts, progress delta, standalone speed/brake/trajectory comparison and JSON curves.
11. **Research interface:** deterministic headless API, complete snapshots, named schema 2 arrays/metadata, recorded supplemental commands, aid-aware evaluation/BC checks, historical visual replay.
12. **Release quality:** regression suite, GPU/software/2D checks, package build, updated examples/docs/screenshots, and public repository publication.

## Local verification

The complete initial regression run passed 58 tests, including reference laps on all eight layouts. Further focused runs verified the added terrain/practice/input/legacy-data checks and bend-impact cases. The final suite contains 80 cases; GitHub Actions runs it on Windows/Linux with Python 3.11/3.13. Build and Ruff checks passed locally.

The five-second Windows RTX 4060 GPU benchmark measured **20.1 ms median / 24.3 ms p95** rendering and a **0.997 simulation-to-wall-time ratio**. Shader/scene warm-up was excluded. This is one local measurement, not a guarantee for other hardware.

## Regression coverage

Tests cover Gymnasium compliance; deterministic replay including reverse/recovery; snapshots; recording alignment and validation; incompatible training configurations; BC save/load; numerical finiteness; individual friction-circle bounds; load transfer; grade/bank forces; integration refinement; acceleration/braking; keyboard pulse/settling and mirrored maneuvers; fixed-step timing at 20/30/60/144 FPS; impact containment on straights and tight bends across all eight tracks; sector timing and reference-controller completion; ghost filtering/persistence; analog mapping/button edges; audio state isolation; and GPU/software/2D rendering without physics mutation.

The reference controller is privileged, conservative and geometric. Passing its lap checks shows that the environment supports completion through the gates, not that an algorithm learned an optimal racing line. Graphics tests skip GPU checks only when no OpenGL context is available. The development PC exercised the GPU path on an RTX 4060. Gamepad mapping was verified with a simulated device; a physical gamepad/steering wheel was not available for hardware testing. Audio was exercised through SDL's dummy device; subjective speaker sound quality remains a user playtest concern.

## Measured simulation behavior

Isolated asphalt integration with default aids, no steering, four substeps per public step:

| Measurement | Result |
| --- | --- |
| 0–100 km/h | 2.925 s |
| Speed after 60 s full throttle | 308.1 km/h |
| 100–0 km/h full brake | 21.0 m / 1.75 s |

These are model outputs, not measured real F1 performance. Track laps include cornering, grade, banking and barriers. Default aids change the response; record their settings when comparing results.

## Reproduce

```sh
python -m pip install -e ".[all,dev]"
python -m ruff check racing_lab tests examples
python -m ruff format --check racing_lab tests examples
python -m pytest -q
python -m build
python examples/benchmark_rendering.py --renderer gpu --seconds 5
```

The full suite includes long simulated laps and can take several minutes. The benchmark excludes scene/shader initialization and writes its local result under `results/`. Longer terrain initialization may take tens of seconds on the first draw. Runtime performance is machine-specific; the real-time scheduler cannot compensate for insufficient physics throughput.

See [physics limits](physics.md#limits-and-references) before presenting this as a real-car simulation. Personal datasets, models, benchmark output and practice ghosts are not bundled or published.
