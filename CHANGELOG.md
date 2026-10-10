# Changelog

## 0.5.0 — Force-based driving (simulator 0.3.0)

- Replaced the arcade yaw/grip rule with a planar dynamic bicycle model: front/rear tyre slip, shared longitudinal/lateral friction circles, yaw inertia, rear-wheel traction, brake balance, quasi-static longitudinal load transfer, aerodynamic downforce and drag, and reduced grass grip.
- Four internal force steps per 60 Hz action. Power/traction-limited acceleration and speed-dependent aero produce approximately 2.4 s 0–100 km/h and 324 km/h terminal speed in isolated straight-line tests, not calibrated F1 performance.
- G-force, axle slip and downforce diagnostics; optional recorded telemetry and explicit parameter provenance.
- Rotating wheel spokes and small telemetry-driven visual body attitude cues. These cues do not simulate suspension.
- Physics compatibility changed: collect new demonstrations and retrain models for simulator 0.3.0. Older visual replays remain supported. Maps remain unchanged: four circuit-inspired layouts.

## 0.4.0 — 3D presentation

- Perspective 3D track and open-wheel car, steering front tyres, curbs, barriers, trees, pit building, grandstand, and start gantry.
- Chase, nose and aerial cameras, telemetry and minimap; C cycles cameras and V switches to the original 2D view.
- Driving, replay and rendered evaluation default to 3D in the CLI. Python rendering retains the 2D default and accepts `view="3d"`.
- Portable software renderer using the existing Pygame extra. No additional dependencies or downloaded artwork.
- Simulator compatibility remains 0.2.0; observations, dynamics, datasets and trained policies are unchanged.

## 0.3.0 — Research tooling

- Public evaluation/configuration API, aligned dataset arrays, discounted feature sums, and trusted simulator snapshots.
- Dataset validation, applied-action recording, reset metadata, duplicate-run detection, and training track/completion filters.
- Versioned evaluation reports with success-aware summaries, runtime provenance, seeds, hashes, and interruption handling.
- Optional play/training dependencies, offscreen RGB rendering, runnable examples, focused documentation, formatting and package checks.
- Numerical dynamics compatibility remains simulator 0.2.0. Existing 0.2.0 recordings remain usable. Evaluation JSON now contains a report object rather than a bare episode list.

## 0.2.0 — Consistent physical scale

- Shared 5.5 × 1.9 m car dimensions and 3.4 m wheelbase; 14–16 m road widths.
- Vehicle envelope in road-edge checks; true-scale rendering and speed-dependent following camera.
- Legacy layouts retained for replay. Simulator 0.1.0 demonstrations require recollection for current-version training.

## 0.1.0 — Initial prototype

- Four circuit-inspired tracks, keyboard driving, labeled recordings/replay, Gymnasium environment, behavior cloning and reference evaluation.
