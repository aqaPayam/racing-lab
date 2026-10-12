# Changelog

## 0.10.0 — TD3 and accounted online benchmarks

- Add F1TENTH-source TD3, numerical initialization/update parity, corrected uniform replay, frozen playback and Progress/CTH reward ablations.
- Add fresh-process SAC–TD3 queues, frozen hashed protocols, five-seed budgets, paired robustness starts and zero-shot circuit tests.
- Record training/evaluation interactions, physics/simulation work, optimizer/sample counts, CPU/wall/component times, native peak memory and inference latency.
- Generate per-seed tables, descriptive seed variation, learning curves and standalone reports; failed times remain null.
- Preserve SAC learning behavior, simulator 0.5.0 and existing ghost compatibility; fix SAC checkpoint track-ID inference.
- Retain Apache-2.0 TD3 attribution and fixtures. Generated training artifacts stay outside Git.

## 0.9.0 — human versus SAC ghost practice

- Add a cached, frozen SAC standing-start lap as a passive ghost for manual driving, with a shared countdown, matched timestep/vehicle aids and strict clean-lap comparisons.
- Add live progress-aligned gaps, orange/teal racing-line overlays in 2D and the minimap, and local HTML/JSON comparison reports with speed/braking/path data.
- Add a Windows ghost launcher and checkpoint/track/configuration compatibility checks. No training or expert-data ingestion occurs in practice mode.
- Use each environment's timestep for ghost playback, camera smoothing and personal-best timestamps; separate personal-best caches by timestep.
- Simulator compatibility remains 0.5.0; existing trained SAC weights and training defaults are unchanged.

## 0.8.0 — online SAC with three-step returns

- Add an optional PyTorch SAC baseline, following pinned racing-author networks, losses and update order; fresh initialization and online-only replay.
- Add a 25 Hz multi-lap task with incremental controls, numerical history, centerline reward, explicit episode endings and strict valid-lap metrics.
- Add training/evaluation CLI commands, Python API, checkpoints, provenance/configuration hashes and CSV/JSON experiment logs.
- Correct n-step truncation tails and reset isolation; document paper/code discrepancies and simulator substitutions.
- Add offline licensed source fixtures and numerical parity/regression tests.
- Add configurable step duration and lap target to RacingEnv; propagate them to snapshots and recording/replay. Default 60 Hz dynamics and the 39-observation base interface remain compatible with simulator 0.5.0.

## 0.7.0 — barriers, four-wheel dynamics and practice

- Simulator 0.5.0: solid road barriers, dissipative impacts, reverse and explicit recovery.
- Eight gears, RPM/engine braking, four wheel speeds and slip, load-sensitive combined grip, reduced pitch/roll suspension and configurable TC/ABS.
- Four additional longer circuit variants with illustrative elevation/banking shared by physics and graphics.
- Redesigned 2D, physical wheel/body animation, shared instruments, gamepad mapping, synthesized audio and optional cockpit motion.
- Sector timing, local clean personal bests, passive ghosts, offline speed/brake/trajectory comparison.
- Schema 2: 39 observations, six reward features, vehicle states, commands, contacts, driving configuration; legacy replay retained. Fresh demonstrations and retraining required.
- Expanded regression tests and updated research/API/driving documentation.


## 0.6.0

- Separate interactive 60 Hz simulation from rendered frame rate; update replay and rendered evaluation pacing.
- Add gradual, speed-dependent keyboard steering with recorded input settings and an explicit raw-steering option. Policy actions remain direct.
- Correct combined tire-force saturation and estimate weight transfer from delivered force. Tune front/rear stiffness and brake balance, smooth low-speed transitions, and remove stationary cornering forces. Simulator compatibility is now **0.4.0**; collect fresh training demonstrations.
- Add optional ModernGL rendering with depth testing, sun shadows, antialiasing and procedural materials. Preserve software 3D and 2D fallbacks.
- Rebuild the car with rounded tires, rims, correctly rotating spokes, steering, wings, cockpit and suspension links. Add braking boards, reflectors, woodland and spectators.
- Add regression tests for steering pulses, lane-change symmetry, corner braking, real-time speed, timing across frame rates, and GPU rendering; include a reproducible rendering benchmark.


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
