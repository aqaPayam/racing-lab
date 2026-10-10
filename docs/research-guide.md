# Research guide

## Tracks

`monza`, `silverstone`, `spa`, and `suzuka` are selectable with `--track`. These are original, hand-authored **approximations inspired by circuit shapes**, not surveyed replicas or official assets. Distances and widths are scaled for this prototype. There is no elevation or banking; Suzuka is explicitly adapted to avoid a bridge crossing. No F1 branding assets are included, and this project is not affiliated with Formula 1 or the circuits.

Add a JSON file under `racing_lab/tracks/` with `name`, `version`, `width` (metres), and an ordered loop of `[x, y]` centerline control points. Coordinates use screen-style positive y downward. The road is interpolated with a periodic Catmull-Rom curve. Avoid intersecting roads and bends closer together than their widths. Increment the version whenever geometry changes; recordings also store a SHA-256 digest of the exact file.

## Observations and dynamics

The 20 float32 observations contain normalized speed, lateral speed, yaw rate, steering angle, lateral road offset, sine/cosine of road-relative heading, road half-width, and six upcoming centerline positions expressed in car coordinates (10, 25, 50, 90, 150, 230 metres ahead). This is a compact local-road description, not global coordinates or images. The preview coordinates and known width describe the nearby road; they do not encode every possible custom-track topology. See `OBS_NAMES` in `racing_lab/env.py` for ordering and normalization.

The simulator uses a force-based dynamic bicycle model with axle slip, friction circles, aero, and longitudinal load transfer. See the [physics specification](physics.md) for equations, parameters, measured simulation behavior, and limitations.

### Scale and compatibility (simulator 0.3.0)

The car remains 5.5 m long and 1.9 m wide, with a 3.4 m wheelbase. Roads are 14 m (Monza/Suzuka) or 16 m (Silverstone/Spa), and circuits are compact 2.2�2.6 km approximations. The 3D view uses metres throughout; the optional 2D view includes a 10 m scale bar. Straight-line terminal speed is now approximately 324 km/h, governed by drive power and aerodynamic drag.

Road-edge checks use the rectangular car envelope projected across the local tangent. The vehicle cannot reverse, and ordered checkpoints discourage but do not eliminate all shortcuts. Legacy track layouts are retained for visual replay. Older recordings are untouched, but models trained on simulator 0.2.0 or earlier must be retrained on new simulator 0.3.0 demonstrations.

## Dataset format and inverse-learning hooks

Each `.npz` file loads with `numpy.load(path, allow_pickle=False)`:

| Array | Shape | Meaning |
| --- | --- | --- |
| observations | T+1 × 20 | Initial observation and every next observation |
| states | T+1 × 7 | x, y, heading, speed, lateral speed, yaw rate, steering |
| actions | T × 3 | Controls applied from observation t to observation t+1 |
| rewards | T | Scalar reward |
| reward_features | T × 4 | Progress delta, negative dt, negative off-road dt, negative squared action change |
| terminated / truncated | T | Gymnasium terminal and timeout flags |
| telemetry (new recordings) | T � 5 | Longitudinal/lateral g, downforce N, front/rear slip degrees |
| timestamps | T+1 | Simulation seconds |
| metadata | scalar JSON | Label, source, track version/hash, seed, schema, names, weights, outcome, stop reason |

Manual exits/interruption are described by `reason` and do not falsely set a simulator timeout flag. Full states support exact visual trajectory replay; replay does not depend on a newly trained model. Replay requires the matching installed track hash.

Reward is a weighted sum of the four recorded features. Defaults are `progress=1`, `time=0.1`, `offtrack=2`, `smoothness=0.02`. Override through `RacingEnv(reward_weights={...})`. This keeps feature-level evidence available for inverse methods; labels make no assumption that a human run is optimal. The physical `states` array does not include all episode bookkeeping; use `snapshot()` / `restore()` for exact in-memory branching, or replay actions from the same seed and reset options for reconstruction rather than treating a state row as a complete environment checkpoint.

For track-generalization experiments, collect training data only on selected circuits and evaluate separately on held-out circuits. The ordinary training split is by run, not automatically by track. Keep the test set out of parameter tuning.

For setup and controls, see the [quick start](../README.md). For evaluation, snapshots and data-loading examples, see the [Python API](api.md). For development checks, see [Contributing](../CONTRIBUTING.md).
