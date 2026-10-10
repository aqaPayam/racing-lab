# Research guide

## Tracks

`monza`, `silverstone`, `spa`, and `suzuka` are selectable with `--track`. These are original, hand-authored **approximations inspired by circuit shapes**, not surveyed replicas or official assets. Distances and widths are scaled for this prototype. There is no elevation or banking; Suzuka is explicitly adapted to avoid a bridge crossing. No F1 branding assets are included, and this project is not affiliated with Formula 1 or the circuits.

Add a JSON file under `racing_lab/tracks/` with `name`, `version`, `width` (metres), and an ordered loop of `[x, y]` centerline control points. Coordinates use screen-style positive y downward. The road is interpolated with a periodic Catmull-Rom curve. Avoid intersecting roads and bends closer together than their widths. Increment the version whenever geometry changes; recordings also store a SHA-256 digest of the exact file.

## Observations and dynamics

The 20 float32 observations contain normalized speed, lateral speed, yaw rate, steering angle, lateral road offset, sine/cosine of road-relative heading, road half-width, and six upcoming centerline positions expressed in car coordinates (10, 25, 50, 90, 150, 230 metres ahead). This is a compact local-road description, not global coordinates or images. The preview coordinates and known width describe the nearby road; they do not encode every possible custom-track topology. See `OBS_NAMES` in `racing_lab/env.py` for ordering and normalization.

The simulator uses a simplified bicycle model, steering slew limits, speed-dependent cornering grip, lateral slip, drag, and off-road deceleration. It is an approachable game model, **not an engineering-accurate F1 model**. Cars cannot reverse. Ordered road checkpoints discourage shortcuts; they do not constitute exact continuous boundary enforcement between gates.

### Scale and speed (simulator 0.2.0)

All positions and dimensions use metres, speeds use metres/second internally, and the HUD converts to km/h. The car is 5.5 m long and 1.9 m wide, with a 3.4 m wheelbase. The renderer uses these same dimensions without enlarging the car. The default following camera uses 8 pixels/metre at rest and widens toward 4 pixels/metre at speed, looking ahead along the heading so braking points stay visible. The overview adds a locator ring around the physically scaled car. Both views include a 10 m scale bar.

Road widths are 14 m (Monza/Suzuka) or 16 m (Silverstone/Spa): about 7.4–8.4 car widths. These are chosen game widths, not measurements of the named real circuits. Layouts remain compact, approximately 2.2–2.6 km. Straight-line terminal speed is about 201 km/h under the drag model, even though the numerical speed cap is 306 km/h. Nominal lateral grip is 17 m/s² (about 1.73 g), so a 50 m radius bend suggests about 105 km/h under the steady-state bicycle approximation. Acceleration, braking and grip are deliberately approachable game settings, not calibrated real F1 performance.

Road-edge checks include the car's width and heading-dependent length projected across the local road tangent. This is a rectangular-envelope approximation, not exact tyre contact simulation on curved boundaries. Legacy track files are retained for replay of version 0.1.0 recordings. Old recordings remain untouched; training rejects incompatible simulator versions instead of silently mixing different dynamics. Current package releases retain simulator 0.2.0 compatibility; record new demonstrations when migrating from simulator 0.1.0.

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
| timestamps | T+1 | Simulation seconds |
| metadata | scalar JSON | Label, source, track version/hash, seed, schema, names, weights, outcome, stop reason |

Manual exits/interruption are described by `reason` and do not falsely set a simulator timeout flag. Full states support exact visual trajectory replay; replay does not depend on a newly trained model. Replay requires the matching installed track hash.

Reward is a weighted sum of the four recorded features. Defaults are `progress=1`, `time=0.1`, `offtrack=2`, `smoothness=0.02`. Override through `RacingEnv(reward_weights={...})`. This keeps feature-level evidence available for inverse methods; labels make no assumption that a human run is optimal. The physical `states` array does not include all episode bookkeeping; use `snapshot()` / `restore()` for exact in-memory branching, or replay actions from the same seed and reset options for reconstruction rather than treating a state row as a complete environment checkpoint.

For track-generalization experiments, collect training data only on selected circuits and evaluate separately on held-out circuits. The ordinary training split is by run, not automatically by track. Keep the test set out of parameter tuning.

For setup and controls, see the [quick start](../README.md). For evaluation, snapshots and data-loading examples, see the [Python API](api.md). For development checks, see [Contributing](../CONTRIBUTING.md).
