# Research guide

## Tracks

Eight bundled layouts are available. All are original hand-authored approximations inspired by circuit shapes, not official assets or surveyed replicas. The four compact layouts retain their geometry and hashes for historical replay. The four `_long` variants extend the proportions and add smooth illustrative height/banking profiles. Suzuka remains a non-crossing adaptation.

| CLI identifier | Approximate length | Road width | Profile |
| --- | --- | --- | --- |
| monza / monza_long | 2.2 km / 5.3 km | 14 m | Flat / gentle height and bank |
| silverstone / silverstone_long | 2.2 km / 5.6 km | 16 m | Flat / rolling |
| spa / spa_long | 2.6 km / 6.8 km | 16 m | Flat / pronounced elevation |
| suzuka / suzuka_long | 2.6 km / 5.8 km | 14 m | Flat / rolling |

Use `Track(name).length` for exact interpolated metres. Add a JSON file with `name`, `version`, `width` and ordered `[x,y]` centerline control points. Coordinates are metres, with positive y downward in the map. The closed curve uses periodic Catmull-Rom interpolation. Do not repeat the first control point or create intersecting/overlapping roads. Optional `shoulder` is metres per side (default 2). Optional `profile` rows are `[lap_fraction,height_m,bank_degrees]`, ordered from fraction 0 to less than 1. A periodic cubic Hermite curve supplies height, its grade derivative, and bank. Increment track versions when geometry changes; recordings store the exact file hash.

## Observations and actions

The **39 float32 observations** begin with the previous 20 fields: normalized forward/lateral speed, yaw rate, steering, lateral offset, sine/cosine of relative heading, road width/40, and six upcoming centerline points in car coordinates (10, 25, 50, 90, 150, 230 m). They add gear, RPM, four wheel angular speeds, four slip ratios, pitch/roll and their rates, local grade/bank, shift interruption, reverse flag, and barrier contact. `racing_lab.env.OBS_NAMES` specifies exact order and normalization. These are direct numerical inputs, never image observations.

Actions remain `[steering,throttle,brake]` in `[-1,1] × [0,1] × [0,1]`. Policies bypass keyboard shaping. Vehicle aids are environment configuration and therefore apply equally to a human or policy using that configuration. Metadata distinguishes them from input-device settings. For manual/reverse/recovery studies, `step(action, commands={...})` supports supplemental commands; the minimal three-output BC baseline deliberately rejects demonstrations requiring them. A custom Gym wrapper can incorporate them into an expanded action space.

The observation is intentionally compact, **not a complete Markov state**. It omits previous action, filtered accelerations, shift cooldown, lap counters, and distant road profiles/topology. Use an observation wrapper/recurrent model where needed, and snapshots for exact branching. See the [physics model](physics.md) and [API](api.md).

## Schema 2 datasets

NPZ files contain numerical arrays and a JSON metadata scalar; `numpy.load(...,allow_pickle=False)` is sufficient. Public `load_trajectory()` additionally validates them. `T` is the number of transitions.

| Array | Shape | Meaning |
| --- | --- | --- |
| observations | T+1 × 39 | Initial and successor observations |
| states | T+1 × 7 | x, y, heading, forward/lateral velocity, yaw rate, steering |
| vehicle_states | T+1 × 22 | Named drivetrain, sprung-body, and four-wheel states |
| actions | T × 3 | Applied continuous controls |
| commands | T × 3 | Requested shift (-1/0/1), direction (-1/0/1), recovery (0/1) |
| contacts | T × 4 | Barrier contact, peak closing speed m/s, collision/recovery counters |
| rewards | T | Scalar weighted reward |
| reward_features | T × 6 | Progress, negative dt, negative offroad dt, negative action-change squared, negative impact speed, negative recovery |
| telemetry | T × 13 | Named g-forces, downforce, slip angles, RPM, gear, pitch/roll, grade/bank, active TC/ABS |
| terminated / truncated | T | Boolean lap-finish and timeout flags |
| timestamps | T+1 | Simulation seconds |
| metadata | JSON scalar | Versions, ordered names, track hash, seed/reset, device/aid configuration, parameters, label/source, outcome |

Metadata is the authoritative field order. Wheel order is front-left, front-right, rear-left, rear-right. Vehicle states contain ten scalar fields followed by four angular speeds, four spin angles, and four slip ratios. New files use schema 2 / simulator 0.5.0. Schema 1 files remain loadable for analysis and visual replay; their dimensions are described by their metadata, not current constants.

Default reward weights are `progress=1,time=.1,offtrack=2,smoothness=.02,collision=1,recovery=20`. Override through `RacingEnv(reward_weights={...})`. These features support linear IO/IRL objectives but do not identify a unique reward or implement an inverse solver. Equal-duration demonstrations do not distinguish a time preference from their time feature alone. A human's good label denotes their judgment, not guaranteed optimality, clean driving, or completion.

Recordings store actual applied actions and requested supplemental commands. To reconstruct current runs, start the same version, track, reset seed/options and `DrivingConfig`, then replay both arrays. Seven-value state rows alone cannot restore the simulator. `snapshot()` / `restore()` includes hidden vehicle dynamics and bookkeeping for exact in-memory counterfactuals.

## Experimental discipline

Collect training runs on chosen tracks and reserve others for evaluation. The BC baseline splits by whole run and rejects duplicates, incompatible simulator versions, mixed vehicle aid configurations, and supplemental-command demonstrations. Keep correlated sessions separate when designing research splits. A model's saved aid configuration is reused in evaluation. Report completion, collisions, recoveries and off-track rates alongside conditional successful lap times. A fast failed run is not a successful fast lap.

The privileged geometric controller uses full track geometry and is a testing tool, not learned evidence. IO/IRL solvers and RL training are left to the researcher through the headless Python/Gymnasium API. Demonstrations, models, practice ghosts, and local reports remain git-ignored and are never uploaded automatically.

## Presentation and practice

Rendering/audio are optional and lazily imported. Scenery uses a private fixed RNG; it never consumes simulation randomness. 2D and 3D share one physics engine. The fixed-step scheduler maintains simulation time under ordinary frame drops and explicitly pauses stalls longer than 0.5 s. A severely overloaded computer cannot guarantee realtime play.

Practice ghosts are local presentation data, never labeled expert demonstrations. A best lap requires completion with no off-road steps, impacts, or recovery and is partitioned by track hash, simulator, aid configuration and initial state. Compare two same-track recordings with `python -m racing_lab.cli compare first.npz second.npz`; the standalone HTML and companion JSON show speed, braking and XY trajectory. The report preserves simulator versions; comparing historical physics is descriptive, not a controlled experiment.
