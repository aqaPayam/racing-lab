# Python API and reproducible experiments

## Environment

```python
from racing_lab import RacingEnv, available_tracks
print(available_tracks())
env = RacingEnv("monza", max_seconds=300,
                reward_weights={"progress": 1., "time": .1, "offtrack": 2., "smoothness": .02})
obs, info = env.reset(seed=42, options={"lateral_jitter": .5})
obs, reward, terminated, truncated, info = env.step([0., .6, 0.])
env.close()
```

`track` accepts a bundled identifier or a path to a custom JSON circuit. Actions are steering [-1,1], throttle [0,1], brake [0,1]; finite out-of-range inputs are clipped. There are 39 float32 observations; their exact order is `racing_lab.env.OBS_NAMES`. `terminated` means finishing all lap gates (including recovered invalid laps); `info["success"]` excludes recovery; `truncated` means the simulation timeout. Reset after either. The default render mode is None; `human` opens a window, while `rgb_array` returns an offscreen RGB array (requires the play extra).

Gymnasium registration is also available after `import racing_lab`: `gymnasium.make("RacingLab-v0", track="monza")`.

`dt` sets the decision interval in seconds (default 1/60), and `laps` sets the required lap count (default one). Physical integration uses substeps no longer than 1/240 second. Snapshots require matching clock and lap settings; recordings retain both. Sector timing in the base environment describes the first lap. The SAC adapter maintains its own per-lap validity and timing.

For online SAC, use `racing_lab.rl.SACRacingEnv`, which adds relative controls, causal history, a 25 Hz clock and the paper-style task reward. Its actions and 168 observations differ from the base environment. See the [SAC guide](sac-baseline.md) for import examples, training and frozen-policy evaluation.

Choose `RacingEnv(render_mode="human", view="3d")` for the perspective renderer or `view="2d"` for the top-down view. The Python API defaults to 2D for compatibility; CLI driving defaults to 3D. `evaluate_policy(..., render=True, view="3d")` displays any policy in 3D. Offscreen RGB output supports both views. Rendering and scenery have no effect on the numerical simulation. C cycles cameras and V changes presentation in the interactive CLI. The optional `renderer` argument accepts `auto` (default), `gpu`, or `software`. Explicit `gpu` raises an error when unavailable; `auto` warns and falls back. Headless `step()` never initializes a renderer.

## Demonstrations and IO/IRL features

```python
from racing_lab import Recorder, load_trajectory, transitions, feature_sum

env.reset(seed=42)
recorder = Recorder(env, source="my-controller")
action = [0., .5, 0.]
transition = env.step(action)
recorder.append(action, transition)
path = recorder.save("data-controller", label="unlabeled", reason="experiment_stop")

trajectory = load_trajectory(path)
observations, actions, next_observations = transitions(trajectory)
phi = feature_sum(trajectory, gamma=.99)
```

Create a recorder immediately after reset and append once after each step. The optional `control_settings` dictionary is copied into recording metadata; CLI driving fills it automatically. It captures the action actually applied by the simulator, including clipping, and retains the reset seed/options. Use an honest `source`; the built-in supervised trainer selects `source=human`, `label=good` only. Imperfect or interrupted demonstrations are allowed and their completion status remains separate from their quality label.

Loading validates array shapes, finite values, action limits, terminal flags, timestamps, feature order, and reward reconstruction. It never loads pickle objects. `transitions` returns views into the loaded arrays. `feature_sum` aggregates `[progress_delta, -dt, -offroad_dt, -squared_action_change, -impact_mps, -recovery]`; this supports linear reward/cost formulations but does not implement an inverse solver or imply feature identifiability. With `gamma=1`, the time feature is simply negative trajectory duration. A fixed-duration dataset cannot identify a time preference from variation in that feature alone.

`python -m racing_lab.cli validate-data --data data` checks all NPZ recordings in a directory. Keep model NPZ files in a separate directory.

## Counterfactual rollouts

```python
snapshot = env.snapshot()
first = env.step([0., 1., 0.])
env.restore(snapshot)
second = env.step([0., 0., 1.])
```

Snapshots copy all wheel/drivetrain/sprung-body state, physical state, previous action, checkpoint bookkeeping, counters, reset metadata, and NumPy RNG state. Restore requires the same simulator, track hash, timeout, reward weights, and DrivingConfig. These are trusted in-memory Python structures, not a portable dataset format. The smaller `env.state()` array contains only seven physical quantities and is not sufficient to restore an episode.

## Evaluation and comparison

```python
from racing_lab import EvaluationConfig, evaluate_policy, save_report

config = EvaluationConfig(tracks=("monza", "spa"), episodes=5,
                          seed=42, max_seconds=300, lateral_jitter=1.)
report = evaluate_policy(my_policy, config, policy_name="my-irl-method")
save_report(report, "results/my-irl-method.json")
```

`my_policy(observation)` returns three controls. A `reset()` method, if present, is called at the beginning of each episode. Seed any randomness in your own model yourself; the environment seed cannot control an arbitrary callable's internals. Pass `None` as the policy to run the privileged geometric reference controller. Pass `driving=DrivingConfig(...)` to choose vehicle aids; a saved BC policy carries its training configuration. An optional `on_episode(row)` callback receives each completed result. `reward_weights` and `render` are keyword arguments.

Reports include:

- Configuration, simulator/package/dependency versions, timestep and reward weights.
- Per-episode seeds, track hashes, return, success, terminal/timeout flags, off-track fraction and mean squared action change.
- Per-track success rates and lap-time mean/population standard deviation **only for successful laps**. Time statistics are null if no lap finishes.
- An interruption flag; manually aborted partial episodes are omitted, while completed results are retained.

CLI evaluation additionally records the model file hash. Compare algorithms under identical settings and report completion rates alongside conditional lap times. Do not rank a policy that fails most laps using only its occasional fast finish. Small sample standard deviations are descriptive, not confidence intervals or evidence of statistical significance.

## Training and held-out tracks

```sh
python -m racing_lab.cli train --data data --tracks monza silverstone --completed-only --output models/bc.npz
python -m racing_lab.cli evaluate --model models/bc.npz --track spa --output results/spa-test.json
```

`--tracks` restricts both training and validation to the named circuits; reserve other circuits for testing. Without it, selected runs from all tracks may enter training. `--completed-only` is optional: leave it out when imperfect partial demonstrations are intentional. The built-in split is 80/20 by whole run (at least one validation run), not by individual frame. Exact duplicate observation/action trajectories are rejected. Near-duplicates and recordings from the same driving session can still be correlated; researchers should design session-level splits when needed.

The exported model report records selected file hashes, train/validation filenames, seed, epochs, runtime versions, and held-out action MSE. The saved NumPy model has no executable pickle. Training uses a fixed seed but bitwise reproducibility across numerical-library versions and hardware is not guaranteed.

## Important modeling limits

The local-road observation is useful for transfer but is **not a complete Markov state**: previous action (used in smoothness reward), lap checkpoints, timeout counters, and distant topology are omitted. A recurrent policy or a custom observation wrapper can expose additional state. Use snapshots when you need exact simulator branching. Do not infer formal MDP guarantees from Gymnasium compatibility alone.

There is no physical calibration or guaranteed globally optimal racing line. Barriers use a substepped rectangular-car constraint solver, not general continuous collision detection. Road checkpoints discourage major shortcuts but do not eliminate every corner cut. Ordinary frame drops preserve simulation timing; stalls over 0.5 seconds pause the interval. See [physics](physics.md) and [driving](driving.md) for the limits.

## Vehicle settings and supplemental commands

```python
from racing_lab import DrivingConfig

env = RacingEnv("spa_long", driving=DrivingConfig(automatic=False, traction_control=False, abs=False))
obs, info = env.reset(seed=42, options={"start_distance":100., "speed":20.})
transition = env.step([0., .5, 0.], commands={"shift":1})
```

`commands` optionally contains `shift` (-1/0/1), `direction` (-1/0/1), and `recover` (0/1). A forward shift applies only with manual gears; a direction change requires speed below 1.5 m/s. Requests are recorded even if a safety interlock prevents application. Omitted commands are zero. Recovery increments its counter and invalidates lap success. Unknown keys/invalid values raise before mutating state. Starting distance changes the start/finish origin; initial speed must be between -6 and 90 m/s. Keep these settings equal when comparing methods.

The standard Gymnasium action space remains three continuous controls with automatic gears by default. Custom algorithms may call the supplemental API directly, or expose commands through their own Gymnasium wrapper. The bundled three-output behavior-cloning learner requires automatic forward driving with no supplemental commands and rejects mixed vehicle-aid settings.

Schema 2 adds `vehicle_states`, `commands` and `contacts`. See the [array table](research-guide.md#schema-2-datasets). For reconstruction, initialize the same configuration/seed/options, then call `step(actions[t], commands=dict(zip(metadata["command_names"], commands[t])))`. Visual replay reads recorded states and does not rerun current physics. Legacy schema 1 files retain their historical field names and shapes.
