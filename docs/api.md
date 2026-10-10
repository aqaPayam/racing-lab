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

`track` accepts a bundled identifier or a path to a custom JSON circuit. Actions are steering [-1,1], throttle [0,1], brake [0,1]; finite out-of-range inputs are clipped. There are 20 float32 observations; their exact order is `racing_lab.env.OBS_NAMES`. `terminated` means lap completion; `truncated` means the simulation timeout. Reset after either. The default render mode is None; `human` opens a window, while `rgb_array` returns an offscreen RGB array (requires the play extra).

Gymnasium registration is also available after `import racing_lab`: `gymnasium.make("RacingLab-v0", track="monza")`.

Choose `RacingEnv(render_mode="human", view="3d")` for the perspective renderer or `view="2d"` for the original view. The Python API defaults to 2D for compatibility; CLI driving defaults to 3D. `evaluate_policy(..., render=True, view="3d")` displays any policy in 3D. Offscreen RGB output supports both views. Rendering and scenery have no effect on the numerical simulation. C cycles cameras and V changes presentation in the interactive CLI.

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

Create a recorder immediately after reset and append once after each step. It captures the action actually applied by the simulator, including clipping, and retains the reset seed/options. Use an honest `source`; the built-in supervised trainer selects `source=human`, `label=good` only. Imperfect or interrupted demonstrations are allowed and their completion status remains separate from their quality label.

Loading validates array shapes, finite values, action limits, terminal flags, timestamps, feature order, and reward reconstruction. It never loads pickle objects. `transitions` returns views into the loaded arrays. `feature_sum` aggregates `[progress_delta, -dt, -offroad_dt, -squared_action_change]`; this supports linear reward/cost formulations but does not implement an inverse solver or imply feature identifiability. With `gamma=1`, the time feature is simply negative trajectory duration. A fixed-duration dataset cannot identify a time preference from variation in that feature alone.

`python -m racing_lab.cli validate-data --data data` checks all NPZ recordings in a directory. Keep model NPZ files in a separate directory.

## Counterfactual rollouts

```python
snapshot = env.snapshot()
first = env.step([0., 1., 0.])
env.restore(snapshot)
second = env.step([0., 0., 1.])
```

Snapshots copy physical state, previous action, checkpoint bookkeeping, counters, reset metadata, and NumPy RNG state. Restore requires the same simulator, track hash, timeout, and reward weights. These are trusted in-memory Python structures, not a portable dataset format. The smaller `env.state()` array contains only seven physical quantities and is not sufficient to restore an episode.

## Evaluation and comparison

```python
from racing_lab import EvaluationConfig, evaluate_policy, save_report

config = EvaluationConfig(tracks=("monza", "spa"), episodes=5,
                          seed=42, max_seconds=300, lateral_jitter=1.)
report = evaluate_policy(my_policy, config, policy_name="my-irl-method")
save_report(report, "results/my-irl-method.json")
```

`my_policy(observation)` returns three controls. A `reset()` method, if present, is called at the beginning of each episode. Seed any randomness in your own model yourself; the environment seed cannot control an arbitrary callable's internals. Pass `None` as the policy to run the privileged geometric reference controller. An optional `on_episode(row)` callback receives each completed result. `reward_weights` and `render` are keyword arguments.

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

There is no physical calibration, exact tyre-contact boundary model, continuous collision detection, or guaranteed globally optimal racing line. Road checkpoints discourage major shortcuts but do not eliminate every corner cut. Human timing follows simulation steps; rendering overload slows wall-clock play. Clearly distinguish these assumptions from properties of real-world driving when reporting results.
