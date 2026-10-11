# SAC with three-step returns

Racing Lab 0.8.0 implements an online, from-scratch adaptation of the SAC baseline in [Remonda et al., NeurIPS 2024](https://arxiv.org/abs/2407.16680). The learning implementation follows the [author's July 2024 source](https://github.com/dasGringuen/assetto_corsa_gym/tree/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4). It is a baseline for this simulator, not a reproduction of Assetto Corsa's vehicle model or reported lap times.

Every training run begins with fresh networks and an empty replay buffer. The runner has no demonstration, dataset, pretrained-policy or expert-controller input. Its replay consists only of transitions collected by its own exploratory actions. The reference used in reward and observations is the existing track centerline geometry, not an expert racing trajectory.

## Install and run

Activate the project's Python environment, then install the optional RL dependency:

```sh
python -m pip install -e ".[rl]"
python -m racing_lab.cli train-sac --track monza --steps 10000 --seed 0 --output results/sac-monza-seed0
python -m racing_lab.cli evaluate-sac --model results/sac-monza-seed0/final.pt --track monza
```

On Windows without activation, replace `python` with `.\.venv\Scripts\python.exe`. PyTorch is optional; ordinary imports, play, recording and the existing supervised trainer do not require it. CPU is the default. Use `--device cuda` with a CUDA-enabled PyTorch installation and supported GPU. Validation here used CPU; CUDA training has not been verified.

Watch a checkpoint with `evaluate-sac ... --render --view 3d`, or select `--view 2d`. Install `.[play,rl]` for rendering. `--track all` evaluates all eight layouts. The environment configuration is loaded from the checkpoint; evaluation uses deterministic actions and never updates networks or collects replay.

The default 10,000-decision run is a pilot. The paper trained for four wall-clock days on an RTX 4090, approximately eight million decisions. To request that approximate step budget use `--steps 8000000`; this is still a step-budget adaptation, not a claim of equal compute or full reproduction. Start by measuring runtime on your machine. Use a different output directory for every run; existing directories are rejected.

## Import from Python

```python
from dataclasses import replace
from racing_lab.rl import SACConfig, RacingTaskConfig, SACRacingEnv
from racing_lab.rl.sac import SACAgent
from racing_lab.rl.training import train_sac, evaluate_sac

result = train_sac("results/my-sac", track="monza", steps=10000, seed=0)
agent, metadata = SACAgent.load("results/my-sac/final.pt")
task = replace(RacingTaskConfig(**metadata["task"]), max_laps=4)
report = evaluate_sac(agent, ("silverstone",), task, episodes=1)

env = SACRacingEnv("monza", task)
observation, info = env.reset(seed=0)
action = agent.act(observation, deterministic=True)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Actions returned by this agent are control increments in [-1, 1]. Apply them through `SACRacingEnv`, which maintains the requested controls and observation history. They cannot be passed directly to the ordinary absolute-action `RacingEnv` or its existing evaluator.

## Preserved learning settings

Defaults use three hidden layers of 256 ReLU units in the actor and each of two critics, Xavier-uniform weights, zero biases, discount 0.992, three-step returns, batch 128, Adam learning rates 0.0003 for actor/critics/entropy, and target coefficient 0.005. Target entropy is minus action dimension, with initial temperature 1. Gaussian log standard deviations are clipped to [-20, 2]; actions use tanh and the source's Jacobian correction with epsilon 1e-6. Updates preserve actor, entropy-temperature, critic, target-critic order. One update is performed per collected decision after 2,000 random-action decisions and sufficient replay availability.

Source: [SAC](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/algorithm/discor/discor/algorithm/sac.py), [networks](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/algorithm/discor/discor/network.py), and [configuration](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/config.yml).

## Paper/source decisions and game adaptations

| Component | Implemented decision and reason |
| --- | --- |
| Policy timing | Fixed 25 Hz decisions. Each 40 ms decision uses ten 4 ms physical integration/contact steps. Normal gameplay retains its default 60 Hz/four-substep timing. This avoids uneven action repeats and preserves the physical three-step horizon of 0.12 seconds. |
| Controls | Preserve source increments: normalized steering rate 600/180 per second, pedal rate 12 per second, requested control clipping, and reset targets [0, -1, -1]. Convert pedals to the game's [0, 1] interface. Existing front-wheel actuator dynamics remain in force; steering-wheel mechanics differ from Assetto Corsa. |
| Observations | Substitute Racing Lab's 39 numerical sensors for Assetto Corsa telemetry/range/curvature signals. Preserve source history structure: current observation, three preceding absolute-control samples in channel-major order, current applied-control feedback, then three preceding observations. Total: 168 numbers. Historical frames contain all 39 game sensors, including road preview; the original history contains only its basic telemetry/rays and excludes its separately appended curvature/out-of-track channels. This content difference is a sensor substitution. Missing history is zero-filled until three preceding samples exist. This follows code; the paper's simpler history description is different. Ordering is exported in every run. |
| Reference/reward | Use existing centerline as the geometric reference. Preserve source reward scale: speed in km/h times (1 minus absolute centerline gap in meters divided by 12), divided by 300. Substitute planar speed magnitude for Assetto Corsa's speed channel. Add no collision, smoothness or recovery reward terms. |
| Episodes | Preserve multi-lap driving and 600-second time limit. End early when at least three wheel centers are off-road or speed stays below 5 km/h for over two seconds. These low-speed settings follow the paper; released code instead uses 1 m/s and ten seconds. There is no launch controller or recovery action. |
| Evaluation | Use separate environments, deterministic mean actions and a four-lap evaluation cap. Ordinary standing starts make repeated evaluations deterministic repetitions, even with different seeds. `--jitter 1` explicitly changes resets for robustness evaluation; do not pool this condition with fixed-start results. Independent training seeds remain necessary. |
| Replay and evaluation schedule | Choose published capacity 10 million and evaluation every 100,000 decisions, checkpoint every 200,000. Released current code uses 8 million and disables periodic evaluation. Exact evaluation multiples are honored even inside episodes; final checkpoint selection is final-budget, not best-test-lap selection. |
| Replay storage | Allocate float32 storage in lazy chunks, while preserving uniform sampling and configured capacity. At 168 observations, full 10-million-row storage approaches 13.7 GB before framework overhead. `--replay-capacity 1000000` reduces this cost but changes the published setting; the override is recorded. |
| Episode-boundary correction | Flush pending returns on every reset or rollout cutoff. True failures have zero bootstrap. Time limits retain bootstrap with gamma raised to the actual tail length. The source uses a fixed gamma cubed and does not consistently flush time-limit boundaries. This correction prevents mixing episodes; full-length and terminal cases match source tests. |
| Vehicle, track and aids | Existing approximate layouts and generic vehicle model replace Assetto Corsa. Automatic gearbox is preserved. Racing Lab's existing TC and ABS defaults remain enabled; original aid equivalence is unverified. `--no-tc`/`--no-abs` are explicit task changes. Four compact/long circuit pairs share geometry families and must stay in the same dataset split. |
| Runtime and random streams | Support headless synchronous CPU/CUDA stepping rather than the original real-time asynchronous Assetto Corsa process. Use separate seeded replay/action-space RNGs; deterministic evaluation does not consume training Torch randomness. These streams differ from the author's global NumPy sampling. Current PyTorch replaces the original 1.12.1 stack; versions, device and thread count are recorded. |

For a three-step transition, the target is discounted reward sum plus gamma cubed times the target soft value. The soft value is the smaller target critic value plus the learned temperature times action entropy. Terminal targets omit this value; a truncated one- or two-step tail uses gamma or gamma squared. The source update order is preserved even though many other SAC packages choose a different order.

## Outputs and evaluation meaning

Each output directory contains:

- `run.json`: immutable run specification, seed, track/configuration/code hashes, source revision, package/runtime versions and the explicit no-expert-data declaration.
- `learning.csv` and `episodes.csv`: losses, temperature, update/replay counts, progress, validity and episode-end reasons.
- `evaluation_initial.json`, periodic evaluation reports, and `evaluation_final.json`: frozen-policy driving measurements.
- `final.pt` and periodic checkpoints: policy, critics, optimizer state and configuration. Replay/simulator/RNG state is not saved for training resume; loading supports inference, not an exact resumed experiment.
- `result.json`: completion phase, decisions, physical substeps, simulated time, evaluation interaction count, elapsed wall time and final evaluation summary. Interruptions save a checkpoint; evaluation failures do not claim completion.

A valid lap must pass every gate, have no off-road steps, no counted collisions and no recovery. This is stricter than the original three-wheel episode-termination rule and stricter than the base game's success flag. Report completion/validity before conditional valid-lap times. Failed laps never receive an invented lap time. Fixed-start repetitions are not independent samples for confidence intervals.

Evaluation during training uses the training circuit with a separate environment and seed. It is not a held-out-track test and does not select a best checkpoint. Choose training/validation/test circuit families before a full benchmark; test outcomes must not tune budgets or settings.

## Verification and limits

Offline test fixtures preserve the exact licensed author source plus SHA-256 hashes. Tests compare initialization and three successive actor/temperature/critic/target updates on fixed numerical inputs, as well as full-length and terminal replay returns. Additional tests cover truncated tails, reset isolation, circular-buffer retention, controls/history, policy timing, checkpoint inference, evaluation isolation and fresh-buffer training reproducibility.

A CPU verification pilot used the default 256-by-256-by-256 networks, 2,000 random decisions followed by 1,000 updates, for 3,000 total decisions on Monza. It used no external data and produced finite losses/checkpoints/evaluation reports. It did not complete a valid lap. This validates the training path, not convergence or the paper's racing performance. Long-budget multi-seed experiments remain future benchmark work.

See the [paper audit](online-rl-paper-audit.md) for broader source comparisons and [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution.
