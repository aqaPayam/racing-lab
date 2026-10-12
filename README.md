# Racing Lab

**A Python racing environment with 3D and 2D views for learning from imperfect experts.**

Drive locally, record and label your runs, train a supervised imitation baseline, and compare policies through one numerical-observation Gymnasium environment. The simulator is designed as a small, inspectable starting point for IO, IRL and RL research.

![3D game view](docs/game-3d.png)

## Quick start

Requires Python 3.10+. From the project folder on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[all]"
.\.venv\Scripts\python -m racing_lab.cli play --track monza
```

After installation, double-click **play.bat**. On Linux/macOS, use `.venv/bin/python` instead of `.venv\Scripts\python`.

For the commands below, activate your environment (`.\.venv\Scripts\Activate.ps1` on PowerShell, or `source .venv/bin/activate` on Linux/macOS). If PowerShell blocks activation, use the virtual environment's Python executable directly; activation is optional.

| Control | Action |
| --- | --- |
| W / up | Throttle |
| S / down | Brake |
| A, D / left, right | Steering |
| C | Cycle chase / cockpit / aerial cameras (following / overview in 2D) |
| V | Switch between 3D and 2D |
| R | End run for labeling |
| G / B / U | Save good / bad / unlabeled, then begin another run |
| 1–4 / 5–8 between runs | Compact / longer Monza, Silverstone, Spa, Suzuka |
| X / H | Reverse at low speed / recover (invalid lap) |
| Q / E | Downshift / upshift in manual mode |
| M / P | Mute audio / toggle ghost |
| Esc | Save unlabeled and exit |

## Your first experiment

The game now starts in **3D**. Use `--view 2d` for the redesigned top-down presentation. Replay and rendered evaluation also support `--view 3d` and `--view 2d`. Camera changes are visual only; they never change dynamics, observations, actions, or recording formats.

Drive at least five varied runs and mark useful demonstrations **good**. Then:

```sh
python -m racing_lab.cli validate-data --data data
python -m racing_lab.cli train --data data --output models/bc.npz
python -m racing_lab.cli evaluate --model models/bc.npz --track monza --episodes 1 --render
python -m racing_lab.cli evaluate --model models/bc.npz --track all --episodes 5
```

Recordings are in `data/`, models in `models/`, and evaluation reports in `results/`. These folders are git-ignored and never automatically uploaded. To inspect a recording visually, use `python -m racing_lab.cli replay data/RECORDING.npz`.

## Import into your research

```python
from racing_lab import RacingEnv, EvaluationConfig, evaluate_policy

env = RacingEnv(track="monza")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step([0., .5, 0.])
env.close()

# Replace this stationary example with your own policy.
report = evaluate_policy(
    lambda observation: [0., 0., 0.],
    EvaluationConfig(tracks=("monza",), episodes=2, max_seconds=2.),
    policy_name="my-policy",
)
print(report["summary"])
```

Headless installation: `pip install -e .`. Add `[play]` for Pygame and ModernGL, `[train]` for scikit-learn, or `[all]` for both. Importing the research API does not initialize a game window or load either optional framework.

## Online RL without demonstrations

Install `pip install -e ".[rl]"`, then run the source-based SAC baseline:

```sh
python -m racing_lab.cli train-sac --track monza --steps 10000 --output results/sac-monza-seed0
python -m racing_lab.cli evaluate-sac --model results/sac-monza-seed0/final.pt --track monza --render --view 3d
```

Training starts from random initialization and an empty replay buffer, using only its own simulator experience. The default budget is a pilot. [SAC documentation](docs/sac-baseline.md) describes preserved paper settings, game adaptations, outputs and limitations. `[rl]` is separate from `[all]` so gameplay installations do not need PyTorch.

## Compare online SAC and TD3

The [online benchmark guide](docs/online-benchmark.md) defines the active no-expert-data
comparison, source adaptations, five-seed protocol and interaction/compute accounting.
Install `pip install -e ".[benchmark]"`, then:

```sh
python -m racing_lab.cli benchmark-online --output results/online-monza-five-seeds --seeds 0 1 2 3 4 --steps 200000
```

Reports include learning curves, every seed's outcomes, held-out tests, training/evaluation
data counts, CPU/wall time and memory. The default queue is sequential; `--workers 4`
enables concurrent CPU training and records that timing condition.

## Drive alongside a trained SAC ghost

On this workspace, double-click **play-sac-ghost.bat** to race the trained 200k Monza checkpoint. For your own checkpoint:

```sh
python -m racing_lab.cli play --track monza --sac-ghost results/sac-monza-seed0/final.pt
```

Press **Space/Enter** for a three-second countdown; both cars start from rest. You drive normally while the frozen SAC lap appears as a passive teal ghost. **V** switches views, **C** selects the camera or 2D overview, **L** toggles racing lines, **P** hides/shows the ghost, and **Space** after a run starts another attempt. Your line is orange; SAC's is teal. The live gap is measured at equal track progress, with negative values meaning you are ahead.

The reference is the agent's first clean standing-start lap, rather than its fastest flying lap. The player matches its track, timestep and vehicle aids. Clean completed laps receive a finish-time comparison; dirty or unfinished laps retain their trajectories but have no valid lap time. HTML/JSON reports and your recordings stay in `results/sac-ghost/sessions`, with reference caches in `results/sac-ghost/cache`. Practice never trains or changes the checkpoint. Checkpoints are local artifacts and are not included in Git. See [driving and practice](docs/driving.md) for details.

## Documentation

- [Driving and validation](docs/driving.md): steering aid, real-time scheduling, GPU setup, and verification.
- [Physics model](docs/physics.md): tyre forces, aero, load transfer, parameters, and compatibility.
- [Research guide](docs/research-guide.md): dynamics, tracks, observations, data schema, and limitations.
- [Online RL paper audit](docs/online-rl-paper-audit.md): the active from-scratch benchmark shortlist, pinned sources, reproduction gaps and proposed implementation chain. [Broader literature](docs/benchmark-literature.md) and the [deferred demonstration protocol](docs/benchmark-protocol.md) retain later research proposals.
- [SAC baseline](docs/sac-baseline.md): online training/evaluation, numerical source parity, and the paper-to-game adaptation ledger.
- [Python API and experiments](docs/api.md): dataset arrays, reward features, snapshots, reproducible evaluation, and held-out tracks.
- [Contributing](CONTRIBUTING.md): checks and compatibility rules.
- [Changelog](CHANGELOG.md): changes to the package, simulator and recordings.
- [Runnable examples](examples/): custom policy evaluation, data loading, and branching rollouts.

## What is implemented?

Eight circuit-inspired layouts; solid barriers with sliding impacts and explicit recovery; detailed 3D and redesigned 2D views; keyboard and configurable gamepad input; eight-speed automatic/manual transmission; individual wheel spin/locking; combined tire grip, aero and reduced pitch/roll load transfer; illustrative elevation/banking on longer tracks; synthetic engine/tire/impact audio; sectors, clean personal bests, passive ghosts, and offline run comparison.

The research API stays headless and deterministic. Versioned NPZ files record numerical observations, controls, supplemental commands, vehicle states, contacts, aid settings and reward features. Supervised behavior cloning, online SAC with three-step returns, and reproducible evaluation helpers are included; additional IO/IRL solvers and RL trainers can use the Python/Gymnasium interface.

**The simulator 0.5.0 upgrade requires fresh demonstrations and retrained models.** It introduced 39 observations, six reward features, and schema 2 recordings. Package 0.8.0 retains this base interface. Older recordings remain visually replayable. Keep new experiments in a directory such as `data-v05`.

```sh
python -m racing_lab.cli play --track monza_long --data data-v05
python -m racing_lab.cli play --track spa_long --manual-gears --no-tc --no-abs
python -m racing_lab.cli play --track suzuka --view 2d
python -m racing_lab.cli controllers
python -m racing_lab.cli compare first.npz second.npz --output results/comparison.html
```

Automatic gears, traction control and ABS default on for accessible driving; their configuration is labeled and recorded. Keyboard steering is smoothed and limited with speed; direct policy actions bypass keyboard shaping. See [controls and practice](docs/driving.md) for gamepad mappings and aid settings.

`--renderer auto` chooses OpenGL 3.3 or warns before software fallback. Graphics/audio use locally generated assets. Headless research requires no graphics initialization. Install upgraded dependencies with `python -m pip install -e ".[all]"`.

![Top-down game view](docs/game.png)

See the [completed improvement checklist and validation](docs/validation.md) for the scope and checks.

**This is a research prototype, not a calibrated F1 simulator or established benchmark.** Layouts and elevation are approximations; Suzuka has no bridge. There is no measured tire calibration, temperature/wear, detailed suspension, damage, ERS, traffic, or weather. The privileged geometric reference controller is a test tool, not a learned baseline. Passing tests does not establish real-car fidelity or superiority to a human expert.

MIT licensed. Not affiliated with Formula 1 or the named circuits.
