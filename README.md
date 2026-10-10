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

## Documentation

- [Driving and validation](docs/driving.md): steering aid, real-time scheduling, GPU setup, and verification.
- [Physics model](docs/physics.md): tyre forces, aero, load transfer, parameters, and compatibility.
- [Research guide](docs/research-guide.md): dynamics, tracks, observations, data schema, and limitations.
- [Python API and experiments](docs/api.md): dataset arrays, reward features, snapshots, reproducible evaluation, and held-out tracks.
- [Contributing](CONTRIBUTING.md): checks and compatibility rules.
- [Changelog](CHANGELOG.md): changes to the package, simulator and recordings.
- [Runnable examples](examples/): custom policy evaluation, data loading, and branching rollouts.

## What is implemented?

Eight circuit-inspired layouts; solid barriers with sliding impacts and explicit recovery; detailed 3D and redesigned 2D views; keyboard and configurable gamepad input; eight-speed automatic/manual transmission; individual wheel spin/locking; combined tire grip, aero and reduced pitch/roll load transfer; illustrative elevation/banking on longer tracks; synthetic engine/tire/impact audio; sectors, clean personal bests, passive ghosts, and offline run comparison.

The research API stays headless and deterministic. Versioned NPZ files record numerical observations, controls, supplemental commands, vehicle states, contacts, aid settings and reward features. A supervised behavior-cloning baseline and reproducible evaluation helpers are included; IO/IRL solvers and RL trainers can use the Python/Gymnasium interface.

**Package 0.7.0 / simulator 0.5.0 requires fresh demonstrations and retrained models.** There are now 39 observations, six reward features, and schema 2 recordings. Older recordings remain visually replayable. Keep new experiments in a directory such as `data-v05`.

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
