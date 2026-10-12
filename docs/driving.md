# Driving and practice

## Start

```powershell
.\.venv\Scripts\python -m racing_lab.cli play --track monza_long --data data-v05
```

Double-click `play.bat` for compact Monza with default settings. Use `--view 2d` for the redesigned top-down view. There are eight layouts; choose `monza`, `silverstone`, `spa`, `suzuka`, or append `_long` for their longer counterparts.

| Input | Action |
| --- | --- |
| W/S or up/down | Throttle/brake |
| A/D or left/right | Steer |
| X | Select reverse/forward below 1.5 m/s; throttle moves in the selected direction |
| H | Recover to nearby centerline; invalidates the lap |
| Q/E | Downshift/upshift with `--manual-gears` |
| C | Cycle chase/cockpit/aerial, or follow/overview in 2D |
| V | Switch 2D/3D |
| M | Mute/unmute audio |
| P | Show/hide personal-best ghost |
| R | Stop run for labeling |
| G/B/U while stopped | Save good/bad/unlabeled and start next run |
| 1–4 / 5–8 while stopped | Select compact / longer circuits in the order above |
| Escape | Save current run unlabeled and exit |

Brake before sharp bends and release the pedal progressively while turning. Tires have limited combined grip. The 2 m shoulders provide a margin before the barriers, but still count as off-track. An impact dissipates speed and can spin the car. Reverse can help you maneuver away; H is an explicit fallback and is recorded.

## Controls and aids

Keyboard steering rises over 0.65 s and returns over 0.24 s. Its maximum requested wheel angle is `min(.45, atan(3.4 * limit / max(speed²,1)))`, with `limit=13 m/s²`. `--steering-limit` accepts 3–30. This is an input envelope, not guaranteed lateral acceleration or automatic racing-line control. `--raw-steering` disables the speed limit but retains the ramp. Research actions always bypass the keyboard adapter.

Automatic gears, traction control and ABS default on. The HUD labels all three. `--manual-gears --no-tc --no-abs` selects a more demanding configuration with wheelspin and lockups. These are explicitly modeled aids for this game, not representations of legal/current F1 equipment. Vehicle configuration and device settings are saved separately in demonstrations and evaluation reports.

```sh
python -m racing_lab.cli controllers
python -m racing_lab.cli play --track monza_long --gamepad 0
```

Default gamepad mapping: axis 0 steering, axis 5 throttle, axis 2 brake; triggers range from -1 (released) to +1 (pressed). Override with `--steer-axis`, `--throttle-axis`, `--brake-axis` after checking your device. Steering has a 0.06 deadzone and 1.7 response exponent, with no keyboard speed limiter. Buttons 4/5 shift down/up, 3 selects reverse/forward, and 6 recovers. Keyboard camera/label controls remain available. Mappings vary by OS/controller; no physical wheel, force-feedback, or controller-specific calibration is bundled.

## Feedback and practice

RPM LEDs, gear, pedal bars, g-forces, sectors, aid settings, and a minimap appear in both views. Synthetic audio follows engine RPM/throttle, tire slip, shifts and impacts. `--volume` accepts 0–1; `--mute` starts silently. Missing audio hardware falls back to silent play. `--camera-motion` adds subtle cockpit motion; it is off by default.

The HUD times three equal-distance sectors. Clean completed personal bests are saved locally in `results/practice` (override `--practice`). Clean means no off-road steps, collision or recovery. The best lap supplies a passive ghost and progress-aligned time delta on later attempts with the same simulator/track/aid/initial-state configuration. GPU ghosts are translucent; the software fallback uses a solid cyan car; 2D uses an outline. Ghosts have no collision or physics effect. Good demonstration labels and clean personal bests are separate concepts.

```sh
python -m racing_lab.cli compare data-v05/FIRST.npz data-v05/SECOND.npz --output results/comparison.html
```

Open the resulting HTML locally for speed, braking and trajectory plots; a companion JSON provides the underlying curves. Both recordings must use the same track hash. Nothing is uploaded.

## Race a frozen SAC ghost

Install `.[play,rl]`, then run:

```powershell
.\.venv\Scripts\python -m racing_lab.cli play --track monza --sac-ghost results/sac-monza-200k-seed0-20261011/final.pt
```

`play-sac-ghost.bat` uses this workspace's trained Monza checkpoint by default; supply another local checkpoint as its first argument. A fresh clone needs its own trained checkpoint. Preparation performs deterministic inference once and caches the **first strictly clean standing-start lap**. It refuses a failed first lap, incompatible simulator/observation layout, different circuit, or mismatched player vehicle aids. It does not search for a fastest lap, import human data, update networks, or alter SAC training settings.

Space/Enter begins a three-second countdown. Until GO, neither simulation time nor the ghost advances. Both cars then share your simulation clock, including pauses after rendering stalls. Your driving uses the checkpoint's timestep (25 Hz for the preserved SAC recipe) and the same automatic gearbox, TC and ABS settings. Presentation targets 60 FPS. Human keyboard/gamepad input shaping still differs from SAC's incremental controls, and is saved in the comparison metadata.

The teal car is a visual ghost with no collision, slipstream or physics effect. It remains at the finish after completing its reference lap. Its car mesh is hidden within 2 m of yours to avoid overlapping graphics; the timing and map marker remain available. Orange is your racing line; teal is SAC's. Both lines appear in the minimap and in 2D; V followed by C selects the track overview. L toggles lines, P toggles the ghost car, C cycles cameras, R stops an attempt, and Space/Enter while stopped saves/restarts it. Esc saves and exits. Circuit-selection and G/B/U labeling are reserved for ordinary recording mode.

The HUD compares first-passage times at equal progress. Negative means ahead of SAC; positive means behind. Final lap times require every gate and zero off-road steps, counted collisions and recoveries, matching the SAC strict validity criterion. Failed/unfinished laps have no invented lap time or finish gap. The trained reference in this workspace is **53.16 s from rest**; its **49.64 s flying best** is a different start condition.

Each finished or stopped attempt saves an unlabeled numerical recording plus standalone HTML/JSON speed, braking, racing-line and lap-time comparisons in `results/sac-ghost/sessions`. Override `--ghost-results` or `--ghost-cache` to choose other local locations. Ghost-practice recordings use the source tag `human_ghost_comparison` and stay separate from the ordinary `--data` directory. These are practice comparisons, with one fixed reference and no independent-trial or algorithm-ranking claim. Personal-best caches now include timestep in their identity; older cache files remain on disk.

## Graphics and timing

Install `.[play]` or `.[all]`. `--renderer auto` attempts OpenGL 3.3 and warns before software fallback; explicit `gpu` requires a compatible context. Both renderers support RGB arrays. The GPU renderer includes depth testing, sunlight/shadows, antialiasing and procedural scenery. Wheels follow independent physical spin angles and front steering; body pitch/roll follows the reduced suspension model. Barriers and road profile use the physics geometry. Terrain is illustrative and graphics remain stylized. The software fallback uses a simpler car mesh and painter ordering; the GPU renderer provides the detailed body and depth-correct scene.

Interactive play executes 60 fixed physics steps per simulated second, independently of presentation FPS. Ordinary 20 FPS rendering runs about three steps per frame. Every executed step is recorded. A stall longer than 0.5 s pauses that interval to avoid applying unseen inputs. Scene/shader initialization can take time, especially for longer terrain layouts; initialization is excluded from driving timing. Repeated stalls require a lighter renderer or 2D. No software can guarantee realtime operation below its required physics throughput.

Replay may skip presentation frames while selecting recorded states by elapsed time. Rendered evaluation drops drawing when behind but computes every policy/physics step. Headless stepping has no wall-clock pacing. Run `python examples/benchmark_rendering.py --renderer gpu --seconds 5` for a machine-specific check; see [validation](validation.md).

## Compatibility

This update changes physics, observations, reward features and recording schema. **Collect fresh simulator 0.5.0 demonstrations and retrain models**, preferably in `data-v05`. Older files are preserved and can be visually replayed when their track hash is installed. The simple BC baseline requires automatic forward driving without supplemental commands; use a custom learner/action wrapper for manual driving or recovery research.
