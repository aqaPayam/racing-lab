# Driving, rendering, and validation

Package 0.6.0 / simulator 0.4.0 addresses the steering sensitivity, slow-motion presentation, and software-graphics limitations of the previous version.

## Start driving

```powershell
.\.venv\Scripts\python -m racing_lab.cli play --track monza --data data-v04
```

Double-clicking `play.bat` also works, using the default `data` folder. W/S control throttle/brake; A/D steer. C cycles chase, nose and aerial cameras. V toggles 2D. R ends a run; G/B/U labels it. Select another circuit with 1–4 while labeling. All four layouts are still available: Monza, Silverstone, Spa and Suzuka.

Begin with a few low-speed laps to learn the compact circuit. Brake before sharp bends and progressively release the brake while turning. The car can still exceed tire grip; the keyboard adapter does not prevent every spin or keep you on the road.

## Steering adapter

Keyboard input ramps from center to full input over 0.65 seconds and returns from full input over 0.24 seconds. The maximum requested wheel angle is:

```text
min(0.45 radians, atan(3.4 * 13 / max(forward_speed², 1)))
```

The 13 m/s² value defines a conservative geometric input envelope, not a guarantee of actual lateral acceleration. Tire forces, aero and understeer still determine the resulting motion. The adapter reads speed and keys only; it does not follow a racing line or correct heading. `--raw-steering` disables the speed-dependent limit while retaining the input ramp.

**Research policies always send direct actions to `env.step`.** Human recordings store those applied actions plus the keyboard settings in `metadata.control_settings`. Replay and imitation therefore need no keyboard emulation. A learned policy should not pass its predictions through this adapter again.

## Timing

Interactive play accumulates elapsed wall time and executes the required number of 1/60-second steps before drawing. Rendering at 20 FPS normally runs three physics steps per frame; rendering at 60 FPS runs one. Every executed step is recorded. Physics itself uses four force-integration substeps per public step.

An individual stall longer than 0.5 seconds pauses that interval and displays a stall message, avoiding an unseen burst of input after window dragging or debugging. This exception intentionally breaks wall-time synchronization. If it happens repeatedly, use 2D or another renderer. Physics cannot maintain real time on a machine too slow to execute the required steps.

Replay selects recorded states according to elapsed time and may skip presentation frames. Rendered policy evaluation also skips drawing when behind, but still computes every policy action and physics step. Headless evaluation runs as quickly as possible. Calling `env.step` yourself retains deterministic fixed-step semantics; your application controls its own wall-time pacing.

## Graphics and setup

Install `.[play]` or `.[all]` to include Pygame and ModernGL. The `auto` renderer attempts OpenGL 3.3 and reports a warning before using the software fallback. Explicit `--renderer gpu` requires a compatible context; `--renderer software` selects the original portable renderer. The HUD shows GPU or SOFTWARE. Both support offscreen RGB output. Headless learning needs neither graphics dependency.

GPU graphics include depth testing, four-sample antialiasing, a sun shadow map, distance haze, and locally generated surface textures. The car has tapered body panels, wings, cockpit/helmet, mirrors, suspension links, rounded tires, rims and rotating spokes. Front wheels follow the actual steering angle. Rotation uses integrated traveled distance divided by a 0.36 m tire radius. It assumes rolling without slip; individual wheelspin/lockup is not simulated.

The scene adds braking-distance boards before detected bends, 25 m reflector spacing, woodland and spectators. Boards refer to distance along these compact layouts; they are not official circuit brake markers or a guarantee that a particular speed is safe. Scenery remains non-collidable. Graphics are procedural and stylized, not photorealistic.

The renderer uses the [ModernGL context and framebuffer API](https://moderngl.readthedocs.io/en/stable/reference/context.html). No third-party art assets or network asset downloads are required.

## Verification

Automated checks cover:

- Exactly 600 simulation steps over ten scheduled seconds at 20, 30, 60 and 144 presentation FPS.
- Approximately 55 m traveled in one wall-scheduled second at 200 km/h, including the 20 FPS case.
- A 150 ms steering tap at 200 km/h, followed by release: less than 1.5 m lateral displacement over three seconds, less than one degree of heading change, and yaw settling below 0.001 rad/s.
- Low-speed constant-radius geometry, mirrored lane-change response, sub-limit turns followed by firm braking, finite aggressive-input behavior, and axle friction-circle bounds.
- Deterministic action replay, snapshot restoration, applied-action recording, BC training, and successful reference-driver laps through every gate on all four tracks.
- GPU rendering, software rendering, camera switching, track rebuilding, wheel rotation, and no rendering-induced change to simulator state.

Run the machine-specific presentation benchmark:

```powershell
.\.venv\Scripts\python examples/benchmark_rendering.py --renderer gpu --seconds 5
```

It writes `results/render-benchmark.json`. On the development Windows PC with an RTX 4060, the five-second offscreen presentation check measured approximately **18 ms median / 22 ms p95** per frame and a **0.999 simulation-to-wall-time ratio**. This is a local measurement, not a performance promise for other machines. Shader/scene warm-up is excluded; physics and presentation run together during measurement.

## Compatibility and remaining limits

The changed tire-force combination, delivered-force load transfer, axle stiffness and brake balance require **fresh simulator 0.4.0 demonstrations and retrained models**. Existing recordings remain untouched and visually replayable. Use `data-v04` to separate new experiments; versioned `data-*` folders are git-ignored.

These tests establish reproducibility and specific handling behaviors. They do not validate the model against an instrumented F1 car. There is still no elevation, suspension dynamics, tire temperature/wear, individual wheel-slip dynamics, damage or barrier collision response. See the [physics specification](physics.md). Your driving feedback remains useful for evaluating the subjective control and camera feel.
