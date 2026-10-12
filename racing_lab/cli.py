import argparse
import json
from pathlib import Path
import numpy as np
from .env import RacingEnv, DT
from .track import names, Track, TRACK_DIR
from .data import Recorder, load
from .learning import Policy, train
from .controls import KeyboardController, GamepadController
from .vehicle_model import DrivingConfig
from .practice import PracticeBook

from .timing import FixedStepClock


def play(args):
    import pygame

    reference = None
    if getattr(args, "sac_ghost", None):
        from .agent_ghost import prepare_sac_ghost

        reference = prepare_sac_ghost(args.sac_ghost, args.track, args.ghost_cache)
    env = RacingEnv(
        args.track,
        dt=reference.identity["dt"] if reference else DT,
        max_seconds=args.seconds,
        view=getattr(args, "view", "3d"),
        renderer=getattr(args, "renderer", "auto"),
        driving=DrivingConfig(
            automatic=not getattr(args, "manual_gears", False),
            traction_control=not getattr(args, "no_tc", False),
            abs=not getattr(args, "no_abs", False),
        ),
    )
    env.reset(seed=args.seed)
    if reference:
        reference.validate_player(env)
    env.render()
    clock = pygame.time.Clock()
    gamepad = getattr(args, "gamepad", None)
    controller = (
        GamepadController(
            gamepad,
            getattr(args, "steer_axis", 0),
            getattr(args, "throttle_axis", 5),
            getattr(args, "brake_axis", 2),
        )
        if gamepad is not None
        else KeyboardController(
            assisted=not getattr(args, "raw_steering", False),
            lateral_acceleration=getattr(args, "steering_limit", 13.0),
        )
    )
    from .audio import DrivingAudio

    audio = DrivingAudio(
        enabled=not getattr(args, "mute", False), volume=getattr(args, "volume", 0.25)
    )
    env.viewer.camera_motion = getattr(args, "camera_motion", False)
    env.viewer.practice = PracticeBook(env, getattr(args, "practice", "results/practice"))
    env.viewer.agent_ghost = reference
    pending_commands = {}
    control_settings = controller.metadata()
    if reference:
        control_settings["ghost_reference"] = reference.identity
    source = "human_ghost_comparison" if reference else "human"
    recorder = Recorder(env, source=source, seed=args.seed, control_settings=control_settings)
    env.viewer.human_line = recorder.states
    scheduler = FixedStepClock(dt=env.dt)
    if reference:
        from .agent_ghost import GhostRaceStart

        race_start = GhostRaceStart()
    else:
        race_start = None
    paused = bool(reference)
    saved_attempt = False
    running = True
    pending_track = args.track

    def save(label, reason):
        nonlocal saved_attempt
        if reference and saved_attempt:
            return
        if not reference:
            env.viewer.practice.consider(recorder)
        directory = Path(args.ghost_results) / "recordings" if reference else args.data
        path = recorder.save(directory, label, reason)
        if path:
            print(f"Saved {path} ({label})", flush=True)
            if reference:
                from .agent_ghost import comparison_report
                from .practice import comparison_html

                report = comparison_report(recorder, reference, path, reason)
                output = Path(args.ghost_results) / f"{path.stem}_comparison.html"
                output.write_text(comparison_html(report), encoding="utf-8")
                output.with_suffix(".json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(f"Saved your SAC comparison: {output}", flush=True)
                saved_attempt = True

    def restart_ghost():
        nonlocal recorder, pending_commands, paused, saved_attempt
        save("unlabeled", "restarted")
        env.reset(seed=args.seed)
        reference.validate_player(env)
        env.viewer.trail = []
        recorder = Recorder(env, source=source, seed=args.seed, control_settings=control_settings)
        env.viewer.human_line = recorder.states
        pending_commands = {}
        controller.reset()
        scheduler.reset()
        saved_attempt = False
        race_start.start()
        paused = True

    try:
        while running:
            elapsed = clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    save("unlabeled", "user_exit")
                    running = False
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_h and not paused:
                    pending_commands["recover"] = True
                if event.key == pygame.K_x and not paused:
                    pending_commands["direction"] = 1 if env.vehicle.gear < 0 else -1
                if event.key in (pygame.K_q, pygame.K_e) and not paused:
                    pending_commands["shift"] = 1 if event.key == pygame.K_e else -1
                if event.key == pygame.K_m:
                    audio.enabled = not audio.enabled
                if event.key == pygame.K_p:
                    env.viewer.show_ghost = not env.viewer.show_ghost
                if event.key == pygame.K_l and reference:
                    env.viewer.show_lines = not env.viewer.show_lines
                if event.key == pygame.K_c:
                    env.viewer.cycle_camera()
                if event.key == pygame.K_v:
                    env.viewer.mode = "2d" if env.viewer.mode == "3d" else "3d"
                if event.key == pygame.K_r and not paused:
                    paused = True
                    if reference:
                        race_start.phase = "finished"
                        save("unlabeled", "user_stop")
                if reference and paused and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if race_start.phase != "countdown":
                        restart_ghost()
                        elapsed = 0
                if not reference and paused and pygame.K_1 <= event.key < pygame.K_1 + len(names()):
                    pending_track = names()[event.key - pygame.K_1]
                if not reference and paused and event.key in (pygame.K_g, pygame.K_b, pygame.K_u):
                    save(
                        {pygame.K_g: "good", pygame.K_b: "bad", pygame.K_u: "unlabeled"}[event.key],
                        "completed" if env.info()["success"] else "interrupted_or_timeout",
                    )
                    env.track = Track(pending_track)
                    env.reset(seed=args.seed)
                    env.viewer.trail = []
                    recorder = Recorder(env, seed=args.seed, control_settings=controller.metadata())
                    env.viewer.human_line = recorder.states
                    env.viewer.practice = PracticeBook(
                        env, getattr(args, "practice", "results/practice")
                    )
                    pending_commands = {}
                    controller.reset()
                    scheduler.reset()
                    elapsed = 0
                    paused = False
            if not running:
                break
            if race_start and race_start.advance(elapsed):
                scheduler.reset()
                controller.reset()
                elapsed = 0
                paused = False
            if not paused:
                keys = pygame.key.get_pressed()
                target = np.array(
                    [
                        int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
                        - int(keys[pygame.K_a] or keys[pygame.K_LEFT]),
                        int(keys[pygame.K_w] or keys[pygame.K_UP]),
                        int(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                    ]
                )
                if gamepad is not None:
                    try:
                        analog, buttons = controller.sample(env)
                    except (pygame.error, ValueError) as exc:
                        save("unlabeled", "controller_disconnected")
                        raise ValueError(
                            "Gamepad unavailable; current run saved unlabeled"
                        ) from exc
                    pending_commands.update(buttons)
                for _ in range(scheduler.advance(elapsed)):
                    action = (
                        analog
                        if gamepad is not None
                        else controller.update(target, env.speed, env.dt)
                    )
                    transition = env.step(action, commands=pending_commands)
                    pending_commands = {}
                    recorder.append(action, transition)
                    paused = transition[2] or transition[3]
                    if paused:
                        if reference:
                            race_start.phase = "finished"
                            save("unlabeled", "completed" if env.info()["finished"] else "timeout")
                        break
            else:
                scheduler.reset()
            if reference:
                env.viewer.race_phase = race_start.phase
                env.viewer.countdown = race_start.remaining
                env.viewer.note = (
                    "Space race / restart | C camera | V view | L lines | P ghost | Esc exit"
                )
            elif paused:
                env.viewer.note = f"G good / B bad / U unlabeled | next: {pending_track} | 1-4 compact, 5-8 GP-length"
            else:
                env.viewer.note = (
                    "RECORDING | R label | C camera | "
                    + (
                        "gamepad"
                        if gamepad is not None
                        else ("assisted keys" if controller.settings.assisted else "raw keys")
                    )
                    + (
                        " | STALL: simulation paused"
                        if scheduler.suspended
                        else f" | physics {1 / env.dt:g} Hz"
                    )
                )
            audio.update(env, paused)
            env.render()
    finally:
        audio.close()
        env.close()


def replay(args):
    import pygame

    data = load(args.file)
    meta = data["metadata"]
    track = next(
        (str(p) for p in TRACK_DIR.rglob("*.json") if Track(str(p)).digest == meta["track_sha256"]),
        None,
    )
    if track is None:
        raise ValueError("Recording track hash does not match an installed layout")
    env = RacingEnv(
        track,
        view=getattr(args, "view", "3d"),
        renderer=getattr(args, "renderer", "auto"),
        dt=meta["dt"],
        laps=meta.get("laps", 1),
    )
    env.reset()
    env.telemetry_available = "telemetry" in data
    if "driving_config" in meta:
        env.driving = DrivingConfig(**meta["driving_config"])
    env.render()
    clock = pygame.time.Clock()
    scheduler = FixedStepClock(dt=meta["dt"])
    i = 0
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(data["states"][:, :2], axis=0), axis=1))]
    progress = np.r_[0, np.cumsum(data["reward_features"][:, 0])]
    offsteps = np.r_[0, np.cumsum(-data["reward_features"][:, 2] / meta["dt"])]
    smoothness = np.r_[0, np.cumsum(-data["reward_features"][:, 3])]
    try:
        while True:
            state = data["states"][i]
            env.distance = float(distances[i])
            if "vehicle_states" in data:
                env.vehicle.restore_vector(data["vehicle_states"][i])
            else:
                from .vehicle_model import VehicleState

                env.vehicle = VehicleState.rolling(float(state[3]))
                env.vehicle.angle[:] = env.distance / 0.36
            if i and "contacts" in data:
                contact, env.impact, collisions, recoveries = data["contacts"][i - 1]
                env.contact = bool(contact)
                env.collisions = int(collisions)
                env.recoveries = int(recoveries)
                env.invalid_lap = env.recoveries > 0
            env.action = data["actions"][i - 1].copy() if i else np.zeros(3, np.float32)
            env.pos = state[:2].copy()
            env.heading, env.speed, env.lateral_speed, env.yaw_rate, env.steering = state[2:]
            env.steps = i
            if i > 0 and "telemetry" in data:
                env.telemetry.update(dict(zip(meta["telemetry_names"], data["telemetry"][i - 1])))
            env.progress = float(progress[i])
            env.offtrack_steps = int(round(offsteps[i]))
            gate = env.next_checkpoint * env.track.length / 40
            if not env.info()["offtrack"] and gate <= env.progress <= gate + env.track.width:
                env.next_checkpoint += 1
            env.smoothness = float(smoothness[i])
            env.viewer.note = (
                f"REPLAY | {meta['label']} | {i + 1}/{len(data['states'])} | ESC exits"
            )
            env.render()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    env.viewer.cycle_camera()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    env.viewer.mode = "2d" if env.viewer.mode == "3d" else "3d"
            if any(
                e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                for e in events
            ):
                break
            if i == len(data["states"]) - 1:
                break
            i = min(len(data["states"]) - 1, i + scheduler.advance(clock.tick(60) / 1000))
    finally:
        env.close()


def evaluate(args):
    from .experiments import EvaluationConfig, evaluate_policy, save_report, file_digest

    policy = Policy(args.model) if args.model else None
    config = EvaluationConfig(
        tracks=tuple(names() if args.track == "all" else [args.track]),
        episodes=args.episodes,
        seed=args.seed,
        max_seconds=args.seconds,
        lateral_jitter=args.jitter,
    )
    report = evaluate_policy(
        policy, config, policy_name=str(args.model), render=args.render, view=args.view
    )
    if args.model:
        report["model_sha256"] = file_digest(args.model)
    save_report(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report: {args.output}")


def validate_dataset(args):
    paths = sorted(Path(args.data).glob("*.npz"))
    if not paths:
        raise ValueError("No NPZ recordings found")
    failures = []
    counts = {"good": 0, "bad": 0, "unlabeled": 0}
    for path in paths:
        try:
            run = load(path)
            counts[run["metadata"]["label"]] += 1
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
    print(json.dumps({"files": len(paths), "labels": counts, "errors": failures}, indent=2))
    if failures:
        raise ValueError(f"{len(failures)} invalid recordings")


def list_controllers(args):
    import pygame

    pygame.init()
    try:
        devices = []
        for i in range(pygame.joystick.get_count()):
            device = pygame.joystick.Joystick(i)
            device.init()
            devices.append(
                {
                    "index": i,
                    "name": device.get_name(),
                    "axes": device.get_numaxes(),
                    "buttons": device.get_numbuttons(),
                }
            )
        print(json.dumps(devices, indent=2))
    finally:
        pygame.quit()


def compare(args):
    from .practice import compare_runs, comparison_html

    report = compare_runs(args.first, args.second)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comparison_html(report), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Racing Lab: drive, record, imitate, evaluate")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("play", help="Drive and record demonstrations")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument("--renderer", choices=["auto", "gpu", "software"], default="auto")
    p.add_argument(
        "--raw-steering",
        action="store_true",
        help="Disable speed-dependent keyboard steering limit",
    )
    p.add_argument("--manual-gears", action="store_true", help="Q/E down/up; automatic by default")
    p.add_argument("--no-tc", action="store_true", help="Disable traction control")
    p.add_argument("--no-abs", action="store_true", help="Disable anti-lock braking")
    p.add_argument(
        "--steering-limit",
        type=float,
        default=13.0,
        help="Keyboard geometric lateral acceleration limit, 3-30 m/s²",
    )
    p.add_argument("--gamepad", type=int, help="Joystick index (use controllers command to list)")
    p.add_argument("--steer-axis", type=int, default=0)
    p.add_argument("--throttle-axis", type=int, default=5)
    p.add_argument("--brake-axis", type=int, default=2)
    p.add_argument("--mute", action="store_true")
    p.add_argument("--volume", type=float, default=0.25)
    p.add_argument("--camera-motion", action="store_true", help="Enable subtle cockpit body motion")
    p.add_argument(
        "--practice", default="results/practice", help="Local personal-best/ghost directory"
    )
    p.add_argument(
        "--sac-ghost", metavar="MODEL", help="Drive against a frozen SAC standing-start ghost"
    )
    p.add_argument("--ghost-cache", default="results/sac-ghost/cache")
    p.add_argument(
        "--ghost-results",
        default="results/sac-ghost/sessions",
        help="Local human/SAC practice comparison directory",
    )
    p.add_argument("--track", choices=names(), default="monza")
    p.add_argument("--data", default="data")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=play)
    p = commands.add_parser("replay")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument("--renderer", choices=["auto", "gpu", "software"], default="auto")
    p.add_argument("file")
    p.set_defaults(func=replay)
    p = commands.add_parser("train")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="models/bc.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tracks", nargs="+", choices=names(), help="Restrict training/validation to these tracks"
    )
    p.add_argument(
        "--completed-only",
        action="store_true",
        help="Exclude interrupted and failed demonstrations",
    )
    p.set_defaults(
        func=lambda a: print(
            json.dumps(
                train(
                    a.data,
                    a.output,
                    a.seed,
                    a.epochs,
                    tracks=a.tracks,
                    completed_only=a.completed_only,
                ),
                indent=2,
            )
        )
    )
    p = commands.add_parser("evaluate")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument(
        "--model", help="NPZ behavior cloning model; omit for geometric reference controller"
    )
    p.add_argument("--track", choices=names() + ["all"], default="all")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=1)
    p.add_argument("--render", action="store_true")
    p.add_argument("--output", default="results/evaluation.json")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("inspect")
    p.add_argument("file")
    p.set_defaults(func=lambda a: print(json.dumps(load(a.file)["metadata"], indent=2)))
    p = commands.add_parser("validate-data", help="Check every recording before training")
    p.add_argument("--data", default="data")
    p.set_defaults(func=validate_dataset)
    p = commands.add_parser("controllers", help="List available joystick devices and axes")
    p.set_defaults(func=list_controllers)
    p = commands.add_parser("compare", help="Export an offline speed/braking/trajectory comparison")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--output", default="results/comparison.html")
    p.set_defaults(func=compare)
    # Register without importing PyTorch; handlers load the optional RL package on demand.
    from .rl_commands import add_sac_commands

    add_sac_commands(commands)
    args = parser.parse_args()
    try:
        args.func(args)
    except ModuleNotFoundError as exc:
        parser.error(
            f"Missing optional dependency {exc.name}. Install with: pip install -e '.[all,rl]'"
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
