"""CLI registration without importing optional PyTorch or RL implementations."""

from .track import names


def _train(args):
    from .rl.cli import train_command

    train_command(args)


def _evaluate(args):
    from .rl.cli import evaluate_command

    evaluate_command(args)


def add_sac_commands(commands):
    p = commands.add_parser("train-sac", help="Train SAC-3step from scratch, without datasets")
    p.add_argument("--track", choices=names(), default="monza")
    p.add_argument("--steps", type=int, default=10_000, help="Policy decisions; default is a pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--output", default="results/sac-monza-seed0")
    p.add_argument("--replay-capacity", type=int, default=10_000_000)
    p.add_argument("--eval-every", type=int, default=100_000, help="0 disables periodic evaluation")
    p.add_argument("--checkpoint-every", type=int, default=200_000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--no-abs", action="store_true")
    p.set_defaults(func=_train)
    p = commands.add_parser("evaluate-sac", help="Evaluate a frozen SAC checkpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--track", choices=names() + ["all"])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--jitter", type=float, default=0, help="Optional lateral reset variation, in m")
    p.add_argument("--seconds", type=float)
    p.add_argument("--render", action="store_true")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="results/sac-evaluation.json")
    p.set_defaults(func=_evaluate)
    p = commands.add_parser("train-td3", help="Train source-based TD3 from scratch")
    p.add_argument("--track", choices=names(), default="monza")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/td3-monza-seed0")
    p.add_argument("--reward-profile", choices=["common", "progress", "cth"], default="common")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.set_defaults(func=_train_td3)
    p = commands.add_parser("evaluate-td3", help="Evaluate or watch frozen TD3")
    p.add_argument("--model", required=True)
    p.add_argument("--track", choices=names() + ["all"], default="monza")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--jitter", type=float, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument("--output", default="results/td3-evaluation.json")
    p.set_defaults(func=_evaluate_td3)
    p = commands.add_parser(
        "benchmark-online", help="Run a frozen, fully accounted SAC–TD3 benchmark"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--track", choices=names(), default="monza")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--workers", type=int, default=1, help="Concurrent CPU runs; timings include contention"
    )
    p.add_argument(
        "--continue-queue",
        action="store_true",
        help="Skip completed runs; never resume interrupted training",
    )
    p.set_defaults(func=_benchmark)
    p = commands.add_parser(
        "report-benchmark", help="Validate raw benchmark artifacts and regenerate the report"
    )
    p.add_argument("--input", required=True)
    p.set_defaults(func=_report_benchmark)
    p = commands.add_parser("diagnose-td3", help="Run all four frozen TD3 diagnostic conditions")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--continue-queue", action="store_true")
    p.set_defaults(func=_diagnose_td3)
    p = commands.add_parser(
        "report-td3-diagnostic", help="Validate and report the diagnostic pilot"
    )
    p.add_argument("--input", required=True)
    p.set_defaults(func=_report_diagnostic)


def _train_td3(args):
    import json
    from .rl.online import train_online

    print(
        json.dumps(
            train_online(
                args.output,
                algorithm="td3",
                track=args.track,
                steps=args.steps,
                seed=args.seed,
                device=args.device,
                reward_profile=args.reward_profile,
            ),
            indent=2,
        )
    )


def _evaluate_td3(args):
    import json
    from dataclasses import replace
    import torch
    from .rl.config import RacingTaskConfig
    from .rl.online import evaluate_online
    from .rl.td3 import TD3Agent
    from .rl.training import write_json

    torch.set_num_threads(1)
    agent, metadata = TD3Agent.load(args.model)
    task = replace(RacingTaskConfig(**metadata["task"]), max_laps=4)
    tracks = tuple(names()) if args.track == "all" else (args.track,)
    report = evaluate_online(
        agent,
        tracks,
        task,
        episodes=args.episodes,
        lateral_jitter=args.jitter,
        render=args.render,
        view=args.view,
        reward_profile=metadata["reward_profile"],
    )
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


def _benchmark(args):
    from .rl.benchmark import BenchmarkConfig, run_benchmark

    held_out = tuple(
        track
        for track in ("monza", "silverstone", "spa", "suzuka")
        if track != args.track.removesuffix("_long")
    )
    run_benchmark(
        args.output,
        BenchmarkConfig(
            track=args.track,
            steps=args.steps,
            seeds=tuple(args.seeds),
            workers=args.workers,
            held_out_tracks=held_out,
        ),
        continue_queue=args.continue_queue,
    )


def _report_benchmark(args):
    from .rl.benchmark_report import generate_report

    generate_report(args.input)


def _diagnose_td3(args):
    from .rl.diagnostic import DiagnosticConfig, run_diagnostic

    run_diagnostic(
        args.output,
        DiagnosticConfig(steps=args.steps, seeds=tuple(args.seeds), workers=args.workers),
        continue_queue=args.continue_queue,
    )


def _report_diagnostic(args):
    from .rl.diagnostic_report import generate_diagnostic_report

    generate_diagnostic_report(args.input)
