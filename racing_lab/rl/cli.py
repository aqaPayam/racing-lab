"""Handlers are imported lazily by the main CLI."""

from dataclasses import replace
import json

from .config import RacingTaskConfig, SACConfig
from .sac import SACAgent
from .training import evaluate_sac, train_sac, write_json
from ..track import names


def train_command(args):
    config = replace(
        SACConfig(),
        replay_capacity=args.replay_capacity,
        eval_interval=args.eval_every,
        checkpoint_interval=args.checkpoint_every,
    )
    task = replace(
        RacingTaskConfig(),
        max_seconds=args.seconds,
        traction_control=not args.no_tc,
        abs=not args.no_abs,
    )
    result = train_sac(
        args.output,
        track=args.track,
        steps=args.steps,
        seed=args.seed,
        config=config,
        task=task,
        device=args.device,
        torch_threads=args.threads,
    )
    print(json.dumps(result, indent=2))


def evaluate_command(args):
    import torch

    torch.set_num_threads(args.threads)
    agent, metadata = SACAgent.load(args.model, device=args.device)
    task = RacingTaskConfig(**metadata["task"])
    task = replace(
        task, max_laps=4, **({"max_seconds": args.seconds} if args.seconds is not None else {})
    )
    track_id = metadata.get("track_id")
    if track_id is None:
        from ..track import Track

        track_id = next(
            (name for name in names() if Track(name).digest == metadata["track_sha256"]), None
        )
    if args.track is None and track_id is None:
        raise ValueError("Checkpoint track is unavailable; specify --track")
    tracks = tuple(names()) if args.track == "all" else (args.track or track_id,)
    report = evaluate_sac(
        agent,
        tracks,
        task,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        view=args.view,
        lateral_jitter=args.jitter,
    )
    report["checkpoint"] = str(args.model)
    report["training_metadata"] = metadata
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
