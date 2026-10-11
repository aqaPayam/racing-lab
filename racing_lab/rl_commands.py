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
