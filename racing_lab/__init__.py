"""Racing Lab: shared simulation for human and machine drivers."""

from gymnasium.envs.registration import register
from .env import RacingEnv
from .data import Recorder, load as load_trajectory, transitions, feature_sum
from .experiments import EvaluationConfig, evaluate_policy, save_report
from .track import names as available_tracks
from .vehicle_model import DrivingConfig

register(id="RacingLab-v0", entry_point="racing_lab.env:RacingEnv")
__all__ = [
    "RacingEnv",
    "DrivingConfig",
    "Recorder",
    "load_trajectory",
    "transitions",
    "EvaluationConfig",
    "evaluate_policy",
    "save_report",
    "available_tracks",
    "feature_sum",
]
