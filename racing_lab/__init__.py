"""Racing Lab: shared simulation for human and machine drivers."""
from gymnasium.envs.registration import register
from .env import RacingEnv

register(id="RacingLab-v0", entry_point="racing_lab.env:RacingEnv")
__all__ = ["RacingEnv"]
