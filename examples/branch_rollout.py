"""Counterfactual one-step comparisons from exactly the same simulator state."""

from racing_lab import RacingEnv


if __name__ == "__main__":
    env = RacingEnv()
    env.reset(seed=42)
    for _ in range(60):
        env.step([0.0, 0.5, 0.0])
    snapshot = env.snapshot()
    for action in ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        env.restore(snapshot)
        observation, reward, terminated, truncated, info = env.step(action)
        print(action, "speed:", env.speed, "reward features:", info["reward_features"])
    env.close()
