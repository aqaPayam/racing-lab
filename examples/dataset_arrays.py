"""Usage: python examples/dataset_arrays.py data/RECORDING.npz"""

import argparse
from racing_lab import load_trajectory, transitions, feature_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    args = parser.parse_args()
    trajectory = load_trajectory(args.recording)
    observations, actions, next_observations = transitions(trajectory)
    print(
        "Observation/action/next-observation shapes:",
        observations.shape,
        actions.shape,
        next_observations.shape,
    )
    print("Feature totals (progress, time, offtrack, smoothness):", feature_sum(trajectory))
