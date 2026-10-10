"""Run after `pip install -e .`; this intentionally stationary policy illustrates the API."""

from racing_lab import EvaluationConfig, evaluate_policy, save_report


def policy(observation):
    return [0.0, 0.0, 0.0]


if __name__ == "__main__":
    config = EvaluationConfig(tracks=("monza", "spa"), episodes=2, max_seconds=2.0, seed=42)
    report = evaluate_policy(policy, config, policy_name="stationary-example")
    save_report(report, "results/custom-policy.json")
    print(report["summary"])
