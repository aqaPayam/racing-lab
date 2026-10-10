"""Measure physics throughput and real-time presentation on the current machine.

Run: python examples/benchmark_rendering.py --renderer gpu --seconds 5
The benchmark never creates or labels human demonstration files.
"""

import argparse
import json
from time import perf_counter
import numpy as np
from racing_lab import RacingEnv
from racing_lab.timing import FixedStepClock
from racing_lab.experiments import runtime_versions, save_report


def benchmark(renderer="auto", seconds=5):
    import pygame

    env = RacingEnv(view="3d", renderer=renderer, render_mode="rgb_array")
    env.reset()
    try:
        env.render()  # Warm up shader compilation and mesh construction before timing.
        timer = FixedStepClock()
        frame_times = []
        start = previous = perf_counter()
        while perf_counter() - start < seconds:
            now = perf_counter()
            for _ in range(timer.advance(now - previous)):
                env.step([0, 0.2, 0])
            previous = now
            before = perf_counter()
            env.viewer.draw()  # No extra RGB-array copy in the interactive presentation path.
            frame_times.append(perf_counter() - before)
            pygame.event.pump()
        wall = previous - start
        return {
            "runtime": runtime_versions(),
            "renderer": env.viewer.renderer_status,
            "device": getattr(env.viewer.scene3d, "renderer_name", "CPU software"),
            "wall_seconds": wall,
            "simulation_seconds": env.steps / 60,
            "simulation_to_wall_ratio": env.steps / 60 / wall,
            "frames": len(frame_times),
            "frame_ms_median": float(np.median(frame_times) * 1000),
            "frame_ms_p95": float(np.percentile(frame_times, 95) * 1000),
        }
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=["auto", "gpu", "software"], default="auto")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--output", default="results/render-benchmark.json")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0.1 <= args.seconds <= 60:
        parser.error("seconds must be between 0.1 and 60")
    report = benchmark(args.renderer, args.seconds)
    save_report(report, args.output)
    print(json.dumps(report, indent=2))
