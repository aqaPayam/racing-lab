# Changelog

## 0.3.0 — Research tooling

- Public evaluation/configuration API, aligned dataset arrays, discounted feature sums, and trusted simulator snapshots.
- Dataset validation, applied-action recording, reset metadata, duplicate-run detection, and training track/completion filters.
- Versioned evaluation reports with success-aware summaries, runtime provenance, seeds, hashes, and interruption handling.
- Optional play/training dependencies, offscreen RGB rendering, runnable examples, focused documentation, formatting and package checks.
- Numerical dynamics compatibility remains simulator 0.2.0. Existing 0.2.0 recordings remain usable. Evaluation JSON now contains a report object rather than a bare episode list.

## 0.2.0 — Consistent physical scale

- Shared 5.5 × 1.9 m car dimensions and 3.4 m wheelbase; 14–16 m road widths.
- Vehicle envelope in road-edge checks; true-scale rendering and speed-dependent following camera.
- Legacy layouts retained for replay. Simulator 0.1.0 demonstrations require recollection for current-version training.

## 0.1.0 — Initial prototype

- Four circuit-inspired tracks, keyboard driving, labeled recordings/replay, Gymnasium environment, behavior cloning and reference evaluation.
