# Contributing

Install Python 3.10+ and run from the repository root:

```sh
python -m pip install -e ".[all,dev]"
python -m ruff format racing_lab tests examples
python -m ruff check racing_lab tests examples
python -m pytest -q
python -m build
```

Tests select SDL's dummy drivers for automated rendering. The full suite drives all four circuits and can take around a minute or longer, depending on hardware. CI runs on Windows and Linux with Python 3.11/3.13. A configured workflow is not a claim that remote CI has already passed.

Keep numerical simulation independent of Pygame and algorithm-specific libraries. Add meaningful regression tests for changes to dynamics, recording alignment, episode outcomes, or model compatibility. Do not add tests that merely repeat implementation constants.

Preserve the observation/action/feature ordering or document a deliberate compatibility change. `SIMULATOR_VERSION` identifies dynamics compatibility; the package version can change for tooling/documentation without changing dynamics. Change the simulator version when dynamics or observation semantics change. Increment track versions when geometry changes and preserve old layouts needed for replay. Never silently mix incompatible demonstrations in training.

Use the existing public API for baseline implementations. Document information available to a policy, random seeds, training tracks, test tracks, objectives, and dependencies. A controller that sees full track geometry must be identified as privileged.

Do not commit personal demonstrations, generated models, virtual environments, local evaluation results, or authentication files. Include a minimal reproducible example with bug reports. Contributions use the project's MIT license.
