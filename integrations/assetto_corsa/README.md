# Assetto Corsa: released SAC reference and numerical research interface

This is an optional, separate integration for the **original Assetto Corsa (2014)**.
It does not replace Racing Lab's simulator, checkpoints, or completed benchmark.
The two simulators have different cars, track geometry and physics: their lap times
must not be compared as if they were the same task.

The immediate milestone is to watch the authors' BMW Z4 GT3 SAC policy on Monza,
verify telemetry, controls and resets, and establish a native-simulator reference.
This milestone performs inference only. It takes no optimizer steps, creates no
replay buffer, and loads no human driving data. The released policy is explicitly
labelled a **pretrained reference**, not a policy we trained from scratch.

## Pinned sources

- [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym), commit
  `ef0fa502bdd257fc2affad9361c02b3dbdfe1632`.
- [Author checkpoint](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym/tree/ad1df12c49cce10f159ba31967744750bb9edd8b/data_sets/monza/bmw_z4_gt3/20241108_SAC/model/checkpoints/step_05400000),
  dataset revision `ad1df12c49cce10f159ba31967744750bb9edd8b`.
- Policy SHA-256:
  `3f30cbb1b88ea221bf686c28b74d531ababe843da2e9529dc89a8a47aa157e84`.
- [Paper and benchmark](https://assetto-corsa-gym.github.io/): Remonda et al.,
  *A Simulation Benchmark for Autonomous Racing with Large-Scale Human Data*,
  NeurIPS 2024.

The current author source is newer than the paper. Matching a released checkpoint
is not proof that we reproduced every Table 2 experiment. The README documents
approximately 112.1 s for this GT3 example, whereas Table 2 lists 112.69 s for SAC
on GT3/Monza. Report our measured results and exact configuration separately.

## Local setup

Follow the [authors' Windows guide](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/INSTALL.md)
for the `sensors_par` plugin, vJoy and Custom Shaders Patch (CSP). Monza and the
BMW Z4 GT3 are included in the base game. Configure Hotlap, the author vJoy control
preset, automatic gears/clutch, no traction control/ABS/stability assistance,
no damage/fuel consumption/tyre wear, tyre blankets, optimum track, mid-clear
weather and 26 C ambient temperature. This local trial uses the stock car setup;
an exact match to the authors' car setup has not been established. Enter **Drive**
before running a policy.

Clone the author source into a local ignored directory and check out the pinned
commit. Download **only** `AssettoCorsaConfigs/tracks/monza_0.1m.pkl` and the three
checkpoint `.pth` files named `policy_net.pth`, `online_q_net.pth` and
`target_q_net.pth`. The repository already includes Monza bounds, reference line
and BMW steering calibration. The track reference line is privileged track
information. Its original derivation has not been audited; declare its use in
comparisons. No files from the released human-driving dataset are loaded.

Use a separate Python 3.9.13 environment with the core author versions:
PyTorch 1.12.1, NumPy 1.23.3, Gym 0.21.0, pandas 1.4.4 and matplotlib 3.5.3.
This local inference setup also pins SciPy 1.9.3, numba 0.56.4 and pyarrow 10.0.1.
Run the local interpreter with `-I` to prevent unrelated user packages from
entering the process. Racing Lab's Python 3.13 environment is separate.
The complete tested CPU package set is in [requirements-cpu.txt](requirements-cpu.txt).
Gym 0.21 requires older build tools; in the separate Python 3.9 environment:

```powershell
python -I -m pip install pip==23.2.1 setuptools==65.5.0 wheel==0.38.4
python -I -m pip install --no-build-isolation -r integrations/assetto_corsa/requirements-cpu.txt
python -I -m pip check
```

The local installation is under `.tools/ac-python39`. Author code is under
`.tmp/assetto-corsa-author-20261014`. Model weights are under
`models/assetto-corsa/monza-gt3-author-step05400000`. None is uploaded to this Git
repository. Original game settings and the installation ledger are under
`.tmp/assetto-setup-20261014/backup` and `installation_ledger.json`.

## Timing is part of the experiment

The original plugin downsamples **render callbacks**, assuming 50 FPS to produce
25 Hz telemetry. An uncapped renderer therefore changes action timing. Set
`video.ini` `[VIDEO] FPS_CAP_MS=20.000000` and the launcher's `[SAVED] FPS_CAP=50`;
the launcher can otherwise overwrite the first setting. Measure timestamps in
the received telemetry instead of assuming the configured rate is achieved.

For this installation, windowed 1280x720 is used to simplify watching the test;
the policy uses numerical observations and screen capture is disabled. CSP is
version 0.2.11. These runtime/rendering choices must appear in native AC reports.
No paper-exact CSP version or calibrated equivalence has been established.

The direct launcher uses Steam's standard `steam_appid.txt` with app ID 244210
and the game directory as the working directory, as implemented by
[Content Manager's AppIdStarter](https://github.com/gro-ove/actools/blob/master/AcManager.Tools/Starters/AppIdStarter.cs).
Steam must be running with access to the purchased game.

## Run the reference

From the project directory:

```powershell
.\.tools\ac-python39\python.exe -I integrations\assetto_corsa\runner.py check-policy
.\.tools\ac-python39\python.exe -I integrations\assetto_corsa\runner.py capture --seconds 5
.\.tools\ac-python39\python.exe -I integrations\assetto_corsa\runner.py evaluate --seconds 600
```

All modes accept `--author`, `--checkpoint` and `--output`. An explicit evaluation
output must be new; existing attempts are preserved. Passive capture sends no
joystick commands or resets. Do not run capture alongside an active evaluator:
the original plugin supports one driving/telemetry client at a time.

`check-policy` checks the checkpoint SHA, native 125-element observation shape,
three actions, and exact equality with the authors' `SAC.exploit` on 100 generated
observations. Evaluation uses the unchanged author Gaussian network, numerical
preprocessing, relative-action conversion and termination rules. CPU FP32 loading
uses `map_location='cpu'`; gradients are disabled. It avoids the original test
entry point's unnecessary multi-million-transition replay allocation. A watchdog
bounds its otherwise unlimited socket retries if the game stops.

Outputs include the effective configuration, native Parquet telemetry,
`evaluation.json`, author episode statistics and lap records. The initial partial
out-lap is excluded from full-lap claims. Lap validity is the native observed
flag/tyre-boundary rule, not Racing Lab's custom simulator's full-car-envelope
criterion. A stopped car on the start screen is a setup failure, not evidence
that the SAC algorithm cannot drive.

## Verified local reference: 14 October 2026

The pinned actor completed one deterministic Monza/BMW Z4 GT3 session. The
authors' four-lap stopping rule produced one initial partial lap and three full
flying laps:

| Full flying lap | Native time | Observed validity |
| --- | ---: | --- |
| 1 | 113.649 s | Clean |
| 2 | 113.543 s | Clean |
| 3 | 113.722 s | Clean |

Best: **113.543 s**; mean: **113.638 s**. This is a single released-policy
reference, not a new training benchmark or an exact Table 2 reproduction.
Observed validity checks the native invalidation flag and more than two tyres
outside at sampled states; it does not establish continuous boundary compliance.

All **11,440** saved policy actions were reproduced with zero difference from
the checkpoint using reconstructed author observations. The raw trajectory
contains 11,441 rows (one retained reset observation plus policy decisions);
three reset decisions are additional simulator work. The mean timestamp rate
was **24.927 Hz**, with a longest interval of 0.312 s and zero reported packet
loss. Maximum recorded speed was 272.0 km/h. Evaluation/cleanup took 461.374 s,
excluding imports and environment construction. Training and optimizer work
were both zero.

The earlier start-screen trial (247 decisions, 11.790 s) is retained separately
and excluded from lap claims. The successful run also retains the inherited
wind configuration (2–40 km/h, direction -1); equality to the paper's weather
and car setup has not been established. Original manual controls, assists,
video, race and launcher preferences were restored byte-for-byte afterwards.

Local evidence is under `results/assetto-corsa-setup-20261014/evaluation-2`:
`report.html`, `validation.json`, `evaluation.json`, a native configuration
snapshot and raw Parquet telemetry. Results remain outside Git. On this PC,
`watch-assetto-sac.bat` starts another inference test **after** you configure
the research profile above and click Drive; it does not alter or restore game
settings automatically.

## Import for a numerical policy

With the compatible environment active, run your script from this project:

```python
from pathlib import Path
import sys

# With python -I, explicitly add only this project to the import path.
sys.path.insert(0, str(Path.cwd()))
from integrations.assetto_corsa.runner import racing_session

with racing_session(".tmp/assetto-corsa-author-20261014", "results/my-ac-run") as env:
    observation = env.reset()
    done = False
    while not done:
        action = my_policy(observation)  # 3 relative controls in [-1, 1]
        observation, reward, done, info = env.step(action)
```

This preserves the author's legacy Gym four-value step API. It does not silently
translate to Gymnasium, transfer Racing Lab's trained agents into AC, or begin
training. A later online-training experiment requires a separate frozen protocol,
fresh weights and replay, declared observation/reward settings, a chosen budget,
multiple seeds, and measured interaction/GPU/CPU/time costs.
