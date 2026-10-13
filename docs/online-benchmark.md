# Online SAC–TD3 benchmark

The active setting is online RL from fresh networks and an empty replay buffer.
There are no demonstrations, expert-controller queries, pretrained weights or launch
controllers. IO/IRL and offline learning remain separate future experiments. These
are source-based learning recipes adapted to Racing Lab, not reproductions of the
original papers' simulators or lap times.

## Run and import

```sh
python -m pip install -e ".[benchmark]"
python -m racing_lab.cli benchmark-online --output results/online-monza-five-seeds --seeds 0 1 2 3 4 --steps 200000
python -m racing_lab.cli report-benchmark --input results/online-monza-five-seeds
```

On Windows, `.\.venv\Scripts\python.exe` works without activation. The default queue
is sequential: a fresh process for each run, one Torch/BLAS thread. `--workers 4`
enables four concurrent CPU processes. Wall time then includes contention and cannot
be interpreted as isolated algorithm speed. CPU time, worker count, memory and hardware
are retained. CUDA requires a compatible installation; numerical verification used CPU.

Use a new output directory. `--continue-queue` skips compatible completed workers
under the frozen manifest. It never resumes interrupted training: replay/simulator/RNG
state is not checkpointed. Incomplete runs are preserved and require a new root for a
retry. Changed Python source or track geometry prevents subsequent workers from running.

```python
from racing_lab.rl.online import train_online
from racing_lab.rl.td3 import TD3Agent, TD3Config
from racing_lab.rl.benchmark import BenchmarkConfig, run_benchmark

train_online("results/td3-pilot", algorithm="td3", track="monza", steps=5000, seed=0)
run_benchmark("results/comparison", BenchmarkConfig(seeds=(0, 1, 2, 3, 4)))
agent, metadata = TD3Agent.load("results/td3-pilot/final.pt")
```

Watch with `evaluate-td3 --model PATH --track monza --render --view 3d`.
Policy outputs are increments for `SACRacingEnv`, not base `RacingEnv` absolute controls.
Ordinary gameplay imports do not require Torch or Matplotlib.

## Frozen first comparison

| Item | Shared rule |
| --- | --- |
| Budget | 200,000 decisions per method/seed; seeds 0–4; ten fresh runs |
| Training circuit | Compact Monza inspired; geometry hash in manifest |
| Inputs | Shared 168 numerical sensors/control-history fields; no images |
| Controls | Three steering/throttle/brake increments; no keyboard shaping |
| Clock | 25 Hz decisions; ten 4 ms physics substeps per decision |
| Aids | Automatic gears, TC and ABS enabled |
| Reward | Existing SAC centerline/speed reward; documented game substitutions |
| Episodes | Multi-lap, 600 s limit, shared wheel-offroad/low-speed failures |
| Update timing | Act, learn from prior data, step, append; evaluation never enters replay |
| Evaluation milestones | Initialization, 100k and 200k decisions |
| Checkpoint selection | Final budget; never select fastest test lap |
| Fixed start | One deterministic four-lap-cap rollout, seed 10000 |
| Robustness | Five single-lap starts on Monza, lateral jitter ±1 m, seeds 20000–20004 |
| Held-out tests | One fixed-start single-lap rollout on each Silverstone/Spa/Suzuka circuit, seed 30000 |
| Valid lap | All gates, zero off-road steps, counted collisions or recovery |
| External data | Zero demonstrations, external transitions or expert queries |

The ten runs collect **2,000,000 unique training transitions**, 80,000 simulated seconds
and 20,000,000 physics substeps. Evaluation is extra; actual cost depends on completion
and failures. Replay samples are repeated presentations, not new driving data. 200k
is below the SAC racing paper's approximately 8M budget and above F1TENTH TD3's 50k:
it is a shared interaction budget. The earlier SAC seed-0 run remains a disclosed pilot;
its unmeasured process CPU/memory/component times are not reconstructed or silently reused.

Compact/long variants share families. Held-out tests are zero-shot and small: they do
not establish broad generalization. No held-out result tunes rewards, hyperparameters
or model selection. Subsequent tuning requires a new declared protocol.

## Learning recipes and adaptations

SAC follows [Remonda et al., NeurIPS 2024](https://arxiv.org/abs/2407.16680) and
[pinned source e560948](https://github.com/dasGringuen/assetto_corsa_gym/tree/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4).
The [SAC ledger](sac-baseline.md) remains authoritative. Tests verify identical learned
weights between the instrumented runner and the original local runner for matched runs.

TD3 follows [Evans et al., Unifying F1TENTH Autonomous Racing](https://arxiv.org/abs/2402.18558)
and [source c973078](https://github.com/BDEvan5/f1tenth_benchmarks/tree/c9730789dd0fa85fc18dd4b600c5246e134ebe22):
100×100 ReLU networks; source-style PyTorch initialization; batch 100; gamma .99;
tau .005; actor/Q Adam .001; capacity 100k; Gaussian actor exploration σ=.1;
target noise σ=.2 clipped to ±.5; one-step targets. Learning starts before decision
101 once 100 transitions exist. There is no uniform warmup. Each eligible call samples
two separate batches: first twin-critic step, actor/targets, second twin-critic step.
The local delay counter starts at zero each call, matching source rather than generic TD3 defaults.

| Per 200k run | SAC | TD3 |
| --- | ---: | ---: |
| Unique training transitions | 200,000 | 200,000 |
| Twin-critic optimizer steps | 198,000 | 399,800 |
| Actor optimizer steps | 198,000 | 199,900 |
| Entropy optimizer steps | 198,000 | 0 |
| Critic minibatch row presentations | 25,344,000 | 39,980,000 |

One twin-critic optimizer step trains both Q networks. Interaction budgets are equal;
network sizes, replay capacity, update counts and compute are source-specific.

| Original TD3 setting | Racing Lab decision |
| --- | --- |
| Random longitudinal starts around the whole circuit | One fixed standing start for every training episode; evaluation jitter changes lateral position only |
| Absolute steering and target speed | Shared three increments; no speed-controller supervision |
| Two LiDAR scans plus speed (41 fields) | Shared 168-number numerical history |
| 10 Hz training, normally 25 Hz evaluation | Shared 25 Hz; source gamma remains .99 per decision |
| Below-1-m/s launch override | Removed; startup is learned and measured |
| Excluded newest replay row; occupancy reset on wrap | Uniform inclusive sampling, capped occupancy; safe beyond 100k inserts |
| Eager float64 replay/global NumPy RNG | Lazy float32 replay, separate seeded random streams |
| Episode-ended mask | Failure removes bootstrap; time-limit cutoff retains gamma bootstrap |
| Scaled RC model/layouts | Existing generic vehicle and inspired layouts; simulator 0.5.0 unchanged |

These task substitutions matter empirically. The October 12 five-seed comparison
completed, but no recorded TD3 training episode reached 19% of Monza. Four final
TD3 policies failed at the first chicane without timely braking; the fifth hit a
barrier near the start and stalled. SAC seeds 0, 1 and 4 completed four laps, but
only seed 0 kept all four strictly valid. See the
[post-benchmark implementation and failure audit](online-rl-paper-audit.md#post-benchmark-failure-audit-13-october-2026).

Preserving a per-decision discount does not preserve its horizon in seconds after
changing control frequency. TD3's source training uses 10 Hz, while this shared task
uses 25 Hz: gamma .99 therefore has an approximately 4-second rather than 10-second
discount horizon. A timing-restored or time-adjusted-discount experiment must be
separately declared, rather than silently changing the completed protocol.

`train-td3 --reward-profile progress` or `cth` provides **reward ablations on the
shared task**, not a full paper-task reproduction. Progress reward is 100 times
normalized unwrapped progress change. CTH is
`max(speed/85 * cos(heading_error) - abs(centerline_gap_metres), 0)`:
85 m/s substitutes for the original RC scale of 8 m/s. Completion gives +1 and a newly
counted collision −1, with completion checked first. Common termination remains;
contact does not automatically end an episode. These ablations are outside the first
algorithm ranking. TAL's controller-supervised reward is excluded.

## Accounting, statistics and artifacts

Each run saves immutable settings, loss/episode CSVs, evaluations, checkpoints and
a completion marker. The queue records active/pending/completed workers. Extra
evaluation NPZ trajectories retain states, applied actions and timestamps separately
from human datasets. The standalone report, CSV, JSON and scientific figures include:

- Per-seed validity/completion, standing-start and conditional lap times, progress,
  collisions/off-road counts, varied-start reliability and zero-shot circuit outcomes.
- Training/evaluation/total unique transitions, simulated seconds, physics substeps;
  replay capacity/occupancy/array bytes and minibatch presentations.
- Actor, twin-critic and entropy optimizer steps and sampled-row presentations.
- Worker wall and process CPU time, process-lifetime peak RSS, component timing,
  inference mean/p95 latency, hardware/threads/dependencies/source/config/track hashes.
- Actual reset-state pairing and frozen checkpoint hashes; reused final/periodic
  evaluations count once because only one rollout occurred.
- Descriptive mean/sample SD/range across independent training seeds, and raw artifact hashes.

Training component timers have disjoint scopes; residual overhead is explicit. Worker
wall/CPU timing begins after framework imports and includes training plus **all**
evaluations. Native peak RSS covers process lifetime including imports; it is per
process, not aggregate system RAM. CPU time is consumption, not simulation time.
CUDA timing synchronizes; unavailable measurements remain null.

Reliability is primary. Failed times remain null, with successful-policy count beside
conditional time statistics. Aggregate resets/laps within each policy first: four laps,
adjacent frames and repeated fixed starts are not independent training seeds. Five
seeds provide descriptive uncertainty, not strong evidence of superiority.

Licensed offline fixtures verify exact TD3 initialization and three successive training
calls. Regression tests cover replay wraparound/truncation, SAC learning parity,
frozen evaluation, accounting, pairing, manifest tampering and report generation.
See [third-party notices](../THIRD_PARTY_NOTICES.md). Generated data/models/results stay
outside Git; a reviewed artifact release is a separate publishing step.
