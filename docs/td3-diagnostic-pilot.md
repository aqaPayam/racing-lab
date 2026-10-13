# TD3 diagnostic pilot

This is a controlled follow-up to the completed five-seed, 200,000-decision
SAC–TD3 experiment. It tests three candidate explanations for TD3's first-chicane
failure. It is not a new algorithm ranking, a reproduction of published scores,
or evidence that 50,000 decisions are sufficient for convergence.

## Frozen run plan

- Monza-inspired compact layout only; no tuning on Silverstone, Spa or Suzuka.
- Four conditions, each with fresh initialization and seeds **5, 6, 7**.
- **50,000 training decisions per run**, 12 runs and **600,000 training decisions**.
- Four concurrent CPU workers in the local experiment; one Torch/BLAS thread each.
- The source, runtime versions, track bytes and all settings are hashed before launch.
- No expert dataset, pretrained model, demonstrations, labels, controller queries,
  reference actions or trajectory-aided learning reward. Replay begins empty.

| Condition | Change from the new pilot baseline |
| --- | --- |
| `baseline` | None: Gaussian exploration from decision one; fixed standing start; common reward. |
| `uniform_warmup` | Uniform independent actions in all three channels for decisions 1–2,000; then normal noisy TD3 actions. |
| `random_starts` | Each training reset starts at a uniform centreline distance around Monza, still at zero speed. |
| `cth_reward` | Replace only the training reward with the existing CTH-style geometry reward. |

Warmup changes **behavior only**. Learning still begins after 100 stored transitions
in every condition: the first update precedes decision 101. Each later decision
performs the same two critic minibatches and one actor/target update. Thus every
50,000-decision run performs **99,800 twin-critic optimizer steps**, **49,900 actor
updates** and **9,980,000 critic minibatch row presentations**. Reused replay rows
are not additional driving data. Exactly 2,000 collected transitions in the warmup
condition come from uniform actions; the other conditions have zero uniform actions.
The actor's exploration RNG is not consumed during uniform-action selection.

Random start distance is drawn from `numpy.random.default_rng(training_seed +
episode_index).uniform(0, track_length)`. Reset speed, lateral offset, heading,
controls, observation history and all driving aids follow the normal reset.
This restores access to downstream turns without supplying a trajectory or policy.
An episode still accumulates travel and gates from its own start, remains multi-lap,
and retains the existing failure/time limits. Used and final unused reset positions
are recorded with their first decision number and physical-state hash.

## Preserved settings and explicit adaptations

The learning rule comes from [Evans et al., Unifying F1TENTH Autonomous Racing:
Survey, Methods and Benchmarks](https://arxiv.org/abs/2402.18558) and the
[pinned author implementation](https://github.com/BDEvan5/f1tenth_benchmarks/tree/c9730789dd0fa85fc18dd4b600c5246e134ebe22).
Numerical source-parity tests remain unchanged. This pilot keeps two 100-unit layers,
batch size 100, replay capacity 100,000, one-step returns, gamma 0.99, tau 0.005,
Adam learning rates 0.001, exploration noise 0.1 and clipped target noise 0.2/0.5.
The author's local actor-update order is preserved.

All conditions share 168 numeric history observations, three incremental controls,
25 Hz decisions, the same simulator/track, automatic gears, TC and ABS. The
600-second time limit, three-wheel off-road failure and low-speed failure stay
unchanged. The source's absolute speed controller, launch assistance, LiDAR input,
single-lap training and 10 Hz training override are **not restored**. This is still
the declared Racing Lab common control task, with one additional factor per variant.
See the [paper audit](online-rl-paper-audit.md) for the complete adaptation ledger.

The common reward is `3.6 * planar_speed * (1 - abs(lateral_gap_m)/12) / 300`.
The existing CTH profile returns +1 on lap completion, -1 on a newly counted
collision, otherwise `max(forward_speed/85 * cos(heading_error) -
abs(lateral_gap_m), 0)`. Completion takes precedence if both events occur in one
decision. The nonnegative clipping follows the
[author reward code](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/drl_racing/reward_functions.py);
85 m/s and nearest-centreline projection are game substitutions. A counted collision
changes the reward without changing the common failure rule. Therefore reward scale,
heading dependence and event rewards change together as one declared reward profile;
this pilot does not separately identify their effects.

## Evaluation and reporting

Compared with the previous 200,000-decision benchmark, **every** pilot condition uses
the same shorter budget, fresh seeds, evaluation every **10,000** decisions and a
checkpoint at **50,000**. Evaluation is deterministic and does not update networks,
consume noisy exploration actions or enter replay. The final-budget checkpoint is
selected regardless of its test performance.

- Initial and each 10,000-decision checkpoint: one fixed standing-start Monza
  rollout, seed 10000, with a four-lap cap. The final periodic evaluation is also
  saved as `evaluation_final.json` but counted only once.
- Final budget: five additional single-lap Monza starts with lateral jitter of
  ±1 m, seeds 20000–20004. These are paired physical starts across all policies.
- Evaluation always uses the common reward and fixed longitudinal starting point,
  including for the random-start and CTH training conditions.
- A valid lap requires all gates and zero vehicle-envelope off-road decision steps,
  counted collisions or recovery. Completed but invalid laps remain visible.
- No held-out-track rollouts or checkpoint selection on other tracks.

Reports include each seed's fixed-start clean laps and standing/flying times, five-start
reliability, progress, first braking time and brake use after the first three seconds.
Training logs report completed episodes, greatest episode progress including partial
episodes, and a histogram of **64 equally spaced centreline-distance bins** visited
after training decisions. Off-road poses also contribute to this exposure histogram;
random spawning and broad coverage alone do not prove the policy can drive a lap.
Training returns under different reward profiles are not compared as a performance score.

Tables and plots use descriptive mean, sample SD/range and paired differences against
the baseline with matching training seeds. Five starts or multiple laps of one policy
are not five independent learned policies. Three seeds and this short budget provide
limited precision and do not isolate interactions between interventions.

Training and evaluation decisions, simulated seconds and physics substeps are separate.
Costs include critic/actor updates, sample presentations, per-phase wall times, worker
CPU time, peak RSS, model capacity and checkpoint size. Queue wall time includes worker
startup and contention; worker meters start after framework imports. An additional
process-lifetime CPU counter includes framework startup for every worker and the
controller. Their wall times
must not be summed and called the queue's elapsed duration. `completion.json` adds
reporting time and controller CPU. Memory is process lifetime peak, including imports.
Infrastructure/unit-test/smoke-test costs are separate from these training costs.

## Run and import

Install `pip install -e ".[benchmark]"`, then:

```sh
python -m racing_lab.cli diagnose-td3 --output results/td3-diagnostic --workers 4
python -m racing_lab.cli report-td3-diagnostic --input results/td3-diagnostic
```

```python
from racing_lab.rl.diagnostic import DiagnosticConfig, run_diagnostic

summary = run_diagnostic("results/td3-diagnostic", DiagnosticConfig(workers=4))
```

Per-run `run.json`, `result.json`, `worker_result.json`, `learning.csv`, `episodes.csv`,
`training_resets.json` and frozen-evaluation trajectories provide raw evidence. The
root includes the manifest and a ZIP with exact tracked source/track bytes (or
installed package source/assets when used without a Git checkout). Reports
include HTML, CSV/JSON metrics, learning/outcome/coverage figures and validation.
Generated results and policies remain git-ignored. The completed original benchmark
is preserved as a separate experiment.

An OS controller lock prevents two live queues for the same output. Continuing a
queue skips only completed compatible workers. An incomplete run is preserved and
requires a fresh run directory; checkpoint weights alone cannot exactly resume
without replay, RNG and simulator state. Do not change frozen source, runtime or tracks
while the queue is active.
