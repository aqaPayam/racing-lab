# Racing Lab draft benchmark protocol

Status: deferred research proposal, 11 October 2026. The separate [online SAC adaptation](sac-baseline.md) is implemented; the demonstration-based exports, methods and comparisons below remain proposals. The [literature review](benchmark-literature.md) supplies primary references and explains the methods' original settings.

Scope update: this demonstration-based protocol is deferred. The active first stage is online RL from scratch without human or expert-controller datasets; its current sources, adaptation requirements and proposed implementation chain are in the [online RL paper audit](online-rl-paper-audit.md). The sections below retain earlier proposals and do not define the active benchmark.

The main task is single-car time-trial driving from numerical observations: finish a valid lap, then minimize its duration. The research question is how well supervised learning, inverse optimization, inverse reinforcement learning and reinforcement learning use a small set of good but imperfect demonstrations. Publish results by information regime so that improvements can be attributed to the method and its access budget.

## Research tasks

| Task | Training evidence | Evaluation question |
| --- | --- | --- |
| Human imitation | Fixed human observation/action sequences; good-only and mixed-quality conditions | Can the policy finish valid laps, including on a held-out circuit family? |
| Improvement from imperfect demonstrations | Demonstrations plus the method's declared rankings, model access or interaction | Can it improve driving performance at a comparable cost in human input and simulation? |
| Human objective learning | Human demonstrations and, in a separate condition, pairwise preferences | Does the inferred objective predict held-out judgments and produce useful driving? There is no known true human reward vector. |
| Synthetic objective recovery | Demonstrations generated under privately sampled objective weights and documented solver quality | Can it recover identifiable objective information and useful behavior under a controlled model? |

Human demonstrations are not certified optima. Distinguish sensor measurement noise, execution/action noise and deliberate suboptimal choices in synthetic experiments. They violate different inverse-model assumptions.

## Proposed environment settings

Use simulator 0.5.0 and package 0.7.0 as the starting physics release. The current source commit is `5e8d2f891629ddf8841126b99c6fdb0e59990422`. Record the final benchmark code commit, configuration hash, dependency versions and exact track hashes when the protocol is frozen.

| Setting | Proposed first version |
| --- | --- |
| Task | One forward lap, no opponents |
| Primary tracks | Compact Monza, Silverstone, Spa and Suzuka families |
| Extended tracks | Their four `_long` variants, reported separately |
| Observations | Current 39 numerical fields; no camera input |
| Actions | Applied steering in [-1,1], throttle and brake in [0,1] |
| Timing | One policy action every 1/60 second; existing internal vehicle substeps |
| Aids | Automatic gearbox, traction control and ABS on for all core data and policies |
| Supplemental controls | No manual shifts, reverse requests or recovery in the core autonomous task |
| Reset | Standing start at the start line; evaluation lateral jitter up to 1 m |
| Timeout | Proposed 300 s on compact layouts, 600 s on long layouts; confirm in the pilot and freeze before comparisons |
| Rendering | Headless for training/evaluation; 2D/3D playback is presentation only |

The 39 fields are a partial observation. They omit, among other quantities, previous action, filtered acceleration and some lap/shift bookkeeping. They must not be described as a complete Markov state. See the [research guide](research-guide.md#observations-and-actions).

For a practical shared history condition, propose the last eight observations and seven previous applied actions, with explicit padding/masks at reset. It uses only causal information. Compare current-observation BC as an ablation; give comparable learned methods the same history condition. Recurrent models with longer memory are a separately labeled condition. Full-state, full-map or snapshot-based methods require an additional access declaration.

Human keyboard shaping is not applied to direct policy commands. Train on recorded applied actions and retain input-device metadata. A simulator-side low-level speed controller, action repeat, alternative decision frequency, disabled aids or expanded action space changes the task and needs its own configuration/result row.

## Information regimes and baselines

| Regime | Initial methods | What the learner may use |
| --- | --- | --- |
| Fixed demonstrations without task reward | BC, history BC, offline IQ-Learn | Published training observations, actions and successor/terminal information. Good-only versus all-data selection is explicit. |
| Reward or cost learning with simulation | AIRL; T-REX and a linear preference baseline followed by SAC | Demonstrations, declared quality/ranking labels, and budgeted reward-masked interactions. Pairwise rankings are extra supervision. |
| Known-model inverse optimization | Candidate-trajectory suboptimality baseline, followed by the shared policy optimizer | Declared feature library, full initial snapshots/model access, feasible alternative trajectories and accounted forward queries. |
| Offline RL with known task reward | IQL, TD3+BC | Fixed transitions labeled with the published training reward. These algorithms receive more supervision than reward-free imitation. |
| Online RL with known task reward | SAC, PPO; later BC+SAC or BC+PPO | Reward-labeled simulator interaction. Demonstration use and initialization are recorded separately. |
| Interactive imitation extension | DAgger or HG-DAgger | Learner rollouts and new human/controller action labels; count queries and human time. |
| Privileged controller references | Existing geometric controller; later trajectory optimization plus tracking or MPCC | Full map/model/state access. Report separately from observation-only learning. |

AIRL with a PPO generator should be labeled **AIRL-PPO** rather than implying exact reproduction of every original implementation choice. Use the same SAC implementation and budget for the linear preference, T-REX and candidate-IO downstream policy stage when their objectives permit this. Report both the complete pipeline and reward-only comparisons; different forward optimizers can otherwise dominate the result.

Access to the six handcrafted reward features is additional training information: they include progress and event bookkeeping not fully present in the 39 observations. Declare it for every feature-based cost learner and match the feature library in direct comparisons. An observation-only neural reward is a different condition. Evaluation policies consume only their declared observations/history; evaluator info must not silently become policy input.

Offline IQ-Learn is reward-free inverse soft-Q imitation. IQL is reward-based offline RL. Their names must not be interchanged. Plain BC, GAIL and IQ-Learn do not automatically use bad labels as negative examples.

Suggested extensions are SPRINQL for quality-grouped offline data, local segment-quality learning inspired by LOPAL, CQL/ReBRAC for stronger offline RL comparisons, and SAC(lambda) or AWAC for demonstration-assisted online improvement. Keep each extension's additional information explicit.

## Demonstration datasets

Begin with compact tracks to keep human collection manageable. A useful pilot is five good runs per family, across multiple sessions, plus retained imperfect or interrupted runs. This is a target collection plan, not an existing dataset. Preserve quality label, completion, validity, driver, session, input device and aid configuration as different fields.

Proposed learning curves use 1, 5, 10 and 20 training runs per training family when available. A single-run condition requires an external validation split rather than splitting that run into training and validation frames. Publish actual run counts, transition counts, duration, quality distribution and all additional validation/annotation data. Larger datasets must not be manufactured by counting adjacent frames as new demonstrations.

Create fixed, hashed good-only and mixed-quality subsets with published sampling seeds. BC-good and BC-all provide essential checks of the value of selecting demonstrations. Keep controller-generated data in separate datasets; source must not be inferred from performance or renamed as human expertise.

For preference learning, collect comparisons such as "run A is preferable to B," with ties and uncertain judgments allowed. Two runs labeled good are not automatically ordered. Lap-time-derived rankings are a separate, declared task-metric supervision condition, not a substitute for unknown human preferences. Compare runs within matched track/reset conditions or use matched segments with a specified horizon and terminal treatment.

Local corner labels can be an extension. A full-run bad label does not mean every action in it is bad. Automated off-road/contact flags provide local safety evidence; they do not reveal the optimal braking point or the human's entire utility function.

Exclude supplemental-command trajectories from the core three-action learner, or cut them at documented episode boundaries. A recovery teleport cannot be treated as the physical consequence of a steering/throttle/brake action. Keep truncation distinct from genuine task termination, and define bootstrapping for each.

## Track and session splits

The eight track names represent four related shape families. A compact track and its `_long` variant share their centerline family; keep both on the same side of any outer split.

| Fold | Training families | Validation family | Untouched test family |
| --- | --- | --- | --- |
| 1 | Spa, Suzuka | Silverstone | Monza |
| 2 | Monza, Suzuka | Spa | Silverstone |
| 3 | Monza, Silverstone | Suzuka | Spa |
| 4 | Silverstone, Spa | Monza | Suzuka |

Use the compact variants for the initial suite. Long variants can extend their assigned family without being counted as independent geometries. Report both held-out-family generalization and held-out-session performance on training families. Four families support a limited generalization claim; new unrelated layouts would strengthen the benchmark later.

Split by driver/session and whole trajectory. Reject exact duplicates and inspect near-duplicates. Fit normalization, feature scaling and reward models using training data only. Keep all segments from a run together. Select configurations and checkpoints without outer-test rollouts, labels or reward outcomes.

For a strict offline result, fitting and checkpoint selection use fixed data and a predefined training recipe. If offline hyperparameters or checkpoints are selected through validation simulator rollouts, label the result **offline fitting with simulation validation** and count those interactions. Pilot tuning uses only training/validation families, with its data and search budget disclosed. Freeze recipes before opening the outer tests.

## Reward access and identifiability

Current schema 2 recordings contain scalar rewards, six reward features and reward weights in metadata. The stored weights describe the game's reward, not a known human objective. The current loader validates weighted reward reconstruction; removing fields directly would produce an incompatible recording. Preserve raw files and implement a distinct benchmark export/view.

For synthetic hidden-objective experiments, privately sample objective weights rather than using the publicly known default vector. The reward-free export must omit true weights, scalar rewards, returns and derived rankings that reveal the target. Training interaction must mask the evaluator reward and associated info/metadata; snapshots and model wrappers also need a sanitized boundary. The held-out objective lives in the evaluator. This is an experimental information contract, not a claim of cryptographic secrecy in an open-source simulator.

The current undiscounted trajectory features are:

`Phi(tau) = sum_t [delta_progress, -dt, -offroad_dt, -squared_action_change, -impact_speed, -recovery]`.

Use `R_w(tau) = w dot Phi(tau)` as the reward convention. Publish the feature units, scaling, signs, discount and terminal treatment. For the interpretable linear track, a nonnegative normalized weight vector can exclude the zero-objective and scale ambiguity; this is a modeling assumption, and scaling coefficients must be mapped back to physical units. It does not remove every reward ambiguity.

Clean completed laps have approximately the same progress feature, with no collision, off-road or recovery penalties. Such data cannot identify all six weights. Audit the rank/variation of trajectory feature differences before reporting coefficient error. Recovery weight is unidentifiable in the core task that forbids recovery; omit it from that recovery target or design a separate task. Useful driving and unique reward recovery are different outcomes. This distinction is supported by the [reward-identifiability literature](benchmark-literature.md#inverse-optimization-and-reward-learning).

Normalization alone also permits a constant objective, for example placing all weight on progress when every candidate completes the same lap. Remove inactive feature dimensions and fix a positive coefficient on a varying feature, such as time, with published bounds on the remaining coefficients. This expresses the intended time-trial model and prevents an uninformative zero-loss solution; it is not evidence that every human has those preferences.

A synthetic suite should include controlled trade-offs in duration, smoothness and safety, plus documented solver-quality levels. Use feasible imperfect trajectories and clearly labeled failures where the task permits them; failed episodes require a fixed completion penalty or matched-horizon treatment to prevent early failure from appearing cheap. Reference trajectories come from a specified approximate solver or candidate search. Call their objective gap **gap to the best found reference**, unless global optimality has actually been certified.

## Inverse optimization comparison

A practical proposed baseline fits a linear reward using a fixed set of alternative feasible trajectories for each demonstrated decision problem. For demonstration `tau_i`, use the same complete initial snapshot, track, constraints and horizon/task definition for every candidate in `C_i`, and include the demonstration itself. Then fit a normalized `w` to minimize the average loss

`L_i(w) = max_{tau in C_i} w dot [Phi(tau) - Phi(tau_i)]`.

This is a restricted empirical suboptimality loss: it penalizes alternative candidates that the learned reward values above the demonstration. With fixed candidates and linear features, fitting weights is convex. Generating candidates in the nonlinear simulator remains a forward-search task. Candidate coverage, model/snapshot access and the rollout budget must be reported. Zero loss only says that the demonstration is best among those candidates; it does not establish global optimality or unique weights.

Declare whether road boundaries are hard constraints or soft penalties in this inverse problem. A demonstration violating a hard constraint cannot be inserted as a feasible candidate. Use a compatible subset or an explicit soft-constraint model; report the selection. Apply the varying-feature anchor above and report alternative fits when the data leave the coefficients ambiguous.

This is a **Racing Lab adaptation**, motivated by [data-driven inverse optimization](https://mohajerinesfahani.github.io/Publications/journal/2015/InvOpt.pdf), rather than an exact implementation of that paper's general robust formulations. Compare it to linear pairwise-preference reward fitting with the same feature library and downstream SAC. Their supervision differs: candidate suboptimality treats demonstrations as approximately preferred to generated alternatives; explicit preference fitting uses supplied comparisons. Report that assumption and ablate candidate quality.

For published convex KKT inverse methods, create a separate smooth constrained-control surrogate with documented objective bases and forward model. Evaluate inference within that surrogate and transfer to the simulator separately. The full race includes nonlinear tire forces, discrete modes and contacts; importing convex guarantees into it would be invalid. Finite-difference local IOC on fixed-mode, contact-free segments is another later extension with separate approximation checks.

## Resource budgets

Proposed publication runs use five independent training seeds, with three reserved for a development pilot. Start by measuring runtime on this PC. A provisional online budget is one million 60 Hz environment steps per training seed, reported with learning curves; confirm and freeze it after the pilot, without looking at test outcomes. The pilot must determine whether this budget yields informative comparisons rather than universal failure or saturation.

Publish separate counts for human driving minutes, runs, preference labels, corrective action queries, offline gradient updates, training environment steps, validation interactions, forward-planning/candidate queries, and wall time/hardware. Count internal rollout stages in IO/IRL and demonstration-assisted RL. Report controller-data generation cost rather than treating unlimited synthetic demonstrations as free human data.

Use paired, published evaluation reset seeds: proposed 20 starts per track per trained policy, with standing speed and lateral jitter up to 1 m. A separate rolling/random-progress-start robustness test can follow. The deterministic simulator does not create independent trials merely by repeating an identical initial state with different unused seed values. Seed policy randomness as well as the environment.

## Evaluation and statistics

The existing `info["success"]` checks lap gates and excludes recovery; it does not exclude all collisions or off-road driving. Add an explicit evaluator validity rule. For the proposed strict condition, a valid lap finishes all gates before timeout with zero off-road steps, zero recorded collisions and zero recoveries. The simulator's off-road flag includes the projected car envelope, so this rule is stricter than a vehicle-center criterion. Pilot it once and freeze any boundary tolerance before results are compared.

| Metric | Reporting rule |
| --- | --- |
| Valid-lap rate | Primary reliability outcome over all attempts; also show gate-completion rate separately |
| Lap duration | Mean and median for valid laps only, with number of valid laps; undefined when none finish |
| Failure behavior | Gate progress/completion fraction, off-road seconds, collision/contact events and peak impact speed, timeout and recovery counts |
| Driving behavior | Speed profile, steering/action change at the fixed control rate, and optional lateral acceleration; no claim that centerline adherence equals an optimal racing line |
| Data and computation | Every budget above, plus policy inference latency and hardware |
| Learned objective | Held-out preference accuracy/calibration; synthetic objective gap and identifiable, normalized coefficient error where applicable |

Show reliability and conditional speed together. A fast rare finish should not outrank reliable driving through a lap-time-only table. Prefer a completion-versus-time comparison over an invented combined scalar; if a penalized score is needed, define its failure penalties before evaluation. Use the same external driving metrics for every algorithm, rather than ranking returns from different learned rewards.

Report each track/fold and training seed, paired differences and 95% uncertainty intervals. Bootstrap at the appropriate independent levels, such as training seeds and separately sampled sessions/datasets, rather than treating correlated timesteps as independent evidence. Five seeds and four families still leave substantial uncertainty. Robust aggregate statistics and [rliable](https://github.com/google-research/rliable) can supplement the per-track results; successful-lap time has selection bias when success rates differ and must retain that qualification.

Action prediction MSE is a diagnostic for BC, not the primary benchmark. Include closed-loop traces and examples of failure modes. Test-track rollouts are evaluation only; any subsequent adaptation belongs to a separate, budgeted result.

## Implementation sequence

1. Implement benchmark configurations, family/session manifests, causal history wrappers and the validity evaluator. Preserve the existing simulator and data API.
2. Add validated reward-free and reward-labeled exports, leakage checks and preference-label files. Hash datasets and distinguish human, controller and synthetic-objective sources.
3. Run a small feasibility pilot with the existing controller, BC/current-history conditions and SAC. Establish achievable validity, suitable horizons and PC runtime before freezing budgets.
4. Add the linear preference and candidate-IO pipelines with matched features and downstream optimizer; then T-REX, AIRL and offline IQ-Learn.
5. Add IQL, TD3+BC and PPO in their information regimes. Validate each port on a small reference problem and document implementation differences.
6. Freeze protocol, versions, seeds, splits, selection rules and resource budgets. Produce baseline reports and uncertainty estimates before adding the user's new method.
7. Extend selectively to mixed-quality SPRINQL, local segment quality, interactive DAgger, MPC and demonstration-assisted RL if those questions match the research contribution.

The current simulator already supplies headless stepping, recordings, reward features, snapshots and a basic BC/evaluation API. The new work is the benchmark layer and validated algorithm comparisons. A publication claim should concern reproducible learning from imperfect demonstrations in this inspectable environment; the game remains an approximate racing model, without evidence of calibrated real-F1 fidelity.
