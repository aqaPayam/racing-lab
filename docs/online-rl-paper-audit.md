# Online RL racing papers and reproduction audit

Research date: 11 October 2026. Status: SAC with three-step returns has been implemented as a documented Racing Lab adaptation; see the [baseline implementation](sac-baseline.md). The other candidates remain proposals, and full benchmark comparisons have not been run.

The active setting is online reinforcement learning from random initialization. Each agent collects its own transitions through the simulator and learns from a fixed task reward. Human demonstrations, expert-controller datasets, behavior-cloning initialization, pretrained policies, inverse optimization and inverse reinforcement learning are deferred. An off-policy replay buffer contains the learning agent's experience; it does not require an external dataset.

## Recommended papers

The ordering below reflects relevance to our numerical racing interface and availability of verifiable implementation details. It is a research recommendation, not an empirical ranking of algorithms.

| Priority | Paper and original benchmark | Relevant method | Reproduction assessment |
| --- | --- | --- | --- |
| 1 | [Remonda et al., A Simulation Benchmark for Autonomous Racing with Large-Scale Human Data](https://proceedings.neurips.cc/paper_files/paper/2024/file/b91ea2ba1b6521940e204b7b81d5ef47-Paper-Datasets_and_Benchmarks_Track.pdf), NeurIPS 2024. Assetto Corsa, multiple cars and circuits, including Formula 3 and Monza. | SAC trained from scratch with three-step returns, numerical telemetry and three continuous controls. | Strongest initial anchor. [Author code](https://github.com/dasGringuen/assetto_corsa_gym) is available, but paper settings and released defaults differ. Select the scratch condition, excluding demonstration-assisted and offline variants. |
| 2 | [Evans et al., Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmarks](https://arxiv.org/html/2402.18558v2), 2024. Scaled racing with numerical range sensors. | TD3 with Progress or cross-track/heading (CTH) reward. | [Original author code](https://github.com/BDEvan5/f1tenth_benchmarks) is available. Default trajectory-aided learning (TAL) reward queries a controller; exclude it from this stage. Controls and a launch heuristic require explicit treatment. |
| 3, conditional | [Czechmanowski et al., On learning racing policies with reinforcement learning](https://arxiv.org/html/2504.02420v2), IROS 2025. Simulation training and physical RC evaluation. | PPO, numerical dynamics/actuator observations, road preview and domain randomization. | Useful dynamics-focused candidate. The [author project](https://grzegorzczput.github.io/rl-racing/) provides the paper and videos; this audit did not verify a paper-specific training-code release. Missing PPO settings prevent an exact-source claim. |
| Later | [Remonda et al., Formula RL: Deep Reinforcement Learning for Autonomous Racing using Telemetry Data](https://arxiv.org/pdf/2104.11106), 2021 preprint version. TORCS, telemetry and steering/throttle/brake. | DDPG variants with history, recurrent components and replay/return changes. | Closely related controls, but no verified author implementation of all reported modifications was found. Bot-derived reference conditions must be distinguished from centerline conditions. |
| Later | [Wurman et al., Outracing champion Gran Turismo drivers with deep reinforcement learning](https://www.cs.utexas.edu/users/pstone/Papers/bib2html-links/nature22.pdf), Nature 2022. Gran Turismo Sport. | Quantile-regression SAC (QR-SAC), rather than ordinary SAC. | Strong racing precedent trained from scratch. The paper supplies pseudocode; its research control API is unavailable in the commercial game. A Racing Lab implementation would be a reimplementation with environment substitutions. |

## Source revisions

Use immutable links in the eventual implementation cards. A repository revision is evidence of released code, not proof that it generated a particular published result.

| Source | Audited revision | Interpretation |
| --- | --- | --- |
| Assetto Corsa Gym, publication-era candidate | [e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4](https://github.com/dasGringuen/assetto_corsa_gym/tree/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4), 19 July 2024 | Useful historical snapshot; not certified as the exact experimental revision. |
| Assetto Corsa Gym, current released source inspected | [ef0fa502bdd257fc2affad9361c02b3dbdfe1632](https://github.com/dasGringuen/assetto_corsa_gym/tree/ef0fa502bdd257fc2affad9361c02b3dbdfe1632), 2 August 2025 | Includes changes after the paper. Git blob comparison confirmed SAC, algorithm-base and replay files identical to the July snapshot; the training-agent file differs. |
| F1TENTH benchmarks | [c9730789dd0fa85fc18dd4b600c5246e134ebe22](https://github.com/BDEvan5/f1tenth_benchmarks/tree/c9730789dd0fa85fc18dd4b600c5246e134ebe22), 22 May 2024 | Verified original author's repository. Earlier notes linked a mirror; use this source for provenance. |

## First candidate: SAC with three-step returns

This paper's scratch condition explicitly uses no prior data. Its task uses relative controls, observation history and an automated gearbox. The reference path is supplied by Assetto Corsa and appears in observations/reward; it is separate from the paper's human driving dataset. These are meaningful parts of the recipe, not interchangeable environment details. See the paper's Sections 4.2 and 5.1 and [author project](https://assetto-corsa-gym.github.io/).

The following values are verified in the [July 2024 configuration](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/config.yml). Agreement with the paper is recorded separately below.

| Component | Released configuration |
| --- | --- |
| Actor and each critic | Three hidden layers, 256 units each |
| Critics | Two Q functions |
| Discount | 0.992 per decision |
| Return | Three-step discounted reward |
| Batch | 128 |
| Actor, critic, entropy learning rates | 0.0003 each |
| Target coefficient | 0.005 |
| Initial exploration | 2,000 steps |
| Update interval | One decision step |
| Checkpoint interval | 200,000 steps |
| Policy frequency | 25 Hz |
| Episode time limit | 600 seconds |
| Scratch switches | Offline loading, pretraining and offline buffer disabled; no model-load argument |

The [SAC source](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/algorithm/discor/discor/algorithm/sac.py) uses twin critics, learned entropy temperature, target entropy equal to minus action dimension, and initial temperature 1. It updates the actor/temperature before the critics. Its [algorithm base](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/algorithm/discor/discor/algorithm/base.py) uses gamma raised to the return length for bootstrapping. The [replay source](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/algorithm/discor/discor/replay_buffer.py) accumulates discounted rewards and flushes pending entries on its supplied episode-end signals. Terminal, truncated and shortened-return cases need numerical checks before a port is accepted.

Released [environment code](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py) normalizes reward as 3.6 times speed in m/s, multiplied by (1 minus absolute path gap divided by 12), then divided by 300. Thus copying only the paper's simplified reward equation misses a scale factor. Relative actions are multiplied by channel-specific limits and added to applied controls; their reset values and clipping must also be preserved in a paper-task adapter.

### Unresolved paper/code differences

| Setting | Published paper | July 2024 source | Audited 2025 source |
| --- | --- | --- | --- |
| Replay capacity | 10 million | 10 million | 8 million |
| Evaluation interval | 100,000 | 0 | 0 |
| Low-speed early termination | Below 5 km/h for over 2 seconds | Below 1 m/s for 10 seconds | Below 1 m/s for 10 seconds |

Sources: paper Appendix E/Section 4.2; [historical configuration](https://github.com/dasGringuen/assetto_corsa_gym/blob/e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4/config.yml), [current configuration](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/config.yml), and the constants in the historical/current environment files.

The paper reports four days of SAC training, roughly two million steps per day; the historical file requests ten million steps and the current file eight million. These are distinct stopping specifications. Select a published-settings profile and a released-code profile explicitly if needed; do not silently combine them and call the result exact. The complete original run configuration remains unverified.

The author's [README](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/README.md) still describes its TD-MPC2 baseline as awaiting upload. The racing paper disables TD-MPC2 planning and changes update timing, so downloading generic TD-MPC2 would not reproduce that racing recipe. Defer this method until the racing adapter is verified.

## Second candidate: TD3-Progress

The [TD3 implementation](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/drl_racing/td3.py) initializes fresh networks and starts learning after 100 collected transitions. Exploration adds Gaussian noise with standard deviation 0.1; it does not contain a separate uniform-random-action warm-up. Batch size is 100, discount 0.99, target coefficient 0.005 and learning rate 0.001. Target noise is 0.2, clipped to 0.5; each training call makes two critic updates. [Training utilities](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/drl_racing/training_utils.py) define 100-by-100 ReLU networks and a 100,000-transition replay buffer. Preserve these choices instead of substituting library defaults.

[Reward code](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/drl_racing/reward_functions.py) confirms that TAL queries pure pursuit, while Progress and CTH do not. CTH includes clipping and Progress includes scaling. Nevertheless, the [agent wrapper](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/drl_racing/EndToEndAgent.py) replaces policy output below 1 m/s with zero steering and a 2 m/s speed command. This is controller assistance without a demonstration dataset. A controller-independent variant should remove it and disclose the change; a source-faithful variant should retain and label it.

The paper/configuration specify 50,000 training steps, while the README says 60,000. The [training script](https://github.com/BDEvan5/f1tenth_benchmarks/blob/c9730789dd0fa85fc18dd4b600c5246e134ebe22/f1tenth_benchmarks/benchmark_results/train_drl_agent.py) uses seeds 12, 13 and 14 and overrides simulation steps per action to ten. At the configured 0.01-second physics step this means 10 Hz training. Do not infer the training rate from the normal 25 Hz planning configuration. The evaluator's default four-physics-step interval corresponds to 25 Hz. Record both clocks and verify the actual invocation in the final experiment card.

Its output is target speed and steering. Racing Lab takes steering, throttle and brake. Adding a speed controller retains a control hierarchy; changing to direct pedal outputs changes the policy interface. Neither substitution should be hidden.

Source audit also found replay details needing attention: sampling excludes the most recent entry, and reported buffer size resets when its write pointer wraps. The published 50,000-step budget does not fill its 100,000-entry buffer; a longer Racing Lab run would exercise that behavior. A correction must be disclosed and checked. Actor/target updates occur at local loop index zero before the second critic update; copying only a generic TD3 delay setting may change the source update order.

## Conditional candidate: the racing PPO paper

The [2025 paper](https://arxiv.org/html/2504.02420v2) specifies 400 parallel environments, 1,024-step rollouts, batch size 1,024, 20 Hz decisions and 120 million transitions. Actor layers are 256-by-256; critic layers 512-by-512, with LeakyReLU. Discount is 0.99 and learning rate decreases from 0.001 to 0.0001. The original simulator was identified using 36 minutes of human driving. This is calibration data, although policy learning uses simulator interaction.

An adaptation can use Racing Lab's existing model without importing those measurements. It must disclose this omission and changes to wheel-speed/actuator controls. PPO clipping, optimization epochs, GAE settings and some observation choices were not resolved from a verified released configuration. Do not invent defaults and attribute them to this paper. Its scale also makes it unsuitable as an assumed small-PC budget without a runtime pilot.

## Other work screened

- [Drive Fast, Learn Faster](https://rlj.cs.umass.edu/2025/papers/Paper90.html), 2025, uses residual SAC to improve a classical base controller. It is relevant for a later controller-assisted condition.
- [Efficient Real-World Autonomous Racing via Attenuated Residual Policy Optimization](https://arxiv.org/abs/2603.12960), 2026, also relies on a base policy/controller. It is not the same experimental setting as a standalone policy learning its controls from scratch.
- Vision-only, imitation and demonstration-assisted racing methods remain outside the active numerical, no-demonstration benchmark. Repository examples without a verified associated paper do not establish a published baseline.

## What a meaningful Racing Lab comparison requires

Replacing the original simulator prevents an exact reproduction of its full experiment and numerical scores. Our valid claim can be a source-based implementation of the learning rule, evaluated in a new environment with documented adaptations. Separate two experiment profiles:

| Profile | Kept common or preserved | Valid interpretation |
| --- | --- | --- |
| Shared Racing Lab benchmark | Same observation/history budget, applied controls, fixed reward, reset/termination rules, aids, tracks and interaction budget for every algorithm. Preserve and identify each algorithm's source-based learning rule. | Compare algorithms on one declared task. Labels must say they are Racing Lab adaptations. |
| Paper-task variants | Preserve each paper's history, relative controls, reward and decision timing where feasible. List unavailable signals and simulator substitutions. | Examine transfer of a published recipe and ablate its design choices. Do not pool differing tasks into one reward ranking. |

The game currently exposes 39 numerical observations, absolute three-channel controls and a 60 Hz environment step. It has approximate circuit geometry and an uncalibrated generic vehicle model. Its four circuit families have compact/long variants; those variants must stay within the same train/validation/test family. Default gearbox, traction-control and ABS settings are shared environment assistance and must be declared. The current success flag does not by itself enforce a collision-free, off-road-free lap.

At 25 Hz, a three-step return spans 0.12 seconds; at 60 Hz it spans 0.05 seconds. Leaving the same step discount also changes its physical time horizon. A simple fixed integer action repeat cannot turn the existing 60 Hz step into exactly 25 Hz. A paper-rate profile needs an explicit timing implementation; alternating step counts is a timing adaptation. Report policy decisions, underlying simulator steps, simulated seconds, gradient updates and wall-clock time separately.

For main results, measure valid-lap completion first and lap time conditional on validity. Also publish progress before failure, collision/off-road counts, learning curves and performance across independent seeds. Define validity, checkpoint selection, training/validation families and any held-out-family test before seeing test performance. Best observed lap time may be a supplementary paper-comparable statistic, but alone does not measure reliability. This paragraph is a proposed Racing Lab evaluation design.

## Next implementation chain

Implementation update: steps 1–3 below are complete for the first SAC adaptation and are documented in the [SAC guide](sac-baseline.md). A 3,000-decision CPU pilot verifies the runner and source-parity tests; it does not establish convergence or freeze a benchmark budget. Step 4 remains the next research experiment. No demonstration-based method has been added to this stage.

1. Freeze a SAC specification with every setting marked paper-confirmed, code-confirmed, differing or unavailable. Resolve the discrepancy table and choose the declared profile.
2. Create an adapter ledger for observation ordering/units/history, action increments/clipping/reset, policy timing, reward scale, launch/reset assistance and episode ends. Require an explanation for every difference.
3. Port SAC with three-step returns. Verify loss/target calculations, actor/temperature update order and replay behavior against the pinned source using fixed numerical inputs. Check terminal and time-limit boundaries explicitly.
4. Run a small from-scratch pilot on training/validation circuits to measure learning and PC cost. Freeze a supported common budget before final comparisons; a shorter pilot is not the paper's full budget.
5. Add TD3-Progress, then the CTH reward variant as a separate task ablation. Resolve its launch/control hierarchy and timing before training.
6. Add PPO only after obtaining its missing configuration or clearly labeling a paper-based reimplementation with stated assumptions. Keep TD-MPC2, QR-SAC, residual methods and demonstration settings as separately scoped extensions.

This audit establishes the shortlist and the known gaps. It does not establish that a port matches the source numerically, that any agent can complete Racing Lab laps, or that any published performance claim transfers to this simulator.
