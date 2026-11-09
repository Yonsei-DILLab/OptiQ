# Table 2 / inference latency — materials for the rebuttal

## What the reviewers ask

| Reviewer | Request |
|---|---|
| R1 (8dm6), W2 | "synchronization, compilation treatment, and timing repetitions also need specification (Appendix C)" |
| R1 (8dm6), Q1 | "What evaluation rule is used for each baseline, and does Table 2 measure the same action-generation procedure used for the return comparison?" |
| R1 (8dm6), Q2 | "how were GPU synchronization, compilation, and warm-up handled?" |
| R2 (uAXn), Q2 | "What returns and inference latency are obtained when sampling both the latent and conditional Gaussian?" |
| R3 (ZHgi), Q5 | "What are the error bars — or σ — for Table 2?" |

## Measurement protocol (proposed text for Appendix C)

- Hardware/software: one RTX 5090, batch size 1, Humanoid-v4, nothing else running on the GPU.
  Each method in its own framework (JAX for iBOLT/SAC/TD3/MFPO, TensorFlow for SQL, PyTorch for
  MEow/DIPO); within JAX all methods use the same JAX version.
- What is timed: the method's own evaluation call, from a NumPy observation to a NumPy action
  (host→device copy, all network evaluations / sampling steps, device→host copy). Reading the action
  back to the host forces the device to finish, so the time includes all GPU work (no asynchronous
  dispatch is left outstanding); device-only times additionally use `block_until_ready()` (JAX) /
  `torch.cuda.synchronize()` (PyTorch).
- Compilation and warm-up: every method/mode is compiled (JAX/XLA jit, TF graph) and run 300 times
  before timing; compilation never enters the latency. Compile time is reported separately.
- Repetitions: 5 seeds (network initialization and observation stream); per seed 30 blocks × 100
  calls, the order of methods/modes randomized in each block (within one framework/process) to
  cancel drift. Reported: mean ± standard deviation across the 5 seed means.
- Latency does not depend on trained weights; randomly initialized networks of the trained
  architecture are used.

## Evaluation rule of each method (from the official code) = what Table 2 times

| Method | Evaluation action (return comparison) | Network evaluations per action |
|---|---|---|
| iBOLT | a = tanh(μ(s, z=0)) (paper); full policy z~N(0,I) + conditional σ also reported | 1 |
| SAC (SBX) | tanh(mean) (`deterministic=True`) | 1 |
| TD3 (SBX) | deterministic actor, no noise | 1 |
| SQL (softqlearning) | `StochasticNNPolicy.get_action`: tanh(f(s, z)), fresh z~N(0,I) per call; one `session.run` | 1 |
| MEow (cleanrl) | `FlowPolicy.sample(deterministic=True)`: prior mean pushed through the conditional flow (4 coupling layers + scaling) | 1 flow pass |
| DIPO | `sample_action(eval=True)`: 100-step reverse diffusion, initial noise only (`noise_ratio=0`), Python loop | 100 |
| MFPO | `eval_actions` with `eval_action_selection=True`: 10 MeanFlow candidates (T=2) scored by twin distributional target critics (101 atoms), argmax of min-Q | 2 actor steps + 2 critics |

Yes: Table 2 times exactly these evaluation calls, i.e. the same action-generation procedure as the
return curves (Figure 6). For iBOLT both z=0 and the full stochastic policy are given below.

## Revised Table 2 (one idle RTX 5090, 5 seeds, mean ± std across seeds; `table2_latency_5090.log`)

| Method | Evaluation action | Steps | Latency (ms) | Paper Table 2 | First call / compile (ms) |
|---|---|---|---|---|---|
| iBOLT | z = 0 (paper eval) | 1 | 0.21 ± 0.01 (0.211 ± 0.009) | 0.20 | 175 |
| iBOLT | z ~ N(0, I), center | 1 | 0.23 ± 0.01 (0.230 ± 0.011) | — | 375 |
| iBOLT | z ~ N(0, I) + conditional σ (full policy) | 1 | 0.23 ± 0.01 (0.231 ± 0.012) | — | 782 |
| SAC | tanh(mean) | 1 | 0.21 ± 0.01 (0.214 ± 0.010) | 0.20 | 75 |
| SAC | sampled | 1 | 0.22 ± 0.01 (0.219 ± 0.011) | — | 204 |
| TD3 | deterministic | 1 | 0.21 ± 0.01 (0.213 ± 0.010) | 0.20 | 48 |
| SQL | tanh(f(s, z)), z ~ N(0, I) | 1 | 0.87 ± 0.05 | 0.69 | 306 |
| MEow | flow, deterministic | 1 | 1.29 ± 0.00 (1.294 ± 0.003) | 1.33 | 413 |
| DIPO | 100-step reverse diffusion | 100 | 33.35 ± 0.12 | 33.35 | 390 |
| MFPO | MeanFlow T=2, 10 candidates + critic selection | 2 | 1.25 ± 0.03 | 1.17 | 2428 |

Architecture as in Appendix C.6 (2×256 actor/critic where applicable; MEow its own flow). With the
repo defaults instead: SQL 2×128 0.88 ± 0.05, DIPO 3×256 35.06 ± 0.26, MFPO 3×256 1.50 ± 0.03 ms.
Frameworks: JAX 0.4.33 (iBOLT, SAC, TD3, MFPO — MFPO's pinned jax 0.4.30 cannot compile for the
RTX 5090), TensorFlow 2 in v1 graph mode (SQL, official policy code unchanged), PyTorch 2.11 (MEow,
DIPO). iBOLT/SAC/TD3: 30 interleaved blocks × 100 calls per seed; SQL/MEow/MFPO 30 × 100; DIPO
10 × 20 (≈35 ms per call). GPU compute per action (JAX, one jitted scan): iBOLT z=0 3.80 µs,
stochastic z 7.52 µs, z+σ 8.76 µs, SAC 3.80 / 6.31 µs, TD3 3.83 µs.

## Mixture / ensemble actors named by R1 (W2, Q3): Gamid-PG and AEAP

Official code: Gamid-PG https://github.com/Pi-Star-Lab/gamid-pg (rl_zoo3 fork, SB3 1.8, `DDPG2`),
AEAP https://github.com/kk0mo/AEAP. Evaluation action as implemented:

| Method | Evaluation action (official code) | Network evaluations per action (Humanoid) |
|---|---|---|
| iBOLT | tanh(μ(s, z)), one actor pass; no critic at inference | 1 actor |
| Gamid-PG | `DDPG2Policy.predict`: all N actor means, each scored by the target critic, pick by Q (N = 4 for Humanoid, paper Table 5) | 4 actors + 4 critic |
| AEAP | `select_action(is_eval=True)`: every active actor, each scored by twin target-critic min-Q (one `.item()` host sync per actor), argmax, then the chosen actor again | K + 1 actors + K twin critics, K = 3 → 1 as pruning proceeds |

Both select the action with the critic at inference, so their per-action cost grows with the number
of components and includes critic passes; iBOLT's mixture is implicit in z and costs one actor pass
regardless of how many modes it represents.

### Humanoid-v4: return, training, inference (`gamid_aeap_latency_5090.log`)

Training: official algorithm code, 1M steps, seeds 0-4, evaluation every 5K steps with 10 episodes
using each method's own evaluation rule (iBOLT protocol). Runners: `baselines/gamid_run.py`,
`baselines/aeap_run.py`. Return = mean over the last 100K steps per run, mean ± std over seeds.
Latency: same protocol as Table 2, measured in one session on an idle RTX 5090 together with an iBOLT
reference (session-to-session offset of the host-bound latency is ~0.02 ms, so compare within the session).

| Method | Return (last 100K) | Latency (ms) | vs iBOLT z=0 (same session) |
|---|---|---|---|
| iBOLT, z = 0 (seeds 0-9) | 4863 ± 403 | 0.230 ± 0.008 | 1.0× |
| AEAP, 3 active actors (all runs ended with 3) | 5101 ± 218 | 0.952 ± 0.008 | 4.1× |
| Gamid-PG, N = 4, TD3 variant (paper) | 4532 ± 311 | 1.045 ± 0.015 | 4.5× |
| Gamid-PG, N = 4, released class (DDPG) | 835 ± 149 | 0.801 ± 0.014 | 3.5× |

AEAP with 2 / 1 active actors: 0.729 ± 0.005 / 0.507 ± 0.001 ms. Training wall time ≈ 4.5 h for all
(GPUs shared with other jobs, not a clean cost comparison).

Notes on running the official code:
- Gamid-PG: the paper (Sec. 3.2, Table 3) builds Gamid on TD3, but the released `DDPG2` subclasses SB3
  DDPG (1 critic, no policy delay, no target smoothing); run as released it reaches only 835 on Humanoid.
  The TD3 variant sets n_critics 2, policy_delay 2, target noise 0.2/0.5 and replaces one line of the
  target-actor selection that assumes a single critic (min over critics; identical for 1 critic).
  The released `predict` computes all N actors and target-critic Q values but selects only between
  actors 0 and 1 (`q_values[0] > q_values[1]`); kept as released. Paper reports 6141 ± 629 at 2M steps.
- AEAP: the repo's main.py imports modules missing from the repository; the runner reproduces its loop
  and defaults with the official agent. Pruning never triggered on Humanoid (3 actors at the end).

From the 3-mode evaluation runs (`results/mujoco_3mode_s0-9`, `results/walker_wandb_check`):
z=0, stochastic z, and z+σ returns over the final 100K steps, mean ± std over seeds.
