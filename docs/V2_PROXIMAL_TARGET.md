# Proximal soft-policy target with sampling and OT

This is an experimental response to concentrated importance weights, not a
claim of improved Humanoid returns. The failed finite screen obtained about
2.6 effective candidates out of 64. Changing Q temperature alone on those same
candidates left density-only concentration almost intact.

## Exact operator and its guarantee

Keep the soft objective temperature T fixed. For the exact old-policy soft Q,
write F(pi,s)=E_pi Q(s,a)+T H(pi(.|s)). For each state choose 0<eta<=1 and
lambda=T(1-eta)/eta. The ideal next policy maximizes

    F(pi,s) - lambda KL(pi(.|s) || pi_old(.|s)).

Its unrestricted optimum is

    p_eta(a|s) proportional to pi_old(a|s)^(1-eta) exp(eta Q(s,a)/T).

The old policy is feasible, so

    F(p_eta,s)-F(pi_old,s) >= lambda KL(p_eta || pi_old) >= 0.

The usual monotone soft Bellman operator/contraction argument therefore gives
V_new>=V_old, assuming exact soft policy evaluation, finite normalizers,
well-defined entropies/KLs, bounded value differences and exact extraction and
distribution-preserving distillation. State-dependent eta is permitted since
the inequality is statewise. Eta=0 denotes keeping the old policy; eta=1 is
the original Boltzmann target. Any exact restricted-class optimizer retaining
the old policy as feasible also has this objective inequality.

This is our application of KL-proximal regularization, not a new guarantee
claimed for an arbitrary OT projection. Related theory: Geist, Scherrer and
Pietquin, [A Theory of Regularized Markov Decision Processes](https://proceedings.mlr.press/v97/geist19a.html).
That paper develops regularized Bellman operators and their connection to
mirror descent. The displayed derivation specifies the objective used here.

## Sampling weights and temperature

The new mode draws fresh teacher candidates from the actual old policy. Then

    w_j proportional to exp(eta * (Q(s,a_j)/T - log pi_old(a_j|s))).

Both terms receive eta. Scaling just the density correction would target a
different energy/entropy balance. Full importance correction is retained for
the proximal target; eta is a policy step, not an independently chosen beta or
the soft TD temperature. The Bellman backup continues to use T=.1.

For each state, 24 bisection iterations select the largest eta in [0,1] with
ESS>=16 out of 64 candidates. ESS decreases monotonically with nonnegative
scaling of a fixed finite score vector. The ESS rule is a sample-budget
heuristic, not a confidence bound. It selects eta from finite sampled actions;
it does not certify an expected or statewise gain after neural projection.

The implementation restricts this mode to the actual finite uniform latent
prior, enumerating all 16 Gaussian components. Teacher sampling uses the
actor's actual conditional sigmas, with no teacher-only floor. Density is
the same full joint mixture after the tanh Jacobian. This matching is required:
using a broader proposal q would instead require weights
`eta*Q/T + (1-eta)*log pi_old - log q`.

Q, candidate actions, density weights and the transport plan are stopped.
The actor still uses the common full-OT conditional NLL update and existing
sampled soft-score guard, with full TrainState rollback on rejection. No
Q-action gradients or additional actor entropy-gradient loss are introduced.
An arbitrary finite OT/NLL projection can still reduce F; the ideal proof
requires the qualifications above. The replay-average guard is an empirical
approximation, not a statewise certificate. Even eta=0 sample weights do not
make a finite neural distillation step exactly identity.

## Candidate configuration and validation scope

`mujoco_v2_proximal` inherits the completed continuous v2 settings: initial
sigma=.5, learned log sigma in [-5,1], actor/critic 256x3, Adam3e-4, grad norm2,
batch256, UTD1, gamma=.99, replay1M, 5K warmup, no subsequent uniform behavior.
It uses a finite prior with16 components for exact marginal density, zero
fixed latent residual, min twin Q for teacher/backup/guard, T=.1, beta1,
16x64 OT and raw squared-action cost with Sinkhorn epsilon=.25/100 iterations.

The failed finite candidate's sigma cap=.2 is removed. This is consistent
with the better completed continuous configuration, and lets variance adapt
to the soft target. It does not ensure that the finite latent means stay
distinct or that modes are preserved. The candidate changes both that
representation restriction and the extraction step, so comparison with the
failed finite run is not a one-factor causal ablation.

Before long training, a paired seed-0 8K GPU check compares this configuration
with `soft_proximal_ess_fraction=0` (the full step). These runs exercise actual
backup, OT, guard, replay, evaluation, saving and online logging. They are too
short to establish return superiority. A separate read-only probe evaluates
eight single OT updates per setting on four frozen 250K finite actors, using
disjoint update and validation states and fresh action draws. It retains the
saved Adam state and sigma cap; it is not a forecast of from-scratch training.

The option defaults to zero, preserving the existing training path. Source:
`optiq_dime/proximal.py`, `optiq_dime/algorithm.py`; theorem, ESS, configuration
and gradient-boundary tests: `tests/test_proximal.py`.

## Completed preflight evidence

The common-loop/guard/finite-policy suite passed 25 tests, and the expanded
proximal suite passed 11 tests (five are repeated from the first suite).
An initial invocation failed before tests ran because the explicit credential
file was missing from that shell; the corrected invocation used the existing
OPTIQ_ENV_FILE and W&B validation project.

Both 8K seed-0 GPU runs finished normally with 3K learning updates and online
W&B sync. Final proximal ESS was 17.10/64 (minimum 16), eta averaged .674,
and cumulative guard acceptance was 46.1%; full-step ESS was 8.56/64 and
acceptance 42.1%. Conditional sigmas were .830 and .842. These are execution and
mechanism checks, not evidence of return superiority.

Frozen 250K actors, eight single-update repeats, held-out learned-Q soft gains:

| Seed | Full-step ESS | ESS-limited eta (target 16) | Full-step gain | Proximal gain |
|---|---:|---:|---:|---:|
|0|2.23|.299|+.00983|+.01198|
|1|2.35|.310|−.00892|−.01191|
|2|2.23|.281|+.00273|+.00583|
|3|2.42|.302|+.00747|+.00791|

All target 16 runs met their sampled ESS constraint. Seed 1's negative held-out
gain remains and gets slightly worse; the representation/optimizer update is
not automatically improving merely because ESS is larger. The sampled guard
therefore stays enabled. The ESS 32 probe also retained seed 1's negative gain,
and gave no consistent gain advantage over 16; 16 is retained for the screen.
Evidence: `proximal_frozen_probe.json`, `proximal_validation.log`,
`proximal_active_path_validation.log`, and `outputs/v2_proximal_validation/`.

## Predeclared training screen

Humanoid-v4 seeds 0/1/2/3 start from scratch on GPUs 0/1/2/3, with a 1M maximum,
10 stochastic evaluation episodes every 5K and actor/critic saves every 50K.
All four seeds remain in every aggregate. Before launch, the group budget
gates are fixed as follows, using the last five shared evaluation checkpoints:

| Checkpoint | Stop if four-seed mean is below this fraction of BOTH stronger references |
|---|---:|
|100K|.60|
|250K|.80|
|500K|.95|

References are completed continuous v2 and historical OptiQ with 10% collection
exploration. These are compute-budget decisions, not statistical tests or
proofs about unseen final performance. Stopping waits for all eight matching
actor/critic checkpoints. The original matched zero-uniform control remains
in reports but cannot alone qualify the candidate as a success.

The primary comparison remains the 900K–1M window (21 shared checkpoints),
followed by 50 new stochastic evaluation episodes per final actor with the
same episode seeds as the previous confirmation. Raw returns, all time-limit
episodes, survival, locomotion and drawdowns must be reported. Passing an
intermediate gate does not achieve the goal.

## Launch record

The four production runs started at 2026-09-11 01:33:35 UTC from clean local
commit `9bf6c36ae484f92145b37579a7dca3ee246e645a`. W&B:
[OptiQ/optiq_mujoco_v2_proximal_screen](https://wandb.ai/OptiQ/optiq_mujoco_v2_proximal_screen).
Seed 0/1/2/3 run IDs are `pzohewyx`, `xodbilj4`, `ddrh0757`, `wvkctogx`.

All four passed 10K with real learning updates. The first diagnostic ESS was
16.25/16.73/16.37/16.10 out of 64, with minimum 16 for each batch and mean eta
.526/.571/.592/.527. This confirms execution of the requested step limiter,
not a return advantage. The comparison reporter successfully retained all 16
candidate/reference runs at the same 10K cutoff. Artifacts:
`proximal_screen_manifest.json`, `proximal_screen_protocol.json`,
`proximal_launch_validation.json`, `proximal_report/step_0010000.json` under
`outputs/v2_improvement/`.

After launch, only the monitor was restarted to require successful MessagePack
parsing of all eight checkpoints before a stop. The common saver writes to
the final filename directly, so existence/nonzero size alone is insufficient.
The truncated-checkpoint test passed; all training processes and frozen core
hashes remained unchanged. No push was performed.

## Final-evaluation automation

The shared finite-policy evaluator and finisher now accept an explicit
protocol path. The proximal protocol keeps the same primary window, 50 final
episodes, environment/policy seeds, cached reference episodes and common
evaluation function as the completed confirmation. Distinct result/lock/status
paths prevent overwriting the previous finite screen. The original defaults
remain available.

Seven routing/completion tests passed. A real readiness check validated all
12 reference models (600 recorded episodes) and correctly reported all four
candidate runs as still RUNNING. Final evaluation has not yet occurred.
A completion marker with fewer than 1M actual steps or a missing primary
evaluation cannot qualify as a full result. If the group stops early, the
finisher waits for its termination and reports the common horizon with every
seed retained, without running or claiming final 1M evaluations.

The finisher uses supervisor and CPU evaluation; it never controls training.
Its `goal_complete` field remains false even when report generation succeeds.
Artifacts: `proximal_final_evaluation_protocol.json`,
`proximal_final_evaluation_readiness.json`,
`proximal_final_evaluation_validation.log`, `proximal_finisher_status.json`.

## First checkpoint review: 50K

All four runs reached 50K and all eight actor/critic checkpoints parsed
successfully. The 30K/35K/40K/45K/50K evaluation means are:

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Four-seed mean |
|---|---:|---:|---:|---:|---:|
| Proximal v2 |440.2|419.9|434.9|468.4|440.9|
| Completed continuous v2 |442.5|445.5|446.1|440.4|443.7|
| Matched OptiQ, uniform 0% |529.2|395.4|438.2|473.4|459.0|
| Historical OptiQ, uniform 10% |517.3|472.4|482.3|417.7|472.4|

The candidate is near continuous v2 and below both OptiQ references. This
does not establish superior returns or stability. The four-seed SD is 20.3,
versus 2.7/56.6/41.3 for the respective references; no confidence claim follows
from these small early samples.

Over the same training interval, ESS averaged 16.00–16.01/64, compared with
2.39–2.66 when applying the untempered full-step weights to those same candidate
sets. Mean eta was .252–.278, sigma .826–.845, and cumulative guard acceptance
averaged 60.4–62.3%. Latent-mean variance contributed only about .0022% of the
pre-tanh variance on these replay batches; a multimodal representation
advantage has not been demonstrated. These observations are not causal
attributions or statewise improvement bounds.

Training core hashes and process PIDs remained unchanged. The W&B API
independently confirmed all four runs receiving live logs in the intended
project. Continue to the predeclared 100K review. Evidence:
`proximal_review_0050000.json`, `proximal_wandb_delivery.json`, and
`proximal_report/step_0050000.{json,md,png,pdf}`.

## First budget gate: 100K

The 80K/85K/90K/95K/100K four-seed comparison is:

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean ± seed SD |
|---|---:|---:|---:|---:|---:|
| Proximal v2 |591.7|787.6|565.4|654.1|649.7 ± 99.2|
| Completed continuous v2 |710.2|638.9|649.3|696.3|673.7 ± 34.9|
| Matched OptiQ, uniform 0% |598.2|527.0|538.1|623.4|571.7 ± 46.5|
| Historical OptiQ, uniform 10% |723.0|721.8|702.4|697.1|711.1 ± 13.2|

The candidate is 3.6% below continuous v2 and 8.6% below historical OptiQ,
while 13.7% above the weaker matched control. Only seed 1 exceeds both stronger
references; seed variation is larger here. Thus neither overall superiority
nor a stability advantage is established. The ratio to the lower stronger
reference is .9644, above the predeclared .60 stopping threshold. The monitor
applied the 100K gate and continued all four runs without changing parameters.
The next resource gate remains 250K at .80 of both stronger reference means.

Over 80K–100K, ESS averaged 16.00–16.01/64, versus 2.02–2.14 for the counterfactual
untempered full step on the same candidate sets. Mean eta fell to .208–.218;
conditional sigma averaged .845–.860. Cumulative guard acceptance averaged
59.9–61.4%. The latent-mean share of pre-tanh variance remained about .0023%
on replay batches. Increased ESS is a verified mechanism effect; none of
these numbers is a true statewise improvement or critic-error certificate.

All eight 100K actor/critic checkpoints parsed successfully, their hashes were
recorded, and all training core hashes/PIDs still matched the launch. The
comparison report and monitor agree numerically. Final-evaluation automation
remains waiting for training, with no final result claimed.
Evidence: `proximal_review_0100000.json`, `proximal_screen_monitor.json`,
`proximal_report/step_0100000.{json,md,png,pdf}`.
