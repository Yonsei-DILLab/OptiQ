# v2 improvement investigation

## Stop decision, 2026-09-10

All four original v2 runs were stopped at about 154k-155k environment steps after
preserving their 150k actor and critic checkpoints. At 140k, returns were
289/281/296/237 for seeds 0/1/2/3, versus 1123/1452/818/722 in historical OptiQ.
The v2 curves were flat or declining over 100k-140k. The historical runs used
10% additional uniform exploration, so this is a resource-allocation decision,
not a controlled causal comparison. Logs and W&B runs are retained and annotated.
Machine-readable evidence: `outputs/v2_improvement/early_stop_evidence.json`.

ESS remained around 28-32/64 while the entropy contribution to the TD target was
around -5, comparable in magnitude to the per-step environment reward. High ESS
therefore did not establish good policy learning.

## Flat-Q diagnosis

`scripts/diagnose_v2_projection.py` fixes Q to zero on 32 stored Humanoid states,
keeps the 17-dimensional action space, and runs 1001 actor updates. The exact
Boltzmann target is uniform on the action box regardless of temperature.

| Projection | Initial entropy estimate | Final entropy estimate | Final mean conditional sigma | Final ESS / 64 |
|---|---:|---:|---:|---:|
| argmax + pointwise MSE | 8.38 | -2.43 | .223 | 32.43 |
| full OT conditional Gaussian NLL | 8.38 | 11.48 | .951 | 3.52 |

The uniform entropy maximum in normalized coordinates is 17*log(2)=11.78 nat.
This isolates a projection failure even without critic errors: high ESS and a
valid transport plan do not imply that a hard argmax student retains its target
distribution. In the MSE run, converting the rows to argmax assignments changed
the column masses by mean TV distance .577 at the final update.

The NLL loss uses the full stopped OT row. Its teacher pre-tanh mean and variance
are sufficient statistics for the Gaussian conditional cross entropy. This
retains the row's variance instead of fitting a single point. The teacher, Q,
weights and plan remain stopped; there are no actor gradients through Q, no
additional actor entropy bonus, and no added uniform behavior exploration.
The tanh Jacobian of fixed teacher samples is constant with respect to student
parameters and need not enter this loss. This is a different projection loss;
it is not claimed to optimize exact Wasserstein distance or reverse KL.

## Bounded screening experiment

Four fresh seed-0 Humanoid runs, maximum 100k steps each, compare:

1. Original `mujoco_setting` OptiQ, T=.25, anchor=true, extra uniform probability 0.
2. v2 conditional OT NLL, T=.5.
3. v2 conditional OT NLL, T=.1.
4. v2 original pointwise MSE, T=.1 (temperature-only control).

All use three stochastic evaluation episodes every 5k steps for screening,
25k checkpoints, 256x3 networks, batch 256, UTD1, 5k warmup, and gradient norm2.
The v2 runs retain 16x64 OT, h=.8, beta1 and M16 entropy components. At 50k and
75k, review the last three evaluation checkpoints. A variant with less than half
the matched reference's return and no improving trend is eligible for early
termination after preserving a checkpoint. This is a pilot selection rule, not
a statistical claim of superiority. Only promising settings advance to the
four-seed, ten-episode evaluation protocol.

W&B: `OptiQ/optiq_mujoco_v2_screen`. Supervisor definition:
`deploy/supervisor/optiq-v2-screen.conf`. No git push is performed.

### Follow-up: learned conditional proposals

At the 50k review, mean returns over the 40k/45k/50k checkpoints were 412.6 for
the matched OptiQ reference, 418.1 for MSE T=.1, 466.4 for NLL T=.1, and 474.5
for NLL T=.5. None met the early-stop criterion. This is still a single-seed
screen, not evidence of stable final superiority.

Saved 25k NLL actors exposed a second issue: their realized-action KDEs had
low importance ESS even after correcting the projection. At T=.5, using the
explicit mixture of learned conditional Gaussians gave ESS 15.04/64 versus
2.84/64 for the KDE; at 256 candidates these were 48.62 and 4.71. These are
fixed-model proposal diagnostics, not comparisons of trained returns.

`ConditionalGaussianProposal` samples from a full mixture of the actor's saved
conditional means and diagonal standard deviations. The teacher-only standard
deviation is `max(actor_sigma, teacher_std_floor)` coordinatewise. Its density
uses those same values, includes the tanh Jacobian, and retains full support.
The entropy estimator remains distinct and uses the actual actor scales. The
actor rollout is unchanged. No realized-action anchors are added. This proposal
is an explicit alternative to the supplied realized-u KDE, not a relabeling of it.

A further 17D known-target test compared flat Q and a quadratic Q whose exact
Boltzmann action distribution is a box-truncated N(0,.15^2 I). With NLL, the
realized KDE with h=.8 converged to mean sigma .696 on the narrow target; a
conditional mixture with floor .8 reached .510, while floor .05 reached .154.
With flat Q, floor .05 reached entropy 11.585 and ESS 39.99/64. Thus a large
fixed teacher floor would recreate the inability to contract the policy.
Evidence: `outputs/v2_improvement/conditional_projection_diagnosis.json`.

`mujoco_v2_conditional` selects the NLL projection and this learned proposal with
floor .05. A separate bounded 100k screen is prepared for T=.1/.5 and seeds 0/1,
using the same no-extra-uniform protocol. It starts only as GPUs become free
from the first screen. It is not the default `mujoco_v2` configuration. Sampling,
matching density, common Humanoid training and checkpoint restoration are tested.

### Completed 100k conditional-proposal screen

All four conditional-proposal runs completed their 100k cap without meeting the
half-reference early-stop rule. Last-three-evaluation mean returns were:

| Setting | Seed 0 | Seed 1 |
|---|---:|---:|
| T=.1, conditional proposal | 794.65 | 916.75 |
| T=.5, conditional proposal | 583.20 | 647.54 |
| Matched OptiQ protocol, reference | 721.67 | not run |

These evaluations have only three episodes per checkpoint. An independent
read-only evaluation of each final 100k checkpoint with 30 stochastic episodes
and shared fresh environment seeds gave:

| Fixed checkpoint | Mean return | Episode SD | Mean episode length |
|---|---:|---:|---:|
| OptiQ seed 0 | 697.60 | 139.26 | 138.23 |
| Conditional T=.1 seed 0 | 726.11 | 188.03 | 139.83 |
| Conditional T=.1 seed 1 | 798.09 | 298.48 | 156.87 |

For seed 0, the paired mean gap is +28.51 with an episode-bootstrap 95% interval
[-59.21, 119.33]. This does not establish superiority. The seed-1 comparison uses
the seed-0 reference and does not substitute for a matched training seed. The
intervals quantify episode noise of these fixed checkpoints, not training-seed
variation. T=.1 is promising relative to original v2, but neither stable OptiQ
superiority nor final 1M performance has been established.

`scripts/report_v2_screens.py` generates PNG/PDF curves and a JSON summary;
`scripts/evaluate_v2_screen_checkpoints.py` reproduces the independent evaluation.
Evidence is under `outputs/v2_improvement/`, particularly
`conditional_screen_results.json` and `independent_100k_evaluation.json`.

## Policy-improvement requirement

Neither the original argmax-MSE nor the new conditional NLL alone provides a
monotonic soft-policy-improvement guarantee. A lower projection loss is not an
acceptance certificate. For a fixed temperature, define

`F_Q(pi,s) = E_{a~pi}[Q(s,a)] + T*H(pi(.|s))`.

With the exact old-policy soft Q, the sufficient condition is
`F_Q(pi_new,s) >= F_Q(pi_old,s)` at every relevant state. The new-policy Bellman
operator is then monotone and contractive, giving nondecreasing soft values.
This condition does not require taking gradients through Q or using KL as the
proposal loss. It does require evaluating the proposed policy, rather than
assuming that OT distillation has improved it.

For an approximate critic satisfying a *verified uniform* error bound epsilon,
the estimated F gap must exceed 2*epsilon. If new entropy has a lower bound and
old entropy has an upper bound, use those sides in the acceptance gap. Sampling
confidence errors must also be subtracted. A replay average or twin-critic
disagreement is not a certificate of either global state coverage or uniform
critic error. Those are unresolved for the actual Humanoid runs. Finite-M IDAC
entropy is a lower bound in expectation; comparing two lower bounds does not
certify that the true entropies improved.

Candidate updates can now be evaluated using these soft value conditions. The
practical sampled acceptance check remains distinct from the theorem under
exact evaluation/error-bound assumptions. There is currently no claim that
practical Humanoid improvement is guaranteed.

`optiq_dime/soft_improvement.py` supplies the paired sampled candidate check and
the explicit error-margin calculation. The original and conditional-proposal
screening runs do not enable this check. Its entropy bracket uses
the generating component for the new-policy lower estimate and an independent
mixture for the old-policy upper estimate. The latter follows Jensen's inequality:
`E_F[-log g_F(a)] >= -log pi(a)` for an action independent of F. The lower side
follows entropy concavity and exchangeability of the generating component.

For completeness, the exact sufficient-condition proof assumes gamma<1 and
bounded rewards/entropy values so the Bellman fixed points exist. Let V_old be
the exact old-policy soft value. The gap condition gives
`T_new V_old >= V_old`. Monotonicity gives `T_new^k V_old >= V_old` for every k;
contraction gives `V_new >= V_old` in the limit, and hence `Q_new >= Q_old`.
The 2*epsilon margin accounts for the two expectations of an approximate Q.
This proof is independent of how the candidate was proposed, so OT proposals
can be used without differentiating Q. It is also a statement about the soft
objective; it does not imply that unregularized environment return beats OptiQ.

Tests compare entropy-bracket sampling to an exactly known two-component
mixture, verify a paired Gaussian candidate check, and solve a finite MDP's
Bellman equations exactly. They reject a flat-Q entropy-collapse proposal and
show why an unknown critic error cannot be replaced by zero. They do not certify
the learned Humanoid critic or turn sample standard errors into rigorous bounds.

References: [SAC soft policy iteration, Appendix B](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
and [IDAC entropy estimation](https://arxiv.org/html/2007.06159). The proposed OT
projection and the approximate-critic acceptance margin above are our analysis,
not claims attributed to the original IDAC algorithm.

## Optional sampled acceptance filter

`mujoco_v2_guarded` enables the experimental filter in the common training path.
An independent replay draw supplies 32 validation states. Eight paired action
draws per state compare the candidate's lower soft score with the old policy's
upper score, using the same frozen minimum of the two current critics. The
candidate is accepted only when the mean gap minus twice its estimated standard
error is positive and all score samples are finite. The error is computed over
state means, since action draws at one state form a cluster. This is an empirical
resource/stability heuristic, not a statistical confidence certificate or a
uniform statewise test. The teacher also uses minimum Q in this configuration.

On rejection, actor parameters, optimizer moments, and optimizer step all return
to their old values. The target actor is updated only from the selected actor;
critic training continues. The disabled path takes no validation replay draw
or additional JAX key. Neither path adds Q gradients, an actor entropy-gradient
loss, or additional uniform behavior exploration. Logs contain acceptance,
estimated gain/error, negative-state fraction, and cumulative acceptance.

The 50k saved T=.1 actor had positive independent 64-draw mean gaps in all 16
probe trials, but only 1/16 passed the 32-state, 8-draw, two-standard-error filter.
Using 96 states and 32 draws still accepted only 1/16 at that same checkpoint.
This reveals a real risk of over-rejection; increasing sample count alone did
not remove it. The T=.5 actor passed 0/16 with either setting. These probes used
fixed saved states, a fixed critic and one proposed update each; they are not
online returns. Evidence: `outputs/v2_improvement/soft_guard_probe.json` and
`soft_guard_probe_96x32_50k.json`. An exploratory probe at 75k gave a different
result and must not be compared as a sample-count ablation against 50k.

A separate bounded screen compares filter on/off at T=.1 with 256 teacher
candidates (OT 16x256), minimum-Q teachers, and seeds 0/1. Increasing candidate
count addresses the approximately 3-4/64 ESS seen during the conditional T=.1
screen; it is applied to both sides of this comparison. Each run is capped at
100k environment steps, with the same 3-episode/5k evaluation and 25k checkpoint
protocol. This is an experiment on the filter, not an announced improvement.
Review poor return and sustained near-zero acceptance before any longer run.
Launcher: `scripts/supervisor_v2_guarded_screen.sh`; W&B project:
`OptiQ/optiq_mujoco_v2_guarded_screen`. No push is performed.

Validation: 13 soft-filter/theory tests (including an actual Humanoid loop and
complete optimizer rollback), plus 51 common-path, critic, mixture, and
distillation regression tests passed. Local logs are under
`outputs/v2_improvement/{soft_guard_tests,guard_regression_tests}.log`.

### Frozen-policy soft-return check at 25k

`scripts/diagnose_v2_soft_critic.py` rolls out frozen saved policies on fresh
environment seeds. Actions follow the actual semi-implicit marginal; generating
and independent mixture components provide lower/upper entropy estimates. The
backward return excludes current-action entropy and includes entropy starting
at the next state, matching the implemented soft Q convention. A dedicated test
checks this indexing and true-terminal treatment. Time-limit episodes would be
excluded and reported; none occurred in this probe.

For K256, T=.1, seed 0, 20 complete episodes at the 25k checkpoint gave:

| Quantity, mean over episodes | Filter on | Filter off |
|---|---:|---:|
| Undiscounted environment return | 316.29 | 412.58 |
| Episode length | 59.55 | 80.30 |
| Initial predicted minimum Q | 231.10 | 251.67 |
| Initial Monte Carlo soft return, lower estimator | 260.75 | 297.46 |
| Mean entropy lower estimate | 5.84443 | 3.42464 |
| Mean entropy upper estimate | 5.84444 | 3.42466 |

The initial Q predictions are low on average in both cases. The finite-mixture
entropy bracket gap is tiny here, so a large entropy-estimator gap does not
explain the filter's frequent rejections at this checkpoint. This does not prove
that action rankings are accurate or that overestimation cannot occur elsewhere.
Each policy induces different visited states and episode lengths: the respective
critic errors are not a controlled comparison on a common state distribution.
The rollout errors cannot be promoted to a verified uniform critic-error bound.
Evidence: `outputs/v2_improvement/guarded_25k_soft_critic.json`.

## Prepared confirmation protocol

`scripts/supervisor_v2_confirmation.sh` prepares fresh, paired training seeds
0/1/2/3 for a selected v2 candidate and the original OptiQ reference.
The variant must be explicitly selected with `OPTIQ_CONFIRMATION_VARIANT`;
preparing this script does not start the queue or select a winner. Each run has
a 1M cap, ten stochastic evaluation episodes every 5k, and 50k checkpoints.
Both methods have zero extra uniform behavior exploration. Network sizes,
critic setup, optimizers, gradient clipping, replay/batch sizes, UTD, warmup,
discount, target update coefficients and evaluation settings are matched.
Resolved configuration checks are saved in
`outputs/v2_improvement/confirmation_common_protocol.json`.

Each supervisor process owns one run; a failure or explicit stop does not launch
another seed. The intended allocation is one v2/reference pair per GPU. The
preceding single-process GPU load was about 46-50% and memory use below 1.5 GiB
per process. This allocation seeks shorter total wall time; shared-GPU elapsed
times must not be used as an isolated algorithm-throughput benchmark.
Long runs remain subject to early-stop review against the matched reference;
the queue is not permission to spend the full cap on a clearly poor candidate.
Final claims must use all prescribed training seeds, fixed-step/tail performance
and variability, not only a selected best checkpoint or episode confidence
intervals. Exact soft-improvement assumptions and practical learned-critic
limitations must remain separate from the empirical return comparison.

### Selected next candidate

The K256/minimum-Q experiments did not outperform the earlier conditional K64
screen. At 100k, seed-0 last-three-evaluation returns were 446.43 with the
two-standard-error filter and 595.98 without it, versus 794.65 for the earlier
K64/mean-Q teacher. K and teacher Q aggregation changed together, so this is not
an isolated sample-count ablation. The strong filter accepted only about 2% of
proposed updates late in training.

After the user clarified that the requested guarantee is the idealized SAC-style
theoretical guarantee plus empirical validation, the next candidate is
`mujoco_v2_checked`: the promising K64 conditional/NLL configuration, T=.1, with
a positive estimated soft-gain check and no extra standard-error margin. The
teacher still uses mean Q; the acceptance score uses the same frozen minimum Q
for old and new policies. This is a new candidate, not a relabeling of any
completed result. It is validated in the common Humanoid loop, but its long-run
performance is not yet established. The four paired comparisons have 1M caps
and remain subject to early termination.

See [the precise soft-improvement theorem and limitations](V2_SOFT_POLICY_IMPROVEMENT.md).
Exact policy evaluation alone is insufficient for arbitrary OT projections;
the exact extraction/distillation or statewise acceptance assumptions must also
be stated. The note includes a tested counterexample to a W2-only guarantee.

### First paired confirmation review: 100k

All eight confirmation runs launched from `8cb4f23` remain running. The
100k comparison averages the 90k, 95k and 100k evaluations, with ten stochastic
episodes at each checkpoint. Slower runs are retained, and both methods are
compared at these same environment steps.

| Training seed | Checked K64 v2 | OptiQ reference | Relative difference |
|---|---:|---:|---:|
| 0 | 746.89 | 571.99 | +30.6% |
| 1 | 703.17 | 534.11 | +31.7% |
| 2 | 679.94 | 524.34 | +29.7% |
| 3 | 731.47 | 629.37 | +16.2% |

Across four training seeds, the means are 715.37 and 564.95, with seed SDs
29.76 and 47.61 respectively. The mean difference is +150.41 (+26.6%). This is
an early milestone, not final superiority or evidence of monotonic improvement.
All four recent v2 trends are nonnegative. Continue all eight runs and review
again at 150k; no run currently warrants termination for poor relative return.

At 100k, v2 importance ESS is only 2.49-2.98 out of 64. The sampled filter has
accepted approximately 50-53% of proposed updates cumulatively. The last logged
backup entropy lower estimates at this step are 2.95-3.96 nat, contributing
+0.30 to +0.40 before discount at T=.1. The old large negative entropy
contribution is absent at this milestone,
but sparse importance weights remain a limitation to monitor. Neither ESS nor
the acceptance fraction is a policy-improvement certificate.

All sixteen 100k actor/critic checkpoint files exist and are nonempty. The
read-only decision record is
`outputs/v2_improvement/confirmation_review_0100000.json`; the paired reporter
is `scripts/analyze_v2_confirmation.py`. Training configuration and algorithms
were not changed for this review. W&B:
`OptiQ/optiq_mujoco_v2_confirmation`.

### Paired confirmation review: 150k

At the next common milestone, the 140k/145k/150k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | Relative difference |
|---|---:|---:|---:|
| 0 | 1241.71 | 750.93 | +65.4% |
| 1 | 870.68 | 717.46 | +21.4% |
| 2 | 1265.33 | 664.96 | +90.3% |
| 3 | 1014.75 | 861.04 | +17.9% |

The seed means are 1098.12 versus 748.59 (+46.7%), with seed SDs 189.08 and
82.89. Every pair remains positive, but v2 has more seed variation at this
milestone. Seed 0's three-evaluation average fell from 1680.03 at 135k to
1241.71 at 150k, and its recent seven-checkpoint slope is negative. Thus the
evidence supports continued investigation, not a claim of stable monotonic
learning. No run meets the inferior-and-flat early-stop rule. Continue the
same eight runs and review at 200k, including seed 0's drawdown.

At 150k, v2 ESS is 2.08-2.40/64, cumulative candidate acceptance is 52.6-54.6%,
and the last logged pre-discount backup entropy term is +0.20 to +0.29. All
sixteen 150k actor/critic checkpoint files are present and nonempty. W&B also
confirmed all eight runs were actively logging during this review period.

The mean episode lengths in these evaluations are approximately 215 steps for
v2 and 148 for OptiQ; return per step remains near 5.1. Longer episodes account
for much of the return advantage. Neighboring collection-time logs estimate
150k completion at approximately 42.4 minutes for v2 versus 29.4 minutes for
OptiQ. These times include evaluation/compilation and paired GPU sharing; they
are observed experiment costs, not isolated algorithm-throughput benchmarks.

Decision and exact diagnostics:
`outputs/v2_improvement/confirmation_review_0150000.json`.
The report and figure are under `confirmation_report/step_0150000.*` in the same
directory. No training algorithm or configuration changed, and no push occurred.

### Frozen-policy critic check after the 150k drawdown

The existing read-only CPU diagnostic evaluated the saved 150k actors and
critics for seeds 0 and 2 on 20 fresh shared environment/policy seeds each.
Seed 0 was selected because of its recent return drawdown; seed 2 was rising.
This is a diagnostic of frozen critics, not an additional benchmark comparison.

| Quantity | Seed 0 | Seed 2 |
|---|---:|---:|
| True-terminal episodes included | 17 | 20 |
| Time-limit episodes excluded | 3 | 0 |
| Mean initial minimum Q prediction | 319.35 | 306.46 |
| Mean initial Monte Carlo soft Q, lower estimator | 479.85 | 480.51 |
| Mean prediction minus Monte Carlo estimate | -160.50 | -174.05 |
| Episodes with positive initial prediction error | 1/17 | 0/20 |
| Mean upper-minus-lower Monte Carlo entropy contribution | .000504 | .000224 |

The observed initial Q values underestimate these true-terminal trajectory
returns on average in both policies. Neither systematic initial-state Q
overestimation nor a large finite-mixture entropy bracket gap explains this
specific probe. It does not rule out inaccurate action rankings, overestimation
on replay states, or other critic feedback problems. The means condition on
true-terminal episodes, especially for seed 0; they are not unconditional
policy-return estimates or uniform critic-error bounds. Each policy visits its
own states, so cross-policy error differences are not causally identified.

During continued training, seed 0's recent evaluation average recovered to
2016.56 at 165k. The 150k dip therefore did not immediately develop into a
sustained collapse. No algorithm or configuration changes were made in response
to this short fluctuation. Raw evidence and limitations are saved in
`outputs/v2_improvement/confirmation_150k_soft_critic{,_summary}.json`.

### Final confirmation assessment, fixed before 200k

The primary completed-run comparison will use the mean evaluation return over
900k through 1M inclusive, first averaged within each training seed and then
across the four prescribed seeds. This fixes the final 100k window before the
outcome is available. Also report the 1M checkpoint, each paired seed difference,
seed SD, the weakest seed, and the full common-step learning curves. Individual
evaluation episodes are not additional independent training seeds.

Stability assessment will include drawdowns of the three-evaluation rolling
mean after 100k, their duration and recovery, and comparisons to the same
baseline measurements. A peak or a favorable short interval is insufficient.
Training failures or early-stopped seeds remain in the record; they cannot be
silently excluded to manufacture a successful four-seed result. The current
interim gains do not complete the goal of stable final superiority.

Observed wall time is reported separately from environment-step efficiency,
with the paired-GPU resource-sharing qualification. The ideal soft-improvement
theorem retains its exact extraction/distillation or exact statewise acceptance
conditions; empirical return gains do not remove those assumptions or certify
the sampled replay-average acceptance filter.

### Paired confirmation review: 200k

The common 190k/195k/200k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | v2 / OptiQ |
|---|---:|---:|---:|
| 0 | 3491.13 | 832.37 | 4.19 |
| 1 | 3041.00 | 786.60 | 3.87 |
| 2 | 2352.62 | 1011.01 | 2.33 |
| 3 | 2109.94 | 874.36 | 2.41 |

Across seeds, the means are 2748.67 versus 876.09 (3.14x), with SDs 632.86 and
96.83. The mean paired difference is +1872.59. All prescribed seeds remain in
the comparison. Seed 0 recovered from the 150k drawdown; the slowest v2 seed
also improved. This still establishes only interim performance, not stable
final superiority. Continue all eight runs, with the next principal review
at 300k and earlier review if a sustained deterioration or runtime failure
appears.

At 200k, v2 ESS is 2.01-2.27/64 and cumulative update acceptance is 54.2-55.9%.
The pre-discount entropy contribution remains positive, +0.15 to +0.24. The
mean evaluation episode length is approximately 545 steps versus 173 for OptiQ;
return per step is near 5.0 in both. Observed mean time to 200k is about 57.6
minutes for v2 and 39.4 minutes for OptiQ under the existing paired-GPU allocation.

All sixteen 200k actor/critic checkpoints are present and nonempty. The decision
record is `outputs/v2_improvement/confirmation_review_0200000.json`, with fixed
figures and the report in `confirmation_report/step_0200000.*`. The CPU diagnostic
is finished, all eight training processes remain live, and no training algorithm,
configuration or remote branch was changed during this review.

The read-only reporter now accepts `--through-step 200000` to reproduce a
fixed milestone from later logs. Both plotted panels end at the shared horizon,
so faster seeds' later checkpoints do not leak into a milestone figure. Runtime
health diagnostics remain current and are distinct from the evaluation cutoff.
Rebuilding the 100k and 200k comparisons reproduced every preserved seed score
within 1e-9; all pair comparisons ended at the requested step. Validation:
`outputs/v2_improvement/confirmation_cutoff_validation.json`.

### Fixed-window and stability assessment tool

`scripts/assess_v2_confirmation.py` reads the same frozen eight-run manifest.
It reports the final 900k--1M comparison only when every prescribed evaluation
checkpoint exists for all four training seeds of both methods. Missing seeds
or evaluations cannot be replaced by a favorable partial tail. Completion
markers and supervisor states are checked separately, and even a complete
dataset is marked ready for scientific review rather than automatically declared
a successful algorithm. This tool never controls training processes.

For intermediate data, it computes normalized learning-curve area and maximum
drawdowns of three-evaluation rolling means from 100k onward, always using the
horizon shared by all eight runs. Drawdown recovery times distinguish observed
recovery from an unfinished interval. Four assessment/alignment tests passed,
including a missing final checkpoint and a collapse with censored recovery.

At the 225k shared horizon, v2's largest rolling-mean drawdowns for seeds 0--3
were 26.1%, 11.2%, 11.1%, and 33.0%; the matched reference values were 8.3%,
17.3%, 15.4%, and 11.5%. The large seed-0 drop recovered by 165k. Seed 3's
205k-to-215k drop had not fully recovered at 225k. This makes the distinction
between higher return and demonstrated stability concrete: the former is
currently supported, while the latter still requires further observation.

Artifacts: `outputs/v2_improvement/confirmation_assessment/step_0225000.json`
and `outputs/v2_improvement/confirmation_assessment_tests.log`.

### Independent 250k checkpoint evaluation

`scripts/evaluate_v2_confirmation.py` restores the exact actor checkpoints and
uses the common stochastic predict/unscale path on CPU. It evaluated all eight
250k models on 50 fresh shared environment/policy seed pairs, 400 episodes in
total. Every episode is included, including time-limit truncations. Checkpoint
hashes and individual episode records are preserved. A dedicated test verifies
that a high-return truncated episode cannot be dropped from the evaluation.

| Training seed | v2 mean return | OptiQ mean return | Paired gap |
|---|---:|---:|---:|
| 0 | 3049.66 | 1532.79 | +1516.87 |
| 1 | 4587.69 | 915.16 | +3672.53 |
| 2 | 3741.93 | 1289.98 | +2451.95 |
| 3 | 2295.75 | 869.64 | +1426.11 |

The four-seed means are 3418.76 versus 1151.89, with seed SDs 977.79 and 316.15.
Episode-bootstrap 95% intervals for the paired gaps were [983, 2035],
[3312, 3995], [2043, 2844] and [1024, 1853]. These intervals describe episode
uncertainty conditional on each frozen model pair, not uncertainty across new
training seeds. The observed checkpoint advantage persists on new evaluation
seeds; it does not establish final 1M superiority or monotonic training.

v2 episode-return SDs range from 1236 to 1542, so ten-episode checkpoint means
can fluctuate substantially. Time-limit episodes for v2 seeds 0--3 numbered
8, 38, 17 and 4 out of 50; all four reference counts were zero. Independent
means should be compared to the original **single 250k checkpoint**, not the
earlier three-checkpoint tail average. The stored validation records make that
comparison explicitly and include all 400 episodes.

Evidence: `outputs/v2_improvement/confirmation_independent_0250000.json`,
`confirmation_independent_0250000_validation.json` and
`confirmation_evaluation_tests.log`. This is a secondary fixed-checkpoint
evaluation; the predefined 900k--1M primary window remains unchanged.

### Paired confirmation review: 300k

The common 290k/295k/300k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | v2 / OptiQ |
|---|---:|---:|---:|
| 0 | 4338.76 | 2716.69 | 1.60 |
| 1 | 4506.99 | 955.60 | 4.72 |
| 2 | 3691.09 | 1060.74 | 3.48 |
| 3 | 3759.92 | 1545.03 | 2.43 |

The four-seed means are 4074.19 versus 1569.51 (2.60x), with seed SDs 409.41
and 806.71. All pairs are positive. OptiQ seed 0 is improving rapidly, so the
earlier gap cannot be assumed to persist to 1M. The v2 largest rolling-mean
drawdowns after 100k are 26.1%, 23.7%, 25.2% and 33.0%; each of these episodes
has recovered, with peak-to-recovery durations of 30k, 40k, 25k and 30k steps.
Current v2 drawdowns from running peaks are 5.0%, 3.2%, 10.6% and 0.0%.
Thus recovery is supported, while monotonic or final stable superiority remains
unproven. Continue all eight runs and review at 400k, sooner if deterioration
or a runtime failure warrants it.

At 300k, v2 ESS is 1.92-2.49/64 and cumulative acceptance is 56.0-57.5%.
The pre-discount entropy term ranges from -0.008 to +0.134; the small negative
value in seed 0 is unlike the original v2's approximately -5 contribution.
Average evaluation episode length is approximately 779 steps versus 308 for
OptiQ. Observed time to 300k is about 89.6 minutes versus 59.9 minutes under
paired GPU sharing, including evaluation and the concurrent bounded CPU probe.

All sixteen 300k actor/critic checkpoints are present and nonempty. The
independent CPU evaluation is finished and all eight training processes remain
live. Evidence: `outputs/v2_improvement/confirmation_review_0300000.json`,
`confirmation_assessment/step_0300000.json` and
`confirmation_report/step_0300000.*`. No training algorithm or configuration
was changed and no push occurred. The final fixed window is not yet available.
