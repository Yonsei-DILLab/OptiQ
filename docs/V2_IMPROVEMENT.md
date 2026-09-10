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

### Active-configuration and OT numerical audit at 350k

The saved configurations for all eight runs confirm that checked v2 uses
un-normalized squared action costs with Sinkhorn epsilon .25 and 100 iterations.
The reference uses mean-normalized costs with epsilon .05 and 30 iterations.
Their epsilon values therefore cannot be compared without accounting for cost
scale. These settings were already active at launch; this audit changes no
training parameter. The Korean current-algorithm reference documents these
settings and distinguishes retained legacy fields from active computations.
The actor, critic, density, distillation and acceptance implementation files
still match the launch commit `8cb4f23`.

For 11 logged diagnostic batches per v2 seed from 300k through 350k, mean row
marginal errors were 5.9e-6--9.9e-6 and mean column marginal errors about 4.2e-9.
All inspected values were finite. The largest **logged batch-mean** row error
was 2.2e-5; this does not bound every individual state or every unlogged update.
Mean squared action costs were about 9.0. These observations give no indication
of a gross mass-conservation failure in the inspected OT diagnostics, but do
not certify OT optimality, NLL distribution fidelity or soft improvement.

The hypothetical hard-argmax projection TV diagnostic averaged .056--.083.
The active loss uses full-row NLL, so this diagnostic is not its actual target
mass error. Evidence: `outputs/v2_improvement/confirmation_algorithm_semantics_audit.json`
and `confirmation_ot_numerics_0300000_0350000.json`.

### Paired confirmation review: 400k

The common 390k/395k/400k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | v2 / OptiQ |
|---|---:|---:|---:|
| 0 | 5080.31 | 4516.47 | 1.12 |
| 1 | 4823.84 | 1614.96 | 2.99 |
| 2 | 4973.05 | 2136.66 | 2.33 |
| 3 | 5267.71 | 3728.41 | 1.41 |

The four-seed means are 5036.23 versus 2999.12 (+67.9%), with seed SDs 186.75
and 1353.26. All pairs are positive at this checkpoint. This is not an
uninterrupted lead: reference seed 0 briefly exceeded v2 at 350k--360k. The
candidate subsequently recovered. Current v2 drawdowns from rolling-mean peaks
are 0.9%, 7.2%, 1.5% and 0%; the largest historical drops since 100k remain
26.1%, 23.7%, 25.2% and 33.0%. These observations support continuing the runs,
while leaving the final 900k--1M comparison and later stability unresolved.

At exactly 400k, v2 ESS is 2.45--3.10/64, cumulative acceptance 56.9--58.3%,
and the pre-discount entropy term -0.105 to +0.086. A negative differential
entropy is possible and this magnitude is much smaller than the original
v2's approximately -5 entropy contribution. The mean evaluation episode lengths
are 957 versus 590 steps. Observed time to 400k is about 123.4 versus 82.4
minutes under paired GPU sharing; this is not an isolated throughput benchmark.

All sixteen 400k actor/critic checkpoints are present and nonempty. Continue
all eight runs without a training change, review again at 500k, and inspect
sooner if sustained deterioration or a runtime failure occurs. Evidence:
`outputs/v2_improvement/confirmation_review_0400000.json` and
`confirmation_report/step_0400000.*`. No push occurred.

### Survival and locomotion check at 400k

The independent CPU evaluator now records environment-reported forward reward,
alive reward, negative control cost and x velocity. Per-model velocity is pooled
over recorded environment steps, avoiding extra weight for short episodes.
Episode records retain component sums and counts. Tests verify terminal-step
inclusion and step-weighted pooling; no training code or random stream changes.

All eight 400k checkpoints were evaluated on ten fresh episodes each, using seed
base 1010000. This is a descriptive reward-composition check, not a replacement
for the predefined final window or the larger fixed-checkpoint evaluation.

| Training seed | v2 return | OptiQ return | v2 x velocity | OptiQ x velocity |
|---|---:|---:|---:|---:|
| 0 | 4992.9 | 4739.4 | 0.407 | 0.132 |
| 1 | 5195.0 | 1864.7 | 0.262 | 0.088 |
| 2 | 4777.7 | 2983.9 | 0.369 | 0.035 |
| 3 | 4929.5 | 4371.2 | 0.343 | 0.176 |

Velocity is in m/s. All v2 checkpoints move forward faster in this sample;
longer survival is also a substantial source of the return advantage, especially
for seeds 1 and 2. Both methods receive the same alive reward of 5 per step.
v2's mean control cost is about .132--.140 per step, versus .084--.102 for
OptiQ. Thus the gain is not a reduction of action cost or an altered reward.
For all 80 episodes, the reported reward components reconstruct total return
within 1e-7, and forward reward equals 1.25 times x velocity. These checks use
the installed Humanoid-v4 implementation and actual environment outputs.

Artifacts: `outputs/v2_improvement/confirmation_independent_0400000.json`,
its `_validation.json` companion, and `confirmation_evaluation_metrics_tests.log`.

### Transient seed-0 event at 430k

v2 seed 0's single ten-episode evaluation mean fell from 4180 at 425k to 2389
at 430k, then rose to 5055 at 435k. The three-checkpoint mean recovered to 5005
at 445k. The logged Q, ESS and entropy term show no comparably large excursion,
but do not identify the cause or rule out policy/critic errors. Different
checkpoints use different policies and episodes. The ordinary 50k checkpoint
series does not retain the 430k actor for independent re-evaluation. Preserve
this event as evidence against a monotonic-stability claim and monitor recurrence;
do not tune or restart the promising confirmation runs around it.

Evidence: `outputs/v2_improvement/confirmation_seed0_event_0430000.json`.

### Paired confirmation review: 500k

The common 490k/495k/500k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | v2 / OptiQ |
|---|---:|---:|---:|
| 0 | 5266.82 | 4881.67 | 1.079 |
| 1 | 4701.59 | 4440.52 | 1.059 |
| 2 | 5249.42 | 4413.64 | 1.189 |
| 3 | 5338.38 | 5099.55 | 1.047 |

The four-seed means are 5139.05 versus 4708.85 (+9.1%), with seed SDs 294.17
and 337.47. All pairs are positive, but the reference has substantially closed
the gap seen at 400k. Current v2 drawdowns from rolling-mean peaks are 1.9%,
10.5%, .4% and 1.4%. Its largest historical rolling drawdowns since 100k remain
26.1%, 23.7%, 25.2% and 33.0%, each with an observed recovery. Single-checkpoint
dips such as the 430k event remain relevant and should not be hidden by averaging.

Normalized area under the evaluation curve from the first evaluation through
500k is 3016.83 for v2 versus 1769.25 for OptiQ, with seed SDs 168.87 and 539.52.
This supports better sample efficiency over the observed horizon; it does not
replace the fixed 900k--1M primary comparison. Both panels of the preserved
figure stop at the common 500k horizon, and the shaded bands are seed SDs.

At 500k, v2 ESS is 2.76--4.48/64, cumulative acceptance 57.3--58.6%, and the
pre-discount entropy term +.070--+.172. Logged actor/critic losses and current/
next Q values in 400k--500k are finite for all eight runs. Mean evaluation
episode length is approximately 963 versus 930 steps. Observed time to 500k is
157.4 versus 106.8 minutes under paired GPU sharing, not an isolated speed test.

All sixteen 500k actor/critic checkpoints are present and nonempty. Continue
all eight runs with the frozen training code and configuration; next major
review is 600k, sooner if a failure or sustained decline occurs. Final-window
results remain unavailable, so the full objective is not yet achieved.
Evidence: `outputs/v2_improvement/confirmation_review_0500000.json`,
`confirmation_assessment/step_0500000.json` and
`confirmation_report/step_0500000.*`. No push occurred.

### Finite-M entropy check on the 500k actors

`scripts/diagnose_v2_entropy.py` restored each frozen v2 actor on CPU, collected
four fresh stochastic trajectories per actor, and selected 64 evenly spaced
visited states from each pooled trajectory sequence. It used 128 independent
action/component draws per state with the common M=16 entropy-bracket function,
32,768 lower/upper pairs in total. Time-limit and true-terminal trajectories
are both retained for this state-sampling diagnostic.

| Seed | Mean lower entropy | Mean upper-minus-lower | Conditional MC SE of gap |
|---|---:|---:|---:|
| 0 | 3.26065 | 0.00001583 | 0.00001753 |
| 1 | 4.18770 | 0.00002037 | 0.00001716 |
| 2 | 3.48131 | 0.00005056 | 0.00001625 |
| 3 | 3.34978 | 0.00001237 | 0.00001734 |

Multiplying these mean gaps by T=.1 gives 1.2e-6--5.1e-6 per step. The selected
states show no indication of a large finite-M entropy gap. Several point
estimates are comparable to their Monte Carlo error, so their precise small
values should not be overinterpreted. The largest observed state-mean gap is
about .00051; this is not a bound on other states or on the true expectation.
These on-policy entropy values should not be directly equated with replay-batch
training metrics, which use a different state distribution.

This diagnostic does not furnish the uniform epsilon_H required by the
theoretical error bound and does not certify soft policy improvement. The
source/checkpoint hashes, seeds, selected state indices and per-state results
are preserved in `outputs/v2_improvement/entropy_diagnostic_0500000.json`;
its `_validation.json` companion checks all four models and state averaging.
No training parameter or running model was changed.

### Paired confirmation review: 600k

The common 590k/595k/600k evaluations give:

| Training seed | Checked K64 v2 | OptiQ reference | v2 / OptiQ |
|---|---:|---:|---:|
| 0 | 5282.82 | 5115.14 | 1.033 |
| 1 | 5263.29 | 4712.82 | 1.117 |
| 2 | 4772.39 | 4919.84 | 0.970 |
| 3 | 5377.46 | 4959.57 | 1.084 |

The four-seed means are 5173.99 versus 4926.84 (+5.0%), with seed SDs 272.34
and 165.71. Three pairs favor v2; seed 2 is 3.0% behind. The narrower overall
gap and the negative seed-2 pair reinforce the need for the final fixed window.
Current v2 drawdowns from rolling-mean peaks are 1.6%, 0%, 9.9% and .7%.
Its largest rolling drawdowns since 100k remain unchanged and have observed
recoveries; this does not exclude single-checkpoint dips or prove monotonicity.

Normalized evaluation AUC through 600k is 3365.19 versus 2254.81 (+49.2%),
with seed SDs 145.05 and 500.65. The earlier sample-efficiency advantage remains
larger than the current return gap. Mean episode length in the recent three
evaluations is 969 versus 975 steps: the current aggregate return advantage is
associated with higher reward per environment step, rather than longer average
survival in this particular window. Aggregate return per step is approximately
5.337 versus 5.054; these are descriptive ratios, not additional success labels.

At 600k, v2 ESS is 3.96--4.79/64, cumulative acceptance 57.6--59.1%, and the
pre-discount entropy term +.089--+.212. Logged actor/critic losses and current/
next Q values in 500k--600k are finite for all eight runs. All sixteen 600k
actor/critic checkpoints are present and nonempty. Observed time to 600k is
191.3 versus 132.4 minutes under paired GPU sharing.

Continue all eight runs with the frozen training code/configuration, reviewing
again at 700k or sooner for a sustained decline or runtime failure. The primary
900k--1M window is still unavailable; final superiority and stability remain
unverified. Evidence: `outputs/v2_improvement/confirmation_review_0600000.json`,
`confirmation_assessment/step_0600000.json` and
`confirmation_report/step_0600000.*`. No push occurred.

### Historical 10% exploration reference: retain the stronger result

The current primary comparison deliberately uses zero extra uniform behavior
for both methods. The earlier completed OptiQ runs with 10% uniform behavior
are a distinct, stronger historical reference and must also appear in the final
report. Their local configurations, all 21 evaluation checkpoints from 900k
through 1M, and completion markers were verified for seeds 0--3.

| Seed | Historical OptiQ: 900k--1M mean | Historical OptiQ: 1M checkpoint |
|---|---:|---:|
| 0 | 5724.90 | 5762.03 |
| 1 | 5590.78 | 5659.10 |
| 2 | 5790.25 | 5898.97 |
| 3 | 5600.04 | 5653.96 |

The historical final-window mean is **5676.49 with seed SD 97.43**. At the common
590k/595k/600k window it averages 5310.65, versus the current v2's 5173.99 and
the current zero-uniform reference's 4926.84. Thus current v2 has not exceeded
the historical result at that matched training horizon. An advantage over the
zero-uniform control alone does not establish superiority over the earlier
10% exploration setting.

The saved algorithm configurations differ between the historical and current
OptiQ controls only in `behavior_uniform_probability` (.1 versus 0). The checked
evaluation/task protocols match, but the recorded source revisions and actual
collection runs differ. This retrospective comparison cannot causally attribute
the score difference solely to uniform exploration. The primary zero-uniform
confirmation protocol remains unchanged; the historical result is an additional
required reference for interpreting the broader performance claim. No extra
uniform behavior is added to v2.

Evidence: `outputs/v2_improvement/historical_behavior010_reference.json`, including
evaluation-file hashes and exact configuration differences. Project:
`OptiQ/optiq_mujoco_scalar_h256x3_anchor_gradnorm2_behavior010_4seed_1m`.

### Historical 1M checkpoints on the final independent episode seeds

The secondary final-checkpoint protocol is now explicit: 50 stochastic episodes
per model, environment seeds 1100000--1100049 and policy seeds 1110000--1110049.
All time-limit and true-terminal episodes count. These are evaluation seeds;
the four training seeds remain 0--3. The primary 900k--1M window is unchanged.

`scripts/evaluate_v2_historical.py` has completed this CPU evaluation for the
four historical 1M actors using the shared predict/unscale and episode evaluator.
The current confirmation models will use the same seeds after reaching 1M:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /workspace/.venv-optiq-mujoco/bin/python scripts/evaluate_v2_confirmation.py \
  --step 1000000 --episodes 50 --seed-base 1100000
```

| Historical training seed | New-episode mean return | Episode SD | Mean length | Time-limit episodes / 50 |
|---|---:|---:|---:|---:|
| 0 | 5304.03 | 1515.11 | 865.46 | 37 |
| 1 | 5670.61 | 48.64 | 1000.00 | 50 |
| 2 | 5901.61 | 17.85 | 1000.00 | 50 |
| 3 | 5656.48 | 16.11 | 1000.00 | 50 |

The mean across four model scores is 5633.18, with training-seed SD 246.53.
The seed-0 actor moves faster (step-weighted mean x velocity .988 m/s) but
terminates before the horizon on 13 of these 50 episodes. The other actors'
velocities are .603, .790 and .587 m/s, and all their episodes reach the horizon.
This is a frozen-1M-model comparison on new episodes, distinct from the
historical 900k--1M score averaged over changing policies.

All 200 episode reward sums match the recorded reward components within 1e-7;
all four actor checkpoints loaded successfully. Hashes and individual episodes
are preserved in `outputs/v2_improvement/historical_behavior010_independent_1000000.json`
and its `_validation.json` companion. Evaluation seeds and the common evaluator
hash are fixed in `confirmation_final_independent_protocol.json`. No training
run was changed and no push occurred.

### Paired confirmation review: 700k

The common 690k/695k/700k evaluations give:

| Training seed | Checked K64 v2 | Current OptiQ, behavior 0 | Historical OptiQ, behavior .1 |
|---|---:|---:|---:|
| 0 | 5387.37 | 5094.55 | 5715.31 |
| 1 | 5147.60 | 4971.31 | 5154.76 |
| 2 | 4936.84 | 4667.26 | 5127.33 |
| 3 | 5437.64 | 4605.71 | 5522.40 |

The current four-seed means are 5227.36 versus 4834.71 (+8.1%), with seed SDs
231.36 and 235.70. All four matched zero-uniform pairs favor v2. However, the
historical 10% behavior reference averages 5379.95 (seed SD 287.11) at these
same steps, placing current v2 2.8% below that stronger historical setting.
The earlier reference is retained as a separate comparison; no causal effect
of exploration alone is inferred from runs with different recorded source
revisions and collection histories. General superiority over the prior stronger
OptiQ result has not been established.

Current v2 drawdowns from rolling-mean peaks are .8%, 2.4%, 6.8% and 1.7%.
There is no early-stop condition. At 700k, ESS is 4.22--5.90/64, cumulative
acceptance 57.8--59.4%, and the pre-discount entropy term +.191--+.334. All
inspected actor/critic losses and current/next Q values in 600k--700k are finite.
Mean evaluation episode lengths are 974 versus 942 steps for the current
paired methods. Observed time to 700k is 225.6 versus 158.4 minutes under paired
GPU sharing, not an isolated speed benchmark.

All sixteen 700k actor/critic checkpoints are present and nonempty. Continue
the frozen confirmation through the primary 900k--1M window; review again at
800k or sooner if a sustained decline or runtime failure warrants it. Final
performance and stability remain unverified. Evidence:
`outputs/v2_improvement/confirmation_review_0700000.json`,
`confirmation_assessment/step_0700000.json` and
`confirmation_report/step_0700000.*`. No push occurred.

Both reporting scripts accept `--through-step 700000` to preserve the same
evaluation horizon while live runs continue. A CLI regression verifies that
future checkpoints cannot enter a frozen report or make its incomplete final
window appear complete. The assessment/alignment checks pass (five tests);
see `outputs/v2_improvement/confirmation_assessment_cutoff_tests.log`.

### Matched reference completion and historical comparison plot

All four current zero-uniform OptiQ references completed 1M normally. Each has
an EXITED supervisor state, a completion marker, all 21 predefined 900k–1M
evaluation checkpoints, and nonempty final actor/critic checkpoints.

| Training seed | OptiQ 900k–1M mean | OptiQ single 1M evaluation |
|---|---:|---:|
| 0 | 4939.75 | 5293.89 |
| 1 | 5107.03 | 5175.79 |
| 2 | 5045.76 | 4866.23 |
| 3 | 5114.08 | 5437.90 |

The reference's primary-window mean is 5051.65, with seed SD 80.67. These are
completed reference results, not a final v2 comparison: v2 is still running
around 760k. The seed-1 single-checkpoint return fell to 3478.87 at 750k
(mean episode length 655), then recovered to 5272.09 at 755k and 5240.39 at
760k (both mean lengths 1000). Continue the frozen runs; this observed recovery
does not establish the cause of the dip or monotonic stability.

`scripts/compare_v2_historical.py --through-step 700000` now plots the current
paired methods and the stronger historical 10% behavior reference together.
All twelve runs use the same intersected evaluation horizon; individual seeds
remain visible and historical evaluation file hashes are checked. The 700k
scores reproduce the existing current and historical reviews to 1e-9. The
plot uses seed SD, not a confidence interval, and labels historical provenance
differences. No final-window score is available until all required checkpoints
exist within the selected horizon.

Evidence: `outputs/v2_improvement/confirmation_all_reference_completions.json`,
`confirmation_historical_comparison/step_0700000.*`, and
`confirmation_historical_comparison_validation.json`. Training code/configs
remain unchanged from the confirmation launch. No push occurred.

### Seed-1 750k dip: independent checkpoint reevaluation

The saved 700k and 750k v2 actors were evaluated on 50 shared new episodes
each, with environment seeds 1200000–1200049 and policy seeds
1210000–1210049. This is a post-hoc diagnostic of one training seed, separate
from the final comparison protocol. CPU evaluation uses the unchanged common
stochastic prediction/unscale helper; all episodes, including time limits, count.

| Frozen checkpoint | Original 10-episode return | New 50-episode return | Episode SD | Mean length | Time limits / 50 |
|---|---:|---:|---:|---:|---:|
| 700k | 5282.29 | 5280.62 | 29.69 | 1000.0 | 50 |
| 750k | 3478.87 | 4350.38 | 1459.81 | 817.5 | 29 |

The paired return difference is −930.23, with a conditional episode-bootstrap
95% interval of [−1351.02, −547.45]. This interval concerns these two fixed
models and shared test episodes, not uncertainty across training seeds. The
original ten-episode score was lower than the fresh checkpoint score, but these
measurements differ in both episodes and potentially one actor update (see the
timing audit below); the entire difference cannot be assigned to sampling noise.
A performance loss and higher early-termination frequency remain in the fresh
comparisons of the saved actors. Later training evaluations recovered;
these results do not identify the learning mechanism causing the temporary dip.

Step-pooled forward velocity increases from .323 to .362 m/s. Net reward per
recorded environment step also rises slightly, while mean episode length falls.
This associates the return loss with reduced survival, not simply slower
locomotion. Exact-step sampled diagnostics at 750k show Q 451.22, ESS 5.55/64,
policy standard deviation .900, and finite actor/critic losses, with no comparably
large change from nearby logged batches. Those sparse diagnostics cannot rule
out localized Q errors or identify a causal mechanism. The raw-return change
also is not a direct test of discounted entropy-regularized policy improvement.

Reproduce with `scripts/evaluate_v2_checkpoint_event.py --seed 1 --steps
700000 750000 --episodes 50 --seed-base 1200000 --output <new-output.json>`.
The actor checkpoint hashes, all 100 episodes, evaluator source hashes, and
reward-component reconstruction validation are retained in
`outputs/v2_improvement/confirmation_seed1_event_0750000_independent.json`
and its `_validation.json` companion; logged metrics are in
`confirmation_seed1_event_0750000_diagnostics.json`. The maximum reward
reconstruction error is 1.1e-11. Training code and settings were not changed.

### Paired confirmation review: 800k

The common 790k/795k/800k evaluation averages are:

| Training seed | Checked K64 v2 | Current OptiQ, behavior 0 | Historical OptiQ, behavior .1 |
|---|---:|---:|---:|
| 0 | 5362.30 | 4807.19 | 5710.21 |
| 1 | 4845.16 | 5109.82 | 4824.84 |
| 2 | 4944.72 | 3857.76 | 5651.12 |
| 3 | 5478.76 | 4857.48 | 5522.30 |
| Mean ± seed SD | 5157.74 ± 309.83 | 4658.06 ± 549.72 | 5427.12 ± 409.11 |

Three current matched pairs favor v2; seed 1 trails. The aggregate is 10.7%
above the current control but 5.0% below the stronger historical setting.
Normalized return AUC through the same 800k horizon is 3833.24 for v2,
2921.45 for the current control, and 3891.91 for the historical reference.
The latter has different collection/source provenance; these comparisons
do not isolate the causal effect of additional uniform collection.

At 800k, v2 ESS is 5.39–6.49/64, cumulative sampled-guard acceptance is
57.9–59.7%, and the pre-discount entropy term is +.115 to +.394. All inspected
700k–800k actor/critic losses and Q values are finite. All sixteen actor/critic
800k checkpoints are present and nonempty. Current three-checkpoint drawdowns
are 1.6%, 8.1%, 8.4%, and 1.0%. These aggregate diagnostics do not negate the
seed-1 checkpoint instability identified below. The matched controls have
completed normally; all four v2 runs continue unchanged toward the predefined
900k–1M window, with the next ordinary review at 900k. No early-stop rule is
currently met, and no final performance/stability claim is warranted.

Evidence: `outputs/v2_improvement/confirmation_review_0800000.json`,
`confirmation_assessment/step_0800000.json`, `confirmation_report/step_0800000.*`,
and `confirmation_historical_comparison/step_0800000.*`. The three reporters
agree on the same cutoff and paired scores. Core training/config files remain
identical to the confirmation launch; no push occurred.

### Seed-1 800k recheck: instability remains on fresh episodes

The saved 800k actor was evaluated on the same 50 new environment/policy seed
pairs used for the 700k/750k diagnostic. Its mean return is 3387.73 (episode SD
1547.41), mean episode length 639.04, with only 13/50 episodes reaching the
time limit. This is below the 4192.88 return from the original ten episodes.
The shared-episode mean differences are −1892.88 versus the 700k model and
−962.65 versus the 750k model. Their conditional episode-bootstrap 95% intervals
are [−2314.88, −1473.06] and [−1539.23, −345.65]. As above, these are post-hoc
comparisons of fixed models from a single training seed, not training-seed
confidence intervals or causal explanations.

The 755k/760k training evaluations recovered after the 750k dip, but this
later checkpoint is unstable again. The completed fresh-episode checks prevent
interpreting the recent smoothed scores as proof that stability is solved.
Step-pooled velocity is .345 m/s; early termination remains the conspicuous
performance difference. The mechanism is unproven, and lower raw return alone
does not disprove the ideal discounted soft-policy-improvement theorem.

All fifty new episodes are included. Source/helper and checkpoint hashes match
the recorded protocol; reward-component reconstruction error is below 1.2e-11.
Evidence: `outputs/v2_improvement/confirmation_seed1_recheck_0800000_independent.json`
and its `_validation.json` companion. Training remains unchanged pending the
frozen full confirmation result, as requested.

### Seed-1 policy ranking under the frozen 700k and 800k critics

The two saved actors generated four new trajectories each. Thirty-two evenly
spaced pre-action states per trajectory were retained, including states near
early termination: 128 states per policy, 256 total. These are recent visited
states, not the fixed landscape-probe replay batch saved at the first 50k
checkpoint. Trajectory seeds are 1300000–1300003 and policy seeds
1310000–1310003. The 700k actor reaches 1000 steps on all four trajectories;
the 800k actor terminates at 327, 150 and 944 steps on three trajectories and
reaches 1000 on the fourth.

For each state bank, both critics score the same old/new actor pair with 256
shared random action draws per state, M=16 entropy estimation and T=.1. The
score is the same minimum-twin-Q plus entropy quantity used by the sampled
guard: new-policy lower minus old-policy upper. It is a learned-Q score,
not an estimate of actual full-episode return.

| States visited by | Frozen critic | Mean new-minus-old soft score | Conditional MC SE | States with negative estimated mean |
|---|---|---:|---:|---:|
| 700k actor | 700k | −.16871 | .00226 | 79.7% |
| 700k actor | 800k | +.08605 | .00201 | 31.2% |
| 800k actor | 700k | −.36854 | .00799 | 74.2% |
| 800k actor | 800k | −.07716 | .00799 | 43.8% |

On exactly the same old-policy states, changing the critic reverses the
average preference. Under the newer critic, the mean score is positive on
old-policy states but negative on new-policy states. Thus a favorable average
can depend on both critic revision and which states are evaluated; individual
state losses can coexist with a positive average. This is consistent with
critic-ranking changes and state-coverage limitations contributing to the
observed instability, but does not establish causality or prove Q overestimation.
The comparison spans 100k training steps, not one accepted update; the state
banks are not the actual validation batches used during those updates. MC
errors are conditional on these fixed states/models and are not uniform error
bounds or training-seed confidence intervals.

All four comparisons completed. Checkpoint/script/state hashes, per-trajectory
state selection, finite scores, and score reductions were verified. Reproduce
with `scripts/diagnose_v2_checkpoint_ranking.py --seed 1 --old-step 700000
--new-step 800000 --seed-base 1300000 --output <new-output.json>`. Evidence:
`outputs/v2_improvement/confirmation_seed1_ranking_0700000_0800000.json`,
its `_states.npz` and `_validation.json` companions. No gradient, parameter,
training process, or experiment configuration was changed by this diagnostic.

### Automatic final processing of the frozen confirmation

Supervisor service `optiq-v2-finisher` waits for all eight prescribed runs to
exit normally, preserving the four already-completed controls. It requires
completion markers, every 900k–1M evaluation checkpoint, and nonempty final
actor/critic files. A final file written by a still-running process does not
trigger evaluation. Missing seeds, stopped runs, and incomplete exits prevent
automatic final processing; the service never starts/stops or restarts training.

After completion it runs all three existing reporters at the fixed 1M cutoff,
then the common CPU stochastic evaluator for all eight final actors: 50 new
episodes per actor, environment seeds 1100000–1100049 and policy seeds
1110000–1110049. These are the previously recorded comparison seeds, also used
by the completed historical evaluation. Core source compatibility and the
common evaluator hash are checked again after waiting. Existing complete
evaluations are verified and reused; partial results are preserved for manual
review rather than overwritten. Do not launch a second final evaluator while
this service is active.

Final window results for all three methods and the matched/historical fresh
episode scores are collected in
`outputs/v2_improvement/confirmation_final_review_inputs.json`. Completing this
pipeline only makes results ready for review; it cannot declare the goal met,
empirical superiority, stability, or a policy-improvement guarantee.

Implementation: `scripts/finish_v2_confirmation.py` and
`scripts/supervisor_v2_finisher.sh`; installed supervisor configuration:
`/etc/supervisor/conf.d/optiq-v2-finisher.conf`. Readiness was checked on the
live runs before activation. The finisher and existing fixed-window checks
pass (five tests), including the still-running-final-files case and missing
checkpoint/seed cases. Logs: `/var/log/portal/optiq-v2-finisher.log` and
`outputs/v2_improvement/confirmation_finisher_tests.log`. The service uses CPU
and has autostart/autorestart disabled. No push occurred.

### Paired confirmation review: 900k

The common 890k/895k/900k evaluation averages are:

| Training seed | Checked K64 v2 | Current OptiQ, behavior 0 | Historical OptiQ, behavior .1 |
|---|---:|---:|---:|
| 0 | 5323.82 | 5196.18 | 5742.74 |
| 1 | 5285.79 | 5077.00 | 5513.82 |
| 2 | 5231.45 | 4947.84 | 5770.60 |
| 3 | 5570.97 | 4729.63 | 5560.22 |
| Mean ± seed SD | 5353.01 ± 150.17 | 4987.66 ± 199.69 | 5646.84 ± 128.73 |

All four matched current pairs favor v2. The mean is 7.3% above that control
but 5.2% below the stronger historical reference; only seed 3 slightly exceeds
its historical counterpart in this window. Normalized return AUC through the
same 900k horizon is 3997.97 for v2, 3136.53 for the matched control, and
4077.51 for the historical reference. Historical provenance/collection
differences remain a limitation; the comparison is not an isolated exploration
ablation. The full predefined 900k–1M window is still unavailable.

At 900k, ESS is 5.24–7.19/64, cumulative acceptance 58.0–59.9%, and the
pre-discount entropy term +.224 to +.400. All inspected 800k–900k actor/critic
losses and Q values are finite. All sixteen actor/critic checkpoints are
present and nonempty. Current rolling drawdowns are 2.3%, .5%, 3.1%, and .1%;
this recent recovery does not erase the independently confirmed seed-1
instability at earlier checkpoints. The sampled diagnostics do not provide a
statewise improvement certificate.

Continue the four unchanged v2 runs for the remaining final window; the four
completed controls remain in the comparison. No early-stop criterion is met.
The already-running `optiq-v2-finisher` owns final processing and independent
evaluation, so a duplicate evaluator must not be launched. Evidence:
`outputs/v2_improvement/confirmation_review_0900000.json`,
`confirmation_assessment/step_0900000.json`, `confirmation_report/step_0900000.*`,
and `confirmation_historical_comparison/step_0900000.*`. All three reporters
agree on the fixed cutoff and current paired scores. Core training/config
source remains unchanged from launch. No push occurred.

### Evaluation/checkpoint timing audit before final interpretation

The inherited SB3 loop calls the evaluation callback during rollout collection,
after incrementing `num_timesteps` but before the following training update.
`OptiQDIME.train` saves actor/critic TrainStates after that update using the
same step label. All eight confirmation configs use UTD=1 and policy delay=1;
the common runner retains DIME's train frequency of one environment step.
Thus an original 10-episode evaluation at step N and the saved N checkpoint
are not necessarily the exact same actor. Fresh-checkpoint evaluation remains
a distinct secondary protocol, as planned.

For the seed-1 diagnostic checkpoints, exact-step logs show one accepted actor
update at 700k and 750k. Their saved policies may differ from the policies in
the original ten-episode evaluations. At 800k the proposal was rejected and
the actor TrainState rolled back, so that update did not change its inference
parameters. The independent 700k/750k/800k comparisons are still valid
comparisons of the explicitly hashed saved actors. However, the difference
between an original score and its fresh-checkpoint score must not generally
be attributed entirely to evaluation sampling; the earlier 750k wording has
been qualified accordingly.

Source order, source hashes, all eight update settings, and the three logged
acceptance decisions are recorded in
`outputs/v2_improvement/confirmation_evaluation_checkpoint_timing.json`.
The predeclared final window, checkpoint selection, and independent episode
seeds are unchanged. All sixteen 950k actor/critic checkpoints were verified
and recorded in `confirmation_checkpoints_0950000.json`. Training continues;
the final evaluation service remains responsible for the full 1M comparison.

### Full confirmation and independent evaluation completed

All eight training runs and the final-processing service exited normally.
All 21 primary-window checkpoints and sixteen final actor/critic files were
verified. The primary 900k–1M means are 5362.67 for v2, 5051.65 for the matched
zero-uniform control, and 5676.49 for the historical 10% behavior reference.
Fresh 50-episode final-model means are 5397.09, 5224.07 and 5633.18, respectively.
All 200 v2 final-model episodes reached the 1000-step horizon, but this does
not erase the earlier independently verified checkpoint instability.

The full result, uncertainty scope, timing caveat, stability comparisons and
theory limitations are in [the Korean final report](V2_CONFIRMATION_RESULTS_KO.md).
All 600 current/historical evaluation episodes, model hashes, shared seeds,
reported means, and reward-component reconstructions were checked in
`outputs/v2_improvement/confirmation_final_independent_validation.json`.
The full training audit is in `confirmation_final_training_audit.json`.

This candidate's confirmation is finished, but the overall goal is not met:
the stronger historical reference remains ahead, and finite sampled checks
do not establish true statewise improvement. Further changes must retain
those unresolved requirements. No new training variant has been launched yet;
the next read-only diagnostic examines the final actors' use of latent z
relative to conditional Gaussian noise. No push occurred.
