# Approved six-target Forward/Reverse KL confirmation, 2026-09-25

## Authorization
The user reviewed the six proposed cases and explicitly approved: "다 좋아보여.
다 진행시켜." Thus L=1048576 confirmation is now authorized for ALL six cases:
t00_reference,n00_spike_ramp,n07_spike_flat_ramp,t01_two_offset,t05_unequal_mass,
t06_minor_mode. This package replaces the previous review gate only for these
cases. The cancelled image-shaped GMM campaign stays cancelled.

## Fixed numerical experiment
Keep N=M128,batch32,temperature.25,100Kupdates,seeds0–3,Adam3e-4,256×256GELU,
mean-head initializer scale3,logsigma[-5,-1],initial-1,action[-10,10]. Copy the
EXACT approved target formulas, shape parameters, masses and mode/core definitions
from their screening configs. Four Gaussian-mixture targets and two genuine
non-Gaussian targets. Log density and score functions are borrowed unchanged;
only the screening class's small-L guard is replaced in a separate adapter.

Reverse is trained FROM THE ORIGINAL RANDOM INITIALIZATION, not resumed from
L1024. Same seed and initial parameter hash as its paired forward run. Its density
bankL1048576 is independently resampled perupdate/group. Stream chunks4096 with
original max-scaled sums and score estimator. No bank reuse, projection, smaller
latent subset, shared bank across32groups, or changed gradient/objective. The
existing high-L training kernel is inherited verbatim. Process5updates/block,
checkpoint every100updates (actor,Adam,RNG). Resume retains sourcecommit and logs
an event. Preemption/timeout can be resumed from that full-state checkpoint.

Forward has no auxiliaryLbank. All24approved forward actors already completed
100K at the identical setting. Verify numerical config and initialparameter hash,
reuse the full final checkpoint, and reevaluate at1Msamples. Record original
training sourcecommit,runpath,checkpointhash separately from new evaluationcommit.
Do not retrain the same forward seeds or pretend that a reused model was trained
under the new sourcecommit. Reverse is rerun for all24cases/seeds including the
reference, making this campaign self-contained while retaining the prior reference
confirmation separately.

## Evaluation
Same target-recovery criteria from screening; no threshold tuning after high-L
results. Intermediate0,1K,10K,25K,50K,75K use32768actualactions. Final100K both
methods use1048576actualactions,512bins,noKDE. For memory control final samples
are64independent chunks of16384, seeds197+104729*i,identical across methods/seeds.
Evaluation does not advance the trainingRNG. Save actions,componentmeans/scales,
histogram masses,target reference,core/basin/valley masses. Compute1DWasserstein
as integral|F_emp-F_target| on131073gridpoints, check against65537gridpoints and
record absolute difference; require<2e-4. GaussianCDFanalytic; non-GaussianCDF
uses previously verified float64Simpson/quad. Final plots must show per-seed
results as well as means, and report mean-of-seedTV separately fromTVofmean.
The L1024outcomes were selected illustrative cases; all high-L outcomes must be
retained and reported, including if a gap disappears. No universal KL claim.

## Verification and execution
CPU validation: exact parameter initialization and gradient parity with BOTH
screening implementations atL1024 for ALL6targets; resume/RNG. GPU validation
on each submitted hardware pool: actualN128,M128,batch32,L1048576 updates in GMM
and non-GMM cases; finite metrics, speed,1Msample evaluation and untouchedRNG.
Forward reuse additionally verifies config/initialhash at runtime. Never train
onlogin node. Commit source/config/protocol/launch before submission toheejoon.
Record manifest,fullSHA,approval and jobindices. Submit each case/seed independently,
using disjoint index subsets for hardware pools (no duplicate training). OneGPU,
2CPU,24GBRAM perrun; ordinarybig_qos and availablePRO6000/A100 pools subject to
accountlimits. 48hwalltime. Full checkpoints and outputs backed up todildata via
the already approved read-only SSH route every180s. Backup stops after all24forward
reevaluations plus24reverse runs complete. No credentials enter source or snapshots.

## Reporting after completion
Per-case2panels (leftForward,rightReverse), true target dashed, learned sample
histograms lightlyfilled,4seedaverage plus separateseedpanels. Table:TV,W1,mode
coverage/masses,referencevsactorbackup if laterrequested; report actual uncertainty.
Keep low-L screening and high-L confirmation labels distinct. Current request is
to launch and verify early progress; final training need not finish this turn.
