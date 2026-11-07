# Diverse forward/reverse KL illustrative targets, 2026-09-25

## Authorization and approval boundary
Only search at reverse L=1024 and 100,000 updates is authorized now. Present up to
six environments INCLUDING the previously established f05 reference, with actual
histograms and all four seeds, to the user. DO NOT submit L=1048576 training before
subsequent user approval. Code asserts L=1024 for reverse and has no high-L launch
path. No claim of success before measured. Preserve failed/reversed/no-gap cases.
This is a deliberately searched illustrative set, not a random benchmark or proof
that forward KL is always superior. Use "recovers modes" with measured density
errors, not "perfect distribution recovery".

## Common learning algorithm
Inherit the validated kl_mode_missing_search_1d numerical update: TRG semi-implicit
truncated Gaussian actor on [-10,10]; z~N(0,1); 256x256 GELU; mean head variance
initializer scale 3; log sigma bounds [-5,-1], initial -1. Adam3e-4; N=M128;
32 independent groups/update; temperature .25; density chunk256. Paired methods
use identical actor initialization/seed and the SAME allocated GPU sequentially
(forward then reverse). No target component identity or samples enter learning;
only Q(a) or its analytic gradient. Geometry and target samples are evaluation only.

f(a)=sum_k rho_k Normal(a;c_k,h_k^2), Q(a)=.25 log f(a).
Target is f normalized over[-10,10]. Widths and masses can now differ. Masses in
config are whole-Gaussian weights before box conditioning; true basin probabilities
are integrated exactly from this target. Inspect actual peaks, not just number of
components. No teacher temperature or actor architecture tuning per target.

## Candidate grid
config.json is authoritative:16 targets,2–6 modes, varying locations, widths,
weights, symmetry. t00 is the previously used f05. All16 get seeds0,1, both methods,
100K fresh updates;32 paired jobs/64 trajectories. Evaluate0,1K,10K,25K,50K,75K,100K.
100K evaluation uses262144 actual action samples (2^18),512 bins,noKDE; intermediate
checkpoints use32768 samples. This cost-effective screening evaluation is not the
final1M-sample figure. Full state checkpoints permit unchanged resume after preemption.

## Gates defined before launch
Basin boundaries: exact-target density minima between neighboring true peaks,
located by dense-grid bracket plus scalar optimization. Core: each peak +/- its
component std, clipped to basin. A mode is missing only if both core and basin mass
are <25% of target. Recovered peaks: all core and basin masses>=50% of target,
and valley/core mass ratio<=max(.5,2*target valley/core ratio).
Forward screening pass requires recovered peaks AND histogramTV<=.15 in BOTH
seeds0,1. PreferTV<=.10, but report the number, not a "perfect" label. Reverse must
miss>=1 mode in BOTH seeds. Keep all outcomes in report. Validate passed candidates
with held-out seeds2,3 at the SAME100K,L1024. A final candidate must meet the gates
in all4seeds; choose no more than6, preferring different mode counts and target
families over largest gap. If fewer qualify, show fewer and retain no-gap results.
Additional search rounds require a new committed config and preserved round history.

## Validation and execution
Check general-target log density/score/CDF/integral and actual peak count for each
candidate; compare reference-case parameter gradients with prior implementation;
check resume/RNG; GPU parity/timing preflight before screen. OneGPU/twoCPU perpair,
Slurm array parallelism, independent cases, no environment completion dependencies.
No training on login node. Commit source, protocol, configuration, launch scripts
before execution on heejoon. Record fullSHA+sourcehashes+jobIDs. dildata is central
storage; use already authorized read-only backup pathway. High-L approval flag stays
false and no high-L jobs are queued. Stop after candidate report for user review.
