# Two-target high-L paper figures and frozen-actor score convergence

User request 2026-09-26: replace the L1024 illustration with the completed
L1048576 experiments, add score-convergence plots, and document exact settings.
Only the three-Gaussian target (t00_reference) and spike+ramp (n00_spike_ramp)
are used. No new training or target selection is performed.

## Density figures
Read all final100K Forward and Reverse checkpoints and evaluations from
kl_six_highL_20260925/runtime/confirm. All16 runs must have COMPLETE and final
metrics. Forward reused the original100K actor; Reverse was trained afresh at
L1048576 from the paired initial parameters. Preserve both original training
and new evaluation provenance. Each density is the mean of four512-bin action
histograms,1048576actualactions perseed,noKDE. Display one row/fourcolumns in
the user's current sans-serif style: method-only panel titles, noTV, nolegend,
noenvironmentheadings, x=a, Density on the first panel only. Preserve existing
independent vertical scales. Report seed metrics and uncertainty separately.

## Frozen score evaluation
Evaluate all8 Reverse final100K checkpoints (two targets x seeds0–3), never
update parameters or overwrite parent files. Verify checkpoint hashes before
and after evaluation. Use the immutable original source for actor restoration.
The auxiliary bank approximates q_theta(a)=E_z[k_theta(a|z)] where conditionals
are normalized Gaussians truncated to[-10,10], and z~Normal(0,1).

For each frozen action a estimate:
s_L(a)=sum_l k_theta(a|z_l)(mu_l-a)/sigma_l^2 / sum_l k_theta(a|z_l).
This is an action score, not the target score or a parameter gradient.
Kernel normalization depends onmu,sigma but notinterioraction, hence its action
score is (mu-a)/sigma^2. Use float32 actor and latents as trained, highest
matmulprecision, and float64 running-max accumulators for density ratios.
Check dense/autodiff parity including near-boundary kernels, and same-bank
float32-versusfloat64 score at L1048576.

Predetermine actions before bank results:128 uniformlyindexed actualactions
from the saved final1M sample; empirical policy10/50/90%quantiles;9target-region
probes listed inconfig. Actions stay fixed acrossL/repeats. External low-policy-
density probes are reported separately, not averaged into the policy error.

L=2^7...2^24;16independent streams, nested bankprefixes within eachstream.
Fouradditional independent2^24banks form an empirical reference andSE. No
deterministic integration claim or exact infinite-mixture reference. Use
independent evaluation RNG fromconfig. Save allscores,logdensity,ESS.

## Score presentation
Seed0 is prescribed as the main illustration for eachtarget, regardless of
outcome. Allfour seeds enter supplementaryplots and error tables. Three blue
curves in threepanels for10/50/90%actions; noforward score plot. Log2Laxis,
vertical dashed trainingL2^20, dotted independent-reference score. Blue10–90%
MC shading remains but nograyreferenceband and noMC-rangelegend label, peruser
styling requests. Supply zoomed and common-y-scale versions. Reference standard
errors and L2^20 numerical deviations remain explicit inMarkdown. Test only
empirical convergence at these frozen checkpoints, not alltraining iterations.

## Execution
Commit sources/config/protocol/launch toheejoon before newmeasurement. Eight
independent GPUjobs,oneGPU/twoCPU/24GB each,onehourlimit, ordinarybig_qos.
Do not resume the explicitlyheld unrelatedarray. Parent/source/checkpointSHA,
Slurmids,rawscores,figures,andMDbackedup todildata. Rendering is versioned
separately from measurement if written after joblaunch.
