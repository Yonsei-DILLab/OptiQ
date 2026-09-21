# Random-latent OptiQ TRG: all40 modes and >=90% mu-only near

User-authorized parameter/initialization search, not an algorithm change.
Keep box-truncated policy, Direct marginal NLL, proposal weighting, gradient
rules, random Gaussian latent train/eval, target and metric definitions.
No fixed bank, no tanh-squashed policy, no sample filtering or altered coverage.
Keep256x2GELU,batch256,Adam3e-4,T1,beta1.
First screen: seed0,100k each; factorial mean-head variance scale16/256 and
N=M64/256. Common log sigma[-5,-2],initial-2.5,teacher floor.05. These are
GMM40-only profiles; defaults and frozen runs are preserved.

Commit before launch; freeze source; validate unchanged update/sample code
and gradients using existing adapters; two-update GPU preflight. The launcher
refuses occupied GPUs and existing runs. Four supervisor services, one per
idle5090, no automatic learner restart. W&B OptiQ/gmm-trg with distinct names,
full metrics gmm40/* and mu-only gmm40_mu/*, raw samples/checkpoints/audits.
Compare all completed100k results. Do not select on near alone: coverage40 is
required. Promising profiles require fresh seeds and independent evaluation
latent keys before declaring the user goal achieved. If no profile passes,
use evidence to register another committed parameter-only round; do not
reinterpret success as partial coverage or full-policy instead ofmu-only.
Actual successes and failures are both preserved and reported.

User additions: matched 256x3 and 512x3 controls, four profiles each (seed0,
100k). Change only width/depth from the four 256x2 profiles above. Each GPU
waits for its verified 256x2 predecessor to exit, then runs256x3 and512x3;
there is no all-GPU barrier. The queue acquires the existing run-gpu lock,
performs a two-update GPU preflight with the actual hidden dimensions, then
trains once. Failed/incomplete predecessors or jobs block that GPU queue and
write a failure sidecar; no learner restart, overwrite, or early termination.
Queue service/source/config is committed and frozen before registration.
CPU checks validate both architectures and matched settings before queuing;
GPU preflight is deferred until the corresponding GPU is free.

Screen1 completed: init16,N=M256 reached40/40 but mu-only near76.74%;
init256 was worse (64.75%,39/40 withN=M256). The init16,N=M256 actor
sigma mean was.0723, max.1353, vs target normalized sigma.0328.
Follow-up: retain init16,N=M256,teacherfloor.05 and compare256x3/512x3
crossed with log-sigma caps-3/-3.5; initial valuescap-.5 (-3.5/-4).
All other parameters and100k budget unchanged, seed0. This tests whether a
narrower conditional family reduces between-mode mu mass while retaining
coverage; it is a hypothesis, not an established causal conclusion.
GPU0..3 follow their existing512x3 jobs; predecessor must exit and pass the
100k audit. The next queue uses its own committed/frozen source and does not
modify the running architecture controls. Failure blocks its slot, no retries.
No algorithm, latent distribution, target, coverage metric or filtering changes.

LATEST USER CONSTRAINT supersedes the N/M screen: N=M must remain64.
The N=M256 architecture jobs and all four pending narrow256 jobs were
cancelled with frozen logs/configs retained. No256 result counts toward the
current goal even if near>=90 andcoverage40. Continue existing512x3 NM64
jobs. Register the four narrow-sigma profiles above with onlyN/M changed to64.
GPU2/3 start when free after cancellation; GPU0/1 wait for their approved
512x3 NM64 predecessors to finish. No automatic restart of cancelled work.

Passing NM64 seed0:256x3,mean-init16,log-sigma[-5,-3.5],initial-4,
teacherfloor.05,N=M64,batch256,T1,Adam3e-4,100k. Original GPU mu-only
near92.18%,coverage40. Independent CPU random-latent sets (five held-out
keys,10k each):91.39,91.68,91.68,91.72,91.28%, allcoverage40. Original
key CPU reproduction92.16% vs GPU92.18%. No filtering or fixed latent.
Replicate identical config on training seeds1,2,3, committed plan; fills idle
GPUs after preflight. Original seed0 remains frozen e0f1ba0; do not retrain
or overwrite it. Final report combines original seed0 with new seeds1..3
and distinguishes fresh evaluation-key robustness from training-seed variance.
Report seed-count labels now derive from plan so the3-new-seed subreport
cannot falsely claim4; the final combined report has4trainingseeds.

User follow-up: test still smaller conditional sigma, since mu(z) can express
within-mode spread. New matched6-run grid:width256/512,depth3 crossed with
caps-3.5/-4/-4.5. Hold initial log sigma=-4.75 strictly inside all bounds,
minimum-5,mean-init16,teacherfloor.05,N=M64,batch256,T1,Adam3e-4,seed0,
100k. Thecap-3.5 controls distinguish cap effects from changing initialization.
All six use the same random prior, target and unchanged algorithm. Keep
teacher exploration floor unchanged. Compare mu-only near AND40/40 coverage,
mode massTV andMMD; retain fullpolicy as supplement. Do not assume smaller
sigma must improve because output variance also comes from the spread ofmu.
Each GPU runs its registered list when free using existing locks. Do not stop
replication/current experiments. Preflight actualparameters beforeeachrun.

Latest user request: reproduce the result with256x2 ifpossible. Four fresh
seed0,100k,256x2 NM64 runs: primary cap-4.5/initial-4.75 matches the successful
512x3 small-sigma run exceptnetworksize. Controls cap-4/initial-4.75 and
cap-3.5/initial-4.75 match the new small-sigma grid. Additional cap-3/initial-3.5
matches the95.22%,40/40 256x3 seed0 configuration exceptnetworkdepth.
Do not describe thecap-3 control as a pure cap comparison withinitial-4.75
runs. Keepmean-init16,teacherfloor.05,randomlatent,batch256,T1,Adam3e-4.
Use freeGPUs without interrupting prior experiments. Comparefinal100kmu-only
near,coverage,MMD,massTV; independentlyreevaluate anypassing256x2checkpoint.
