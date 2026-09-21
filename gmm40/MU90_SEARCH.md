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
