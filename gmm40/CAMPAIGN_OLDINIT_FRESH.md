# Current TRG policy with previous initialization/exploration settings

User request (2026-09-21): retain current OptiQ structure, do not revert to the
old tanh-squashed Gaussian, and rerun with the previous initialization settings.
Use the existing box-truncated Gaussian actor/proposal/marginal NLL unchanged.
Its existing bounded-center tanh remains; no post-noise tanh transformation is added.
Latents are freshly sampled Gaussian codes in both training and evaluation.

Approved GMM40-only profile: log sigma [-5,+1], initial sigma .5 (log -.693147...),
mean-head variance scale1 (Xavier), teacher-only std floor .05. Teacher sampling
and log density both use the same effective floor; actor sampling does not use
the teacher floor. Sigma remains hard-clipped, initialized inside its bounds.
All other settings: seeds0..3,100k actor updates,256x2GELU,N=M64,batch256,T1,
beta1 density correction,Adam3e-4,10,000 independent full-policy evaluation
samples and separate centers-only evaluation. No changes to RL defaults or old
frozen sources. The added teacher_std_floor argument defaults to exp(-5), so
all prior TRG profiles retain their behavior.

Commit code/plan/protocol before training, create a detached frozen source,
validate with gmm40.validate_oldinit_fresh (no training), prepare using
python -m gmm40.campaign prepare --root <root> --plan gmm40/campaign_oldinit_fresh_plan.json,
then execute the existing two-update GPU preflight through run-gpu.sh on a free
GPU. After passing, register gmm40.campaign run with the existing user supervisor.
Preserve manifest, source SHA, dependency pins, preflight and registration.
Use free GPUs without interrupting existing campaigns; locks are respected and
free slots are checked every five seconds. Failure blocks pending jobs; preserve
active workers and investigate without automatic learner retries.

Server vast-heechan-180. Root /home/heechan/optiq-experiments/gmm40-trg-oldinit-fresh-100k-4seed-20260921.
Supervisor and W&B group use that campaign name. W&B OptiQ/gmm-trg.
Four completed runs must pass100k optimizer audits. Existing reporter saves
seed means/sample SD, learning curves, distributions, JSON/CSV/Korean report,
and SHA256-addressed final archive. Collect under artifacts/gmm40_oldinit_fresh_queue.

This imports numerical sigma settings into box coordinates; the old settings
were in pre-tanh coordinates and therefore do not represent identical action
space widths. This is not exact reproduction of the old policy family.
Bounds, initialization and teacher floor change together; compare their joint
result without claiming individual causality. Track mode coverage as well as
near fraction/MMD/mass TV; inspect raw sigma saturation if the problem remains.
