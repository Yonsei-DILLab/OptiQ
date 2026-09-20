# gmm-trg sequential campaign (2026-09-21)

Branch direct-gmm-trg, W&B OptiQ/gmm-trg. Existing completed runs are copied,
not deleted or retrained; imported provenance remains fc6067b. Running originals
finish unchanged. Preserve DIPO and Yonsei jobs. No automatic early stopping,
no unapproved retries. One scientific process per GPU; check nvidia-smi first.

Frozen baseline: bounded tanh centers, box-truncated diagonal Gaussian, continuous
normal latent, shared 256x2 GELU mu/logstd heads, logstd[-5,-1], initial=-1,
64 components/64 exact candidates, direct mixture NLL, no OT/gradient clip,
scalar twin critic, mean extraction / min plain TD, no entropy backup.
All existing UTD=1, batch=256, lr_actor=lr_critic=3e-4, warmup=5K, buffer=1M,
total_steps=1M, evaluation every5K with10episodes remain unchanged.

## Stages (all seeds 0,1,2,3,4)

1. beta=1, no DACER: HalfCheetah .25/.1, Walker2d .25/.1, Hopper .05/.1,
   Ant .25, Humanoid .25. Total40, reuse12 originals (non-Hopper seeds0..2),
   so28 newly trained runs. Existing Hopper .25 does not enter selection.
2. Same six HalfCheetah/Walker/Hopper configurations + DACER:30 runs.
3. Using stage1's environment-specific best temperature, beta=.5/.9/1,
   no DACER. Reuse beta1, so10 new runs.
4. Using stage1's best temperature, beta1 + DACER. Reuse three environments'
   stage2 runs, add Ant/Humanoid:10 new runs.

90 unique configurations/seeds including12 reused originals;78 new runs.
Never select individual best seeds. Rank temperatures by arithmetic mean over
five seeds of eval/stochastic_z/mean_reward averaged over eval steps
(900000,1000000] (20 evaluations, each10episodes). No zero-z ranking, no max
checkpoint, no smoothing. Wait for complete five-seed cohorts; ties prefer lower
temperature. Record all cohort scores, selected temperatures and reused IDs.
Later stages are globally gated, not independently advanced on each host.

## DACER behavior-only adapter

Read paper https://arxiv.org/html/2405.15177v4, Sec4.3, Eqs15/16, Table3,
and official implementation https://github.com/happy-yan/DACER-Diffusion-with-Online-RL
commit9f22f29fa91b1bed8ee07177a51598d6bd413cb5.

H_target=-.9*action_dim, GMM3 full covariance components fitted by sklearn EM,
200 noisy/clipped policy actions per replay state, batch256. Entropy proxy is
H(component label)+mean component entropy; it is NOT exact mixture entropy.
Official deterministic fitting seed42. Adam on log(alpha),lr=.03, every10000
learner updates (first measurement at update0, as official code). Low entropy
increases alpha. alpha_initial=.27 from paper Table3. Official repository uses
initial3 (comment suggests5) instead; this difference is intentionally disclosed.
lambda=.15 for HalfCheetah/Humanoid, .1 otherwise, per Table3 (official get_action
hardcodes .15 with comment for other tasks).

User confirmed behavior-only isolation: a_env=clip(a_policy+lambda*alpha*noise).
Replay stores a_env. Teacher/proposal/IS/NLL, plain TD actions, and dual mean-only
evaluation remain unchanged. Official DACER also noises TD/policy-Q actions;
we intentionally do not import that coupling, actor Q gradients, distributional
critic, altered warmup, network width, actor lr, or policy-update delay.
Regulator has no backprop into the actor. It samples on separate RNG streams.

## Operations

campaign.py worker is Supervisor-managed; state files are atomic and local queue
claims locked. No failed/interrupted attempt is silently retried. Source hash and
commit verified before each run. Results require final actor/critic checkpoints
and complete finite dual evaluation arrays, not just W&B status. W&B upload-only
failure is distinguished from training failure. import-old uses local W&B binary
history with explicit target project and does not delete original logs.

Advance via coordinator only after all registered hosts have been inspected.
Unreachable vast3 is reserved, not assumed empty; never duplicate its seeds.
