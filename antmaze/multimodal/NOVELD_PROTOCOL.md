# v1 / seed0 / 100k dense NovelD pilot

Latest user scope: OptiQ, SAC, MFPO, MEOW only, v1 only, seed0, exactly100000
training environment interactions each. Finish and report before extending.
No v3/v4/1M/additional seeds are launched by this campaign.

Use DDiffPG7edd06c's NovelD equation and collection schedule:256 independent
environments, batch4096,8 optimizer updates after each vector collection round.
Warmup8192 interactions (32 vector rounds, from the upstream baseline profile)
is included in100k. Do not use DDiffPG's128k warmup in a100k experiment.
The final round collects160 transitions, preserving an exact100k total;
359 post-warmup rounds yield2872 actual learner and RND updates.
Preflight: same256env/batch4096,8192warmup,8704total,16updates,2rollouts per mode.

Keep the existing dense environment reward -nearest-goal Euclidean distance,
map/robot/dynamics/termination. Add0.01*max(n(next)-.5*n(current),0) to each
critic minibatch reward. RND uses full29D observation with xy positional
encodingL10, no normalization, frozen target,512-256-128-128 ELU MLP,
AdamW1e-4/clip1. Novelty is computed before each predictor update. Each method
has independent RND/replay/state; evaluation adds no intrinsic reward.
Original code provenance: ddiffpg/utils/intrinsic.py and models/mlp.py,
Apache2 license retained in vendor/LICENSE-APACHE2.

Native architectures, learning rates, discount, target updates and loss functions
remain as recorded in native config; this is an adaptation of the DDiffPG
exploration/collection setup, not a complete algorithm reproduction. OptiQ
T.25/beta1/DACERtrue/mean-init1/random latent/N=M64 remain unchanged. DACER
entropy diagnostic keeps256states,200draws/state,GMM3 and10k learner-update
interval. With2872updates its initial alpha update occurs once. Keep its
behavior noise in collection and exclude it from learned-policy evaluation.

Collection batches policy inference and steps256 independent MuJoCo instances
in8 bounded CPU threads. The replay stores true terminal next-observations;
timeouts bootstrap, successful terminals do not. Shared reward bonus is computed
on replay samples, not permanently cached with transitions.

Evaluation: initial, approximately25k/50k/75k vector boundaries,final100k;
10 stochastic episodes each. Final100 natural-reset plus100 identical-full-state
episodes per policy. OptiQ conditional-noise-free mu-only is supplemental.
No seed pooling. Save checkpoints, RND weights/optimizer, intrinsic statistics,
actor/critic change audit, all rollout NPZ, training coverage and W&B config.
Verify raw rewards from XY, terminal success, checkpoint hashes, exact100k
coverage count,2872RND/learner updates, frozen RND target and changed predictor.

Compare to old dense-only100k separately. Collection/update schedule, warmup,
batch and NovelD all change together, so a difference is not a pure intrinsic
reward ablation. Zero successes is unresolved goal reaching, not proof of
deterministic mode collapse. One training seed is preliminary evidence.
