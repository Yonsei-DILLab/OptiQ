# 250k horizon / temperature screen, unchanged OptiQ algorithm

Authorization: the active user goal permits reward/hyperparameter experiments,
early rejection around250k and revised hypotheses, while absolutely prohibiting
algorithm changes. Preserve the explicit16-job positive DACER target sweep and
let its queued jobs take GPU slots first. This new eight-job study tests a
different hypothesis, using only configuration overrides and the same TD/NLL
implementation. It is an exploratory screen, not a successful final result.

Evidence observed before registration:

- Current v3 H/d+.7 at300032 total transitions: first left gate39, right1,
  left goal successes1/40. Entropy target changes alone have not yet retained
  two successful routes. v4 H/d+.9 still uses both corridors at300k, success0;
  do not treat that run as a failed completed experiment.
- Historical v3 control at400128: left8/right27, actual successful left0/right12.
  Same saved trajectories have route-mean discounted progress returns546.12
  versus652.12 atgamma.99. Atgamma.999 the means are1319.92 versus1353.91;
  undiscounted means1577.49 versus1571.80. Endpoint progress alone is therefore
  insufficient to characterize the current preference. This is observational
  re-scoring, not proof that changing gamma will train a better policy.
- Current actual extra noise std remains only about.02-.04 in early training.
  A higher entropy target does not directly constrain actor route entropy.

## Conditions

v3 and v4 each: (gamma.999,T1), (gamma.99,T3), (gamma.999,T3).
v1: (gamma.999,T1), (gamma.999,T3). Seed0, eight policies.
Compare with the current matching H/d+.7, gamma.99, T1 controls, including all
failures. v2 remains covered by the existing16-job target sweep.

All new conditions: DACER ON, H/d+.7, interval500, initial alpha.27, alphaLR.03,
noise_scale.1, unchanged3-component/200samples estimator. NovelD OFF.
Reward100*(nearest-goal Euclidean distance decrease), no bonus/step penalty;
no discount is inserted into the reward itself. Only the existing model gamma
parameter and teacher temperature are varied. Discount.999 changes the planning
objective as well as TD propagation; inspect Q scale/convergence if promising.
Actor/twin critics256x3, learning rates3e-4/5e-4, Adam, tau.005, replay1M,
random latent,N=M64, log sigma[-5,-1]/initial-1, mean-init1, beta1 remain fixed.
256env,batch4096,8updates/256transitions. Preserve native physics/termination.

## Budget and decision

250000 requested post-warmup interactions, using native strict-greater block
accounting:250112 post-warmup,258304 total,7816 learner updates,16 DACER updates.
No automatic extension: inspect the final saved state and trajectories before
choosing a longer confirmatory run. Each50k total save an evaluation policy and
evaluate40 episodes/mode; final100. v1 native random starts; v2-v4 original fixed
full state. Direct policy includes conditional sigma; native mu-only is separate;
no external DACER noise during evaluation. Mere entry into two corridors is not
goal achievement or sustained multimodality. Inspect path/goal counts, failures,
closest-goal distances and repeated later checkpoints; do not combine policies.

## Launch and invariants

`antmaze-optiq-horizon-temperature-250k-s0-20260925`, W&B OptiQ/antmaze.
Run `register_horizon_temperature --shard 0 --host vast-heechan-180`, or shard1
host199, only from a committed/shared immutable source. Commit/push/share before
preflight. Registration verifies that core algorithm, policy, loss, sampling,
transport, base DIME and DACER regulator files are byte-identical to
2564b59faa0d319eece496b93eff0f19359efc37 and records their SHA256 values.
Per-job preflight checks the live model.gamma, temperature, replay reward,
optimizer architecture, actual updates and checkpoint round-trip.

Independent backfill every2seconds. Wait only while the predecessor campaign
has unassigned pending jobs; once all are assigned, take free GPUs without an
all-completed barrier. On predecessor failure hold the new pending queue.
Predecessor running GPU assignments also reserve their slots before the child
acquires its OS lock, closing the dispatch/lock-acquisition race. A controller
source replacement before the first training job is recorded in a separate
controller-provenance.json; the frozen training manifest/source remains unchanged.
Never restart/cancel old jobs through this controller, and never use vast1/4090.
Full failed/finished/early-screen data and frozen sources are preserved.
