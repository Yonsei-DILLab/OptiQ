# DDiffPG-based fast route learning, v1 seed 0

User authorization: use DDiffPG as the reference; modifications/additions are
allowed to find routes quickly. This is an assisted goal-reaching experiment,
not an MFPO reproduction or an unassisted-exploration benchmark.

OptiQ, SAC, MFPO and MEOW each train from scratch for exactly 100,000 online
transitions, seed 0, with independent replay and NovelD modules. Four free GPUs
run these jobs concurrently. No other environment, seed, or continuation is
launched by this profile. W&B: OptiQ/gmm-trg;
group antmaze-v1-routefast-100k-s0-20260921.

Retained from DDiffPG: v1 maze, low-gear Ant XML, action/observation spaces,
dt=.1, goal radius .5, 500-step horizon, full-start xy reset uniform[-2,2]^2,
and the existing exact RND/NovelD port (.01 coefficient). Modern MuJoCo port
provenance remains in vendor/provenance.json.

Changes shared by all four methods:

* Obstacle-aware distance is computed on a .25m grid, with .6m wall clearance,
  eight-connected shortest paths and no diagonal corner cutting. The upper and
  lower corridors are symmetric. The map is additional training knowledge;
  no waypoint, route identifier, action, or planner is fed to the policy.
* Training reward: 10*(d_before-d_after) + .1*healthy - .05*sum(action^2)
  - .01 + 10*success - 5*fall. Here d is obstacle-aware distance. This explicitly
  changes the objective; it is not claimed to preserve the sparse-reward optimum.
  Fallen training episodes terminate at torso z outside [.2,1]. Successful
  terminals also terminate; time limits bootstrap.
* Reset curriculum: before 60k, 20% native full starts and 80% starts sampled
  uniformly from reachable safe grid points at distance 1..R from the goal;
  R grows linearly from 2m to the origin's route distance by 60k. Both corridors
  use the same field without a selected preferred route. From 60k to 80k the
  full-start probability is 75%. At/after 80k all newly reset episodes use the
  native full-start distribution. Ongoing episodes are not teleported mid-rollout.
  Curriculum and full-start training successes/episodes are logged separately.
* Eight environments, batch256, eight learner updates per collection round:
  UTD1. Warmup8192 is included in100k, yielding 91,808 actual learner and RND
  updates. Native learner architectures/LR/tau/discount remain unchanged.
  OptiQ T.25/beta1/DACERtrue/mean-init1/random-z/N=M64/logsigma[-5,-1] retained.

Evaluation remains the existing unmodified DDiffPG geometry/reset/termination
adapter with no curriculum, no reward shaping in reported returns, no RND, and
no external DACER action noise. Report success within .5m and geometric routes,
not a shaped training return. Primary action evaluation directly samples the
learned policy. OptiQ mu-only is supplemental. No planner or action selection
is introduced at inference. Final100 natural reset +100 identical-full-state
rollouts per method. These full-start scores are never mixed with curriculum
training successes. Evaluate10 episodes every10k; save50k/100k checkpoints.

Run meaningful environment contract checks before training: obstacle detour,
reflection symmetry, collision-free curriculum starts, curriculum removal,
success/fall masks, and unchanged native evaluation reset. Then run all-four
preflight with512 interactions,256warmup,256updates and2rollouts/mode.
Freeze/commit exact source before preflight and real runs. Verify actor/critic
changes, RND target immutability, all raw evaluation rewards from XY, actual
successes, final checkpoint hashes and exact update/transition counters.

Preserve prior frozen results. Report the combined effect of shaping,
curriculum, fall handling and update schedule, not an isolated NovelD effect.
Zero or one training seed cannot establish superiority or absence of mode collapse.
