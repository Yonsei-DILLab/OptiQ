# AntMaze distance-progress reward variants (implementation, not a launch)

User request 2026-09-24: prepare Euclidean and geodesic variants of
`r = d(current) - d(next) - 0.01 + success_bonus` for v1-v4.
No new queue, training run, restart, or change to active frozen sources is
requested. Existing sparse/dense defaults and historical analyses stay unchanged.

## Profiles

- `--reward-profile progress_euclidean`: distance to nearest goal center in XY.
- `--reward-profile progress_geodesic`: nearest goal-center shortest distance
  through free XY space around the original axis-aligned maze walls.
- Both require `--noveld off`; no intrinsic reward is silently added.
- Every step, including successful and time-limit transitions, pays0.01.
- Success bonus is upstream: v1(-8,0)=10; v2(8,0)=10,(-8,8)=20;
  v3(-12,12)/(12,-12)=10; v4(-16,4)/(-16,-4)=10.
- Original Euclidean radius0.5 success detection and automatic termination stay.
  Bonus is paid once on the terminating transition. No new STOP action.
- Measure distance to the goal point, not its success ball. Do not set the final
  distance to zero upon success or timeout. No gamma inside the reward formula.
  Learner gamma remains0.99. Timeout retains the existing bootstrap semantics.
- Rewards are computed before Gym auto-reset. No hidden stateful goal assignment.
  The nearest goal may change between steps; min distance remains continuous.
  Goal bonus asymmetry is NOT encoded into the nearest-distance shaping term.

## Geometry and computation

Read unmodified official MAZE_v1-v4 definitions by AST;4m cells, reset marker
as coordinate origin, rectangular wall footprints. Build a visibility graph
with exposed obstacle corners and goals, precompute graph shortest distances,
and query visible vertices from each continuous position. This computes a
point-agent polygonal geodesic, not a cell-count approximation. A1e-6m numerical
wall margin prevents paths along internal shared-wall seams or through touching
corners. Body inflation is0: this is a torso-XY geometric approximation, not
Ant joint-space feasibility or Habitat's cylindrical NavMesh. Invalid/wall or
unreachable queries raise an error rather than silently reverting to Euclidean
reward. Continuous positions and all reward computations usefloat64; observations
retain the existingfloat32 format.

Graph construction is cached per maze and performed before fork. Every worker
inherits immutable geometry. No full grid raster interpolation or per-step maze
rebuilding. Return only distance, never a chosen path/waypoint to the actor.
The same geometry/reward is used by training and all evaluation modes.

## Integration and provenance

`envs.py` exposes both profiles. The runner/controller validate explicit profile
names, record `reward_specification` (formula, coordinates, per-goal bonuses,
geometry SHA256/margin, endpoint convention) in config, evaluations, and checkpoint
config. Transition info separately records progress/time-cost/success components.
Final checkpoint verification reconstructs every replay reward from current and
next XY (handlesfloat32 rounding at the success boundary using the terminal bit).
Old collectors that assume reward=-distance are ONLY for their historical dense
campaigns; they must not be used on these new profiles.

All actor/critic optimizers, architectures, temperature, action/latent sampling,
sigma bounds, replay, update ratio, and evaluation starts are unchanged by choosing
a reward profile. For DIPO only, new profiles need signed C51 support with a positive
endpoint above20: derive conservative bounds from maze-distance maximum, gamma.99,
and the one-time20 bonus. Existing sparse[0,5] and dense[-6000,5] are untouched.
This is explicitly recorded in native config, not a default change to old runs.

Tests: analytic obstacle-free/single-rectangle detours, shared-wall shortcut
rejection, symmetric v1 paths, v4 goal distances, progress/retreat/stationary
rewards, real terminal distance, goal10/20, return telescoping atgamma1; actual
CPU MuJoCo physics equality across four profiles, reset/state restoration,
terminal auto-reset and timeout bootstrap, plus forked vector checks.

No reproduction claim: Habitat ICCV2019 uses a single designated goal and its
NavMesh, +10 when STOP within0.2m. These AntMaze profiles retain multiple goals,
upstream10/20 bonuses,0.5m automatic success, and Ant physics. Neither variant
guarantees multimodality; a shorter/better-return route may still dominate.
