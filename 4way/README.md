# 4-Way Maze source and task profile

`upstream/simpler_path_finding.py` is the unmodified environment file from
Zhang, Yu and Turk, *Learning Novel Policies For Tasks*, ICML 2019. Source:
<https://github.com/yunbozhangGT/Task_Novelty_Bisector>, Git commit
`6d686397b9cb5f9dd395ec0a5ce1ada0725d3ede`. The original project license
is copied to `upstream/LICENSE.md`. The original file SHA256 is
`d6c8c16b6826611fe738574362f197fc5c4f2b16286689b0c1466aa5c3da774a`.

`environment.py` ports the 27×27 grid, central reset, 4D position/velocity
observation, clipped 2D force action, 5-substep dynamics, wall collision and
500-step horizon to Gymnasium. `FourWayBatch` advances many independent
replicas in one NumPy call, and `FourWayEnv` is the scalar adapter. The
`native` reward profile preserves the author's unequal goal rewards and
single-use hint cells. The default `symmetric` profile keeps the dynamics,
walls and terminal conditions but gives all four goals the same +500 bonus
and disables hints. This is an explicit adaptation for comparing whether one
policy retains four behaviors: the original native profile strongly favors
the south goal (500) over the west goal (7.8125).

The original author's novelty penalty depends on previously learned policies;
it vanishes when no previous policies are installed. These comparisons train
each method as a single policy and do not use that cross-policy mechanism.

Goal order in evaluation is east, west, north, south. Always report goal reach
counts from complete sampled-policy rollouts from the same central reset
distribution. First-action direction is a diagnostic, not a reached goal.
