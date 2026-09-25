# Four-goal PointMaze task

`upstream/point_maze.py` is the unmodified Farama Gymnasium-Robotics
`PointMazeEnv` source at commit
`4d1ebecbc6436806cfbc0e42ebc36f594d05844e` of
<https://github.com/Farama-Foundation/Gymnasium-Robotics>.
Its SHA256 is `ea4e3d106579a9fe1927f1538eb911485fca3019ce2f95a30cd311f055c5935b`;
the repository license is in `upstream/LICENSE`. The actual simulator uses
the installed Gymnasium-Robotics `PointMazeEnv`, whose dynamics come from
MuJoCo.

`environment.py` supplies a disclosed **new task**, not a stock Farama
environment ID. The maze is a 7×7 four-arm cross with one central reset cell
and four equal-distance goals. The policy observes the native point's 4D
position/velocity, with no desired-goal coordinate. A rollout succeeds at
the first goal whose distance is at most 0.45. Dense reward applies the
official `exp(-distance)` formula to the nearest of the four goals; sparse
reward is 0/1 at success. Each episode ends on success or at 300 steps.

Goal order is east, west, north, south. Evaluation must use repeated full
policy samples from the common center and report four goal counts, failures,
success rates and raw trajectories. The hidden native target sampled by
`PointMazeEnv` is ignored by this any-goal wrapper and never fed to a policy.
