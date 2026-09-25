# DrAC multi-goal PointMaze source

The files in `envs/mgmaze/` are copied without modifications from
[`PneuC/DrAC`](https://github.com/PneuC/DrAC), commit
`4e718983ea29aa3a856955f553a99795fcb4e94d` (2026-09-26 checkout).
The upstream source headers identify Apache License 2.0; its text is included
here as `LICENSE`. The OptiQ integration lives outside this upstream directory.

For the three maps, the upstream `maps.py` defines four, four and eight goals.
`point_maze.py` defines horizons 150, 300 and 600, while `maze.py` awards
100 and terminates on reaching any goal. With `reward_type="sparse"`, all
other rewards are zero. Cells coded `2` are passable during training and become
walls only in the obstacle evaluation mode. The training starts are fixed.

The paper defines removal robustness by removing half the goals at test time,
and obstacle robustness by adding the latent walls. Both are five-trial success
probabilities. See [Wang et al., "Learning Intractable Multimodal Policies with
Reparameterization and Diversity Regularization"](https://arxiv.org/abs/2511.01374),
§5.1 and Appendix A.2.
