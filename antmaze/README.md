# AntMaze policy diversity

The active experiment is [multimodal/PROTOCOL.md](multimodal/PROTOCOL.md):
DDiffPG layouts with an explicit dense nearest-goal reward, online learned policies,
and repeated stochastic trajectories of each individual policy.

The earlier single-goal Gymnasium UMaze pilot at source811e0e2 was canceled after
the user clarified the multi-goal/multi-path objective. Its code and logs are
retained for provenance, and its runs/queue must not be resumed or reported as
the requested benchmark. Current entry points are in `antmaze.multimodal`.
