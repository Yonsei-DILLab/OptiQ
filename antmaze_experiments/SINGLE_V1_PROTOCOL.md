# Single-environment AntMaze v1, 2026-09-25

User requested one v1 test with 100,000 total training interactions, UTD1,
and no vectorized collection. Seed0, current basic Direct GMM/TRG algorithm.
One native environment stepped in-process, no Gym VectorEnv or workers for
collection. Evaluation uses the existing evaluator and is outside training count.
Warmup10000 is included: 90,000 learner updates follow, one immediately after
each transition. User explicitly selected batch256; preserve replay1M, N=M64, T1, 256x2 networks,
log sigma[-5,-1], DACER OFF, NovelD OFF, random v1 starts.
Reward =100*(nearest-goal Euclidean distance before minus after), no step cost
or success bonus; native success termination and timeout bootstrapping preserved.
Evaluate every10k,20 episodes each native(mu-only) and full-policy mode;
final100 per mode including zero-z. Save intermediate policy and full final state.
User also requests videos every5000 total training transitions: full-policy
native v1 rollout, fixed evaluation seed90500 for paired comparison. Save MP4
and per-video return/success/trajectory metadata, isolate policy/training RNG.
Twenty main videos from5k to100k; these rollouts do not enter replay or counts.
W&B OptiQ/jaehun-antmaze, group antmaze-v1-single-utd1-100k-20260925.
Videos are also uploaded as eval/rollout_video with their training-step label.
Use free vast1 GPU2 only; do not stop/resume other jobs. Commit frozen source
before GPU preflight (10000 random transitions plus8 real batch256 updates,
checkpoint readback and six evaluation episodes), then launch fresh main run
only if preflight passes. No automatic restart on failure.
