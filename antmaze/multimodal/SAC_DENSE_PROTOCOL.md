# SAC dense AntMaze-v1 500k confirmation

Latest user authorization, 2026-09-22: extend SAC-only dense reward test to
500k environment interactions, record whether any goal is reached.
One training seed0; preserve the100k checkpoint for comparison.

The current environment already implements -nearest-goal Euclidean distance.
This is a fresh run of that explicit dense-only setting, not a new reward
implementation. Prior seed0 dense-only100k had0/100 evaluation successes;
preserve it. Record training-time success counts in this run as well.

Environment: antmaze.multimodal.env, v1, DDiffPG map/low-gear robot port.
Observation29/action8,goal(-8,0),goal radius.5,horizon500,
resetxy uniform[-2,2]^2,default robot pose/zero velocity.
Reward=-distance(next_xy,goal),scale1,success termination,timeout bootstrap.
Keep map/reset/dynamics unchanged. No NovelD,RND,DACER,geodesic shaping,
locomotion bonuses,goal-near curriculum,demonstrations or pretrained policy.

SAC: existing SB3 learner,256x2,Adam3e-4,batch256,UTD1,
gamma.99,tau.005,auto entropy,replay1M,warmup5000 included in500k.
Single training environment. Exactly495000 optimizer updates after warmup.
Fresh initialization. Evaluation:10 stochastic episodes every5000steps;
final100 natural-reset and100 identical-full-state stochastic episodes,
reported separately. Save checkpoints every50k through500k,raw trajectories,coverage,
per-training-episode successes/lengths/distances,and first success step.
Only instrumentation and a restricted SAC-only controller profile are added;
training algorithm,environment reward,andhyperparameters are unchanged.

Campaign: antmaze-v1-sac-dense-500k-s0-20260922 onvast-heechan-180.
W&B:OptiQ/gmm-trg. GPU0 through existing run-gpu.sh locks.
Commit/freeze before launch. Existing cancelled campaigns remain stopped.
Preflight: same committed source;controller --smoke,272steps/256warmup,
16updates,2evaluation episodes per reset mode; W&B disabled.
Require environment contract checks and verified preflight completion.

Launch via managed supervisor (autostart=false,autorestart=false):
python -m antmaze.multimodal.sac_dense_campaign --root <campaign>
The pipeline preflights and then invokes the controller with
--profile sac-dense --methods sac --tasks v1 --seeds 0.
Source path and exact argv are stored in the manifest. Only one GPU is used.

On500k completion verify updates,checkpoint hashes,policy parameter changes,
coverage transition count,and raw XY-based reward/success reconstruction.
Run antmaze.multimodal.report and collect artifacts plusfinalcheckpoint locally.
Report training success count and final success rates without claiming that
one seed establishes impossibility or mode collapse. No follow-on experiments.
