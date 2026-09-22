# AntMaze v3 OptiQ: NovelD 50 and 100, 100k steps

User authorization (2026-09-22): run NovelD coefficients 50 and 100 after
reviewing the 0.1/1/5/10 sweep. The objective is coefficient selection in our
existing OptiQ + dense reward setting, not MaxEntDP or DDiffPG reproduction.

Exactly two fresh runs, v3 OptiQ seed 0, 100,000 environment interactions each,
on server180 GPU0/1 through supervisor and the existing GPU lock wrapper.
No new seeds or environments, no extension, no implicit restart. The default
coefficient remains .01. Preserve all existing frozen sources and results.

Only NovelD coefficient changes relative to training commit
262a10d6280eb6e79eec24e6170541a7b0f58e72. Training implementation is unchanged.
Keep dense reward=-nearest-goal distance, NovelD c*max(n(next)-.5*n(current),0),
normalize=false, RND L2 error / xy10bands / AdamW1e-4 / clip1. Replay stores
only environment reward and recomputes bonuses at sampling. No soft TD,
entropy backup, reward scaling, extra shaping, or mode-specific Q changes.

Keep 256x2 GELU, mean-init1, random z, N=M64, logstd[-5,-1]/initial-1,
T=.25,beta1,DACERtrue,batch256,UTD1,LR3e-4,gamma.99,tau.005,warmup5k,
replay1M,one env. DACER target-.9,noise_scale.1,initial_alpha.27,alpha_lr.03,
interval10k,GMM3/200samples. Thus each run has 95,000 learner/RND updates.

Commit this code/protocol before execution. Coefficient-linearity/default
compatibility check and each setting's 272-step smoke followed by a new-process
resume to280 must pass before production. Compare initial model/RND hashes and
all non-ablation configuration between new runs and against the completed
0.1 reference (ignore source/output paths only). Do not rerun the old suite.
Failure holds pending jobs, preserves live jobs, and is reported without restart.

W&B OptiQ/gmm-trg; group antmaze-v3-optiq-noveld-high-100k-s0-20260922.
Evaluation unchanged: every25k, native random-z mu-only and full policy including
conditional sigma, 10episodes each. Final100episodes per evaluation/reset mode
and separate zero-z. No external DACER noise or NovelD at evaluation. Do not
pool natural/fixed rollouts as independent starts; v3 natural reset is fixed.

Save final replay/model/target/optimizer/DACER/RND/RNG/simulator checkpoint
and raw rollout archives; verify SHA256/full-state/replay reward identities.
Collect with antmaze.noveld_strength_high.collect_report --source <frozen SHA>.
Compare all six coefficients using final100episode success/goal/route counts,
training first success and distance to each goal, coverage, actual bonus size,
and learning curves. A wider visited area is not evidence of two successful
routes. Keep failed trajectories visible; one training seed/100k is a probe.
