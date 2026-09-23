# Historical dense OptiQ: fresh random-start reevaluation

User request 2026-09-24: evaluate available checkpoints with random starts,
rather than infer trajectory diversity from one fixed initial position.

Read-only policy inference, no learner update, no resume, no new training.
Each checkpoint uses its original frozen policy sampler and environment code.
Initial xy is sampled IID uniform[-2,2]^2 for every episode; original reset pose
and velocity are retained. This is an explicit evaluation-only override for
legacy v2/v3/v4, whose training starts were fixed. Same sampled full initial
states are paired across direct-policy and mu-only modes. No external DACER
action noise and no NovelD reward during evaluation. Fresh action latent every
step; conditional sigma is included only in direct-policy evaluation.

- Legacy dense + NovelD0.01 source19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5:
  v1 final1M:1000 episodes/mode, v2/v3/v4 final1M:500 episodes/mode.
- Official dense + NovelD OFF source26336810f7ea4ca61210ea70c6aeeae9f7acaed0:
  v1 final3,008,256:1000 episodes/mode. Interrupted v2-v4 have no final checkpoint.
- Single training seed0.7,000 new episodes across five policies and two modes.
- Seed sequence[20260924,90177,maze number];50 simultaneous inference observations.
- CPU-only, four CPUs/process, nice10, no interference with occupied GPU locks.
- Save raw trajectories/actions/initial states, goal hits, dense return,
  progress, training config, both source SHAs, checkpoint/NPZ SHA256, readback
  and model-unchanged evidence under a separate reevaluation campaign.
- Verify exact restored weights, random start diversity, paired full initial
  states, finite bounded actions, actual goal distances and all dense returns.
- Report failures, success/goal/corridor fractions and dependence on initial xy.
  Distinguish random-start behavior diversity from conditional action modality.
- Do not overwrite historical training or current16-run dense campaign artifacts.

Launch through the existing per-user supervisor, autorestart=false,
autostart=false, startretries=0. CPU-only smoke4 episodes/mode before the
registered full evaluation; smoke results are not experiment outcomes.
