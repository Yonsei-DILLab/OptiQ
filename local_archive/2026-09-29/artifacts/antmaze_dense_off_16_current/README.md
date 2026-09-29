# Dense / NovelD OFF 16-policy campaign

- Campaign: antmaze-dense-off-16-current-s0-20260924
- Source: a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3, direct-gmm-trg; pushed and shared with both hosts.
- v1/v2/v3/v4 × OptiQ/SAC/DIPO/MFPO, seed0, sixteen fresh policies.
- 256env, batch4096, eight updates per256 transitions, native3M/3M/4M/5M budgets.
- Dense reward = negative next-state distance to nearest goal. NovelD/RND OFF.
- OptiQ256x3 actor/critic,T=.01,actorLR3e-4,criticLR5e-4,log sigma[-5,-1],DACER on.
- Interim40 episodes/mode every250k, final100/mode/reset. Random-start primary evaluation.
- Host180: v1/v3. Host199: v2/v4. Each runs two DIPO and two OptiQ jobs, then SAC/MFPO backfill independently every2seconds.
- Eight initial real preflights passed; all eight main learners updated with RND updates0. Eight baseline jobs remain queued. No duplicate/cancelled jobs resumed.
- launch-verification.json and latest/*.json preserve launch/config/checkpoint proof and actual GPU processes.
- Read-only collection: /tmp/optiq-antmaze-report-20260922/bin/python artifacts/antmaze_dense_off_16_current/collect_status.py
- Final full checkpoints and complete logs remain under the same campaign root on each server. This status collector does not download model/replay archives.
