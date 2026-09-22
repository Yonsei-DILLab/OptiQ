# AntMaze route concentration audit, 2026-09-22

User requested new rollouts and a detailed investigation of single-route behavior.
This is inference only; preserve all 16 production jobs, queues, frozen code and raw results.

- Restore trusted 1M checkpoints from production source `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`; verify SHA256 and exact restored model digest.
- Run 100 new fixed-full-state direct-policy episodes for all four v3 methods with an independent evaluation RNG stream. Save every observation, action and xy coordinate.
- OptiQ v3 additionally: 100 normal fresh-z mu-only episodes, and 100 diagnostic episodes holding a distinct latent constant within each episode, mu-only. The held-latent condition changes the temporal sampling protocol and is not the production policy score.
- OptiQ v1: 100 new ordinary direct-policy episodes as a positive control for detecting two routes.
- Sample 1024 actions at the identical starting observation. Check unique draws and action variance; do not equate a projection or variance with proof of action multimodality.
- Evaluation excludes DACER external noise and intrinsic rewards. No updates, no new W&B training runs, no resumption of training.
- Four independent inference workers use available GPU locks on server 180 under supervisor. No automatic restarts. Commit code and launch protocol before execution.
- Separately audit original training episode goal counts, replay spatial coverage and NovelD bonus magnitudes. Separate observed facts from causal hypotheses; single seed per method.
- Save a separate diagnostic artifact and figures, never overwrite production reports or checkpoints.
