# Active experiment registration

- Twelve new runs: v2/v3/v4 x OptiQ/SAC/DIPO/MFPO, NovelD0.1, seed0, 1M.
- Frozen source `0d23377f31bdd5c360e9c79d44bb9f4b1341012b`; branch direct-gmm-trg;
  committed/pushed and both canonical servers synchronized before launch.
- Separate v1 DIPO0.01: frozen source `a4ea6c1e3284199bbfab7057282fdc57fd3742a2`.
- 8 GPUs total; one free slot starts the next queued job independently.
  DIPO v1/v3 run on180; v2/v4 on199, while other methods backfill around them.
  No full-maze or all-method completion barrier; polls every2seconds.
- Every new job validates its own272->280 smoke resume before production.
- Eval250k, final1M full checkpoint only. Models, native LR, batch256, UTD1,
  dense reward and NovelD implementation unchanged; no new shaping.
- W&B OptiQ/gmm-trg group `antmaze-v234-noveld01-1m-s0-20260922`.
- Run `python collect.py` for live state; use compatible Python3.12 runtime
  `/tmp/optiq-antmaze-report-20260922/bin/python collect.py --archive-completed`
  to download and verify complete archives. v1 uses its separate collect_status.py.
- Completion report should combine the existing v1 OptiQ/SAC/MFPO coefficient0.01
  controls (source19fc37a) with v1DIPO0.01 and these12 coefficient0.1 runs, clearly
  recording per-environment coefficient and per-run source. Evaluate routes from
  direct-policy samples, with OptiQ mu-only shown separately. Never merge training
  coverage and final-policy diversity or hide failures. All runs are single seed0.

Earlier literature investigation is retained in
`../antmaze_v1_dipo_noveld001_1m/LITERATURE_KO.md`; its candidate suggestions were
superseded by the user's explicit choice of0.1 for all three remaining mazes.
