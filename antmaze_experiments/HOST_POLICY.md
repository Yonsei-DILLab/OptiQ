# AntMaze server allocation

Latest user instruction, 2026-09-25:

- `vast1` (175.155.64.161:19329, RTX4090 x4) is reserved for GMM40.
  Do not use it for AntMaze training, preflight, evaluation or watchers.
- AntMaze may use the two existing RTX5090 hosts, `vast-heechan-180` and
  `vast-heechan-199`, within the currently approved campaign scope.
- Keep current 5090 learners and their frozen sources unchanged.
- Preserve the 4090 host's old AntMaze results, source snapshots, supervisor
  configurations, and all NM/ablation data. Archive AntMaze service configs
  outside the active supervisor jobs directory and unload those services.
- This reservation does not authorize shutting down the instance, deleting
  experiment data, stopping unrelated work or launching a new GMM40 experiment.

Earlier three-host manifests and protocols describe historical experiments;
they do not override this policy. The historical progress100 registrar now
rejects new vast1 registrations, while its pure manifest builder remains
available to inspect/reproduce the recorded old allocation.

Operational receipt: `/home/heechan/OptiQ-ops/host-reservation-gmm40-20260925.json`.
Archived supervisor configs live under
`/home/heechan/OptiQ-ops/supervisor/disabled-antmaze-20260925/`.
