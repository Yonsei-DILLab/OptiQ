# AntMaze logging destination

User instruction, 2026-09-23: separate all AntMaze logs from `OptiQ/gmm-trg`
into `OptiQ/antmaze`, and use that project for subsequent AntMaze experiments.

`settings.WANDB_ENTITY` and `settings.WANDB_PROJECT` are shared by online
initialization, offline synchronization, and the OptiQ adapter's recorded
configuration. The launch wrapper sets the same destination. Other benchmarks
continue using their own projects; the vendored `antmaze/` repository is unchanged.

Historical runs are moved with the W&B application's Move to project feature,
keeping run IDs, histories, configurations and groups. Original frozen training
source, local configurations, binary W&B logs and model checkpoints are preserved.
Record the new run URLs and migration provenance in separate metadata sidecars.
Never restart learning or change a hyperparameter just to change a logging project.

For a run which was already active when moved, verify that subsequent steps reach
the new project. If live streaming does not follow the move, retain its local
W&B event file and synchronize it to the new location with the same run ID.
Historical internal `wandb-history` artifacts retain their original provenance;
verify that their history remains readable from each moved run.
