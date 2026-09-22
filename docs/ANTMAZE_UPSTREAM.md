# AntMaze: official DDiffPG source

`antmaze/` contains all 159 tracked files from the official
[supersglzc/ddiffpg](https://github.com/supersglzc/ddiffpg) repository at
`7edd06c4799abbab0f8fa534c21deb56253b018e`, imported on 2026-09-23 KST.
The original source layout, README, licenses, configuration, environment assets,
learners, vectorized collection, and trajectory tools are unchanged. This is a
vendored source snapshot, not a Git submodule or a new implementation. Upstream
Git metadata is not nested inside `antmaze/`.

The file manifest in [antmaze-ddiffpg-upstream.json](antmaze-ddiffpg-upstream.json)
records every upstream file's SHA256, size, mode, and Git blob ID. From the OptiQ
repository root, check the snapshot without importing its training dependencies:

```sh
python3 analysis_tools/verify_antmaze_upstream.py
```

## Upstream entry points and settings

Follow the original [installation instructions](../antmaze/README.md). Run from
the imported repository root after installing its dependencies:

```sh
cd antmaze
python scripts/ddiffpg_main.py algo=ddiffpg_algo env.name=antmaze-v1
python scripts/baselines_main.py algo=sac_algo env.name=antmaze-v1
python scripts/baselines_main.py algo=dipo_algo env.name=antmaze-v1
```

These are the upstream PyTorch learners. The upstream environment uses the older
MuJoCo 2.1 / `mujoco_py` stack; its declared Conda environment uses Python 3.8.
This import does not install those dependencies into existing experiment virtual
environments and does not claim compatibility with their newer MuJoCo/JAX stack.
OptiQ and MFPO adapters are not part of the official repository and have not been
added to this unchanged snapshot.

Upstream defaults are preserved, including NovelD coefficient 0.01,
`normalize: False`, and 256 training environments. The AntMaze environment returns
sparse goal bonuses (0/10/20), not the previous nearest-goal dense reward. Merely
passing `reward_type=dense` does not implement dense rewards in this upstream
fork. Configuration import is not authorization to start these default runs.

## Previous experiments and provenance

The previous custom `antmaze/` implementation was removed from the active branch.
Its complete tracked version remains in OptiQ commit
`0d23377f31bdd5c360e9c79d44bb9f4b1341012b` and earlier Git history. Before replacing
each working folder, its actual contents (including ignored files) were moved to
a separate backup. Existing experiment data and immutable source snapshots are
preserved.

- Local backup: `artifacts/antmaze_upstream_migration_20260923/legacy-workspace/antmaze/`.
- Each server backup: `/home/heechan/OptiQ-ops/archives/antmaze-legacy-20260923/antmaze/`.
- v1 DIPO NovelD 0.01 was stopped by user request on 2026-09-23; preserve its original frozen source
  `/home/heechan/OptiQ-ops/sources/a4ea6c1e3284199bbfab7057282fdc57fd3742a2`.
- All cancelled v2-v4 and earlier experiment queues remain cancelled.

Old `python -m antmaze.multimodal...` commands belong to the legacy source. For
historical reporting, use that frozen checkout, or on the local machine expose
the archived package explicitly. For example, to collect and verify the existing
v1 DIPO run using the validated reporting runtime:

```sh
PYTHONPATH="$PWD/artifacts/antmaze_upstream_migration_20260923/legacy-workspace" \
  /tmp/optiq-antmaze-report-20260922/bin/python \
  artifacts/antmaze_v1_dipo_noveld001_1m/collect_status.py --archive-completed
```

The import verification checks exact source identity and Python syntax. It does
not launch training, validate simulation dynamics at runtime, or restart jobs.

The separately approved 64-environment integration is documented in
[antmaze_experiments/PROTOCOL.md](../antmaze_experiments/PROTOCOL.md). It does not
alter the vendored upstream snapshot.
