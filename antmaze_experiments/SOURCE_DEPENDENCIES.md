# Preparing MFPO for AntMaze

MFPO is a Git submodule at `gmm40-baseline/MFPO`. Creating a new Git worktree
copies the gitlink but does not populate its contents. This caused the
`configs.mfpo_config` import failure in the 2026-09-24 frozen campaign.

In each **new** source checkout, before scheduling jobs, run:

```sh
python -m antmaze_experiments.dependencies --prepare
```

This initializes only the pinned MFPO dependency and checks its exact gitlink
revision, required files and clean tracked state. Existing dirty or mismatched
dependencies are rejected rather than overwritten. Omit `--prepare` for a
read-only check. The dense campaign registrars prepare dependencies before
writing a manifest or starting a controller; the controller and standalone
runner also check them before allocating environments/GPU work. Manifests and
run configs record the verified dependency commit.

The MFPO config is loaded by its explicit file path, so an unrelated module
named `configs` cannot shadow it. The imported upstream files, model settings,
reward and optimizer remain unchanged.

Do not repair an already running frozen source in place or automatically
restart a failed controller. Preserve the old failure/provenance and use a new
committed source for any separately authorized replacement job.

Regression checks, requiring only Python and Git:

```sh
python -m unittest antmaze_experiments.test_dependencies -v
```

They exercise a real fresh worktree with an empty submodule, exact revision
materialization, a conflicting `configs` module, and dirty/wrong-revision
protection. A full MFPO preflight additionally checks actual initialization,
batch4096 learner updates, evaluation and checkpoint readback on the GPU runtime.
