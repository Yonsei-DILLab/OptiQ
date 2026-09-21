# AntMaze policy diversity

The active experiment is [multimodal/PROTOCOL.md](multimodal/PROTOCOL.md):
DDiffPG layouts with an explicit dense nearest-goal reward, online learned policies,
and repeated stochastic trajectories of each individual policy.

The earlier single-goal Gymnasium UMaze pilot at source811e0e2 was canceled after
the user clarified the multi-goal/multi-path objective. Its code and logs are
retained for provenance, and its runs/queue must not be resumed or reported as
the requested benchmark. Current entry points are in `antmaze.multimodal`.

## Active experiment and runtime

| Item | Recorded setting |
|---|---|
| Server | `vast-heechan-199`, GPUs 0–3 |
| Training source | `f47cedad3c757f37893fa6de0466882ebb9673e3` |
| Reporting source | Current committed snapshot recorded in campaign `reporting-source.json` |
| Git branch | `direct-gmm-trg` |
| Campaign / supervisor | `antmaze-multimodal-100k-20260921` |
| Python | `/home/heechan/.venv-optiq-antmaze/bin/python`, Python 3.11.16 |
| Simulator | MuJoCo 3.3.7, Gymnasium 1.2.3 |
| Learner runtimes | JAX 0.4.33, Flax 0.9.0, Optax 0.1.7; Torch 2.7.1+cu128 |
| Scope | v1/v4 × OptiQ/SAC/MEOW/SQL/MFPO/DIPO × seed 0–3 = 48 runs |
| Budget | 100,000 environment interactions per run |
| W&B | `OptiQ/gmm-trg`, group `antmaze-multimodal-100k-20260921-<task>` |

The isolated AntMaze environment installs [requirements-overlay.txt](requirements-overlay.txt)
ahead of the existing GMM40 environment's read-only site-packages fallback. The
original GMM40 environment was not modified. [requirements.lock.txt](requirements.lock.txt)
records all resolved package versions, including the CUDA wheels; per-run config
also records the package versions used by that process.

The full repository is needed: `agents.py` imports the existing OptiQ training
code and native baseline implementations. Pinned baseline sources are DIPO
`c6d8d1b`, MEOW `b786d27`, and MFPO `d8b3977`; SQL uses `gmm40/sql_jax.py`.
Robot XML, maze maps, upstream hashes and licenses are in `multimodal/vendor`.

The frozen source lives under `/home/heechan/OptiQ-ops/sources/<commit>`.
Campaign data lives under
`/home/heechan/optiq-experiments/antmaze-multimodal-100k-20260921`.
The already registered controller owns the queue and fills free GPU slots every
2 seconds; a failed run holds pending jobs while preserving other running jobs.

## Inspection, reporting and local archive

On the server, inspect `status.json`, `failure.json` (when present),
`jobs/<task>-<method>-s<seed>.json`, and each run's `progress.json`.
The supervisor command uses the existing experiment supervisor configuration:

```sh
supervisorctl -c /home/heechan/OptiQ-ops/supervisor/supervisord.conf status antmaze-multimodal-100k-20260921
```

From the **reporting** source directory, use the AntMaze Python runtime to render:

```sh
/home/heechan/.venv-optiq-antmaze/bin/python -m antmaze.multimodal.report --root /home/heechan/optiq-experiments/antmaze-multimodal-100k-20260921 --partial
```

Remove `--partial` for the final report: all 48 runs must have completed and passed
raw trajectory/checkpoint verification. Reporting code is versioned separately;
it does not alter the frozen training source. `reporting-source.json` records this
post-launch distinction and `report/validation.json` records the source/file hashes.

On the local machine, from the parent directory of this folder:

```sh
python3 antmaze/collect.py
python3 antmaze/collect.py --final-checkpoints --verify
```

The first command collects current JSON/NPZ/figures. The second is for the final
verified report: it also collects final policy/critic checkpoints, compares every
audited data hash and writes `results/local-integrity.json`.

Each run retains raw `rollouts/100000-policy-natural.npz` and
`100000-policy-fixed.npz`: XY trajectories, returns, episode lengths, reached
goals, RNG seeds, initial observations and complete initial simulator states.
OptiQ's `mu_only-*` files are supplemental. No training seeds are pooled into one
policy's trajectory figure. See the protocol for reward, route definitions,
evaluation details, native algorithm differences and interpretation limits.
