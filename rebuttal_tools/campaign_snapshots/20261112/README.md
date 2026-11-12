# Fetch F1 / Adroit A1 code snapshot, 2026-11-12

This is a source archive for the non-anonymous `heechan-rebuttal` branch. It
preserves the code used by the experiments separately from the evolving root
implementation. It does not start, resume, stop, or change any experiment.

## Contents

- `index.json` maps 12 frozen source trees to their original file names and SHA256
  hashes. `objects/` deduplicates identical source bytes across those trees.
  `manifests/` preserves each original source manifest byte for byte.
- `operations/` preserves campaign preparation, controllers, constrained archive
  gateways, queue movements, selection, monitoring and Q/ESS diagnosis scripts.
  These are historical operator scripts with their original relative paths,
  machine addresses and filesystem assumptions. They are not a new launch CLI.
  Their original paths and hashes are listed in the index. Deployment protocols
  are configuration records; their embedded queued/running fields are not live
  status. Live status remains in the server plans and central collection.
- `validation/` preserves the numerical JAX SAC evidence and paired-config
  checks, including the earlier Fetch CPU comparison that failed its existing
  tolerance. It is not presented as a passing result. The accepted GPU-reference
  validation and Adroit validation are separate records.
- Training logs, model checkpoints, demonstrations, simulator states, figures,
  SSH private keys and environment credentials are not in this archive.

The archive contains source only. Data and complete result provenance remain in
the approved central experiment repository and the local ignored `outputs/`.

## Verify and restore

From the repository root, verify every source and operator-file hash:

```sh
python3 rebuttal_tools/campaign_snapshots/restore.py
```

Restore one complete, frozen runtime to a **new** directory:

```sh
python3 rebuttal_tools/campaign_snapshots/restore.py \
  --snapshot fetch-adroit-sac-alpha0-20261112/alpha0-adroit-source \
  --destination /tmp/adroit-alpha0-frozen-source
```

Restoration verifies hashes before writing, never overwrites an existing
destination, and does not invoke any training or deployment code. Restored
historical AGENTS.md files document their original context; the current user's
instructions and root repository instructions govern any new execution.

## Experiment distinctions

| Campaign | Scope |
| --- | --- |
| Temperature | DACER on; T={0.05,0.1,0.25,0.5,1}; seeds5/6; independent selected-T seeds0-4; automatic-alpha JAX SAC controls |
| DACER target entropy | H/d={-0.9,0,0.25,0.5,0.75}; early-frozen T; seeds5/6; independently confirmed selection; integer-count tie handling |
| N/M | DACER H/d=-0.9; original final selected T; N64/M256 and N16/M64; seeds5/6; original N64/M64 controls |
| SAC alpha=0 | Shared validated JAX SAC; fixed zero, no alpha optimizer; stochastic collection retained; seeds0/1 only |

All training budgets are 5,000,192 environment transitions. F1 uses Fetch-v4
sparse/h50, Slide explicitly 20cm, corrected HER normalization, critic LayerNorm
off, actor/critic256x2, vec256/worker4/batch256/UTD1/8. A1 uses native Sparse-v2/h200,
25 distinct saved-replay-verified expert-policy demonstrations per task, equal
128:128 demonstration/online minibatches, BC0, critic LayerNorm, actor/critic256x2,
vec256/worker4/batch256/UTD1/4. Human-action replay and older 5cm Slide results are
historical and are not equivalent to these protocols.

The current root `use_min_candidate_q` option is experimental and defaults to
false. These frozen F1/A1 runs score actor candidates with the twin-Q **mean**;
their TD target uses the twin-Q minimum. Do not relabel the frozen runs as
minimum-candidate-Q experiments.

The CrossQ preparation and normalization-diagnostic tools under `rebuttal_tools/`
are separate work. Archiving them does not release the preparation-only queue.
