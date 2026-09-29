from pathlib import Path
repo=Path('/home/heechan/OptiQ-direct-gmm-trg')
for name in ('audit_teacher_weights.py','audit_v5_adapter.py'):
 p=repo/'gmm40'/name
 s=p.read_text().replace('from .evaluation import atomic_json','from . import V5_SOURCE\nfrom .evaluation import atomic_json')
 s=s.replace('"optiq_dime.algorithm.', '"gmm40._v5.optiq_dime.algorithm.').replace("'optiq_dime.algorithm.", "'gmm40._v5.optiq_dime.algorithm.")
 p.write_text(s)
p=repo/'gmm40/README.md';s=p.read_text()
s=s.replace('# OptiQ v5 GMM40 experiments','# OptiQ v5 GMM40 experiments (imported into direct-gmm-trg)')
s=s.replace('## Dependencies','''## Source and algorithm

Imported from `v5-gmm40` commit `a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71`.
[IMPORT_SOURCE.json](IMPORT_SOURCE.json) records the files and dependency trees.
The historical v5 implementation and Hydra profiles are pinned under `_v5/`;
its Python namespace is `gmm40._v5.optiq_dime`. Shared `common`, `models` and
`diffusion` dependencies were checked byte-for-byte against the source branch.

`--method optiq` uses the original mean-action OT/Sinkhorn adapter, including
its historical actor sigma settings. It does not select the Direct GMM/TRG
trainer. Active TRG defaults and running frozen snapshots are separate.

## Dependencies''')
start=s.index('Clone the upstream sources at the revisions')
end=s.index('## Running')
s=s[:start]+'''The four original baseline repositories are Git submodules, pinned to the
revisions in [baselines.json](baselines.json): DiKL, DIPO, MEOW and MFPO.
After checking out this branch, initialize their source code with:

```bash
git submodule update --init --recursive
```

Their paths are `gmm40-baseline/DiKL`, `gmm40-baseline/DIPO`,
`gmm40-baseline/meow`, and `gmm40-baseline/MFPO`. The upstream files are
unmodified and retain their upstream licenses. DiKL initializes the target,
including for OptiQ-only runs. Generated results are excluded from Git.

'''+s[end:]
start=s.index('## Checks')
s=s[:start]+'''## Checks

```bash
# CPU-only target, gradient, MEOW Q/V, v5 NLL and navigation invariants.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \\
  python -m gmm40.validate

# CLI/configuration discovery (does not train).
python -m gmm40.run --help
```

The historical diagnostic and comparison scripts still require their named
local checkpoints. Model archives and generated results are not included.
'''
p.write_text(s)
p=repo/'README.md';s=p.read_text();s+='''

## GMM40 experiments and baseline sources

The imported fixed-Q density and navigation experiments are in
[`gmm40/`](gmm40/README.md). Original DiKL, DIPO, MEOW and MFPO sources are
pinned under `gmm40-baseline/`; populate them with
`git submodule update --init --recursive`. These are the historical v5 GMM40
adapters, with an isolated v5 implementation, rather than a change to TRG RL
training defaults.
''';p.write_text(s)
