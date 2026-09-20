# OptiQ v5 GMM40 experiments

On `v5-direct-gmm`, `--method direct_gmm` selects the heejoon marginal GMM
objective with the same v5 teacher and density correction. `--method optiq`
retains the OT baseline; SAC/DIPO/MEow/MFPO remain available. See
[the Direct GMM audit](../docs/v5_direct_gmm/AUDIT_KO.md). The two RTX 5090 hosts
use CUDA PyTorch 2.7.1+cu128 for hardware support, with the original JAX 0.4.33
core. Historical experiment budgets, permissions and results below are context,
not instructions to resume or stop new experiments.

Fixed-Q density reconstruction and a separate 100-step navigation task using
the DiKL seed-0, 40-component Gaussian mixture. The default OptiQ adapter reuses
the v5 actor, Gaussian proposal, mean-action Sinkhorn assignment and full-row
conditional Gaussian NLL. Baselines include SAC, DIPO, MEow and MFPO.

## Dependencies

Use the repository's Python 3.11 / JAX environment (`requirements-mujoco.in`)
with `gym==0.26.2` for the upstream MFPO adapter. The recorded environment used
JAX/JAXlib 0.4.33, Flax 0.9.0, Optax 0.1.7, NumPy 1.26.4 and SciPy 1.11.4.
The PyTorch agents require a CUDA-enabled PyTorch build; the experiment used
PyTorch 2.4.1+cu124, whereas the MuJoCo requirements specify CPU PyTorch.

Clone the upstream sources at the revisions in [baselines.json](baselines.json)
into `gmm40-baseline/<name>`. Run from the repository root:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path
for source in json.loads(Path('gmm40/baselines.json').read_text()):
    destination = Path('gmm40-baseline') / source['name']
    subprocess.run(['git', 'clone', source['url'], str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', source['commit']], check=True)
PY
```

DiKL is required to initialize the target, including for OptiQ-only runs.
The imported baseline source files have no local patches. Baseline repositories
and generated results are stored locally and excluded from this branch.

## Running

```bash
# Fixed Q = log p_GMM, T=1: 100K actor updates, one selected GPU.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m gmm40.run --method optiq --name optiq_seed0_100k \
  --seed 0 --steps 100000 --n 16 --m 64 --temperature 1

# Navigation: 5K random warmup followed by 100K actor updates.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m gmm40.run --method optiq --navigation --name navigation_optiq_seed0_100k \
  --seed 0 --steps 100000 --warmup 5000 --temperature .25
```

Use a fresh name for every run. On managed instances, launch long runs through
supervisor. `python -m gmm40.run --help` lists explicit experimental overrides,
including N/M, Sinkhorn iterations, NLL row selection and sigma row balancing.
These options are disabled by default. Fixed-Q T other than 1 changes the
target to a tempered density; navigation learns its critic separately.

Results, target metadata, source hashes, checkpoints, stochastic-policy samples
and evaluations are written under `gmm40-results/`. Reference target samples
are used for evaluation only. Physical actions are scaled by 40; the primary
density target is conditioned on the box (-40, 40)^2.

`dispatch.py` is the original instance-specific campaign launcher. It expects
a local `gmm40-results/queue.json`, `/root` virtual environments and CPU affinity
assignments; use `gmm40.run` directly on other machines. Historical diagnostic
and comparison scripts require the named local checkpoints/results in their
source. Those large artifacts are not distributed in Git.

## Checks

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_v5.py tests/test_distributional_distillation.py
```

These check the underlying v5 configuration, update semantics and NLL. Short
adapter smoke runs check execution and checkpoint restoration, not convergence.
