# Baseline implementations

## Shared JAX SAC

`baselines/jax_sac.py:JaxSAC` is the implementation for new Fetch and Adroit SAC
runs. Actor inference, twin-critic updates, Adam, automatic entropy adjustment,
and Polyak targets run in JAX. SB3 supplies the reference initialization and
collector/replay interfaces. Adroit additionally supports critic LayerNorm.

Automatic entropy adjustment is the baseline. The explicit fixed-alpha=0
ablation disables its optimizer while retaining stochastic policy collection;
the zero coefficient uses a static mode and finite checkpoint placeholder,
without evaluating log(0). Restore a training run from its JAX state sidecar,
not just the SB3 inference zip.

`rebuttal_tools/check_fetch_jax_sac.py` compares aligned noise/batches, losses,
gradients, optimizer moments, target updates and serialization against SB3.
It also supports the existing Adroit environment shapes. Recorded validation
and frozen campaign sources are indexed in
`rebuttal_tools/campaign_snapshots/20261112/README.md`.

## CrossQ comparison

Two runs per task and seed, under the same protocol as the main iBOLT results
(Gymnasium v4, 1M interactions, 5K random warmup, evaluation every 5K steps with
10 episodes using the paired reset seeds of `ibolt/dual_evaluation.py`).

1. **CrossQ (Gaussian policy), reference implementation** — `baselines/crossq_sbx.py`
   runs `sbx.CrossQ` with its defaults (BatchRenorm critic and actor, no target
   network, Adam lr 1e-3 / b1 0.5, policy delay 3, automatic entropy, UTD 1).
   Logged evaluations: `deterministic` (mean action) and `stochastic` (sampled).

   ```bash
   uv venv -p 3.11 ~/sbx-venv
   VIRTUAL_ENV=~/sbx-venv uv pip install sbx-rl "gymnasium[mujoco]==0.29.1" mujoco==2.3.7 "jax[cuda12]"
   ~/sbx-venv/bin/python baselines/crossq_sbx.py --task ant --seed 0 --critic-width 1024
   ```

2. **iBOLT with the CrossQ critic** — the same critic recipe inside the iBOLT
   learner (`+critic=crossq`): BatchRenorm critic, joint (s,a)/(s',a') forward pass,
   no target network, critic Adam lr 1e-3 / b1 0.5. The iBOLT actor, its optimizer,
   temperature, N=M=64, DACER exploration and the plain TD backup are unchanged.
   Logged evaluations: `zero_z`, `stochastic_z`, `stochastic_z_sigma`.

   ```bash
   python train.py benchmark=ant +critic=crossq "alg.critic.hs=[1024,1024]" seed=0
   ```

Both use the same critic width (1024x2 by default here, the SBX default; the
CrossQ paper uses 2048x2, available via `--critic-width 2048` and
`alg.critic.hs=[2048,2048]`).
