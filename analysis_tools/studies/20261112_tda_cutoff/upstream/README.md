# iBOLT vs SAC mode counting (basin-graph H0 persistence) — reproduction bundle

The bundle has three zips. Only the first one is needed to reproduce the reported tables and to change the
1% peak cutoff.

| zip | contents | needed for |
|---|---|---|
| `tda_repro_core.zip` (this one) | code, evaluation states, per-state/per-mode results (`bp32k/*.pkl`), SAC mu/sigma on the same states, logs, tables | re-scoring K(tau) with a different cutoff, re-running SAC from scratch |
| `tda_repro_mixtures.zip` | iBOLT mixture parameters at N=32768 on the evaluation states (`data/*_n32768.npz`, ~0.9 GB) | re-running `basin_ph.py` for iBOLT from scratch (no checkpoint needed) |
| `tda_checkpoints.zip` | 1M-step checkpoints: iBOLT Hopper/Humanoid s0-4 (actor, critic, config), SAC Hopper s0-4, SAC Humanoid s0-1 | regenerating states / mixtures with `extract.py` / `sac_ph_extract.py` |

SAC Humanoid seeds 2-4 were still training when this bundle was made. The Humanoid SAC row therefore uses 2 seeds
(s0, s1); all other rows use 5 seeds.

## 1. Changing only the 1% peak cutoff (no GPU, seconds)

The cutoff is **not** applied inside `basin_ph.py`'s saved results. Every `bp32k/*.pkl` stores, for every state:

- `pairs[:, 0]`: birth = log p(peak), for every non-global maximum
- `pairs[:, 1]`: death = log p(saddle)
- `pers`: birth − death
- `fmax`: log p of the global maximum
- `births`, `basin_mass`, `n_components`, `tomato`

K(tau) is computed at table time in `cmp5_table.py`:

```python
HF = np.log(100.0)        # peak must be >= 1% of the global max  (birth >= fmax - log 100)
K(tau) = 1 + #{pers >= tau and birth >= fmax - HF} + (n_components - 1)
```

To use another cutoff c, set `HF = np.log(1 / c)`; for example c = 0.1% gives `HF = np.log(1000)`. To disable the
cutoff, use `HF = np.inf`. Then run:

```bash
cd tda_repro_core
python cmp5_table.py      # needs numpy + matplotlib; writes figs/cmp5_summary.txt, figs/cmp5_Ktau.png
```

With the unchanged script, `figs/cmp5_summary.txt` is reproduced exactly; the shipped copy is
`figs/cmp5_summary_reference.txt`.

The ToMATo metrics (`K5`, `M_eff`) are stored per tau in `r["tomato"]` and do **not** use the peak cutoff. They
merge basins whose prominence is below tau, then measure the share of the 32,768 policy samples.

## 2. What each result file is

| path | meaning |
|---|---|
| `bp32k/ibolt_{hopper,humanoid}_s{i}.pkl` | iBOLT seed i, on its own evaluation states (Hopper 64, Humanoid 32), mixture N=32768, M=32768 samples, k=15 |
| `bp32k/sac1m{j}_{env}_s{i}.pkl` | SAC seed j (1M) on iBOLT seed i's evaluation states |
| `ph_sac/{env}_sac1m{j}_on_s{i}.npz` | SAC pre-tanh `mu`, `sigma` on those states (input to `basin_ph.py sac`) |
| `ph_sac/exact_*.txt` | closed-form SAC table (`sac_exact.py`); this agrees with the graph estimate |
| `states/{env}_s{i}_states.npz` | the evaluation states actually used (`states`, `src_idx` into the 512-state pool) |
| `data/hopper_s{i}.npz` | original 512-state pool per Hopper seed (plus the N=1024 mixture); Hopper uses rows `0, 8, ..., 504` (`data/hopper_idx_stride8.txt`) |
| `logs_cmp/*.log` | per-run logs, with timing and per-state progress |
| `figs/cmp32k_summary.txt` | earlier matched table (SAC 2 seeds); `figs/cmp5_summary_reference.txt` is the 5-seed table |

`res` in each pkl is a list with one dict per state:

| key | meaning |
|---|---|
| `state` | index |
| `pairs` | (birth, death) |
| `pers` | persistence |
| `fmax` | log p of the global maximum |
| `births` | log p of every verified mode |
| `basin_mass` | share of policy samples whose ascent reaches each mode |
| `n_modes` | number of verified modes |
| `n_components` | graph components; always 1 in practice |
| `far` | samples whose ascent ended more than 1e-2 from a verified mode |
| `tomato[tau]` | dict with keys `q`, `K`, `K5`, `meff` |

## 3. Pipeline (how the pkls were produced)

1. **States** (`extract.py`, JAX, CPU):
   - Roll out the iBOLT zero-z evaluation policy with `env.reset(seed=10000+i)`.
   - Pool the visited states and subsample 512 of them uniformly (rng seed 0).
   - Hopper uses every 8th state (64). Humanoid uses 32 states.
2. **iBOLT mixture**: `extract.py --n 32768`, with z ~ N(0, I) from `PRNGKey(12345)` (the same z for all states).
   - Component i is a box-truncated diagonal Gaussian with center `tanh(mu(s, z_i))`.
   - Its `log sigma` is clipped to [-5, -1].
   - Weights are uniform 1/N.
3. **SAC** (`sac_ph_extract.py`, sbx): pre-tanh `mu`, `sigma` of the actor on exactly the same states.
4. **`basin_ph.py`** (PyTorch, CUDA). See the docstring for the full description.
   - **Modes:**
     - iBOLT: projected Modal EM from all 32,768 policy samples (fp32, tol 1e-4), then dedup (L∞ 1e-3), fp64 polishing (1e-10), and a check that each point is a strict local max (KKT on the bounds plus negative-definite Hessian on the free coordinates).
     - SAC: exact product of the 1D modes.
   - **Basin label:** each sample gets the mode its ascent reaches.
   - **Graph:**
     - Nodes are the samples, the modes, and 8 points on each segment between the 6 heaviest modes.
     - Symmetric kNN with k=15.
     - Edge height = min(f_i, f_j, f(midpoint)), where f = log p is computed exactly.
   - **Saddle(A, B):** the max edge height over edges joining basin A and basin B.
   - **H0:** union-find over basins with the elder rule.
5. **`cmp5_table.py`**:
   - mean ± std over training seeds;
   - SAC seed j pooled over all 5 iBOLT state sets.

Shell drivers: `cmp32k.sh` (iBOLT N=32768 extraction + matched runs), `sac5_post.sh` (SAC seeds 2-4),
`sac1m_run.sh` (SAC seeds 0-1).

## 4. Re-running from scratch

```bash
# iBOLT (GPU; needs tda_repro_mixtures.zip unpacked into data/)
python basin_ph.py ibolt data/hopper_s0_n32768.npz bp32k/ibolt_hopper_s0.pkl --samples 32768
python basin_ph.py ibolt data/humanoid_s0_n32768.npz bp32k/ibolt_humanoid_s0.pkl --samples 32768
# SAC (GPU, ~30 s per file)
python basin_ph.py sac ph_sac/hopper_sac1m0_on_s0.npz bp32k/sac1m0_hopper_s0.pkl --samples 32768
# closed-form SAC table (CPU, scipy)
python sac_exact.py ph_sac/hopper_sac1m*_on_s*.npz
```

- `extract.py` and `sac_ph_extract.py` need the OptiQ repo (`train.py`, `ibolt/`) and sbx respectively, plus the
  checkpoints. They are only needed to regenerate states and mixtures.
- `basin_ph.py` uses a fixed sampling seed (`--seed 0`). Re-runs on the same GPU and software reproduce the pkls.
  Different hardware may change the last digits of the samples.
- Runtime on one RTX 5090: Hopper iBOLT about 8 min per seed (64 states); Humanoid about 3 min per seed (32 states);
  SAC about 30 s per file.
- `basin_ph.py` in this bundle has the streamed `dedup` (no n×n matrix). It gives results identical to the earlier
  all-pairs version (checked on Hopper s3 and Humanoid s0) and is required for M ≥ 262k.

## 5. Environment used

- GPU: RTX 5090, driver 615.71
- `basin_ph.py`: torch 2.11.0+cu128, numpy 2.4.6
- `extract.py` / tables: jax 0.4.33, numpy 1.26.4, matplotlib 3.8.4, scipy 1.11.4
- SAC: sbx 0.28.0, jax 0.10.2
- OptiQ commit 1c8d7a4 (with local modifications)

## 6. Known caveats

- **Humanoid spikes:**
  - iBOLT K(tau) on Humanoid grows with the sample count (M = 32k → 262k → 524k gives K(0.5) of 4.5 → 12.8 → 15.8 on 4 test states).
  - The extra maxima are single-component spikes with sigma at the floor and basin mass below 0.1%. Some of them exceed 1% of the global max, so the cutoff does not remove them.
  - The mass-based K5 and M_eff stay unchanged.
  - In 1 of the 4 test states, the global max itself is such a spike, with basin mass 1e-4.
- **SAC Humanoid seed 1:** its second modes sit at the tanh boundary (median |a| between 0.98 and 0.9997, for sigma > 1/√2). They are squashing artifacts, not distinct behaviours.
