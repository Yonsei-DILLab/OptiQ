# Launchers

`mujoco_5env_5seed.sh` maps one Slurm array of 25 tasks to five environments
and seeds 1 through 5:

| Array IDs | Environment |
| --- | --- |
| 0-4 | Hopper-v4 |
| 5-9 | Walker2d-v4 |
| 10-14 | HalfCheetah-v4 |
| 15-19 | Ant-v4 |
| 20-24 | Humanoid-v4 |

The launcher intentionally does not encode a site-specific partition, QOS, or
node exclusion list. Supply those options at submission time. This also makes
it possible to route Humanoid to faster GPUs without changing the experiment:

```bash
sbatch --array=0-19%8 --partition=<standard-gpu> launch/mujoco_5env_5seed.sh
sbatch --array=20-24%5 --partition=<fast-gpu> launch/mujoco_5env_5seed.sh
```

Use a dry run to inspect any task without starting training:

```bash
DRY_RUN=1 bash launch/mujoco_5env_5seed.sh 20
```

The default Conda environment is `flowrl-ogbench`. Override it with
`OPTIQ_CONDA_ENV`, and set `OPTIQ_ROOT` when launching from a relocated checkout.
W&B uses `optiq_mujoco5` by default; `WANDB_PROJECT` and `WANDB_ENTITY` override
the project and entity.
