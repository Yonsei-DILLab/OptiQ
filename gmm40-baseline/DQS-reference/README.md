## Sampling from Energy-based Policies using Diffusion

[![arXiv](https://img.shields.io/badge/arXiv-2410.01312-b31b1b.svg)](https://arxiv.org/abs/2410.01312)

JAX implementation of Diffusion Q-Sampling (DQS) from the RLC 2025 paper [Sampling from Energy-based Policies using Diffusion](https://arxiv.org/abs/2410.01312). Includes DeepMind suite tasks, a custom 2D Gaussian mixture environment, and sparse reward PointMaze.

### Installation

The recommended way to install the environment is using Conda:
```bash
git clone https://github.com/vineetjain96/Diffusion_Q_Sampling.git
cd Diffusion_Q_Sampling
conda env create -f environment.yaml
conda activate dqs
```

### Running experiments

The experiments from the paper can be reproduced with the provided config files:
- `configs/gmm.yaml` for the 2D GMM navigation task
- `configs/pointmaze.yaml` for PointMaze
- `configs/defaults.yaml` for the DeepMind suite tasks

```bash
python main.py --config configs/gmm.yaml          # 2D GMM navigation
python main.py --config configs/pointmaze.yaml    # PointMaze
python main.py --config configs/defaults.yaml     # DeepMind suite task
```

Optionally, you can override parameters using a custom config file (via the `--config` flag) or specify them as command line arguments. For example, if you 
want to run the experiment for PointMaze with a different temperature schedule, you can use the following command:
```bash
python main.py --env_name PointMaze --init_temperature 10.0 --final_temperature 1.0 --temperature_steps 250000
```

### Logging

The code uses Weights & Biases for logging. By default, logs and artifacts are written to `dqs_logs/` in the repository root. Change via `--wandb_dir`.
W&B logging is enabled when all are true:
  - `--debug=false` (or `debug: false` in config)
  - `--wandb_entity <entity>` is provided (or `WANDB_ENTITY` env var is set)

Setup example:
```bash
export WANDB_API_KEY=<your_wandb_api_key>
python main.py --config configs/gmm.yaml \
  --wandb_entity <your_team_or_username> --wandb_project dqs.gmm
```

If the entity is not provided or `--debug=true`, W&B is disabled and runs are logged locally only.

### Project Structure

- `main.py`: Entry point. Loads configs, applies CLI overrides, constructs env/agent/trainer, and launches training.
- `configs/`: YAML configs for defaults and per-environment overlays.
  - `defaults.yaml`: Global defaults, used for DMC experiments.
  - `gmm.yaml`, `pointmaze.yaml`: Environment-specific overlays.
- `src/trainer.py`: Training loop, evaluation, and logging hooks.
- `src/agent/`:
  - `dqs.py`: DQS agent and sampling logic.
  - `critic.py`: Q/energy function implementation.
- `src/components/`: Core modules for DQS internals.
  - `score_estimator.py`, `sde_integration.py`, `noise_schedule.py`, `prior.py`, `buffer.py`, etc.
- `src/envs/`: Environment factories and wrappers.
  - `gmm_env.py`: 2D GMM navigation.
  - `dmc_env.py`: DeepMind Control Suite wrapper.
  - `wrappers/`: Common wrappers (episode monitor, action repeat, precision, etc.).
- `src/utils/`: Utilities
  - `logger.py`, `plotting_utils.py`, `grad_utils.py`
- `environment.yaml`: Reproducible Conda environment.

### Results

![DMC with classical baselines](assets/dmc_classic.png)
![DMC with diffusion baselines](assets/dmc_diff.png)

### Citation

If you find this work useful, please cite the paper:
```bibtex
@article{jain2024sampling,
  title={Sampling from energy-based policies using diffusion},
  author={Jain, Vineet and Akhound-Sadegh, Tara and Ravanbakhsh, Siamak},
  journal={arXiv preprint arXiv:2410.01312},
  year={2024}
}
```
