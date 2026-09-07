The Code for the DIME paper submission at ICML2025.

## Scalar-critic ablation (`mujoco-setting`)

This branch starts from `critic-dime-no-anchor` and makes the default entry
Humanoid-v4 with **256x3 scalar twin critics and TD MSE**. No-anchor sampling,
actor learning are preserved. Critics use GELU, standard Adam betas (0.9,0.999),
target-network min-Q backup and Polyak=0.005. Batch renorm is removed;
there is one update per environment step (UTD=1).
All five MuJoCo tasks use fixed beta=1 and seeds0,1,2 (15 runs). See
[MUJOCO-README.md](MUJOCO-README.md) for the protocol and launch commands,
and [critic implementation details](docs/MUJOCO_SCALAR_SETTING.md).

```bash
python run_optiq_dime.py --config-name=mujoco_setting benchmark=humanoid
python run_optiq_dime.py --config-name=mujoco_setting benchmark=ant
```

## No-anchor OptiQ with the DIME critic (`critic-dime-no-anchor`)

The retained `optiq_dime_no_anchor` experiment is `myoHandPenTwirlRandom-v0` (pen-twirl-hard),
with Ant-v4 and Humanoid-v4 selectable through `benchmark=ant` / `benchmark=humanoid`.
MyoHand reach-hard and object-hold-hard are also available as
`benchmark=reach_hard` / `benchmark=obj_hold_hard`.
The actor defines a truncated Gaussian KDE from 16 policy samples, draws four
random candidates per component, and distills a **16 × 64** transport plan.
Candidate anchors are disabled; fixed density beta is 0.1. The launcher defaults
to **seeds 0, 1, 2**.

Use the separate environment and [baseline setup and protocol](docs/NO_ANCHOR_BASELINE.md).
Credentials belong in an ignored `.env`; a template is provided in [.env.example](.env.example).

```bash
bash scripts/setup_no_anchor_env.sh
bash scripts/run_no_anchor.sh --list
```

## Ant-v4 / Humanoid-v4 fixed-beta experiments (`heechan`)

The retained `heechan` configuration provides a JAX OptiQ+DIME path for Ant-v4 and Humanoid-v4,
comparing stratified and IID KDE-mixture proposals at fixed density beta
0.1, 0.25, 0.5, 0.75 and 1.0. See [setup, protocol and launch commands](docs/MUJOCO_BETA_SWEEP.md).
These legacy scripts select `--config-name=optiq_dime_mujoco` explicitly and keep
their original four-seed sweep. The Dog configuration below is also retained.
## DIME: Diffusion-Based Maximum Entropy Reinforcement Learning 

This repository accompanies the paper "[DIME: Diffusion-Based Maximum Entropy Reinforcement Learning](https://arxiv.org/pdf/2502.02316)" published at ICML 2025.

### Learning Curves are available in *paper_results* 
**Update:** We have uploaded all the learning curve data to our repository. You can find the data in the *paper_results* folder.
Additionally, we have run DIME on all remaining DMC enviroinments and added the results to the same folder. 

### Installation
The file setup.sh provides a convenient way to set up the conda environment and install the required packages automatically via
```bash
chmod +x setup.sh
./setup.sh
```

After installation is finished, the conda environment can be activated, and the code can be run using 

```python
python run_dime.py
```

### Running DIME

Specific parameters can be set in the terminal such as the learning environment using hydra's multirun function

```python
python run_dime.py --multirun env_name=dm_control/humanoid-run
```

Detailed hyperparameter specifications are available in the config directory.
The current config file is adapted to the hyperparameters used for DMC. If you want to run DIME on the gym environemtns, 
you only need to change the v_min and v_max parameters of the critic as specified in the appendix of the paper. We used the same values for both gym environments. 
For example if you would like to run DIME on gym's Humanoid-v3 environment, you can do so by running

```python
python run_dime.py env_name=Humanoid-v3  alg.critic.v_min=-1600 alg.critic.v_max=1600
```

### OptiQ actor with the DIME critic

`run_optiq_dime.py` keeps DIME's distributional CrossQ critic, replay buffer,
batch renormalization, and UTD=2 update loop, while replacing the 16-step
diffusion policy with OptiQ's one-step implicit actor and density-corrected OT
distillation. The legacy `optiq_dime_dog` configuration covers DMC Dog with the supplied OptiQ
actor settings (N=16, R=5, anchor, temperature 0.25, argmax assignment):

```bash
source /workspace/.venv-dime/bin/activate
python run_optiq_dime.py --config-name=optiq_dime_dog task=run seed=0
```

The supported tasks are `run`, `trot`, `walk`, and `stand`. The helper script
accepts a task, seed, and optional step count:

```bash
scripts/run_dog_experiment.sh trot 1 1000000
```


## Acknowledgements
Portions of the project are adapted from other repositories: 
- https://github.com/DenisBless/UnderdampedDiffusionBridges is licensed under MIT,
- https://github.com/adityab/CrossQ is licensed under MIT and is built upon code from "[Stable Baselines Jax](https://github.com/araffin/sbx/)"
