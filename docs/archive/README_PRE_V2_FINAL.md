The Code for the DIME paper submission at ICML2025.

## OptiQ v2: IDAC entropy with an OT actor

Branch `v2` adds a one-step tanh-Gaussian semi-implicit actor and self-inclusive
IDAC entropy estimates for soft TD targets. The current Humanoid confirmation
candidate is **`mujoco_v2_checked`**: fixed T=0.1, a learned conditional-Gaussian
mixture teacher with standard-deviation floor 0.05, full-OT conditional NLL,
and a sampled soft-value update check. It uses scalar 256x3 twin critics,
16x64 OT, beta=1, M=16 entropy components, no teacher anchors and no extra uniform
behavior exploration. All four paired seeds (0/1/2/3) completed 1M. The fixed
900k–1M mean is 5363 versus 5052 for the matched OptiQ control, but below the
stronger historical 10%-exploration reference (5676). Fresh final-policy
evaluations average 5397 for v2, 5224 for the matched control and 5633 for the
historical reference. See the [full results](../v2/RESULTS_KO.md).

See the [investigation and confirmation protocol](v2/V2_IMPROVEMENT.md),
[current algorithm explanation in Korean](../v2/ALGORITHM_KO.md),
[precise theorem assumptions](../v2/THEORY.md), and
[W&B confirmation project](https://wandb.ai/OptiQ/optiq_mujoco_v2_confirmation).
Neither OT NLL nor the sampled replay-average check alone certifies policy
improvement. To explicitly inspect or run this candidate:

```bash
scripts/run_v2.sh --list
OPTIQ_CONFIG=mujoco_v2_checked scripts/run_v2.sh 0 --check
OPTIQ_CONFIG=mujoco_v2_checked scripts/run_v2.sh 0
```

The original `mujoco_v2` config remains reproducible with T=0.5, realized-action
pre-tanh KDE bandwidth 0.8 and pointwise MSE. It is still the bare launcher's
default for historical reproduction; explicitly select the checked candidate
with `OPTIQ_CONFIG=mujoco_v2_checked`. Its original runs
were stopped for poor performance; see [original design](v2/V2_IDAC.md) and
[initial calibration](v2/V2_CALIBRATION_RESULTS.md) for historical details.

## Scalar-critic ablation (`mujoco-setting`)

This branch starts from `critic-dime-no-anchor` and makes the default entry
Humanoid-v4 with **256x3 scalar twin critics and TD MSE**. The `mujoco_setting`
configuration includes candidate anchors and retains pointwise-MSE actor
distillation. Critics use GELU, standard Adam betas (0.9,0.999),
target-network min-Q backup and Polyak=0.005. Batch renorm is removed;
there is one update per environment step (UTD=1).
All five MuJoCo tasks use fixed beta=1 and seeds0,1,2 (15 runs). See
[MUJOCO-README.md](../../MUJOCO-README.md) for the protocol and launch commands,
and [critic implementation details](../MUJOCO_SCALAR_SETTING.md).

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

Use the separate environment and [baseline setup and protocol](../NO_ANCHOR_BASELINE.md).
Credentials belong in an ignored `.env`; a template is provided in [.env.example](../../.env.example).

```bash
bash scripts/setup_no_anchor_env.sh
bash scripts/run_no_anchor.sh --list
```

## Ant-v4 / Humanoid-v4 fixed-beta experiments (`heechan`)

The retained `heechan` configuration provides a JAX OptiQ+DIME path for Ant-v4 and Humanoid-v4,
comparing stratified and IID KDE-mixture proposals at fixed density beta
0.1, 0.25, 0.5, 0.75 and 1.0. See [setup, protocol and launch commands](../MUJOCO_BETA_SWEEP.md).
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
