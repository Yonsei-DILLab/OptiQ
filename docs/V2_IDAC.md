# OptiQ v2: semi-implicit policy, IDAC entropy, OT distillation

Branch `v2` starts from `mujoco-setting` commit `48bd296`. The common
`run_optiq_dime.py` → `OptiQDIME` → SB3 replay/evaluation/checkpoint path is
retained. The supplied specification is preserved in `V2_IDAC_PSEUDOCODE.txt`.

## Three distributions

- **Actual policy:** draw independent z and eps, evaluate the actor once to get
  mu and log_std, and execute tanh(mu + exp(log_std) * eps). Gaussian components
  are conditional on z; their exact marginal density is not assumed available.
- **Entropy estimate:** draw M independent conditional components at the next
  state, generate one action from component zero, and include that component in
  the log-mean-density calculation. Sum action dimensions before logsumexp over
  components. Retain pre-tanh values and subtract the stable tanh log Jacobian.
- **Teacher proposal:** build a separate equal-weight Gaussian KDE from the
  actual 16 student pre-tanh samples (not their mu values). Sample 64 fresh
  candidates, apply tanh, and use the transformed KDE density in the IS weights.
  No student anchors or hard proposal cutoff are added.

The entropy calculation uses normalized action coordinates, the same coordinates
as replay actions and Q inputs. Humanoid's physical action range is [-0.4,0.4];
SB3 unscales the selected behavior action when calling the environment.

## Updates

Critic: the **current actor** supplies one next action and a self-inclusive
log-density estimate. Each target critic evaluates that action once. The target
is `r + gamma*(1-terminal)*(min(Q1_target,Q2_target) - T*log_g)`.
No TD smoothing or teacher noise is added. TimeLimit truncations continue to
bootstrap through the existing SB3 replay handling. The categorical path retains
its original projection/loss and shifts its support by the same entropy term;
the Humanoid experiment uses scalar twin critics.

Actor: weights are `softmax(Q_aggregate/T - log_q_teacher)` with beta=1.
Squared distances are measured in normalized action space. The v2 config uses
the raw squared distance from the supplied pseudocode; the old mean-normalized
cost remains available to legacy configurations. Sinkhorn row argmax targets
are stopped, then pointwise squared action errors train mu and sigma through the
**same z/eps realization**. No actor Q-gradient or extra entropy loss is added.

The backup entropy coefficient and teacher Boltzmann temperature share one
configuration field. Validation rejects coefficient mismatches, teacher anchors,
non-unit beta, and TD smoothing on this path.

## Preserved experimental protocol

Humanoid-v4, seeds 0/1/2/3, 1M environment steps, 5k random warmup, batch 256,
replay capacity 1M, UTD 1, gamma .99, actor/critic hidden layers 256x3 GELU,
Adam 3e-4 with betas (.9,.999), separate actor/critic global gradient clipping 2,
critic Polyak .005, policy update every step, no batch/layer norm or dropout.
Evaluation at the start and each 5k steps, 10 stochastic episodes, diagnostics
every 5k, checkpoints every 50k and final. Actor and critic checkpoints contain
their optimizer states; the common path does not save the complete replay/RNG
state required for bit-for-bit training continuation.

There is no extra uniform exploration after warmup (`behavior_uniform_probability=0`).
The supplied semi-implicit policy is the rollout/evaluation policy. The existing
optional collection hook remains available for legacy experiments but is disabled
in both the v2 pilots and production. Initial behavior=.1 pilots are retained as
superseded records and excluded from the v2 temperature selection.

## Hyperparameter calibration

The new policy is calibrated independently of the old action-space KDE. Starting
the learned log-std head with random state-dependent outputs saturated its clamps
on some unnormalized Humanoid observations. It now starts with zero kernel and
constant log(.5) bias; the mu output uses variance-scaling initialization 1e-4.
Both heads remain fully trainable. Log-std clamp bounds are explicit in the config.

`scripts/calibrate_v2.py` compares initial std {.3,.5,1}, pre-tanh bandwidth
{.2,.4,.6,.8,1}, and raw-cost Sinkhorn epsilon {.05,.1,.25} on 128 Humanoid states.
With controlled initialization and initial std .5, h=.2 yielded ESS 2.98/64,
while h=.8 yielded 18.36/64. Policy entropy was approximately 8.4, and no student
coordinate exceeded abs(action)=.99 in this sample. Epsilon .25 and 100 iterations
gave small transport marginal errors. This establishes initial numerical behavior,
not long-run return or universal hyperparameter optimality.

Four independent 20k-step pilot runs compare T {.1,.25,.5,1} from seed0 and fresh
initializations. They retain full training dimensions and warmup, use three eval
episodes and more frequent diagnostics, and log to a separate calibration project.
`scripts/analyze_v2_calibration.py` reports ESS, entropy contribution, sigma,
evaluation trajectories and paired M=1/4/16/64 entropy estimates from final actors.
Final production settings and the selection evidence are recorded in
`V2_CALIBRATION_RESULTS.md` after calibration, before launching production.

## Launch and records

```bash
scripts/run_v2.sh --list
OPTIQ_CONFIG=mujoco_v2 scripts/run_v2.sh 0 --check
OPTIQ_CONFIG=mujoco_v2 scripts/run_v2.sh 0
# The same script accepts seeds 1,2,3 and explicit Hydra overrides.
```

Credentials are read using the existing ignored env file (`OPTIQ_ENV_FILE`).
Production W&B: `OptiQ/optiq_mujoco_v2_scalar_h256x3_no_anchor_idac_4seed_1m`.
Calibration W&B: `OptiQ/optiq_mujoco_v2_calibration`.
Supervisor config: `deploy/supervisor/optiq-v2.conf`; production uses one worker
per GPU. All run directories receive resolved config, runtime provenance, CSV,
TensorBoard, evaluation arrays and checkpoints. Production launch records live
under `outputs/v2_validation`.

## Validation and interpretation

`tests/test_semi_implicit.py` checks joint-mixture/Jacobian math against SciPy,
state isolation, self inclusion, both proposal modes, nontruncated Gaussian
sampling, soft TD terminal masking, absence of TD noise, both-head gradients,
no actor gradient through Q, and actual Humanoid collection/training/checkpoints.
Legacy scalar, behavior-exploration and gradient-clipping regressions are retained.

Entropy is a lower bound **in expectation** over sampled components and actions;
individual log-density samples need not be bounds. Finite-M bias is recorded
separately from teacher KDE entropy. This implementation does not assert exact
Boltzmann samples or monotonic policy improvement under finite OT/argmax/MSE.
See the supplied specification for IDAC references and the distinction between
the entropy estimator and the original IDAC actor-gradient algorithm.
