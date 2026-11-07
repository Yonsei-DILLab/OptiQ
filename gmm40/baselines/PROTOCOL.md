# GMM40 bandit baselines

These adapters use the target in `gmm40/target.json`, a constant zero observation,
normalized actions in `[-1,1]^2`, and reward `log p_GMM40(40*a)`. Every transition
terminates. Training uses oracle log-density values, never target samples.

For MEow, entropy coefficient 1 makes the optimal bandit policy proportional to
the bounded target density. The official MEow energy-based flow and its two
reward-shifting heads are retained. The squared TD target reduces exactly to the
observed reward for terminal transitions. This is a one-step adaptation of MEow,
not a replacement of its objective with a generic reverse-KL flow loss.

TD3 retains the Stable-Baselines3 implementation, twin critics, delayed actor
updates, and target-policy smoothing. Its entropy coefficient is zero. Its
deterministic actor optimizes reward and is not a Boltzmann density sampler.
Primary evaluation repeats the deterministic action at the fixed observation;
exploration noise is used during collection only. Do not label noisy exploration
actions as samples from a learned stochastic TD3 policy.

| Setting | MEow | TD3 |
|---|---|---|
| Actor optimizer updates | 100,000 unified flow updates | 100,000 actor updates |
| Critic optimizer updates | same unified optimizer | 200,000 |
| Batch / learning rate | 256 / 3e-4 | 256 / 3e-4 |
| Warmup / replay capacity | 1,000 / 1,000,000 | 1,000 / 1,000,000 |
| Collected transitions per optimizer step | 1 per flow update | 1 per critic update |
| Discount / entropy coefficient | 0 / 1 | 0 / 0 |
| Architecture | official toy MEow flow | SB3 TD3 default (400, 300), ReLU |
| Gradient clipping | official MEow norm 30 | SB3 TD3 default |
| Evaluation | stochastic flow samples | deterministic actor |

MEow retains its own prior log-scale bounds `[-2,1]`, flow dropout, layer
normalization and initializers. The iBOLT conditional-Gaussian log-scale bounds
and proposal floor do not apply to these baselines. Source provenance and both
upstream MIT notices are in `meow/UPSTREAM.json` and the accompanying licenses.

Budgets are explicitly counted as actor updates. Equal actor-update counts do
not imply equal oracle-query counts or compute: these replay-based adapters query
the oracle once per collected transition. The logs include actual training
`Q_evaluations`, which exclude evaluation queries. At the default budget these
are 101,000 for MEow and 201,000 for TD3.

Use seeds 0 through 4, 10,000 policy evaluation samples, the original 2,048-sample
five-bandwidth MMD, and the existing coverage and sliced-Wasserstein definitions.
The reference seed is 20260917 and policy evaluation seed is `900000+seed`.
Evaluation preserves training RNG state. Only full-policy metrics are reported;
there is no Gaussian mean-only counterpart for a flow or a deterministic TD3
actor. Checkpoints support evaluation; this runner does not provide replay-buffer
training resume.

```bash
python -m gmm40.baselines.run --baseline meow --seed 0 --device cuda --output outputs/gmm40-meow-s0
python -m gmm40.baselines.run --baseline td3 --seed 0 --device cuda --output outputs/gmm40-td3-s0
```

The runner refuses to overwrite existing output directories. Short runs using
`--steps`, `--warmup`, or `--eval-samples` are execution checks, not the full
performance benchmark. These adapters do not change the iBOLT learner or its
evaluation functions.
