# v3 collection exploration

Version: `v3-exploration`, based on v3 `b41b024`. Config:
`mujoco_v3_exploration` / `configs/v3/exploration.yaml`.

## Behavior and learning

After the unchanged 5K random-action warmup:

```python
a_base = sample_current_actor(s)                 # original rollout RNG
a_exec = clip(a_base + 0.1 * alpha * Normal(0,I), -1, 1)
env.step(unscale(a_exec))
replay.add(s, a_exec, reward, s_next, done)        # actual executed action
```

Only `CollectionExplorationOptiQDIME._sample_action` adds the noise.
Evaluation, current-actor next actions, twin-min ordinary TD, mean-Q teacher,
full beta=1 proposal correction, 16x64 OT and conditional Gaussian NLL retain
the original functions. There is no policy entropy reward, entropy gradient
into actor/critic, acceptance guard, rollback or added target smoothing.
The learner changes indirectly through the collected transitions.

## Alpha controller

The fixed variant holds alpha=1.5, giving external noise standard deviation
0.15 in normalized action coordinates. Adaptive variants start at the same
alpha and use target entropy per action dimension -0.9, -0.5 or 0.0.

Before the first collection action after each 10K completed learner updates:

```python
states = sample_without_replacement(last_10000_collected_states, 256)
for s in states:
    actions = clip(sample_actor(s, 200) + 0.1*alpha*Normal(0,I), -1, 1)
    weights, means, covs = fit_full_covariance_GMM(actions, K=3)
    h[s] = -sum(weights*log(weights)) + sum(weights*GaussianEntropy(covs))
H_estimate = mean(h)
H_target = target_entropy_per_dim * action_dim
log_alpha = Adam(log_alpha, gradient=H_estimate-H_target, lr=.03)
```

For Ant's 8 actions the targets are -7.2, -4.0 and 0.0. Target entropy is a
fixed hyperparameter; current behavior entropy is estimated. Adam uses
beta1=.9, beta2=.999 and epsilon=1e-8; only its scalar log-alpha state is
updated. Momentum can delay reversal when the entropy error changes sign.
The first default controller update is after 10K learner updates, at env
step 15K before the next collection action. There are 99 alpha updates in a
full 1M run. The fixed variant never estimates entropy or updates alpha.

GMM settings: sklearn 1.7.2, full covariance, K=3, 200 samples/state,
reg_covar=1e-6, random_state=42, n_init=1, max_iter=100, tol=1e-3. Fits are
separate per state. The formula is the DACER proxy H(A,Z), an upper bound
on fitted-mixture differential entropy, not exact H(A). With clipped actions
it is a smooth proxy for a distribution that can also have boundary atoms.
Convergence rate and clipping fraction are logged. The current collection
state window, not the full replay buffer, controls alpha in this adaptation.
State selection, entropy sampling and executed noise have independent RNGs;
none advance the actor rollout, learner or global replay RNG streams.

The scalar controller, diagnostics and RNG states are saved beside each
actor/critic checkpoint and included in the final W&B artifact. As in base
v3, this is not an exact training-resume checkpoint (no replay/environment).

## Automatic campaign

The predecessor is the 20-run Ant-v4 v3 temperature campaign with temperatures
0.05, 0.1, 0.25, 0.5, 1.0 and seeds 0,1,2,3. Selection waits for **all 20**
to complete 1M successfully. Failed, partial or missing runs do not enter
selection and do not trigger a fallback to intermediate scores.

For each seed, average the 10 evaluation episodes at each of the 21 evaluation
points from 900K through 1M inclusive. Average those seed scores equally over
the four seeds. Select the temperature with the highest average, breaking an
exact tie by lower temperature. Freeze this one value for all 16 follow-up
runs. Retain selection inputs, scores and artifact hashes.

Follow-up order per GPU/seed: fixed std .15, adaptive H/d=-.9, adaptive H/d=-.5,
adaptive H/d=0. Each run starts fresh and runs 1M, without performance-based
early stopping. Keep the existing baseline queue/source/dependencies intact.
W&B project `OptiQ/v3_test`, separate timestamped exploration group. Four
workers maximum, one seed per GPU; a supervisor controller waits and launches.

Install the three exploration dependencies into an isolated campaign `deps/`
directory using `requirements-exploration.in` and `--no-deps`; set that path in
the follow-up worker PYTHONPATH. Do not upgrade the running baseline's venv.

## References

- [DACER paper, Section 4.3 and Table 3](https://papers.neurips.cc/paper_files/paper/2024/file/6174c67b136621f3f2e4a6b1d3286f6b-Paper-Conference.pdf)
- [Official DACER implementation](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL/blob/9f22f29fa91b1bed8ee07177a51598d6bd413cb5/relax/algorithm/dacer.py)

DACER's official `get_action` also adds noise for target and actor-loss actions.
This experiment intentionally adopts its entropy regulator for collection
only. Initial alpha=1.5 and recent-state entropy estimation are our choices;
this is not a full DACER reproduction.
