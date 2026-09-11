# Finite latent candidate: exact policy density, sampling/OT improvement

This is an experimental successor to the completed continuous-latent v2
confirmation. It has not beaten OptiQ in Humanoid. Select it explicitly with
`OPTIQ_CONFIG=mujoco_v2_finite`; existing v2 configs retain their original paths.

The actual latent prior is uniform on 16 fixed code vectors. The codebook is
generated with seed 20260911, then centered and standardized per coordinate.
It is shared across training seeds. The actor remains one network conditioned
on observation and latent, with conditional mean and log-standard-deviation
heads. A mean residual `0.5*z` supplies initial component diversity; the network
can subsequently change or cancel that residual. Conditional sigma starts at
0.2 and has upper bound 0.2, lower bound exp(-5).

For a state, the actual policy is

`pi(a|s) = (1/16) sum_i tanh-Normal(a; mu(s,z_i), sigma(s,z_i))`.

Rollout chooses one code uniformly, evaluates the network once, adds Gaussian
noise, then applies tanh. Stochastic evaluation uses this same policy with no
extra behavior exploration. Deterministic prediction selects code 0 and zero
Gaussian noise; it is not the mean of the mixture. Normalized actions are
unscaled through the existing environment path.

TD backup enumerates all 16 actual components, samples one action, and computes
its full joint mixture density with the pre-tanh value and tanh Jacobian.
It uses the current actor and the minimum target Q as before. The density at
that action is directly evaluable; the entropy expectation remains sampled.
The compatibility log names `backup_entropy_lower` and `policy_entropy_lower`
now contain sampled exact-density entropy for this variant, distinguished by
`backup_policy_density_exact=1` and `policy_density_exact=1`.

OT uses one student realization per actual component with equal 1/16 mass.
The teacher is the full conditional mixture with teacher-only sigma floor .05,
from which 64 new candidates are drawn. No student anchors are added. Full
importance correction beta=1, temperature .1, mean live twin Q, raw squared
action cost, Sinkhorn epsilon .25/100 iterations and conditional OT NLL are
retained. Teacher, Q and transport are stopped in the actor loss: there is no
Q-action gradient or additional entropy-gradient objective. The latent-mean
share of pre-tanh variance is logged to detect loss of component diversity.

The sampled soft-score guard uses all 16 actual component densities for both
policies. Thus its old upper/new lower density expressions coincide with their
respective actual policy log densities; the finite random-mixture bracket gap
is removed. It still uses 32 replay states, 8 draws and zero SE multiplier.
Rejection restores the full actor TrainState including optimizer state.

**Theory boundary:** exact density does not make OT/NLL projection automatically
improving. Exact old-policy soft Q and a nonnegative true soft-score change in
every relevant state imply the standard ideal soft-policy improvement result.
The actual learned Q, finite action samples and replay-average guard do not
certify that statewise condition. Raw undiscounted return also need not increase
monotonically. See `V2_SOFT_POLICY_IMPROVEMENT.md` for the explicit conditions.
Uniform finite component weights also restrict the representable policy family.

The new TrainState's component count/codebook seed are static metadata restored
from the saved run config. Preserve that config alongside the checkpoint.
Old continuous-policy checkpoints retain the original TrainState type and RNG
path. Neither checkpoint format is an exact full-replay/all-RNG resume mechanism.

## Selection evidence and intended comparison

The preceding exact-target and entropy-overlap probes are documented in
`V2_PROJECTION_DIAGNOSIS_KO.md`. With the finite prior and initial residual scale
1, a 1D bimodal probe obtained final W2-squared .00684/.00885/.00482 across three
seeds with sigma capped at .2; allowing sigma to expand again lost the modes.
These are synthetic projection results, not Humanoid returns.

Production-size GPU preflights used seed 0, 5k ordinary warmup and 1k updates:

| Change from finite16 / residual 1 / 64 candidates | Guard acceptance | Last logged ESS | Last sampled policy H |
|---|---:|---:|---:|
| None | .9% | 2.68/64 | −15.83 |
| 256 teacher candidates | 1.1% | 3.15/256 | −15.76 |
| Minimum-Q teacher | 1.0% | 2.70/64 | −16.02 |
| Initial residual scale .5 | 19.7% | 3.22/64 | −6.13 |

The residual-.5 variant was selected for a longer check. These very short runs
do not establish comparative return performance. Temperature stays .1: in the
first preflight changing the counterfactual teacher temperature .05–1 barely
changed ESS, while density-only ESS was also small. Raising temperature alone
does not address that observed concentration. No beta or gradient ablation is
added. The selected variant's exact-target check is separately saved as
`bimodal_finite_selected_projection.json`.

The longer comparison uses Humanoid-v4 seeds 0–3, at most 1M steps, 10 stochastic
evaluation episodes every 5k, and the same common optimizer/critic/replay
settings as the completed v2. The finite screen monitor compares each seed with
both the completed continuous v2 and matched OptiQ at identical steps, using
the latest five evaluations. It stops only if below half of **both** reference
means with nonpositive recent slope after 100k, or below 60% of both after 200k.
These are declared compute-budget heuristics, not statistical significance
tests. A matching periodic actor checkpoint is required before SIGINT; records
and prior checkpoints are preserved. Other services are not controlled.

W&B project: `OptiQ/optiq_mujoco_v2_finite_screen`.
Launcher: `scripts/supervisor_v2_finite.sh`; monitor:
`scripts/monitor_v2_finite.py`; service template:
`deploy/supervisor/optiq-v2-finite.conf`.

## Adaptive budget decision after the 200k review

At 200k, the five-evaluation four-seed mean is 1256.7, compared with 2473.1
for continuous v2 and 2282.5 for historical OptiQ. The original per-seed
stop rule does not trigger because the matched zero-exploration control is
weaker (864.7). Beating that control alone is not the goal. Seed 2 has just
risen to 3208 at the single 200k evaluation, so allow a bounded extra 50k.

At 250k, compare all four seeds over 230k/235k/240k/245k/250k. If the candidate
mean is below 70% of BOTH continuous-v2 and historical-OptiQ means, stop the
entire candidate group after all 250k actor/critic checkpoints are present.
This is an **adaptive compute-budget decision made after observing 200k**.
It is not described as a launch-time rule or as evidence about unseen 1M
performance. The original protocol and monitor history are preserved in
`finite_screen_protocol_before_200k_adaptation.json` and
`finite_screen_monitor_before_strong_gate.json`. Training parameters and
primary/final evaluation protocols remain unchanged.
