# OptiQ v7: conditional Boltzmann extraction through latent OT

Canonical objective: `ot_conditional_sac`, restored at the user's request.
The goal is Boltzmann extraction through OT-assigned latent conditionals.
It is not a claim that the actor objective equals marginal SAC. The separate
full-mixture change is archived; see [CONDITIONAL_RESTORE_KO.md](CONDITIONAL_RESTORE_KO.md).
The Korean specification is [ALGORITHM_KO.md](ALGORITHM_KO.md), with original
notation in [NOTATION_KO.md](NOTATION_KO.md).

## Policy and sample counts

Each latent selects a diagonal Gaussian from the ordinary shared MLP. Its
continuous mixture and finite latent approximation are

\[
\pi_\theta(a\mid s)=\int p_0(z)\pi_\theta(a\mid s,z)dz,
\qquad \pi_{\theta,H}(a\mid s)=\frac1H\sum_i\pi_i(a\mid s).
\]

The finite mixture is useful for theory; the current actor loss evaluates the
selected conditional density, not log density of the whole H-bank mixture.

| Quantity | Default |
|---|---:|
| Fixed normal OT latent sites H | 4096 |
| Fresh teacher latents / original actions M | 256 / 256 |
| Teacher actions per Gaussian | 1, stratified across components |
| Importance-resampled teacher occurrences K | 16 |
| OT matrix / new actor actions per state | 4096 × 16 / 16 |
| Actor Gaussian outputs used for its loss | Selected 16 |
| Soft-TD fresh mixture components | 16, self-inclusive |
| OT epsilon / persistent dual updates | .1 / one per actor update |
| Dual MLP / Adam | ReLU 256×2→H / 1e-4 |
| Actor and critic Adam / UTD / train frequency | 3e-4 / 1 / 1 |
| T=alpha | RL .25 / fixed-Q GMM 1 |

Teacher actions are b_j with pre-tanh coordinates v_j. Normalized action-density
correction weights are W_j. P_ij is the coupling, with column sums 1/K.
R_ij=P_ij/sum_k(P_ik) is the row conditional over teacher j.
Pr(i|a,s) instead normalizes over source i. Do not conflate their directions.

H=4096 coordinates are fixed and used by OT and the assignment query. They do
not require 4096 actor Gaussian outputs in the current loss. Teacher latents,
resampling, actor source indices and Gaussian noise refresh every update.
Potential parameters and Adam state persist. Collection/stochastic-z evaluation
use fresh continuous normal latents; zero-z evaluation fixes z=0.

## One update

```python
# Once: fixed standard-normal latent integration coordinates.
H, M, K, L = 4096, 256, 16, 16
Z = normal(seed=ot_latent_seed, shape=(H, action_dim))

# RL only: current actor, fresh self-inclusive L-component density estimate.
Z_next = normal(shape=(batch, L, action_dim))
mu_next, log_sigma_next = actor(theta, s_next, Z_next)
u_next = mu_next[:, 0] + exp(log_sigma_next[:, 0]) * fresh_normal_noise()
a_next = tanh(u_next)
log_pi_next = (logmeanexp_over_L_components(gaussian_log_density(
    u_next, mu_next, log_sigma_next)) - tanh_log_jacobian(u_next))
target = stopgrad(reward + gamma * (1 - terminal) * (
    min(Q_target(s_next, a_next)) - T * log_pi_next))
update_critic_once(target)
polyak_update_target_critics()

# Teacher and actor share the configured current-Q aggregation.
Q_current = current_twin_mean  # source_q_eval=min is an explicit alternative
# Freeze critic parameters during actor backward, preserving action derivatives.

# Use the same pre-update f for source sampling, actor query and dual gradient.
f, dual_pullback = vjp(lambda phi: dual_network(phi, s), phi)

with stop_gradient():
    Z_teacher = normal(shape=(batch, M, action_dim))
    mu_old, log_sigma_old = actor(theta, s, Z_teacher)
    sigma_proposal = maximum(exp(log_sigma_old), 0.05)
    v = mu_old + sigma_proposal * fresh_normal_noise()  # one per Gaussian
    b = tanh(v)
    log_q = full_256_mixture_log_density(v, mu_old, log(sigma_proposal))
    # log_q includes tanh and, for physical GMM coordinates, scale Jacobian.
    W = softmax(Q_current(s, b) / T - log_q)
    chosen = stratified_importance_resample(W, count=K)
    v_tilde, b_tilde = gather(v, chosen), gather(b, chosen)
    # Duplicates are retained as occurrences of equal mass 1/K.
    # W has already entered multiplicity; do not multiply it again.

    cost[i, j] = squared_norm(Z[i] - v_tilde[j])
    log_column_assignment = log_softmax_over_i((f[:, :, None] - cost) / epsilon)
    log_P = log_column_assignment - log(K)
    log_row_mass = logsumexp_over_j(log_P)
    P = exp(log_P)
    row_mass = exp(log_row_mass)
    R = exp(log_P - log_row_mass[:, :, None])  # explanatory row conditional
    for j in range(K):
        index[j] = categorical(log_column_assignment[:, j])
        source_importance[j] = exp(-log(H) - log_row_mass[index[j]])
    # Importance weights are not clipped or normalized within the 16 samples.

# Actor forward for selected 16 latents only, with actual actor variance.
mu, log_sigma = actor(theta, s, Z[index])
u_new = mu + exp(log_sigma) * fresh_normal_noise()
a_new = tanh(u_new)
log_pi_conditional = (gaussian_log_density(u_new, mu, log_sigma)
                      - tanh_log_jacobian(u_new))
# For physical GMM density, also subtract action_dim * log(40).

# f and Z frozen; u_new remains differentiable through the assignment query.
log_assignment_all = log_softmax_over_H_sources(
    (stopgrad(f) - squared_norm(stopgrad(Z) - u_new)) / epsilon)
log_assignment_selected = gather_sources(log_assignment_all, index)
local_loss = (T * log_pi_conditional - Q_current(s, a_new)
              - T * log_assignment_selected)
actor_loss = mean(stopgrad(source_importance) * local_loss)
update_actor_once(actor_loss)

# Semi-dual derivative at the SAME pre-update f and fresh teacher.
dual_grad = dual_pullback(stopgrad((row_mass - 1 / H) / batch))
phi, dual_Adam_state = Adam_step(phi, dual_Adam_state, dual_grad, lr=1e-4)
```

GMM has fixed Q=`log p_GMM40(40*a)`, T=1, scale=40 and no critic/TD step.
All lanes represent the same state, so one dual forward is shared and its
pullback receives `mean_batch(row_mass) - 1/H`. RL requires f(s) for each replay
state. Only the integration coordinates are fixed; teacher and dual keep changing.

The selected Gaussian means/scales, new-action Q input and new-action assignment
input all have live gradients. Do not detach the whole assignment query: that
would remove the conditional target's allocation force. Teacher proposal sigma
floor is absent from actor conditional density and soft-TD density.

## Conditional targets and sufficient recovery conditions

Fix Q and the map. Define the population allocation and normalized local target:

\[
\pi_B=e^{Q/T}/Z,\qquad
\bar P_i=\int\pi_B(a\mid s)\Pr(i\mid a,s)da,\qquad
t_i(a\mid s)=\pi_B(a\mid s)\Pr(i\mid a,s)/\bar P_i.
\]

After exact source correction in expectation, the actor minimizes

\[
J_{\rm OT}=\frac1H\sum_i E_{\pi_i}
[T\log\pi_i-Q-T\log\Pr(i\mid a,s)]
=\frac TH\sum_i{\rm KL}(\pi_i\|t_i)+\text{constant}.
\]

If population allocation satisfies bar P_i=1/H and all pi_i=t_i, then
pi_H=pi_B. These are sufficient conditions for the desired Boltzmann recovery;
a finite batch's row balance does not establish population balance, and one
Gaussian need not be capable of representing a multimodal t_i. Neural SGD and
a finite latent bank introduce further optimization/approximation limits.

For actual actor posterior rho_theta(i|a,s)=pi_i/(H*pi_H),

\[
J_{\rm OT}=J_{\rm SAC}(\pi_H)+
T E_{\pi_H}{\rm KL}(\rho_\theta(\cdot\mid a,s)\|\Pr(\cdot\mid a,s))+T\log H.
\]

This is the current loss's decomposition, not an extra regularizer added to it.
The method fits the marginal and OT allocation together. It does not inherit
marginal SAC's exact policy-improvement claim from arbitrary joint-loss descent.
That distinction does not change its conditional Boltzmann extraction goal.

## Two importance corrections and finite-sample limits

W corrects proposal actions toward the teacher target through 256→16 resampling.
The detached factor (1/H)/sum_j(P_ij) separately corrects source selections to
the quadrature prior. With a frozen map, positive source probabilities and fresh
actor noise independent of the teacher slot, any local expected gradient G_i obeys

\[
\frac1K\sum_j\sum_i\Pr(i\mid\tilde b_j,s)
\frac{1/H}{\sum_kP_{ik}}G_i=\frac1H\sum_iG_i.
\]

This does not establish population source balance or local target fitting.
Nor does it make finite 16-pair gradients or Adam steps equal to uniform sampling.
The persistent path has no upper bound on source importance; log-space masses do
not remove statistical problems from rarely sampled sources with huge weights.
Track weight mean/max/ESS and full-map log second moment
`log sum_i ((1/H)^2 / sum_j P_ij)`.

For a frozen f, teacher coordinates no longer enter the new-action local loss
as direct NLL targets; W influences expected actor gradients through changes
to the potential and assignment over training. Source correction removes the
unadjusted source-frequency effect at that fixed map.

## Evaluation and provenance

Preserve 256×2 actor/critic, learning rates, UTD/FREQ, actor sigma limits,
paired RL noise-free evaluations, and isolated evaluation RNG. Measure near-mode
fraction together with coverage, mode weights, MMD² and sliced W₂. Good source
balance or high near-mode fraction alone does not prove full target recovery.

Existing conditional GMM40 runs remain relevant records of this objective.
The marginal-density comparison is archived and not silently resumed as a
conditional run. Restoration does not start a new training campaign.

```bash
bash scripts/run_v7.sh 0 --check benchmark=ant
bash scripts/run_v7.sh 0 --check benchmark=ant alg.actor.ot_potential_mode=fresh_sinkhorn
/root/.venv-optiq-mujoco/bin/python -m gmm40.train_v7 --help
```
