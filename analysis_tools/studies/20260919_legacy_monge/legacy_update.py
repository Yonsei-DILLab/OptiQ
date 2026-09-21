"""Archived legacy update_actor with only the assignment branch extended.

Baseline argmax remains numerically identical to the archived production update.
No actor, KDE, importance weighting, raw-action MSE, or optimizer changes.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from legacy_optiq.critic_utils import critic_expectation
from legacy_optiq.transport import (GaussianKDE,TruncatedGaussianKDE,clip_action,
    sample_truncated_gaussian,sample_truncated_gaussian_mixture,select_density_beta_for_ess,sinkhorn)
from solvers import monge_plan,exact_1d_plan
class LegacyMonge:
    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "num_policy_samples",
            "proposals_per_policy_sample",
            "proposal_sampling_mode",
            "include_anchor",
            "density_correction",
            "adaptive_density_beta",
            "density_beta_grid_size",
            "sinkhorn_iterations",
            "source_q_eval",
            "transport_target_mode",
            "unbounded_actions",
            "return_clouds",
        ],
    )
    def update_actor(
        actor_state: TrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        key,
        z_atoms: jax.Array,
        num_policy_samples: int,
        proposals_per_policy_sample: int,
        proposal_sampling_mode: str,
        proposal_std: float,
        proposal_clip: float,
        include_anchor: bool,
        density_correction: bool,
        density_beta: float,
        adaptive_density_beta: bool,
        minimum_source_ess: float,
        density_beta_grid_size: int,
        temperature: float,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
        source_q_eval: str,
        transport_target_mode: str,
        unbounded_actions: bool = False,
        policy_latents: jax.Array | None = None,
        proposal_centers: jax.Array | None = None,
        return_clouds: bool = False,
        source_actor_params=None,
    ):
        # GMM-style density benchmarks can opt out of action/proposal bounds.
        # RL callers retain their bounded path through the default False value.
        if unbounded_actions and proposal_sampling_mode != "stratified":
            raise ValueError("Unbounded proposals currently require stratified sampling")
        key, latent_key, proposal_key, dropout_key = jax.random.split(key, 4)
        batch_size, observation_dim = observations.shape

        def actor_loss(actor_params):
            output_layer = f"Dense_{len(actor_params) - 1}"
            action_dim = actor_params[output_layer]["bias"].shape[0]
            if policy_latents is None:
                latents = jax.random.normal(
                    latent_key,
                    (batch_size, num_policy_samples, action_dim),
                    dtype=observations.dtype,
                )
            else:
                # Optional externally sampled prior draws (e.g. randomized
                # Gaussian strata in unconditional density benchmarks).
                if policy_latents.shape != (batch_size, num_policy_samples, action_dim):
                    raise ValueError('Unexpected policy_latents shape')
                latents = policy_latents.astype(observations.dtype)
            repeated_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_policy_samples, observation_dim),
            )
            raw_actions = actor_state.apply_fn(
                {"params": actor_params},
                repeated_observations.reshape(batch_size * num_policy_samples, -1),
                latents.reshape(batch_size * num_policy_samples, action_dim),
            ).reshape(batch_size, num_policy_samples, action_dim)
            policy_samples = raw_actions if unbounded_actions else clip_action(raw_actions)
            # The slow actor supplies BOTH OT rows and (by default) KDE centers.
            # Online predictions keep the same latent/row correspondence and
            # remain the only network differentiated by the regression loss.
            transport_sources = policy_samples
            if source_actor_params is not None:
                source_raw = actor_state.apply_fn(
                    {"params": source_actor_params},
                    repeated_observations.reshape(batch_size * num_policy_samples, -1),
                    latents.reshape(batch_size * num_policy_samples, action_dim),
                ).reshape(batch_size, num_policy_samples, action_dim)
                transport_sources = jax.lax.stop_gradient(
                    source_raw if unbounded_actions else clip_action(source_raw)
                )
            # Define the KDE from actor centers BEFORE drawing its random candidates.
            # Density evaluation below uses this very same distribution, never a
            # KDE fitted to the newly drawn candidate cloud.
            kde_centers = transport_sources if proposal_centers is None else proposal_centers
            if kde_centers.shape != policy_samples.shape:
                raise ValueError("Proposal centers must preserve the configured center count")
            if unbounded_actions:
                proposal_kde = GaussianKDE(jax.lax.stop_gradient(kde_centers), proposal_std)
            else:
                proposal_kde = TruncatedGaussianKDE.from_centers(
                    jax.lax.stop_gradient(kde_centers), proposal_std, proposal_clip
                )
            if proposal_sampling_mode == "stratified":
                sampled = proposal_kde.sample_stratified(
                    proposal_key, proposals_per_policy_sample, include_anchor
                )
            elif proposal_sampling_mode == "exact":
                sampled = sample_truncated_gaussian_mixture(
                    proposal_key, proposal_kde.centers, proposals_per_policy_sample,
                    proposal_std, proposal_clip, include_anchor, return_component_indices=True,
                )
            else:
                raise ValueError(
                    f"Unknown proposal_sampling_mode: {proposal_sampling_mode}"
                )
            if proposal_sampling_mode == "exact":
                proposals, proposal_component_indices = sampled
            else:
                proposals = sampled
                proposal_component_indices = jnp.broadcast_to(
                    jnp.arange(num_policy_samples)[None, :, None], proposals.shape[:-1]
                )
            proposal_component_indices = proposal_component_indices.reshape(batch_size, -1)
            proposals = proposals.reshape(
                batch_size,
                num_policy_samples * proposals_per_policy_sample,
                action_dim,
            )
            num_proposals = proposals.shape[1]
            proposal_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_proposals, observation_dim),
            )
            source_distributions = qf_state.apply_fn(
                {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
                proposal_observations.reshape(batch_size * num_proposals, -1),
                proposals.reshape(batch_size * num_proposals, action_dim),
                rngs={"dropout": dropout_key},
                train=False,
            ).reshape(2, batch_size, num_proposals, -1)
            source_qs = critic_expectation(source_distributions, z_atoms)
            if source_q_eval == "mean":
                source_q = source_qs.mean(axis=0)
            elif source_q_eval == "min":
                source_q = source_qs.min(axis=0)
            else:
                raise ValueError(f"Unknown source_q_eval: {source_q_eval}")
            source_q = jax.lax.stop_gradient(source_q)

            proposal_log_density = jnp.zeros_like(source_q)
            if density_correction:
                proposal_log_density = proposal_kde.log_prob(jax.lax.stop_gradient(proposals))
            q_score = source_q / temperature
            density_score = -proposal_log_density
            if adaptive_density_beta and density_correction:
                selected_density_beta, selected_beta_ess = (
                    select_density_beta_for_ess(
                        q_score,
                        density_score,
                        minimum_source_ess,
                        density_beta_grid_size,
                    )
                )
            else:
                selected_density_beta = jnp.full(
                    (batch_size,), density_beta, dtype=q_score.dtype
                )
                fixed_logits = (
                    q_score + selected_density_beta[:, None] * density_score
                )
                fixed_weights = jax.nn.softmax(fixed_logits, axis=-1)
                selected_beta_ess = 1.0 / jnp.sum(
                    jnp.square(fixed_weights), axis=-1
                )
            selected_density_beta = jax.lax.stop_gradient(selected_density_beta)
            effective_density_score = selected_density_beta[:, None] * density_score
            logits = q_score + effective_density_score
            source_weights = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))

            squared_costs = jnp.sum(
                jnp.square(transport_sources[:, :, None, :] - proposals[:, None, :, :]),
                axis=-1,
            )
            costs = squared_costs / (
                squared_costs.mean(axis=(-2, -1), keepdims=True) + 1.0e-8
            )
            if transport_target_mode == "monge_quantile":
                transport = jax.vmap(monge_plan)(transport_sources, proposals, source_weights)
            elif transport_target_mode == "exact_argmax":
                transport = jax.vmap(exact_1d_plan)(transport_sources, proposals, source_weights)
            else:
                transport = sinkhorn(costs, source_weights, sinkhorn_epsilon, sinkhorn_iterations)
            transport = jax.lax.stop_gradient(transport)
            row_distribution = transport / jnp.maximum(
                transport.sum(axis=-1, keepdims=True), 1.0e-20
            )
            if transport_target_mode in ("argmax", "exact_argmax", "monge_quantile"):
                selected_indices = jnp.argmax(row_distribution, axis=-1)
                selected_actions = jax.vmap(lambda actions, indices: actions[indices])(
                    proposals, selected_indices
                )
            elif transport_target_mode == "categorical":
                # Independent draw per row. Preserve the original RNG split and
                # returned key so actor/proposal streams match the argmax run.
                target_key = jax.random.fold_in(key, 0x434154)
                row_logits = jnp.where(
                    row_distribution > 0, jnp.log(row_distribution), -jnp.inf
                )
                selected_indices = jax.random.categorical(
                    target_key, row_logits, axis=-1
                )
                selected_actions = jax.vmap(lambda actions, indices: actions[indices])(
                    proposals, selected_indices
                )
            elif transport_target_mode == "barycentric":
                selected_actions = jnp.einsum(
                    "bnm,bma->bna", row_distribution, proposals,
                    precision=jax.lax.Precision.HIGHEST,
                )
            else:
                raise ValueError(
                    f"Unknown transport_target_mode: {transport_target_mode}"
                )
            selected_actions = jax.lax.stop_gradient(selected_actions)
            loss = jnp.mean(
                jnp.sum(jnp.square(raw_actions - selected_actions), axis=-1)
            )
            source_ess = 1.0 / jnp.sum(jnp.square(source_weights), axis=-1)
            q_only_weights = jax.nn.softmax(q_score, axis=-1)
            q_only_ess = 1.0 / jnp.sum(jnp.square(q_only_weights), axis=-1)
            density_only_weights = jax.nn.softmax(density_score, axis=-1)
            density_only_ess = 1.0 / jnp.sum(jnp.square(density_only_weights), axis=-1)
            effective_density_only_weights = jax.nn.softmax(
                effective_density_score, axis=-1
            )
            effective_density_only_ess = 1.0 / jnp.sum(
                jnp.square(effective_density_only_weights), axis=-1
            )
            centered_q = source_q - source_q.mean(axis=-1, keepdims=True)
            centered_density = density_score - density_score.mean(
                axis=-1, keepdims=True
            )
            q_density_correlation = jnp.mean(centered_q * centered_density, axis=-1) / (
                jnp.std(source_q, axis=-1) * jnp.std(density_score, axis=-1) + 1.0e-8
            )
            q_logit_std = jnp.std(q_score, axis=-1)
            density_logit_std = jnp.std(density_score, axis=-1)
            effective_density_logit_std = jnp.std(
                effective_density_score, axis=-1
            )

            def weighted_q_gain(weights, q_values):
                return jnp.sum(weights * q_values, axis=-1) - q_values.mean(axis=-1)

            q_only_q_gain = weighted_q_gain(q_only_weights, source_q)
            density_only_q_gain = weighted_q_gain(density_only_weights, source_q)
            full_q_gain = weighted_q_gain(source_weights, source_q)

            # Cross-evaluate selection by each twin critic with the other critic.
            # This is still not a fully independent estimate, but is less optimistic
            # than evaluating a Q-weighted selection with the exact same Q values.
            q1, q2 = source_qs[0], source_qs[1]
            q1_weights = jax.nn.softmax(
                q1 / temperature + effective_density_score, axis=-1
            )
            q2_weights = jax.nn.softmax(
                q2 / temperature + effective_density_score, axis=-1
            )
            cross_critic_q_gain = 0.5 * (
                weighted_q_gain(q1_weights, q2)
                + weighted_q_gain(q2_weights, q1)
            )

            # Group local diagnostics by the component that actually generated
            # each proposal. This is identical to the old layout reduction for
            # stratified draws and remains valid for IID mixture draws.
            component_mask = (
                proposal_component_indices[:, None, :]
                == jnp.arange(num_policy_samples)[None, :, None]
            )
            component_count = component_mask.sum(axis=-1)

            def grouped_values(values, fill_value):
                return jnp.where(component_mask, values[:, None, :], fill_value)

            grouped_q = grouped_values(source_q, -jnp.inf)
            component_max_q = grouped_q.max(axis=-1)
            component_min_q = grouped_values(source_q, jnp.inf).min(axis=-1)
            local_q_range = jnp.where(
                component_count > 0, component_max_q - component_min_q, 0.0
            )
            local_top_two = jax.lax.top_k(grouped_q, min(2, num_proposals))[0]
            local_q_top_gap = jnp.where(
                component_count > 1,
                local_top_two[..., 0] - local_top_two[..., -1],
                0.0,
            )

            grouped_q1 = grouped_values(q1, -jnp.inf)
            grouped_q2 = grouped_values(q2, -jnp.inf)
            q1_argmax = jnp.argmax(grouped_q1, axis=-1)
            q2_argmax = jnp.argmax(grouped_q2, axis=-1)
            valid_components = component_count > 0
            twin_local_argmax_agreement = jnp.sum(
                valid_components * (q1_argmax == q2_argmax)
            ) / jnp.maximum(valid_components.sum(), 1)

            local_best_q_gain = jnp.zeros_like(local_q_range)
            local_improvement_fraction = jnp.asarray(0.0)
            local_anchor_argmax_fraction = jnp.asarray(0.0)
            twin_local_delta_correlation = jnp.asarray(0.0)
            twin_local_improvement_sign_agreement = jnp.asarray(0.0)
            if include_anchor:
                anchor_indices = (
                    jnp.arange(num_policy_samples) * proposals_per_policy_sample
                )
                anchor_q = source_q[:, anchor_indices]
                anchor_q1 = q1[:, anchor_indices]
                anchor_q2 = q2[:, anchor_indices]
                local_best_q_gain = component_max_q - anchor_q
                local_anchor_argmax_fraction = jnp.mean(
                    jnp.argmax(grouped_q, axis=-1) == anchor_indices[None, :]
                )

                anchor_slot_mask = jnp.zeros(
                    (num_policy_samples, proposals_per_policy_sample), dtype=bool
                ).at[:, 0].set(True).reshape(num_proposals)
                random_mask = ~anchor_slot_mask[None, :]
                sample_anchor_q = jnp.take_along_axis(
                    anchor_q, proposal_component_indices, axis=1
                )
                sample_anchor_q1 = jnp.take_along_axis(
                    anchor_q1, proposal_component_indices, axis=1
                )
                sample_anchor_q2 = jnp.take_along_axis(
                    anchor_q2, proposal_component_indices, axis=1
                )
                local_improvement_fraction = jnp.sum(
                    random_mask * (source_q > sample_anchor_q)
                ) / jnp.maximum(random_mask.sum() * batch_size, 1)

                q1_deltas = q1 - sample_anchor_q1
                q2_deltas = q2 - sample_anchor_q2
                random_component_mask = component_mask & random_mask[:, None, :]
                random_count = random_component_mask.sum(axis=-1)
                safe_count = jnp.maximum(random_count, 1)
                q1_mean = jnp.sum(
                    random_component_mask * q1_deltas[:, None, :], axis=-1
                ) / safe_count
                q2_mean = jnp.sum(
                    random_component_mask * q2_deltas[:, None, :], axis=-1
                ) / safe_count
                centered_q1 = q1_deltas[:, None, :] - q1_mean[..., None]
                centered_q2 = q2_deltas[:, None, :] - q2_mean[..., None]
                covariance = jnp.sum(
                    random_component_mask * centered_q1 * centered_q2, axis=-1
                ) / safe_count
                variance_q1 = jnp.sum(
                    random_component_mask * jnp.square(centered_q1), axis=-1
                ) / safe_count
                variance_q2 = jnp.sum(
                    random_component_mask * jnp.square(centered_q2), axis=-1
                ) / safe_count
                correlations = covariance / (
                    jnp.sqrt(variance_q1 * variance_q2) + 1.0e-8
                )
                valid_correlations = random_count > 1
                twin_local_delta_correlation = jnp.sum(
                    jnp.where(valid_correlations, correlations, 0.0)
                ) / jnp.maximum(valid_correlations.sum(), 1)
                twin_local_improvement_sign_agreement = jnp.sum(
                    random_mask * ((q1_deltas > 0.0) == (q2_deltas > 0.0))
                ) / jnp.maximum(random_mask.sum() * batch_size, 1)
            metrics = {
                "actor_loss": loss,
                "source_ess_fraction": (source_ess / num_proposals).mean(),
                "source_ess_absolute": source_ess.mean(),
                "source_ess_min": source_ess.min(),
                "source_ess_target": jnp.asarray(
                    minimum_source_ess, dtype=source_q.dtype
                ),
                "q_only_ess_fraction": (q_only_ess / num_proposals).mean(),
                "density_only_ess_fraction": (density_only_ess / num_proposals).mean(),
                "effective_density_only_ess_fraction": (
                    effective_density_only_ess / num_proposals
                ).mean(),
                "density_beta_mean": selected_density_beta.mean(),
                "density_beta_min": selected_density_beta.min(),
                "density_beta_max": selected_density_beta.max(),
                "density_beta_at_one_fraction": jnp.mean(
                    selected_density_beta >= 1.0 - 1.0e-6
                ),
                "density_beta_infeasible_fraction": jnp.mean(
                    q_only_ess < minimum_source_ess
                ),
                "selected_beta_search_ess": selected_beta_ess.mean(),
                "max_source_weight": source_weights.max(axis=-1).mean(),
                "source_q_std": jnp.std(source_q, axis=-1).mean(),
                "neg_log_proposal_std": jnp.std(density_score, axis=-1).mean(),
                "q_logit_std": q_logit_std.mean(),
                "density_logit_std": density_logit_std.mean(),
                "effective_density_logit_std": effective_density_logit_std.mean(),
                "combined_logit_std": jnp.std(logits, axis=-1).mean(),
                "q_to_density_logit_std_ratio": (
                    q_logit_std / (density_logit_std + 1.0e-8)
                ).mean(),
                "q_to_effective_density_logit_std_ratio": (
                    q_logit_std / (effective_density_logit_std + 1.0e-8)
                ).mean(),
                "q_neglogq_correlation": q_density_correlation.mean(),
                "q_only_weighted_q_gain": q_only_q_gain.mean(),
                "density_only_weighted_q_gain": density_only_q_gain.mean(),
                "full_weighted_q_gain": full_q_gain.mean(),
                "cross_critic_weighted_q_gain": cross_critic_q_gain.mean(),
                "twin_q_abs_diff": jnp.abs(q1 - q2).mean(),
                "source_q_mean": source_q.mean(),
                "source_q_global_range": (
                    source_q.max(axis=-1) - source_q.min(axis=-1)
                ).mean(),
                "local_q_range": local_q_range.mean(),
                "local_q_top_gap": local_q_top_gap.mean(),
                "local_best_q_gain_over_anchor": local_best_q_gain.mean(),
                "local_improvement_fraction": local_improvement_fraction,
                "local_anchor_argmax_fraction": local_anchor_argmax_fraction,
                "twin_local_argmax_agreement": twin_local_argmax_agreement,
                "twin_local_delta_correlation": twin_local_delta_correlation,
                "twin_local_improvement_sign_agreement": (
                    twin_local_improvement_sign_agreement
                ),
                "selected_delta_l2": jnp.linalg.norm(
                    policy_samples - selected_actions, axis=-1
                ).mean(),
                "policy_spread_l2": jnp.linalg.norm(
                    policy_samples.std(axis=1), axis=-1
                ).mean(),
            }
            for label, counterfactual_temperature in (
                ("0p05", 0.05),
                ("0p1", 0.1),
                ("0p2", 0.2),
                ("0p25", 0.25),
                ("0p5", 0.5),
                ("1p0", 1.0),
            ):
                counterfactual_weights = jax.nn.softmax(
                    source_q / counterfactual_temperature + effective_density_score,
                    axis=-1,
                )
                counterfactual_ess = 1.0 / jnp.sum(
                    jnp.square(counterfactual_weights), axis=-1
                )
                metrics[f"counterfactual_ess_T{label}"] = (
                    counterfactual_ess / num_proposals
                ).mean()
            if return_clouds:
                metrics.update(cloud_proposals=proposals, cloud_weights=source_weights,
                    cloud_transport=transport, cloud_selected=selected_actions,
                    cloud_centers=kde_centers, cloud_policy=policy_samples,
                    cloud_log_density=proposal_log_density,
                    cloud_transport_sources=transport_sources)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, loss, key, metrics

