"""AntMaze-only forwarding and checks for an existing teacher std-floor option.

No policy or learning implementation lives here. Scratch numerical checks use
their own fixed RNG key and never update learner parameters, optimizers or RNG.
"""
import hashlib
import json
import math
from pathlib import Path


def validate_floor(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Teacher std floor must be finite and positive')
    return value


def verify_update_summary(runtime_floor, info, expected, actor_bounds=(-5., -1.)):
    """Periodic diagnostic metrics may be absent even after valid updates."""
    expected = validate_floor(expected)
    actual = float(runtime_floor)
    assert actual == expected
    logged = info.get('train/proposal_std_pretanh')
    if logged is not None:
        assert math.isclose(float(logged), actual, rel_tol=1e-6)
    # This metric is in the unchanged algorithm's always-logged core set.
    mean = float(info['train/actor_std_mean'])
    lower, upper = actor_bounds
    assert math.isfinite(lower) and not math.isnan(upper) and lower < upper
    assert math.isfinite(mean) and math.exp(lower)-1e-6 <= mean <= math.exp(upper)+1e-6
    return dict(runtime_cfg_teacher_floor=actual, logged_teacher_floor=logged,
                periodic_diagnostic_available=logged is not None, actor_std_mean=mean,
                unchanged_train_call_argument='model.cfg.alg.actor.proposal_std')


def verify_teacher_floor(learner, folder, expected):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    from optiq_dime.box_gaussian import sample_box, mixture_log_prob
    expected = validate_floor(expected)
    actor = learner.model.cfg.alg.actor
    assert actor.type == 'semi_implicit' and actor.teacher_distribution == 'conditional_mixture'
    assert actor.distillation_loss == 'direct_gmm_nll'
    assert float(actor.get('soft_proximal_ess_fraction', 0.)) == 0.
    assert actor.density_correction and actor.density_beta == 1.
    assert actor.teacher_std_floor == actor.proposal_std == actor.proposal_std_pretanh == expected
    from .actor_sigma_profile import settings as sigma_settings
    sigma = sigma_settings(getattr(learner, 'actor_sigma_profile', None))
    assert (actor.log_std_min, actor.log_std_max, actor.initial_log_std) == tuple(sigma.values())
    assert (actor.num_policy_samples, actor.proposals_per_policy_sample) == (64, 1)
    means = jnp.asarray([[[-.85], [0.], [.85]]], dtype=jnp.float32)
    logs = jnp.asarray([[[-5.], [-1.], [-2.]]], dtype=jnp.float32)
    key = jax.random.PRNGKey(41357)
    proposal = ConditionalGaussianProposal(means, logs, expected)
    samples, _, indices = proposal.sample(key, 4096, 'exact')
    effective = jnp.maximum(logs, math.log(expected))
    _, noise_key = jax.random.split(key)
    selected_means = jnp.take_along_axis(means, indices[:, :, None], axis=1)
    selected_logs = jnp.take_along_axis(effective, indices[:, :, None], axis=1)
    np.testing.assert_array_equal(samples, sample_box(noise_key, selected_means, selected_logs))
    np.testing.assert_array_equal(proposal.log_prob(samples), mixture_log_prob(samples, means, effective))
    assert np.isfinite(samples).all() and np.isfinite(proposal.log_prob(samples)).all()
    assert np.max(np.abs(samples)) <= 1.
    grid = jnp.linspace(-1., 1., 16385).reshape(1, -1, 1)
    density = np.exp(np.asarray(proposal.log_prob(grid))[0])
    integral = float(np.sum((density[:-1] + density[1:]) / 2) * (2 / 16384))
    assert abs(integral - 1.) < 5e-4
    # Default exp(-5) must be exactly the identity throughout the allowed range.
    control = ConditionalGaussianProposal(means, logs, math.exp(-5))
    no_floor = ConditionalGaussianProposal(means, logs, 0.)
    np.testing.assert_array_equal(control.sample(key, 4096, 'exact')[0], no_floor.sample(key, 4096, 'exact')[0])
    result = dict(verified=True, teacher_std_floor=expected,
        actor_log_std_bounds=[sigma['log_std_min'], sigma['log_std_max']],
        actor_initial_log_std=sigma['initial_log_std'],
        exact_sampling_and_density_scales=True, normalized_density_integral=integral,
        default_floor_identity=True, finite_bounded_samples=True,
        sample_sha256=hashlib.sha256(np.asarray(samples).tobytes()).hexdigest(),
        learner_rng_untouched=True, scratch_rng_seed=41357,
        scope='Existing teacher-only proposal scale; actor bounds are verified separately and recorded above')
    (Path(folder) / 'teacher-proposal-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
