"""Explicit AntMaze actor-scale ablations; the existing actor/sampler is unchanged."""
import hashlib
import json
from pathlib import Path

PROFILES = {'capm2': (-5., -2., -2.), 'capm3': (-5., -3., -3.)}


def settings(name=None):
    values = (-5., -1., -1.) if name is None else PROFILES[name]
    return dict(zip(('log_std_min', 'log_std_max', 'initial_log_std'), values))


def parameter_digest(params):
    """Use the same sorted-leaf byte order as learners.audit."""
    import numpy as np
    def leaves(tree):
        if hasattr(tree, 'items'):
            for key in sorted(tree):
                yield from leaves(tree[key])
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                yield from leaves(value)
        else:
            yield tree
    digest = hashlib.sha256()
    count = 0
    for leaf in leaves(params):
        array = np.asarray(leaf)
        assert np.isfinite(array).all()
        digest.update(array.tobytes())
        count += array.size
    return dict(sha256=digest.hexdigest(), parameters=count)


def verify(learner, folder, stage):
    import flax.core
    import flax.serialization as fs
    import jax
    import jax.numpy as jnp
    import numpy as np
    from optiq_dime.box_gaussian import sample_box
    from optiq_dime.latent import sample_latents
    from .learners import audit
    assert stage in ('initial', 'final')
    wanted = settings(learner.actor_sigma_profile)
    model, policy = learner.model, learner.model.policy
    cfg = model.cfg.alg.actor
    for key, value in wanted.items():
        assert float(cfg[key]) == value
    assert cfg.distillation_loss == 'direct_gmm_nll'
    assert cfg.teacher_distribution == 'conditional_mixture'
    assert cfg.density_correction and cfg.density_beta == 1.
    assert float(cfg.get('soft_proximal_ess_fraction', 0.)) == 0.
    assert model.cfg.alg.critic.backup_mode == 'td'
    assert learner.latent_profile is None
    assert (cfg.num_policy_samples, cfg.proposals_per_policy_sample) == (64, 1)
    states = dict(actor=policy.actor_state, critic=policy.qf_state,
                  target_actor=policy.target_actor_state, entropy=model.ent_coef_state)
    before = hashlib.sha256(fs.to_bytes(states)).hexdigest()
    rng_before = [np.asarray(k).copy() for k in (policy.key, policy.noise_key, model.key)]
    state = policy.actor_state
    key = jax.random.PRNGKey(71291)
    observations = jax.random.normal(jax.random.PRNGKey(71292), (64, 29))
    latent_key, noise_key = jax.random.split(key)
    z = sample_latents(state, latent_key, (64, 8), jnp.float32)
    mu, logs = state.apply_fn({'params': state.params}, observations, z)
    assert np.isfinite(mu).all() and np.isfinite(logs).all()
    assert np.min(logs) >= wanted['log_std_min'] - 1e-6
    assert np.max(logs) <= wanted['log_std_max'] + 1e-6
    direct = policy.sample_action(state, observations, key, False, True)
    native = policy.sample_action(state, observations, key, False, False)
    np.testing.assert_allclose(direct, sample_box(noise_key, mu, logs), rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(native, mu, rtol=1e-5, atol=2e-6)
    assert np.max(np.abs(direct)) <= 1. and np.isfinite(direct).all()
    restored = fs.from_bytes(state, fs.to_bytes(state))
    np.testing.assert_array_equal(policy.sample_action(restored, observations, key, False, True), direct)
    control_digest = None
    if stage == 'initial':
        assert int(state.step) == int(policy.qf_state.step) == 0
        np.testing.assert_array_equal(logs, np.full((64, 8), wanted['initial_log_std'], dtype=np.float32))
        # Only change a scratch copy of the constant sigma-head bias back to -1.
        # Its hash must match the archived control at registration verification.
        params = flax.core.unfreeze(state.params)
        np.testing.assert_array_equal(params['log_std']['bias'],
                                      np.full((8,), wanted['initial_log_std'], dtype=np.float32))
        params['log_std']['bias'] = jnp.full_like(params['log_std']['bias'], -1.)
        control_digest = parameter_digest(params)
    assert hashlib.sha256(fs.to_bytes(states)).hexdigest() == before
    for actual, wanted_rng in zip((policy.key, policy.noise_key, model.key), rng_before):
        np.testing.assert_array_equal(actual, wanted_rng)
    record = dict(verified=True, stage=stage, profile=learner.actor_sigma_profile,
        settings=wanted, parameters=audit(learner),
        control_actor_after_restoring_only_initial_sigma_bias=control_digest,
        actor_updates=int(state.step), critic_updates=int(policy.qf_state.step),
        observed_log_std_min=float(np.min(logs)), observed_log_std_max=float(np.max(logs)),
        direct_sampler_verified=True, native_sampler_verified=True,
        serialization_verified=True, model_optimizer_rng_unchanged=True,
        scratch_rng_seed=71291, teacher_std_floor=float(cfg.proposal_std),
        scope='Actor cap and initial sigma only; random latent and teacher proposal floor preserved')
    (Path(folder) / f'actor-sigma-{stage}-verification.json').write_text(json.dumps(record, indent=2)+'\n')
    return record
