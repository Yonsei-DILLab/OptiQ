"""Adapter checks for the existing fixed finite latent option; no learning code."""
import hashlib
import json
from pathlib import Path


def evaluation_mode_label(learner, mode):
    # The existing finite-prior sampler's deterministic control is component0.
    return ('component0_mu' if mode == 'zero_z' and
            getattr(learner, 'latent_profile', None) == 'fixed64' else mode)


def verify_fixed_latent(learner, folder, stage):
    import flax.serialization as fs
    import jax
    import jax.numpy as jnp
    import numpy as np
    from optiq_dime.latent import FiniteMixtureTrainState, finite_latent_codes, sample_latents, stratified_finite_latents
    from optiq_dime.box_gaussian import sample_box
    from .learners import audit
    assert stage in ('initial', 'final') and learner.latent_profile == 'fixed64'
    model, policy = learner.model, learner.model.policy
    actor = model.cfg.alg.actor
    assert (actor.latent_prior,actor.latent_components,actor.latent_codebook_seed)==('finite',64,20260911)
    assert actor.distillation_loss == 'direct_gmm_nll' and actor.teacher_distribution == 'conditional_mixture'
    assert (actor.num_policy_samples,actor.proposals_per_policy_sample)==(64,1)
    assert (actor.log_std_min,actor.log_std_max,actor.initial_log_std)==(-5.,-1.,-1.)
    assert float(actor.get('soft_proximal_ess_fraction',0.)) == 0.
    assert model.cfg.alg.critic.backup_mode == 'td' and actor.density_correction and actor.density_beta == 1.
    assert not model.cfg.dual_mu_eval  # unused standalone evaluator; AntMaze runs both modes itself
    states=dict(actor=policy.actor_state,critic=policy.qf_state,target_actor=policy.target_actor_state,
                entropy=model.ent_coef_state)
    before=hashlib.sha256(fs.to_bytes(states)).hexdigest()
    rng_before=[np.asarray(k).copy() for k in (policy.key,policy.noise_key,model.key)]
    for state in (policy.actor_state,policy.target_actor_state):
        assert isinstance(state,FiniteMixtureTrainState)
        assert (state.latent_components,state.latent_codebook_seed)==(64,20260911)
    state=policy.actor_state
    codes=finite_latent_codes(state,8,jnp.float32)
    assert codes.shape==(64,8) and np.isfinite(codes).all()
    np.testing.assert_allclose(np.asarray(codes).mean(0),0,atol=2e-7)
    np.testing.assert_allclose(np.asarray(codes).std(0),1,atol=2e-7)
    stratified=stratified_finite_latents(state,3,64,8,jnp.float32)
    np.testing.assert_array_equal(stratified,np.broadcast_to(np.asarray(codes),(3,64,8)))
    key=jax.random.PRNGKey(194831)
    sampled=sample_latents(state,key,(8192,8),jnp.float32)
    indices=jax.random.randint(key,(8192,),0,64)
    np.testing.assert_array_equal(sampled,codes[indices])
    counts=np.bincount(np.asarray(indices),minlength=64)
    assert np.all(counts>0)
    observations=jnp.zeros((64,29),dtype=jnp.float32)
    latent_key,noise_key=jax.random.split(key)
    z=sample_latents(state,latent_key,(64,8),jnp.float32)
    means,logs=state.apply_fn({'params':state.params},observations,z)
    direct=policy.sample_action(state,observations,key,deterministic=False,sample_conditional_noise=True)
    native=policy.sample_action(state,observations,key,deterministic=False,sample_conditional_noise=False)
    np.testing.assert_allclose(direct,sample_box(noise_key,means,logs),atol=2e-6,rtol=1e-5)
    np.testing.assert_allclose(native,means,atol=2e-6,rtol=1e-5)
    component0=policy.sample_action(state,observations,key,deterministic=True,sample_conditional_noise=False)
    expected0,_=state.apply_fn({'params':state.params},observations,jnp.broadcast_to(codes[0],(64,8)))
    np.testing.assert_allclose(component0,expected0,atol=2e-6,rtol=1e-5)
    assert np.isfinite(direct).all() and np.max(np.abs(direct))<=1.
    restored=fs.from_bytes(state,fs.to_bytes(state))
    assert isinstance(restored,FiniteMixtureTrainState) and restored.latent_components==64
    assert restored.latent_codebook_seed==20260911
    np.testing.assert_array_equal(finite_latent_codes(restored,8),codes)
    np.testing.assert_array_equal(policy.sample_action(restored,observations,key,False,True),direct)
    assert hashlib.sha256(fs.to_bytes(states)).hexdigest()==before
    for actual,wanted in zip((policy.key,policy.noise_key,model.key),rng_before):
        np.testing.assert_array_equal(actual,wanted)
    np.save(Path(folder)/'latent-codebook.npy',np.asarray(codes))
    result=dict(verified=True,stage=stage,latent_prior='finite',latent_components=64,
        latent_codebook_seed=20260911,codebook_sha256=hashlib.sha256(np.asarray(codes).tobytes()).hexdigest(),
        action_sampling='Uniform codebook index independently at each action; conditional sigma retained for policy/train',
        native_sampling='Uniform same-codebook index; conditional sigma removed',
        deterministic_control='component0_mu, not z=0',full_stratified_components=True,
        sample_counts=counts.tolist(),direct_sampler_verified=True,native_sampler_verified=True,
        serialization_roundtrip_verified=True,model_optimizer_rng_unchanged=True,
        scratch_rng_seed=194831,actor_updates=int(state.step),parameters=audit(learner))
    (Path(folder)/f'latent-profile-{stage}-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
