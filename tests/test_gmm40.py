"""Guard target identity and the unconditional adapter to the existing update."""
import importlib.util
from pathlib import Path
import types
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from benchmarks.gmm40.sampler import make_target, log_prob, initialize, update, draw, default_config
from benchmarks.gmm40.metrics import evaluate_sample_tensor


def test_hyperparameter_schedule_preserves_disabled_path_and_endpoints():
    from benchmarks.gmm40.sampler import scheduled_config
    cfg = default_config()
    assert scheduled_config(cfg, 100) is cfg
    cfg.update(anneal_updates=400, temperature_start=10., temperature=1.,
               proposal_std_start=32., proposal_std=2.,
               sinkhorn_epsilon_start=.1, sinkhorn_epsilon=.001)
    assert scheduled_config(cfg, 0)['temperature'] == 10.
    mid = scheduled_config(cfg, 200)
    np.testing.assert_allclose([mid['temperature'], mid['proposal_std'], mid['sinkhorn_epsilon']],
                               [np.sqrt(10), 8., .01])
    for field in ('temperature', 'proposal_std', 'sinkhorn_epsilon'):
        assert scheduled_config(cfg, 400)[field] == pytest.approx(cfg[field])
        assert scheduled_config(cfg, 800)[field] == pytest.approx(cfg[field])
    assert cfg['temperature'] == 1.  # No mutation of recorded final configuration.


def test_resume_key_and_optimizer_reproduce_uninterrupted_actor_update():
    import flax.serialization
    cfg = default_config()
    cfg.update(batch_size=1, num_policy_samples=8, proposals_per_policy_sample=4,
               coordinate_scale=1., unbounded_actions=True, proposal_std=8.)
    initial, oracle, key = initialize(17, [8, 8], coordinate_scale=1.)
    first, _, next_key, _ = update(initial, oracle, key, cfg)
    expected = update(first, oracle, next_key, cfg)
    restored = flax.serialization.from_bytes(initial, flax.serialization.to_bytes(first))
    recovered_key = jax.lax.fori_loop(0, int(restored.step), lambda _, k: jax.random.split(k, 4)[0], key)
    actual = update(restored, oracle, recovered_key, cfg)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)


def test_paper_spatial_tv_is_two_dimensional_and_retains_overflow():
    from benchmarks.gmm40.paper_metrics import spatial_tv, sample_w2
    first = np.array([[0.,0.],[1.,1.]])
    second = np.array([[0.,1.],[1.,0.]])
    assert spatial_tv(first,first,bins=2)['tvd'] == 0.
    # Same 1D coordinate marginals, disjoint 2D distributions.
    assert spatial_tv(first,second,bins=2)['tvd'] == 1.
    assert spatial_tv(torch.tensor(first,requires_grad=True),second,bins=2)['tvd'] == 1.
    assert spatial_tv(first+100,first,bins=2)['generated_overflow'] == 1.
    assert spatial_tv(first+100,first,bins=2)['tvd'] == 1.
    points = torch.tensor([[0.,0.],[10.,0.]])
    assert sample_w2(points,points+torch.tensor([0.,3.])) == 3.


def test_broad_initialization_preserves_rng_and_calibrates_both_output_directions():
    original, original_oracle, key = initialize(19, [32,32], coordinate_scale=1.)
    broad, oracle, broad_key = initialize(19, [32,32], coordinate_scale=1., initial_output_std=25.)
    np.testing.assert_array_equal(key,broad_key)
    for name in ['Dense_0','Dense_1']:
        for field in ['kernel','bias']:
            np.testing.assert_array_equal(original.params[name][field],broad.params[name][field])
    for a,b in zip(jax.tree_util.tree_leaves(original_oracle.params),jax.tree_util.tree_leaves(oracle.params)):
        np.testing.assert_array_equal(a,b)
    samples=np.asarray(draw(broad,jax.random.PRNGKey(777),10000,1.,True))
    covariance=np.cov(samples,rowvar=False)
    np.testing.assert_allclose(np.linalg.eigvalsh(covariance),[625.,625.],rtol=.15)
    assert np.max(np.abs(samples.mean(axis=0)))<2.
    assert int(broad.step)==0


def test_identity_weight_initialization_is_target_free_and_keeps_existing_actor():
    original, original_oracle, key = initialize(19, [32,32], coordinate_scale=1.)
    linear, oracle, same_key = initialize(19, [32,32], coordinate_scale=1.,
        initial_output_std=8., initialization='identity', initialization_noise=0.)
    noisy, _, noisy_key = initialize(19, [32,32], coordinate_scale=1.,
        initial_output_std=8., initialization='identity', initialization_noise=.01)
    np.testing.assert_array_equal(key,same_key)
    np.testing.assert_array_equal(key,noisy_key)
    for a,b in zip(jax.tree_util.tree_leaves(original_oracle.params),jax.tree_util.tree_leaves(oracle.params)):
        np.testing.assert_array_equal(a,b)
    original_shapes = jax.tree_util.tree_map(lambda x:x.shape, original.params)
    assert jax.tree_util.tree_map(lambda x:x.shape, linear.params) == original_shapes
    z = jax.random.normal(jax.random.PRNGKey(901), (2048,2)) * 3.
    observations = jnp.zeros((len(z),0))
    expected = 8.*z
    actual = linear.apply_fn({'params':linear.params}, observations,z)
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-6)
    perturbed = noisy.apply_fn({'params':noisy.params}, observations,z)
    assert float(jnp.sqrt(jnp.mean((perturbed-expected)**2))) < .05
    assert np.count_nonzero(np.asarray(noisy.params['Dense_0']['kernel'][:,4:])) > 0
    # The same OptiQ update trains this parameter initialization without a new
    # forward model, target-gradient objective or output transformation.
    cfg=default_config()
    cfg.update(batch_size=1,num_policy_samples=8,proposals_per_policy_sample=4,
        coordinate_scale=1.,unbounded_actions=True,proposal_std=8.)
    trained,loss,_,_=update(noisy,oracle,key,cfg)
    assert np.isfinite(loss)
    assert int(trained.step)==1


@pytest.mark.parametrize('widths', [[16,32], [32,16,32]])
def test_wide_identity_initialization_uses_existing_units_and_exact_initial_map(widths):
    original, _, original_key = initialize(19, widths, coordinate_scale=1.)
    linear, oracle, key = initialize(19, widths, coordinate_scale=1.,
        initial_output_std=8., initialization='identity_wide', initialization_noise=0.)
    assert jax.tree_util.tree_map(lambda x:x.shape, linear.params) == jax.tree_util.tree_map(
        lambda x:x.shape, original.params)
    np.testing.assert_array_equal(key, original_key)
    z = jax.random.normal(jax.random.PRNGKey(901), (2048,2)) * 3.
    actual = linear.apply_fn({'params':linear.params}, jnp.zeros((len(z),0)), z)
    np.testing.assert_allclose(actual, 8.*z, rtol=5e-6, atol=2e-5)
    # Every existing hidden unit participates in the map, including the
    # output weights; there is no new parameter or frozen runtime transform.
    for i, width in enumerate(widths):
        kernel = np.asarray(linear.params[f'Dense_{i}']['kernel'])
        assert kernel.shape[1] == width
        assert np.all(np.linalg.norm(kernel, axis=0) > 0)
    assert np.all(np.linalg.norm(np.asarray(linear.params[f'Dense_{len(widths)}']['kernel']),axis=1)>0)
    noisy, _, _ = initialize(19, widths, coordinate_scale=1.,
        initial_output_std=8., initialization='identity_wide', initialization_noise=.01)
    perturbed = noisy.apply_fn({'params':noisy.params}, jnp.zeros((len(z),0)), z)
    assert float(jnp.sqrt(jnp.mean((perturbed-8.*z)**2))) < .5
    cfg = default_config()
    cfg.update(batch_size=1,num_policy_samples=8,proposals_per_policy_sample=4,
               coordinate_scale=1.,unbounded_actions=True,proposal_std=8.)
    trained,loss,_,_ = update(noisy,oracle,key,cfg)
    assert int(trained.step)==1 and np.isfinite(loss)
    assert any(not np.array_equal(a,b) for a,b in zip(
        jax.tree_util.tree_leaves(trained.params),jax.tree_util.tree_leaves(noisy.params)))


def test_wide_identity_initialization_is_independent_of_target(monkeypatch):
    import benchmarks.gmm40.sampler as sampler
    actor, _, key = initialize(19,[16,16],coordinate_scale=1.,initial_output_std=8.,
                             initialization='identity_wide')
    target = make_target()
    different = types.SimpleNamespace(locs=target.locs+100., scale_trils=target.scale_trils*2.)
    monkeypatch.setattr(sampler,'make_target',lambda:different)
    other, _, other_key = initialize(19,[16,16],coordinate_scale=1.,initial_output_std=8.,
                                    initialization='identity_wide')
    np.testing.assert_array_equal(key,other_key)
    for a,b in zip(jax.tree_util.tree_leaves(actor.params),jax.tree_util.tree_leaves(other.params)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='even widths'):
        initialize(19,[15,16],coordinate_scale=1.,initial_output_std=8.,initialization='identity_wide')


def test_jax_target_matches_ported_official_torch_density():
    target = make_target()
    points = np.array([[0., 0.], [-39., 32.], [49., -49.], [20., 20.]], dtype=np.float32)
    scales = torch.diagonal(target.scale_trils, dim1=-2, dim2=-1).numpy()
    actual = log_prob(jnp.asarray(points), jnp.asarray(target.locs.numpy()), jnp.asarray(scales))
    expected = target.log_prob(torch.from_numpy(points)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    assert target.locs.shape == (40, 2)
    np.testing.assert_allclose(scales, np.logaddexp(0, 1), atol=1e-7)


def test_existing_actor_update_accepts_empty_observation_and_keeps_oracle_frozen():
    cfg = default_config()
    cfg.update(batch_size=2, num_policy_samples=4, proposals_per_policy_sample=4)
    actor, oracle, key = initialize(123, hidden_dims=[8, 8])
    before = jax.tree_util.tree_map(np.array, actor.params)
    oracle_before = jax.tree_util.tree_map(np.array, oracle.params)
    actor, loss, key, metrics = update(actor, oracle, key, cfg)
    assert np.isfinite(loss)
    assert 1 <= float(metrics['source_ess_absolute']) <= 16.0001
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(actor.params)))
    for a, b in zip(jax.tree_util.tree_leaves(oracle_before), jax.tree_util.tree_leaves(oracle.params)):
        np.testing.assert_array_equal(a, b)
    samples = np.asarray(draw(actor, key, 10))
    assert samples.shape == (10, 2) and np.isfinite(samples).all()
    assert np.max(np.abs(samples)) <= 50


def test_common_evaluator_identity_and_rng_isolation():
    target = make_target()
    with torch.random.fork_rng():
        torch.manual_seed(20260821)
        reference = target.sample((256,))
    rng = torch.random.get_rng_state().clone()
    result = evaluate_sample_tensor(reference, target)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert result['eval/mean_log_prob_abs_error'] == 0
    assert result['eval/energy_w1'] == 0
    assert result['eval/energy_ks'] == 0
    assert result['eval/sliced_w2'] == 0


def test_port_matches_optic_source_when_available():
    source = Path('/workspace/OptiC/DiKL/benchmark_eval.py')
    if not source.exists():
        import pytest
        pytest.skip('Optional original OptiC checkout absent')
    helper = types.ModuleType('algorithms.optiq')
    helper.nearest_mode = lambda x, means: torch.cdist(x, means).argmin(1)
    old = sys.modules.get('algorithms.optiq')
    sys.modules['algorithms.optiq'] = helper
    try:
        spec = importlib.util.spec_from_file_location('optic_reference_evaluator', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            del sys.modules['algorithms.optiq']
        else:
            sys.modules['algorithms.optiq'] = old
    target = make_target()
    samples = torch.linspace(-35, 35, 256).reshape(128, 2)
    assert evaluate_sample_tensor(samples, target) == module.evaluate_sample_tensor(samples, target)


def test_exact_wasserstein_and_histogram_tail_handling():
    from benchmarks.gmm40.evaluate import exact_w2, energy_tvd
    first = torch.tensor([[0., 0.], [10., 0.]])
    second = torch.tensor([[0., 3.], [10., 3.]])
    assert exact_w2(first, second) == 3.
    assert exact_w2(first, first.flip(0)) == 0.
    tvd, with_tails, overflow = energy_tvd(np.array([100., 101.]), np.array([0., 1.]), bins=2)
    assert (tvd, with_tails, overflow) == (1., 1., 1.)


def test_matched_256_by_256_uses_the_original_actor_update():
    from benchmarks.gmm40.sampler import training_candidates_per_update
    cfg = default_config()
    cfg.update(batch_size=1, num_policy_samples=256, proposals_per_policy_sample=1, proposal_std=.08)
    actor, oracle, key = initialize(123, hidden_dims=[8,8])
    actor, loss, key, metrics = update(actor, oracle, key, cfg)
    assert int(actor.step) == 1 and np.isfinite(loss)
    assert 1 <= float(metrics['source_ess_absolute']) <= 256.0001
    assert training_candidates_per_update(cfg) == 256
    np.testing.assert_allclose(cfg['proposal_std'] * cfg['coordinate_scale'], 4.)


def test_anchor_setting_reaches_original_sampler_and_updates_actor():
    cfg = default_config()
    cfg.update(batch_size=1, num_policy_samples=8, proposals_per_policy_sample=1,
        include_anchor=True, proposal_std=.16)
    actor, oracle, key = initialize(123, hidden_dims=[8,8])
    # This is the original sampler's guard; it proves the flag is forwarded.
    with pytest.raises(ValueError, match='At least one random draw'):
        update(actor, oracle, key, cfg)
    cfg['proposals_per_policy_sample'] = 2
    actor, loss, key, metrics = update(actor, oracle, key, cfg)
    assert int(actor.step) == 1 and np.isfinite(loss)
    assert 1 <= float(metrics['source_ess_absolute']) <= 16.0001
    assert 0 <= float(metrics['local_anchor_argmax_fraction']) <= 1
    np.testing.assert_allclose(cfg['proposal_std'] * cfg['coordinate_scale'], 8.)


def test_unbounded_kde_matches_torch_and_has_no_old_cutoffs():
    from optiq_dime.transport import GaussianKDE
    centers = jnp.array([[[0., 0.], [2., -3.]]])
    kde = GaussianKDE(centers, .16)
    samples = np.asarray(kde.sample_stratified(jax.random.PRNGKey(71), 20001, True))
    np.testing.assert_array_equal(samples[:,:,0,:], centers)
    noise = samples[:,:,1:,:] - np.asarray(centers)[:,:,None,:]
    assert np.max(np.abs(noise)) > .5  # Old +/-25 physical perturbation limit.
    assert np.max(np.abs(samples)) > 1  # Old +/-50 physical action limit.
    np.testing.assert_allclose(noise.std(axis=(0,1,2)), .16, rtol=.03)
    query = jnp.array([[[1.2, -1.2], [2.8, -3.9], [-.6, .1]]])
    mixture = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(torch.ones(2)),
        torch.distributions.Independent(torch.distributions.Normal(
            torch.tensor(np.array(centers[0])), torch.full((2,2), .16)), 1))
    expected = mixture.log_prob(torch.tensor(np.array(query[0]))).numpy()
    np.testing.assert_allclose(np.array(kde.log_prob(query))[0], expected, rtol=2e-6, atol=2e-5)


def test_unbounded_update_and_evaluation_ignore_action_and_proposal_clips():
    cfg = default_config()
    cfg.update(batch_size=1, num_policy_samples=8, proposals_per_policy_sample=5,
        include_anchor=True, proposal_std=.16, unbounded_actions=True)
    actor, oracle, key = initialize(123, hidden_dims=[8,8])
    params = jax.tree_util.tree_map(lambda x:x, actor.params)
    last = f'Dense_{len(params)-1}'
    params[last]['kernel'] = jnp.zeros_like(params[last]['kernel'])
    params[last]['bias'] = jnp.array([1.2, -1.2])
    actor = actor.replace(params=params)
    np.testing.assert_allclose(np.asarray(draw(actor, key, 10, 50., True)),
        np.tile([60., -60.], (10,1)), atol=1e-5)
    np.testing.assert_allclose(np.asarray(draw(actor, key, 10)),
        np.tile([50., -50.], (10,1)), atol=1e-5)
    first = update(actor, oracle, key, cfg)
    cfg['proposal_clip'] = .00001
    second = update(actor, oracle, key, cfg)
    assert np.isfinite(first[1]) and int(first[0].step) == 1
    for a,b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        np.testing.assert_array_equal(a,b)


def test_bounded_update_matches_prechange_source(monkeypatch):
    import subprocess
    from optiq_dime.algorithm import OptiQDIME
    # Pin the actual prechange implementation so committing the GMM adapter
    # cannot silently turn this regression check into a self-comparison.
    baseline = 'e3853c0565316c2fcd91dd9ead2f30578ad149e7'
    source = subprocess.check_output(
        ['git', 'show', f'{baseline}:optiq_dime/algorithm.py'],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    module = types.ModuleType('optiq_dime._gmm_reference_algorithm')
    module.__package__ = 'optiq_dime'
    exec(compile(source, '<committed OptiQ algorithm>', 'exec'), module.__dict__)
    actual_update = OptiQDIME.update_actor
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return actual_update(**kwargs)
    monkeypatch.setattr(OptiQDIME, 'update_actor', staticmethod(capture))
    cfg = default_config()
    cfg.update(batch_size=1, num_policy_samples=8, proposals_per_policy_sample=5, include_anchor=True)
    actor, oracle, key = initialize(123, hidden_dims=[8,8])
    actual = update(actor, oracle, key, cfg)
    captured.pop('unbounded_actions')
    captured.pop('policy_latents')
    expected = module.OptiQDIME.update_actor(**captured)
    for a,b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a,b)


def test_gaussian_latent_grid_covers_equal_probability_cells():
    from benchmarks.gmm40.latent_sampling import gaussian_grid
    import jax.scipy as jsp
    z = gaussian_grid(jax.random.PRNGKey(77),4096)
    assert z.shape == (4096,2) and np.isfinite(z).all()
    # Check the intended probability measure, not physical-coordinate bins.
    cells = np.floor(np.asarray(jsp.special.ndtr(z))*64).astype(int)
    assert np.unique(cells,axis=0).shape[0] == 4096
    np.testing.assert_allclose(np.asarray(z).mean(axis=0),0.,atol=.02)
    np.testing.assert_allclose(np.cov(np.asarray(z),rowvar=False),np.eye(2),atol=.04)


def test_supplied_iid_latents_preserve_full_actor_update(monkeypatch):
    from optiq_dime.algorithm import OptiQDIME
    cfg = default_config()
    cfg.update(batch_size=1,num_policy_samples=8,proposals_per_policy_sample=4,
               coordinate_scale=1.,unbounded_actions=True,proposal_std=2.)
    actor,oracle,key = initialize(31,[8,8],coordinate_scale=1.)
    actual_update = OptiQDIME.update_actor
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return actual_update(**kwargs)
    monkeypatch.setattr(OptiQDIME,'update_actor',staticmethod(capture))
    expected = update(actor,oracle,key,cfg)
    latent_key = jax.random.split(key,4)[1]
    captured['policy_latents'] = jax.random.normal(latent_key,(1,8,2))
    actual = actual_update(**captured)
    for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a,b)
    cfg['latent_sampling'] = 'grid'
    result = update(actor,oracle,key,cfg)
    assert int(result[0].step) == 1 and np.isfinite(result[1])
    np.testing.assert_array_equal(result[2],expected[2])


@pytest.mark.parametrize('unbounded',[False,True])
def test_default_update_matches_frozen_goal_source_when_available(monkeypatch,unbounded):
    import hashlib
    from optiq_dime.algorithm import OptiQDIME
    root = Path(__file__).resolve().parents[1]
    source = next(root.glob('outputs/gmm40_tuning/hp3/002-continue-width512-eps0001-150k/*/source/optiq_dime/algorithm.py'),None)
    if source is None:
        pytest.skip('Optional immutable pre-grid experiment source is absent')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '5ac2994eace1fea17960d4585d1234c85f9eda53bb87fd732eba9a791e6ee89b'
    module = types.ModuleType('optiq_dime._gmm_frozen_algorithm')
    module.__package__ = 'optiq_dime'
    exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    current = OptiQDIME.update_actor
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return current(**kwargs)
    monkeypatch.setattr(OptiQDIME,'update_actor',staticmethod(capture))
    cfg = default_config()
    cfg.update(batch_size=1,num_policy_samples=8,proposals_per_policy_sample=5,
               include_anchor=True,unbounded_actions=unbounded)
    actor,oracle,key = initialize(123,[8,8])
    actual = update(actor,oracle,key,cfg)
    assert captured.pop('policy_latents') is None
    expected = module.OptiQDIME.update_actor(**captured)
    for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a,b)
