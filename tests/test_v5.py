"""Mean-action OT semantics and the actual v5 Ant training/evaluation path."""
import copy
import csv
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.distillation import conditional_ot_nll
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.transport import sinkhorn
import optiq_dime.policy as policy_module
from run_optiq_dime import create_algorithm
from scripts.verify_v4 import verify as verify_v4
from scripts.verify_v5 import verify
from test_semi_implicit import actor_state, critic_state


def update(actor, critic, obs, key, mode):
    return OptiQDIME.update_actor(actor, critic, obs, key, jnp.array([-3600.]),
        16, 4, 'exact', .05, .5, False, True, 1., False, 16., 257,
        .25, .25, 100, 'mean', 'argmax', True, False,
        'conditional_ot_nll', 'conditional_mixture', 0., False, mode)


def test_default_and_alias_apply_requested_no_clip_sinkhorn_defaults():
    cfg = OmegaConf.to_container(verify(['benchmark=ant']), resolve=True)
    alias = OmegaConf.to_container(verify(['benchmark=ant'], 'v5/final'), resolve=True)
    expected = OmegaConf.to_container(verify_v4(['benchmark=ant', 'alg.actor.temperature=.25']), resolve=True)
    assert expected['alg']['actor'].get('ot_student_action', 'sample') == 'sample'
    assert expected['alg']['actor']['sinkhorn_epsilon'] == .25
    assert expected['alg']['optimizer']['ac_grad_norm'] == 2.
    expected['alg']['actor']['ot_student_action'] = 'mean'
    expected['alg']['actor']['sinkhorn_epsilon'] = .1
    expected['alg']['optimizer']['ac_grad_norm'] = None
    for key in ('run_name', 'output_root', 'wandb'):
        expected[key] = cfg[key]
    assert cfg == expected == alias
    assert cfg['alg']['actor']['temperature'] == .25
    assert cfg['alg']['actor']['sinkhorn_iterations'] == 100
    assert cfg['alg']['behavior_uniform_probability'] == 0
    assert 'temperature_schedule' not in cfg['alg']['actor']
    assert cfg['wandb']['project'] == 'v4-test'
    assert 'v5_meanOT' in cfg['wandb']['group']


@pytest.mark.parametrize('override', [
    'alg.actor.ot_student_action=sample', 'alg.actor.ot_student_action=invalid',
    'alg.behavior_uniform_probability=.1', 'dual_mu_eval=false',
    'alg.actor.teacher_distribution=realized_kde',
    'alg.actor.distillation_loss=pointwise_mse',
    '+alg.actor.temperature_schedule={enabled:true,final_temperature:0.25,anneal_steps:40000}',
])
def test_v5_launcher_rejects_conflicting_variant(override):
    with pytest.raises(ValueError):
        verify([override])


@pytest.mark.parametrize('seed', [0, 1])
def test_mean_cost_full_nll_and_preserved_teacher_rng(seed):
    actor, critic = actor_state(), critic_state()
    obs, key = jnp.ones((4, 3)), jax.random.PRNGKey(seed + 17)
    original = update(actor, critic, obs, key, 'sample')
    changed = update(actor, critic, obs, key, 'mean')
    np.testing.assert_array_equal(original[2], changed[2])
    for metric in ('teacher_log_density_mean', 'source_ess_absolute',
                   'student_action_saturation_fraction', 'actor_std_mean'):
        np.testing.assert_allclose(original[3][metric], changed[3][metric], atol=1e-6)

    # A constant critic makes beta=1 weights exactly softmax(-log q).
    @jax.jit
    def reconstruct(params):
        _, latent_key, proposal_key, _ = jax.random.split(key, 4)
        zk, _ = jax.random.split(latent_key)
        z = jax.random.normal(zk, (4, 16, 2))
        mu, ls = actor.apply_fn({'params': params}, jnp.repeat(obs, 16, axis=0), z.reshape(-1, 2))
        mu, ls = mu.reshape(4, 16, 2), ls.reshape(4, 16, 2)
        teacher = ConditionalGaussianProposal(mu, ls, .05)
        actions, u, _ = teacher.sample(proposal_key, 4, 'exact')
        weights = jax.nn.softmax(-teacher.log_prob(u), axis=-1)
        cost = ((jnp.tanh(mu)[:, :, None] - actions[:, None])**2).sum(-1)
        plan = sinkhorn(cost, weights, .25, 100)
        rows = plan / jnp.maximum(plan.sum(-1, keepdims=True), 1e-20)
        return conditional_ot_nll(mu, ls, u, rows), cost.mean()

    nll, cost = reconstruct(actor.params)
    np.testing.assert_allclose(changed[1], nll, atol=2e-6)
    np.testing.assert_allclose(changed[3]['ot_cost_mean'], cost, atol=2e-6)
    assert abs(float(original[3]['ot_cost_mean'] - cost)) > .01
    for head in ('mu', 'log_std'):
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(actor.params[head]),
            jax.tree_util.tree_leaves(changed[0].params[head])))
    qgrad = jax.grad(lambda p: update(actor, critic.replace(params=p), obs, key, 'mean')[1])(critic.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(qgrad))


def test_real_ant_routes_mean_ot_and_keeps_td_and_paired_evaluation(tmp_path, monkeypatch):
    cfg = verify(['benchmark=ant', f'output_root={tmp_path}',
        'alg.batch_size=4', 'alg.buffer_size=32', 'alg.learning_starts=2',
        'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
        'diagnostic_interval=4', 'checkpoint_interval=4'])
    seen = []
    optimizer_limits = []
    original_optimizer = policy_module.adam_with_grad_clip
    def record_optimizer(*args, **kwargs):
        optimizer_limits.append(kwargs['max_grad_norm'])
        return original_optimizer(*args, **kwargs)
    monkeypatch.setattr(policy_module, 'adam_with_grad_clip', record_optimizer)
    original = OptiQDIME.update_actor
    signature = inspect.signature(original)
    def record_update(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        seen.append(arguments['ot_student_action'])
        return original(*args, **kwargs)
    OptiQDIME._train.clear_cache()
    monkeypatch.setattr(OptiQDIME, 'update_actor', staticmethod(record_update))
    model, callbacks = create_algorithm(cfg)
    # Capture the numeric epsilon before entering JIT; inside update_actor it
    # is a tracer, whereas the OT mode above is a static argument.
    sinkhorn_epsilons = []
    original_train = model._train
    train_signature = inspect.signature(original_train)
    def record_train(*args, **kwargs):
        arguments = train_signature.bind(*args, **kwargs).arguments
        sinkhorn_epsilons.append(float(arguments['sinkhorn_epsilon']))
        return original_train(*args, **kwargs)
    monkeypatch.setattr(model, '_train', record_train)
    model.set_logger(configure(str(tmp_path/'test_logs'), ['csv']))
    cb = callbacks.callbacks[0]
    cb.eval_env.envs[0].env._max_episode_steps = 2
    model.get_env().envs[0].env._max_episode_steps = 2
    before = copy.deepcopy(model.policy.actor_state.params)
    try:
        model.learn(total_timesteps=8, callback=callbacks)
        assert seen and set(seen) == {'mean'}
        assert len(sinkhorn_epsilons) == 6 and np.allclose(sinkhorn_epsilons, .1)
        assert optimizer_limits == [None, None]  # Actual critic and actor construction.
        assert model._n_updates == int(model.policy.actor_state.step) == 6
        assert model.behavior_uniform_count == 0 and model.backup_mode == 'td'
        assert model.soft_guard_attempts == 0
        for head in ('mu', 'log_std'):
            assert not np.array_equal(before[head]['kernel'], model.policy.actor_state.params[head]['kernel'])
        assert cb.evaluations_timesteps == [1, 4, 8]
        for mode in cb.MODES:
            with np.load(cb.directory/f'evaluations_{mode}.npz') as data:
                assert data['results'].shape == (3, 2)
                assert np.isfinite(data['results']).all()
        np.testing.assert_array_equal(cb.histories['zero_z']['env_seeds'], cb.histories['stochastic_z']['env_seeds'])
        np.testing.assert_array_equal(cb.histories['zero_z']['policy_seeds'], cb.histories['stochastic_z']['policy_seeds'])
        with (tmp_path/'test_logs/progress.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert all(float(r['train/backup_entropy_term']) == 0 for r in rows if r.get('train/backup_entropy_term'))
        assert model.replay_buffer.timeouts[:8].sum() > 0
        assert not np.asarray(model.replay_buffer._get_samples(np.arange(8)).dones).any()
    finally:
        cb.eval_env.close()
        model.get_env().close()
        model.logger.close()
        OptiQDIME._train.clear_cache()
