"""Independent winner law, stopped uniform OT projection, and profile isolation."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.distillation import conditional_ot_nll
from optiq_dime.winner_distillation import sample_winner_teacher, select_winner_atoms, winner_ot_targets
from scripts.verify_v5 import verify
from test_semi_implicit import actor_state
from test_best_of_k import directional_state


def update(actor, critic, obs, key):
    return OptiQDIME.update_actor(actor, critic, obs, key, jnp.array([-3600.]),
        16, 4, 'exact', 0., .5, False, False, 0., False, 16., 257,
        0., .1, 100, 'min', 'argmax', True, False,
        'conditional_ot_nll', 'best_of_k_winners', 0., False, 'mean', 8)


def test_winners_selected_per_group_by_min_and_keep_saturated_pretanh():
    u = jnp.array([[[[10.], [11.], [-10.]], [[-12.], [13.], [14.]]]])
    # Mean Q favors index 1 in both groups; min Q picks indices 2 and 0.
    qs = jnp.array([[[[1., 20., 4.], [3., 20., 1.]]],
                    [[[1., -2., 4.], [3., -2., 1.]]]])
    winners, metrics = select_winner_atoms(u, qs)
    np.testing.assert_array_equal(winners, [[[-10.], [-12.]]])
    assert np.all(np.isfinite(winners)) and np.all(np.tanh(winners) == -1.)
    assert float(metrics['teacher_winner_q_gain']) > 0
    assert np.count_nonzero(jax.grad(lambda x: select_winner_atoms(x, qs)[0].sum())(u)) == 0


def test_teacher_uses_independent_full_actor_draws_and_no_scale_floor():
    actor, critic = actor_state(), directional_state()
    params = copy.deepcopy(actor.params)
    params['log_std']['bias'] = jnp.full_like(params['log_std']['bias'], -5.)
    actor = actor.replace(params=params)
    obs = jnp.array([[1., 0., 0.], [-1., 0., 0.]])
    key = jax.random.PRNGKey(101)
    winners, metrics = sample_winner_teacher(actor, critic, obs, key, 64, 8)
    zk, ek, _ = jax.random.split(key, 3)
    shape = (2*64*8, 2)
    repeated = jnp.repeat(obs, 64*8, axis=0)
    z = jax.random.normal(zk, shape)
    mu, ls = actor.apply_fn({'params': actor.params}, repeated, z)
    u = (mu + jnp.exp(ls)*jax.random.normal(ek, shape)).reshape(2, 64, 8, 2)
    # The live minimum here is obs[0]*action[0]; mean and target prefer its negative.
    score = np.asarray(obs[:, 0])[:, None, None]*np.asarray(jnp.tanh(u)[..., 0])
    expected = np.asarray(u)[np.arange(2)[:, None], np.arange(64)[None, :], score.argmax(-1)]
    np.testing.assert_allclose(winners, expected, atol=2e-7)
    assert np.unique(np.asarray(winners[0, :, 0])).size == 64
    assert float(metrics['teacher_winner_q_gain']) > 0


def test_uniform_ot_full_row_nll_updates_both_heads_without_teacher_gradients():
    actor, critic = actor_state(), directional_state()
    obs, key = jnp.ones((4, 3)), jax.random.PRNGKey(41)
    changed, loss, next_key, metrics = update(actor, critic, obs, key)
    expected_key, lk, tk, _ = jax.random.split(key, 4)
    zk, _ = jax.random.split(lk)
    z = jax.random.normal(zk, (4, 16, 2))
    mu, ls = actor.apply_fn({'params': actor.params}, jnp.repeat(obs, 16, axis=0), z.reshape(-1, 2))
    mu, ls = mu.reshape(4, 16, 2), ls.reshape(4, 16, 2)
    u, _ = sample_winner_teacher(actor, critic, obs, tk, 64, 8)
    rows, ot_metrics = winner_ot_targets(mu, u, .1, 100)
    np.testing.assert_allclose(rows.mean(1), np.full((4, 64), 1/64), atol=2e-6)
    np.testing.assert_allclose(loss, conditional_ot_nll(mu, ls, u, rows), atol=1e-5)
    np.testing.assert_array_equal(next_key, expected_key)
    hard = jax.nn.one_hot(rows.argmax(-1), 64)
    assert abs(float(loss - conditional_ot_nll(mu, ls, u, hard))) > 1e-3
    assert metrics['teacher_winner_count'] == 64 and metrics['teacher_candidate_count'] == 512
    assert metrics['teacher_best_of_k'] == 8 and metrics['teacher_uniform_mass'] == 1
    assert 'temperature' not in metrics and 'teacher_log_density_mean' not in metrics
    assert float(ot_metrics['ot_col_marginal_error']) < 1e-6
    for head in ('mu', 'log_std'):
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(actor.params[head]), jax.tree.leaves(changed.params[head])))
    qgrad = jax.grad(lambda p: update(actor, critic.replace(params=p), obs, key)[1])(critic.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(qgrad))
    teachergrad = jax.grad(lambda p: sample_winner_teacher(actor.replace(params=p), critic, obs, tk, 64, 8)[0].sum())(actor.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(teachergrad))


@pytest.mark.parametrize('override', [
    'alg.actor.temperature=.25', 'alg.actor.density_correction=true',
    'alg.actor.density_correction_beta=1.', 'alg.actor.teacher_best_of_k=4',
    'alg.actor.teacher_best_of_k=true', 'alg.actor.teacher_best_of_k=1.5',
    'alg.actor.source_q_eval=mean', 'alg.actor.teacher_std_floor=.05',
    'alg.actor.ot_student_action=sample', 'alg.actor.normalize_ot_cost=true',
    '+alg.actor.temperature_schedule={enabled:true,final_temperature:.25,anneal_steps:40000}',
])
def test_winner_profile_rejects_old_boltzmann_settings_and_mismatched_selection(override):
    with pytest.raises(ValueError):
        verify([override], 'mujoco_v5_bestof8')
