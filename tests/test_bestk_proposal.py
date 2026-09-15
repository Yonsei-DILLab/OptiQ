"""Best-k guidance must change the proposal, not the OptiQ Boltzmann target."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.special import logsumexp

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.bestk_proposal import BestKGuidedProposal, make_bestk_proposal
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from scripts.verify_v5 import verify
from run_optiq_dime import validate_config
from test_best_of_k import directional_state
from test_semi_implicit import actor_state, critic_state


def test_independent_groups_use_original_gaussian_scales_and_configured_q():
    base = ConditionalGaussianProposal(jnp.array([[[-.5, .1], [.4, -.3]]]),
        jnp.log(jnp.array([[[.02, .8], [.3, .6]]])), .05)
    obs, key, critic = jnp.array([[1., 0., 0.]]), jax.random.PRNGKey(119), directional_state()
    proposal, metrics = make_bestk_proposal(base, critic, obs, key, 8, .5, 'mean')
    sample_key, _ = jax.random.split(key)
    actions, u, origins = base.sample(sample_key, 8, 'exact')
    # Mean( a[0], 10-3*a[0] ) = 5-a[0], opposite to twin-min here.
    expected_indices = (-np.asarray(actions[..., 0]).reshape(1, 2, 8)).argmax(-1)
    expected_u = np.asarray(u).reshape(1, 2, 8, 2)[np.arange(1)[:, None], np.arange(2)[None], expected_indices]
    expected_origins = np.asarray(origins).reshape(1, 2, 8)[np.arange(1)[:, None], np.arange(2)[None], expected_indices]
    np.testing.assert_allclose(proposal.guide_means, expected_u, atol=1e-7)
    np.testing.assert_array_equal(proposal.guide_origins, expected_origins)
    np.testing.assert_allclose(proposal.guide_log_std, np.asarray(base.effective_log_std())[0, expected_origins[0]][None], atol=1e-7)
    _, final_u, final_origins = proposal.sample(jax.random.PRNGKey(120), 4, 'exact')
    assert final_u.shape == (1, 8, 2) and np.all(np.asarray(final_origins) < 2)
    assert not np.any(np.all(np.asarray(final_u)[:, :, None] == expected_u[:, None], axis=-1))
    assert metrics['proposal_pilot_count'] == 16 and metrics['proposal_pilot_winner_q_gain'] > 0


def test_mixture_density_matches_independent_joint_gaussian_calculation():
    base = ConditionalGaussianProposal(jnp.array([[[-.5, .1], [.4, -.3]]]),
        jnp.log(jnp.array([[[.02, .8], [.3, .6]]])), .05)
    guide_mu = jnp.array([[[1., .5], [-1., .2]]])
    guide_ls = jnp.log(jnp.array([[[.5, .3], [.2, .4]]]))
    proposal = BestKGuidedProposal(base, guide_mu, guide_ls, jnp.array([[0, 1]]), .3)
    u = np.array([[[-.2, .7], [1.3, -.6], [9., -8.]]], dtype=np.float32)
    mu = np.concatenate([base.means, guide_mu], axis=1)
    ls = np.concatenate([base.effective_log_std(), guide_ls], axis=1)
    terms = (-.5*((u[:, :, None] - mu[:, None])*np.exp(-ls[:, None]))**2
             - ls[:, None] - .5*np.log(2*np.pi)).sum(-1)
    terms += np.log([.35, .35, .15, .15])
    jac = (2*(np.log(2)-u-np.logaddexp(0, -2*u))).sum(-1)
    expected = logsumexp(terms, axis=-1) - jac
    np.testing.assert_allclose(proposal.log_prob(jnp.asarray(u)), expected, atol=2e-4, rtol=2e-6)
    # q_mix >= (1-rho)*q_base: no removal of original proposal support.
    assert np.all(np.asarray(proposal.log_prob(jnp.asarray(u))) >= np.asarray(base.log_prob(jnp.asarray(u)))+np.log(.7)-1e-5)


def test_sampling_and_correction_recover_boltzmann_target_not_winner_law():
    base = ConditionalGaussianProposal(jnp.array([[[-.3], [.3]]]), jnp.full((1,2,1), np.log(.9)), .05)
    proposal = BestKGuidedProposal(base, jnp.array([[[.8], [1.1]]]),
        jnp.full((1,2,1), np.log(.7)), jnp.array([[0,1]]), .5)
    actions, u, _ = proposal.sample(jax.random.PRNGKey(491), 100000, 'exact')
    a = np.asarray(actions[0, :, 0])
    grid = np.linspace(-1., 1., 100001)
    q = lambda x: -(x-.55)**2
    temperature = .18
    target = np.exp(q(grid)/temperature)
    expected = np.trapz(grid*target, grid)/np.trapz(target, grid)
    logits = q(a)/temperature-np.asarray(proposal.log_prob(u))[0]
    weights = np.exp(logits-logsumexp(logits))
    actual = (weights*a).sum()
    assert abs(actual-expected) < .008
    wrong_logits = q(a)/temperature-np.asarray(base.log_prob(u))[0]
    wrong_weights = np.exp(wrong_logits-logsumexp(wrong_logits))
    assert abs((wrong_weights*a).sum()-expected) > .025
    # Independently verify the sampler's pre-tanh mixture moments.
    expected_mean = .5*0+.5*.95
    expected_second = .5*(.9**2+.3**2)+.5*(.7**2+(.8**2+1.1**2)/2)
    assert abs(np.asarray(u).mean()-expected_mean) < .01
    assert abs((np.asarray(u)**2).mean()-expected_second) < .015


def test_random_pilots_use_same_pools_and_scales_without_q_selection():
    base = ConditionalGaussianProposal(jnp.array([[[-.5, .1], [.4, -.3]]]),
        jnp.log(jnp.array([[[.02, .8], [.3, .6]]])), .05)
    obs, key, critic = jnp.array([[1., 0., 0.]]), jax.random.PRNGKey(119), directional_state()
    random, metrics = make_bestk_proposal(base, critic, obs, key, 8, .5, 'mean', 'first')
    best, best_metrics = make_bestk_proposal(base, critic, obs, key, 8, .5, 'mean')
    actions, u, origins = base.sample(jax.random.split(key)[0], 8, 'exact')
    expected_origins = np.asarray(origins).reshape(1, 2, 8)[:, :, 0]
    np.testing.assert_array_equal(random.guide_means, np.asarray(u).reshape(1, 2, 8, 2)[:, :, 0])
    np.testing.assert_array_equal(random.guide_origins, expected_origins)
    np.testing.assert_array_equal(random.guide_log_std,
        np.asarray(base.effective_log_std())[0, expected_origins[0]][None])
    scores = 5 - np.asarray(actions[..., 0]).reshape(1, 2, 8)
    np.testing.assert_allclose(metrics['proposal_pilot_selected_q_gain'],
        (scores[:, :, 0] - scores.mean(-1)).mean(), atol=1e-6)
    assert metrics['proposal_pilot_selects_best'] == 0
    assert best_metrics['proposal_pilot_selects_best'] == 1
    np.testing.assert_array_equal(metrics['proposal_pilot_winner_q_gain'],
        best_metrics['proposal_pilot_winner_q_gain'])
    assert metrics['proposal_pilot_count'] == best_metrics['proposal_pilot_count'] == 16
    assert not np.array_equal(random.guide_means, best.guide_means)
    # Reversing the critic changes winners but cannot change random-pilot centers.
    reverse, _ = make_bestk_proposal(base, critic.replace(params={'sign': -critic.params['sign']}),
        obs, key, 8, .5, 'mean', 'first')
    np.testing.assert_array_equal(random.guide_means, reverse.guide_means)


def update(actor, critic, obs, key, k=8, fraction=.5, selection='best'):
    return OptiQDIME.update_actor(actor, critic, obs, key, jnp.array([-3600.]),
        16, 4, 'exact', .05, .5, False, True, 1., False, 16., 257,
        .25, .1, 100, 'mean', 'argmax', True, False,
        'conditional_ot_nll', 'conditional_mixture', 0., False, 'mean', 1, k, fraction, selection)


@pytest.mark.parametrize('selection', ['best', 'first'])
def test_actor_updates_both_heads_without_gradients_through_pilot_q(selection):
    actor, critic = actor_state(), directional_state()
    obs, key = jnp.ones((4,3)), jax.random.PRNGKey(661)
    changed, loss, next_key, metrics = update(actor, critic, obs, key, selection=selection)
    assert np.isfinite(loss) and metrics['density_beta_mean'] == 1
    assert metrics['temperature'] == .25 and metrics['proposal_best_of_k'] == 8
    assert metrics['proposal_guided_fraction'] == .5
    assert metrics['proposal_pilot_count'] == 128
    assert metrics['proposal_pilot_selects_best'] == float(selection == 'best')
    for head in ['mu','log_std']:
        assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(actor.params[head]),jax.tree.leaves(changed.params[head])))
    qgrad = jax.grad(lambda p: update(actor, critic.replace(params=p), obs, key, selection=selection)[1])(critic.params)
    assert all(np.count_nonzero(g)==0 for g in jax.tree.leaves(qgrad))
    base = ConditionalGaussianProposal(jnp.zeros((4,16,2)), jnp.zeros((4,16,2)), .05)
    pilot_grad = jax.grad(lambda mu: make_bestk_proposal(base._replace(means=mu), critic,obs,key,8,.5,'mean',selection)[0].guide_means.sum())(base.means)
    assert np.count_nonzero(pilot_grad)==0
    np.testing.assert_array_equal(next_key,jax.random.split(key,4)[0])


def test_profile_only_changes_proposal_and_metadata_and_alias_matches():
    baseline = OmegaConf.to_container(verify(['benchmark=hopper','alg.actor.temperature=.01']),resolve=True)
    guided = OmegaConf.to_container(verify(['benchmark=hopper','alg.actor.temperature=.01'],'mujoco_v5_bestk_proposal'),resolve=True)
    expected = copy.deepcopy(baseline)
    expected['alg']['actor'].update(proposal_best_of_k=8,proposal_guided_fraction=.5)
    for key in ['run_name','wandb','output_root']: expected[key]=guided[key]
    assert guided==expected
    assert guided==OmegaConf.to_container(verify(['benchmark=hopper','alg.actor.temperature=.01'],'v5/bestk_proposal'),resolve=True)


def test_random_profile_only_changes_pilot_selection_and_metadata():
    best = OmegaConf.to_container(verify(['benchmark=ant'], 'mujoco_v5_bestk_proposal'), resolve=True)
    random = OmegaConf.to_container(verify(['benchmark=ant'], 'mujoco_v5_random_proposal'), resolve=True)
    expected = copy.deepcopy(best)
    expected['alg']['actor']['proposal_pilot_selection'] = 'first'
    for key in ['run_name', 'wandb', 'output_root']: expected[key] = random[key]
    assert random == expected
    assert random == OmegaConf.to_container(verify(['benchmark=ant'], 'v5/random_proposal'), resolve=True)


def test_combined_profile_only_adds_collection_to_corrected_teacher():
    guided = OmegaConf.to_container(verify(['benchmark=ant'], 'mujoco_v5_bestk_proposal'), resolve=True)
    combined = OmegaConf.to_container(verify(['benchmark=ant'], 'mujoco_v5_bestk_combined'), resolve=True)
    expected = copy.deepcopy(guided)
    expected['alg']['behavior_best_of_k'] = 8
    for key in ['run_name', 'wandb', 'output_root']: expected[key] = combined[key]
    assert combined == expected
    assert combined == OmegaConf.to_container(verify(['benchmark=ant'], 'v5/bestk_combined'), resolve=True)


@pytest.mark.parametrize('profile,selection', [
    ('mujoco_v5', 'first'), ('mujoco_v5_bestk_proposal', 'unknown'),
])
def test_invalid_pilot_selection_rejected(profile, selection):
    cfg = verify([], profile)
    OmegaConf.update(cfg, 'alg.actor.proposal_pilot_selection', selection, force_add=True)
    with pytest.raises(ValueError, match='proposal_pilot_selection'):
        validate_config(cfg)


@pytest.mark.parametrize('override', [
    'alg.actor.proposal_best_of_k=0','alg.actor.proposal_best_of_k=true',
    'alg.actor.proposal_best_of_k=1.5','alg.actor.proposal_guided_fraction=1.',
    'alg.actor.proposal_guided_fraction=0.','alg.actor.density_correction=false',
    'alg.actor.density_correction_beta=0.','alg.actor.proposal_sampling_mode=stratified',
    'alg.actor.teacher_distribution=best_of_k_winners',
])
def test_invalid_guidance_rejected(override):
    with pytest.raises(ValueError): verify([override],'mujoco_v5_bestk_proposal')
