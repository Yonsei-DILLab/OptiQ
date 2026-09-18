import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import softmax

from optiq_dime.proximal import proximal_policy_weights
from test_soft_improvement import entropy, exact_value


def test_exact_proximal_operator_improves_soft_value_with_statewise_step_sizes():
    rng = np.random.default_rng(7101)
    for _ in range(20):
        transitions = rng.dirichlet(np.ones(5), size=(5, 7))
        rewards = rng.normal(size=(5, 7))
        old = rng.dirichlet(np.full(7, .3), size=5)
        temperature, gamma = .1, .99
        value = exact_value(old, rewards, transitions, temperature, gamma)
        q = rewards + gamma*np.einsum('san,n->sa', transitions, value)
        eta = rng.uniform(.01, 1., (5, 1))
        target = softmax((1-eta)*np.log(old) + eta*q/temperature, axis=-1)
        gain = ((target-old)*q).sum(-1) + temperature*(entropy(target)-entropy(old))
        kl = (target*(np.log(target)-np.log(old))).sum(-1)
        penalty = temperature*(1-eta[:, 0])/eta[:, 0]
        assert np.all(gain >= penalty*kl-1.e-10)
        assert np.all(exact_value(target, rewards, transitions, temperature, gamma) >= value-1.e-9)


def test_target_is_stationary_at_soft_boltzmann_policy():
    q = np.array([[-.3, 0., .2, .5]])
    old = softmax(q/.2, axis=-1)
    for eta in [0., .03, .5, 1.]:
        np.testing.assert_allclose(softmax((1-eta)*np.log(old)+eta*q/.2, axis=-1), old)


def test_ess_search_attains_largest_feasible_step_and_full_step_when_possible():
    q = jnp.array([[0., 0., 0., 30.], [0., 0., 0., 0.]])
    logp = jnp.full_like(q, -2.)
    weights, eta = jax.jit(proximal_policy_weights)(q, logp, .1, 3.)
    ess = 1/np.square(weights).sum(-1)
    np.testing.assert_allclose(ess, [3., 4.], atol=1.e-4)
    assert 0 < eta[0] < .01 and eta[1] == 1.
    increased = softmax((float(eta[0])+1.e-4)*(np.asarray(q[0])/.1-np.asarray(logp[0])))
    assert 1/np.square(increased).sum() < 3.
    full, eta_full = proximal_policy_weights(q, logp, .1, 1.)
    np.testing.assert_array_equal(eta_full, [1., 1.])
    np.testing.assert_allclose(full, softmax(np.asarray(q)/.1-np.asarray(logp), axis=-1), atol=1.e-6)
    identity, eta_identity = proximal_policy_weights(q, logp, .1, 4.)
    np.testing.assert_array_equal(eta_identity, [0., 0.])
    np.testing.assert_allclose(identity, np.full((2,4), .25))


def test_proposal_ratio_requires_both_q_and_density_terms_scaled():
    # Independent discrete quadrature: old(a)*w(a) must equal the ideal target.
    old = np.array([.8, .15, .05]); q = np.array([.1, .4, .2]); eta=.2; temperature=.1
    target = softmax((1-eta)*np.log(old)+eta*q/temperature)
    weighted = old*np.exp(eta*(q/temperature-np.log(old)))
    np.testing.assert_allclose(weighted/weighted.sum(), target)
    wrong = old*np.exp(q/temperature-eta*np.log(old))
    assert np.max(np.abs(wrong/wrong.sum()-target)) > .1


def test_q_shift_invariance_and_no_gradients_through_target():
    q = jnp.array([[.2, .5, .9, -.1]])
    lp = jnp.array([[-1., -2., -3., -.7]])
    a, eta = proximal_policy_weights(q, lp, .1, 2.)
    b, eta_b = proximal_policy_weights(q+100., lp, .1, 2.)
    np.testing.assert_allclose(a,b,atol=3.e-5)
    np.testing.assert_allclose(eta,eta_b,atol=3.e-5)
    gradients = jax.grad(lambda x,y: proximal_policy_weights(x,y,.1,2.)[0][0,0],argnums=(0,1))(q,lp)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_common_actor_path_uses_exact_policy_and_stops_q_gradients():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import gymnasium as gym
    from optiq_dime import OptiQDIME
    from optiq_dime.policy import OptiQPolicy
    from test_semi_implicit import critic_state
    from run_optiq_dime import validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg=compose(config_name='mujoco_v2_proximal',overrides=['alg.actor.hidden_dims=[16,16]'])
    assert validate_config(cfg)
    policy=OptiQPolicy(gym.spaces.Box(-1.,1.,(3,),dtype=np.float32),
                      gym.spaces.Box(-1.,1.,(2,),dtype=np.float32),cfg)
    policy.build(jax.random.PRNGKey(9),lambda _: .0003,.0003)
    actor=policy.actor_state; critic=critic_state()
    def update(q, floor, fraction):
        return OptiQDIME.update_actor(actor,q,jnp.zeros((4,3)),jax.random.PRNGKey(4),
            jnp.zeros(1),16,4,'exact',floor,.5,False,True,1.,False,16.,257,
            .1,.25,100,'min','argmax',True,False,'conditional_ot_nll','conditional_mixture',fraction)
    new,loss,_,metrics=update(critic,.05,.25)
    assert np.isfinite(loss) and int(new.step)==1
    assert metrics['source_ess_min'] >= 16.-1.e-4
    assert metrics['proximal_actual_policy_proposal']==1.
    assert metrics['proposal_std_pretanh']==0.
    # A teacher floor cannot silently change the proximal sampling policy.
    large_floor=update(critic,8.,.25)
    for a,b in zip(jax.tree_util.tree_leaves(new.params),jax.tree_util.tree_leaves(large_floor[0].params)):
        np.testing.assert_array_equal(a,b)
    gradients=jax.grad(lambda p:update(critic.replace(params=p),.05,.25)[1])(critic.params)
    for gradient in jax.tree_util.tree_leaves(gradients):
        np.testing.assert_array_equal(gradient,np.zeros_like(gradient))


@pytest.mark.parametrize('override',[
    'alg.actor.soft_proximal_ess_fraction=-0.1','alg.actor.soft_proximal_ess_fraction=1.1',
    'alg.actor.soft_proximal_ess_fraction=0.001','alg.actor.latent_prior=normal',
    'alg.actor.teacher_distribution=realized_kde'])
def test_invalid_proximal_config_rejected(override):
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    from run_optiq_dime import validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg=compose(config_name='mujoco_v2_proximal',overrides=[override])
    with pytest.raises(ValueError):
        validate_config(cfg)
