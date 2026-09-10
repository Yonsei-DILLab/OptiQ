"""Distribution identities, gradient boundaries and common-loop regressions."""

import csv
from pathlib import Path

from flax import serialization
from flax.training.train_state import TrainState
import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
from stable_baselines3.common.logger import configure

from common.type_aliases import RLTrainState
from optiq_dime import OptiQDIME
from optiq_dime.policy import SemiImplicitActor, OptiQPolicy
from optiq_dime.semi_implicit import (
    PretanhTeacherKDE, conditional_mixture_log_prob,
    idac_action_and_log_density, tanh_log_jacobian,
)
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="mujoco_v2", overrides=list(overrides))


def actor_state(action_dim=2):
    model = SemiImplicitActor(action_dim, (16, 16), -5., 1., -.69314718)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)), jnp.zeros((1, action_dim)))["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(.01))


def critic_fn(variables, observations, actions, rngs=None, mutable=False, train=False):
    q = variables["params"]["q"][:, None, None]
    result = jnp.broadcast_to(q, (2, observations.shape[0], 1))
    return (result, {"batch_stats": {}}) if mutable else result


def critic_state():
    return RLTrainState.create(apply_fn=critic_fn, params={"q": jnp.array([2., 6.])},
        batch_stats={}, target_params={"q": jnp.array([10., 14.])}, target_batch_stats={}, tx=optax.sgd(.01))


def test_joint_mixture_density_jacobian_and_state_isolation():
    mu = np.array([[[-1., 2.], [1., -2.]], [[10., 12.], [13., 9.]]], dtype=np.float32)
    std = np.array([[[.2, .4], [.3, .7]], [[1., .6], [.8, .5]]], dtype=np.float32)
    u = np.array([[[.1, 1.], [-1., -2.]], [[11., 10.], [12., 11.]]], dtype=np.float32)
    component = norm.logpdf(u[:, :, None], loc=mu[:, None], scale=std[:, None]).sum(-1)
    jac = (2*(np.log(2)-u-np.logaddexp(0, -2*u))).sum(-1)
    expected = logsumexp(component, axis=-1)-np.log(2)-jac
    actual = conditional_mixture_log_prob(jnp.array(u), jnp.array(mu), jnp.log(std))
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    for i in range(2):
        single = conditional_mixture_log_prob(jnp.array(u[i:i+1]), jnp.array(mu[i:i+1]), jnp.log(std[i:i+1]))
        np.testing.assert_allclose(actual[i], single[0], atol=2e-5)
    assert np.isfinite(tanh_log_jacobian(jnp.array([[-100., 100.]]))).all()


def test_generating_component_is_included_and_original_pretanh_is_used():
    actor = actor_state()
    obs = jnp.ones((7, 3))
    key = jax.random.PRNGKey(31)
    action, log_g = idac_action_and_log_density(actor, obs, key, 16)
    latent_key, noise_key = jax.random.split(key)
    z = jax.random.normal(latent_key, (7, 16, 2))
    mu, ls = actor.apply_fn({"params": actor.params}, jnp.repeat(obs, 16, axis=0), z.reshape(-1, 2))
    mu, ls = mu.reshape(7,16,2), ls.reshape(7,16,2)
    u = mu[:,0] + jnp.exp(ls[:,0])*jax.random.normal(noise_key, (7,2))
    expected = conditional_mixture_log_prob(u[:,None], mu, ls)[:,0]
    np.testing.assert_array_equal(action, jnp.tanh(u))
    np.testing.assert_allclose(log_g, expected, atol=1e-6)
    # Widely separated components make omission of the generating one observable.
    centers = jnp.array([[[0.], [100.]]])
    lp = conditional_mixture_log_prob(jnp.array([[[0.]]]), centers, jnp.zeros_like(centers))
    np.testing.assert_allclose(lp, [[norm.logpdf(0)-np.log(2)]], atol=1e-6)


@pytest.mark.parametrize("mode", ["exact", "stratified"])
def test_teacher_sampling_no_anchors_full_gaussian_and_matching_density(mode):
    centers = jnp.broadcast_to(jnp.arange(16)[None,:,None]/4, (4,16,2))
    kde = PretanhTeacherKDE(centers, .5)
    action, u, idx = kde.sample(jax.random.PRNGKey(22), 512, mode)
    assert action.shape == u.shape == (4,8192,2)
    selected = np.take_along_axis(np.asarray(centers), np.asarray(idx)[:,:,None], axis=1)
    noise = np.asarray(u)-selected
    assert abs(noise.mean()) < .01 and abs(noise.std()-.5) < .01
    assert np.max(np.abs(noise)) > 3*.5  # No local hard cutoff.
    assert not np.any(np.all(noise == 0, axis=-1))
    counts = np.bincount(np.asarray(idx[0]), minlength=16)
    if mode == "stratified":
        np.testing.assert_array_equal(counts, np.full(16,512))
    else:
        assert np.std(counts) > 0 and np.max(np.abs(counts-512)) < 100
    expected = conditional_mixture_log_prob(u[:,:64], centers, jnp.full_like(centers, np.log(.5)))
    np.testing.assert_array_equal(kde.log_prob(u[:,:64]), expected)


def test_soft_td_target_min_q_entropy_terminal_and_stop_gradient():
    actor, critic = actor_state(), critic_state()
    obs, actions = jnp.zeros((2,3)), jnp.zeros((2,2))
    rewards, dones, key = jnp.array([1., 3.]), jnp.array([0., 1.]), jax.random.PRNGKey(42)
    temperature = .25
    def update(a=actor, td_std=0.):
        return OptiQDIME.update_critic(False, False, .9, a, critic, obs, actions, obs,
            rewards, dones, 1, jnp.array([-3600.]), -3600., 3600., 0., td_std, .5, key,
            True, 16, temperature)
    _, log_g = idac_action_and_log_density(actor, obs, jax.random.split(key,6)[1], 16)
    expected_y = np.array([1.+.9*(10.-temperature*log_g[0]), 3.])
    result, metrics, _ = update()
    expected_loss = np.square(np.array([2.,6.])[:,None]-expected_y).mean(1).sum()
    np.testing.assert_allclose(metrics["critic_loss"], expected_loss, rtol=2e-6)
    np.testing.assert_allclose(metrics["backup_entropy_lower"], -log_g.mean(), rtol=2e-6)
    assert metrics["ent_coef"] == temperature
    # The semi-implicit path ignores the legacy smoothing arguments entirely.
    np.testing.assert_array_equal(metrics["critic_loss"], update(td_std=100.)[1]["critic_loss"])
    grad = jax.grad(lambda p: update(actor.replace(params=p))[1]["critic_loss"])(actor.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(grad))


@pytest.mark.parametrize("mode", ["exact", "stratified"])
def test_ot_updates_both_heads_without_q_gradients(mode):
    actor, critic = actor_state(), critic_state()
    def update(q=critic):
        return OptiQDIME.update_actor(actor, q, jnp.ones((3,3)), jax.random.PRNGKey(5),
            jnp.array([-3600.]), 16, 4, mode, .5, .5, False, True, 1., False, 16., 257,
            .25, .1, 100, "mean", "argmax", True, False)
    new, loss, _, metrics = update()
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in metrics.values())
    for head in ("mu", "log_std"):
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(actor.params[head]),
                                                        jax.tree_util.tree_leaves(new.params[head])))
    assert 1 <= metrics["source_ess_absolute"] <= 64.001
    # Reconstruct the saved noise realization: centers must be realized u,
    # not mu, and IS must use the density after the tanh Jacobian correction.
    _, latent_key, proposal_key, _ = jax.random.split(jax.random.PRNGKey(5), 4)
    zk, ek = jax.random.split(latent_key)
    z = jax.random.normal(zk, (3,16,2))
    mu, ls = actor.apply_fn({"params":actor.params}, jnp.ones((48,3)), z.reshape(48,2))
    mu, ls = mu.reshape(3,16,2), ls.reshape(3,16,2)
    u = mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape)
    kde = PretanhTeacherKDE(u,.5)
    _, v, _ = kde.sample(proposal_key,4,mode)
    log_q = kde.log_prob(v)
    weights = jax.nn.softmax(-log_q,axis=-1)  # Constant critic cancels.
    np.testing.assert_allclose(metrics["teacher_log_density_mean"],log_q.mean(),atol=2e-6)
    np.testing.assert_allclose(metrics["source_ess_absolute"],(1/(weights**2).sum(-1)).mean(),rtol=2e-6)
    q_grad = jax.grad(lambda p: update(critic.replace(params=p))[1])(critic.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(q_grad))


def test_initial_std_is_controlled_even_for_large_observations_and_no_uniform_default():
    cfg = config()
    assert cfg.alg.behavior_uniform_probability == 0
    actor = actor_state()
    mu, ls = actor.apply_fn({"params":actor.params}, jnp.full((8,3),10000.),
                            jax.random.normal(jax.random.PRNGKey(9),(8,2)))
    np.testing.assert_allclose(jnp.exp(ls), .5, atol=1e-6)


def test_categorical_backup_shifts_support_by_the_same_entropy_term():
    def distributional(variables, observations, actions, rngs=None, mutable=False, train=False):
        probs=jax.nn.softmax(variables['params']['logits'],axis=-1)
        out=jnp.broadcast_to(probs[:,None,:],(2,len(observations),3))
        return (out,{'batch_stats':{}}) if mutable else out
    live=jnp.array([[0.,1.,2.],[2.,1.,0.]])
    target=jnp.array([[1.,2.,0.],[0.,1.,2.]])
    critic=RLTrainState.create(apply_fn=distributional,params={'logits':live},batch_stats={},
        target_params={'logits':target},target_batch_stats={},tx=optax.sgd(.01))
    actor=actor_state()
    obs,actions=jnp.zeros((2,3)),jnp.zeros((2,2))
    rewards,dones=jnp.array([.2,.7]),jnp.array([0.,1.])
    key=jax.random.PRNGKey(101)
    _,lp=idac_action_and_log_density(actor,obs,jax.random.split(key,6)[1],16)
    _,metrics,_=OptiQDIME.update_critic(False,False,.9,actor,critic,obs,actions,obs,rewards,dones,
        3,jnp.array([-2.,0.,2.]),-2.,2.,0.,0.,0.,key,True,16,.25)
    target_probs=np.asarray(jax.nn.softmax(target,axis=-1)).mean(axis=0)
    expected=np.zeros((2,3))
    for row in range(2):
        for atom,prob in zip([-2.,0.,2.],target_probs):
            value=np.clip(float(rewards[row])+.9*(1-float(dones[row]))*(atom-.25*float(lp[row])),-2,2)
            pos=(value+2)/2; low=int(np.floor(pos)); high=int(np.ceil(pos))
            if low==high:
                expected[row,low]+=prob
            else:
                expected[row,low]+=prob*(high-pos);expected[row,high]+=prob*(pos-low)
    expected_loss=-(expected[None]*np.asarray(jax.nn.log_softmax(live,axis=-1))[:,None,:]).sum(-1).mean(-1).sum()
    np.testing.assert_allclose(metrics['critic_loss'],expected_loss,rtol=2e-6)


@pytest.mark.parametrize("override", ["alg.actor.include_anchor=true", "alg.actor.density_correction_beta=0.1",
    "alg.actor.entropy_samples=0", "alg.actor.td_noise_std=0.2", "alg.ent_coef.init=0.0",
    "alg.actor.log_std_min=2.0", "alg.actor.proposal_std_pretanh=-1"])
def test_invalid_v2_config_rejected(override):
    with pytest.raises(ValueError):
        validate_config(config([override]))


def test_common_humanoid_loop_timeout_replay_heads_and_checkpoint(tmp_path):
    cfg = config(["alg.batch_size=4", "alg.buffer_size=32", "alg.learning_starts=2",
                  "alg.actor.learning_starts=2", "diagnostic_interval=1"])
    assert validate_config(cfg)
    env = gym.make("Humanoid-v4", max_episode_steps=2)
    model = OptiQDIME("MlpPolicy", env, str(tmp_path), 4, cfg)
    model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
    before = model.policy.actor_state.params
    try:
        model.learn(total_timesteps=8)
        assert model._n_updates == 6 and int(model.policy.actor_state.step) == 6
        assert model.replay_buffer.timeouts[:8].sum() > 0
        sample = model.replay_buffer._get_samples(np.arange(8))
        assert np.count_nonzero(sample.dones.numpy()) == 0  # TimeLimit still bootstraps.
        # Diagnostic dumps clear the logger's in-memory values; verify the
        # actual emitted records instead of a fallback/default logger value.
        with (tmp_path / "logs/progress.csv").open() as stream:
            coefficients = [float(row["train/ent_coef"]) for row in csv.DictReader(stream)
                            if row.get("train/ent_coef")]
        assert coefficients and all(x == cfg.alg.actor.temperature for x in coefficients)
        for head in ("mu", "log_std"):
            assert not np.array_equal(before[head]["kernel"], model.policy.actor_state.params[head]["kernel"])
        files = list(tmp_path.glob("actor_state_8.msgpack"))
        assert len(files) == 1
        restored = serialization.from_bytes(model.policy.actor_state, files[0].read_bytes())
        obs, key = jnp.zeros((1,376)), jax.random.PRNGKey(9)
        np.testing.assert_array_equal(OptiQPolicy.sample_action(restored,obs,key),
                                      OptiQPolicy.sample_action(model.policy.actor_state,obs,key))
    finally:
        model.get_env().close()
        model.logger.close()
