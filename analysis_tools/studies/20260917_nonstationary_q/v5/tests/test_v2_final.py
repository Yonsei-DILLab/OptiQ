"""Release regression: frozen settings, independent arithmetic and common loop."""
import importlib.util
import json
from pathlib import Path

from flax import serialization
import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from optiq_dime import OptiQDIME
from optiq_dime.distillation import conditional_ot_nll
from optiq_dime.policy import OptiQPolicy
from optiq_dime.semi_implicit import ConditionalGaussianProposal, conditional_mixture_log_prob
from optiq_dime.transport import sinkhorn
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_reference_math', ROOT/'docs/v2/reference_math.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def config(name='mujoco_v2', overrides=()):
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize('name', ['mujoco_v2', 'mujoco_v2_checked', 'v2/final'])
def test_final_entries_match_completed_algorithm(name):
    cfg = config(name)
    saved = json.loads((ROOT/'docs/v2/REFERENCE_CONFIG.json').read_text())
    assert OmegaConf.to_container(cfg.alg, resolve=True) == saved['alg']
    for key in ('env_name','task','total_steps','eval_interval','num_eval_episodes',
                'checkpoint_interval','diagnostic_interval','stochastic_eval','eval_at_start'):
        assert cfg[key] == saved[key]
    assert cfg.alg.actor.get('latent_prior','normal') == 'normal'
    assert cfg.alg.actor.get('soft_proximal_ess_fraction',0) == 0
    assert validate_config(cfg)


@pytest.mark.parametrize('old,new', [
    ('mujoco_v2','archive/v2/original'),
    ('mujoco_v2_conditional','archive/v2/conditional'),
    ('mujoco_v2_guarded','archive/v2/guarded'),
    ('mujoco_v2_finite','archive/v2/finite'),
    ('mujoco_v2_proximal','archive/v2/proximal'),
])
def test_archives_preserve_resolved_settings(old,new):
    before=json.loads((ROOT/'tests/data/v2_pre_final_configs.json').read_text())[old]
    assert OmegaConf.to_container(config(new),resolve=True) == before
    if old != 'mujoco_v2':
        assert OmegaConf.to_container(config(old),resolve=True) == before


@pytest.mark.parametrize('benchmark,env', [('humanoid','Humanoid-v4'),('ant','Ant-v4'),
    ('halfcheetah','HalfCheetah-v4'),('walker2d','Walker2d-v4'),('hopper','Hopper-v4')])
def test_final_benchmark_override(benchmark,env):
    cfg=config(overrides=[f'benchmark={benchmark}'])
    assert cfg.env_name == env
    assert validate_config(cfg)


def test_numpy_reference_agrees_with_jax_density_ot_and_nll_gradients():
    rng=np.random.default_rng(84)
    mu=(rng.normal(size=(2,16,3))*.2).astype(np.float32)
    ls=rng.uniform(-4,-.2,size=mu.shape).astype(np.float32)
    proposal=ConditionalGaussianProposal(jnp.array(mu),jnp.array(ls),.05)
    key=jax.random.PRNGKey(73)
    action,u,indices=proposal.sample(key,4,'exact')
    ck,nk=jax.random.split(key)
    noise=np.asarray(jax.random.normal(nk,(2,64,3),dtype=jnp.float32))
    a_ref,u_ref,lp_ref=reference.teacher_from_draws(mu,ls,np.asarray(indices),noise)
    np.testing.assert_allclose(action,a_ref,atol=2e-6)
    np.testing.assert_allclose(u,u_ref,atol=2e-6)
    np.testing.assert_allclose(proposal.log_prob(u),lp_ref,atol=2e-4,rtol=2e-5)
    # Multiple coordinates/components, with the floor active only on some scales.
    np.testing.assert_allclose(conditional_mixture_log_prob(u,jnp.array(mu),jnp.array(ls)),
                               reference.log_mixture(np.asarray(u),mu,ls),rtol=2e-5,atol=3e-4)
    q1=rng.normal(size=(2,64))*.03; q2=rng.normal(size=(2,64))*.03
    weights,ess=reference.importance_weights(q1,q2,lp_ref)
    student=np.tanh(mu+np.exp(ls)*rng.normal(size=mu.shape))
    cost=((student[:,:,None]-a_ref[:,None])**2).sum(-1)
    plan=sinkhorn(jnp.array(cost,dtype=jnp.float32),jnp.array(weights,dtype=jnp.float32),.25,100)
    expected=reference.sinkhorn(cost,weights)
    np.testing.assert_allclose(plan,expected,rtol=5e-5,atol=2e-7)
    rows=reference.row_probabilities(expected).astype(np.float32)
    loss,dmu,dls=reference.conditional_nll_and_gradients(mu,ls,np.asarray(u),rows)
    got=conditional_ot_nll(jnp.array(mu),jnp.array(ls),u,jnp.array(rows))
    grad=jax.grad(conditional_ot_nll,argnums=(0,1))(jnp.array(mu),jnp.array(ls),u,jnp.array(rows))
    np.testing.assert_allclose(got,loss,rtol=2e-5)
    np.testing.assert_allclose(grad[0],dmu,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(grad[1],dls,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(loss,reference.direct_conditional_nll(mu,ls,np.asarray(u),rows),rtol=2e-5)
    reference.self_check()


def test_final_common_humanoid_loop_guard_timeout_and_checkpoint(tmp_path):
    # Keep final 256x3 networks, 16x64 OT, M16 and J8; shorten data collection only.
    cfg=config(overrides=['alg.batch_size=4','alg.buffer_size=32','alg.learning_starts=2',
        'alg.actor.learning_starts=2','diagnostic_interval=1'])
    assert validate_config(cfg)
    env=gym.make('Humanoid-v4',max_episode_steps=2)
    model=OptiQDIME('MlpPolicy',env,str(tmp_path),4,cfg)
    model.set_logger(configure(str(tmp_path/'logs'),['csv']))
    try:
        model.learn(total_timesteps=8)
        assert model._n_updates == 6
        assert model.soft_guard_attempts == 6
        assert int(model.policy.actor_state.step) == model.soft_guard_accepts
        assert model.replay_buffer.timeouts[:8].sum() > 0
        assert not np.asarray(model.replay_buffer._get_samples(np.arange(8)).dones).any()
        saved=(tmp_path/'actor_state_8.msgpack').read_bytes()
        restored=serialization.from_bytes(model.policy.actor_state,saved)
        obs,key=jnp.zeros((1,376)),jax.random.PRNGKey(12)
        np.testing.assert_array_equal(OptiQPolicy.sample_action(restored,obs,key),
            OptiQPolicy.sample_action(model.policy.actor_state,obs,key))
        assert (tmp_path/'critic_state_8.msgpack').is_file()
    finally:
        model.get_env().close();model.logger.close()


def test_config_migration_audit_rejects_changed_source_or_unknown_old_config():
    import hashlib
    from scripts.verify_v2_final import verify_launch_sources
    migration=json.loads((ROOT/'docs/v2/CONFIG_MIGRATION.json').read_text())
    expected={name:wanted for name,wanted in migration['before_sha256'].items()
              if name in {'configs/mujoco_v2.yaml','configs/mujoco_v2_checked.yaml'}}
    name='optiq_dime/algorithm.py'
    expected[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert len(verify_launch_sources(ROOT,expected)) == 2
    with pytest.raises(AssertionError,match='Learning source changed'):
        verify_launch_sources(ROOT,{**expected,name:'0'*64})
    with pytest.raises(AssertionError):
        verify_launch_sources(ROOT,{'configs/mujoco_v2.yaml':'0'*64})
