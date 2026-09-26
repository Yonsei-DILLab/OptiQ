import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import train
from scipy.stats import truncnorm
from scipy.integrate import trapezoid
from ibolt.box_gaussian import sample_box, mixture_log_prob, component_log_prob
from ibolt.distillation import direct_gmm_nll
from ibolt.policy import SemiImplicitActor, IBOLTPolicy

def test_density_normalizes_and_matches_scipy():
    x=np.linspace(-1,1,20001)
    for mu,std in [(0,.367879),(.99,.05),(-1,.00673795),(1,.367879)]:
        ls=jnp.array([[[np.log(std)]]]); m=jnp.array([[[mu]]])
        got=np.asarray(mixture_log_prob(jnp.array(x[None,:,None]),m,ls))[0]
        expected=truncnorm.logpdf(x,(-1-mu)/std,(1-mu)/std,loc=mu,scale=std)
        np.testing.assert_allclose(got,expected,rtol=2e-5,atol=.02)
        assert abs(trapezoid(np.exp(got),x)-1)<2e-4

def test_sampling_bounds_and_moments():
    for mu,std in [(0,.367879),(.99,.05),(-1,.00673795),(1,.367879)]:
        samples=np.asarray(sample_box(jax.random.PRNGKey(12),jnp.full((100000,),mu,dtype=jnp.float32),jnp.full((100000,),np.log(std))))
        assert np.isfinite(samples).all() and samples.min()>=-1 and samples.max()<=1
        a,b=(-1-mu)/std,(1-mu)/std
        mean,var=truncnorm.stats(a,b,loc=mu,scale=std,moments='mv')
        assert abs(samples.mean()-mean)<.01*std
        assert abs(samples.var()-var)<.01*std**2

def test_float32_support_stress():
    key=jax.random.PRNGKey(413)
    mu=jax.random.uniform(key,(1000000,),minval=-1.,maxval=1.)
    ls=jnp.full_like(mu,-1.)
    for seed in range(4):
        samples=sample_box(jax.random.PRNGKey(seed),mu,ls)
        assert np.isfinite(samples).all() and np.min(samples)>=-1 and np.max(samples)<=1

def test_nll_normalizer_gradient_and_stops():
    mu=jnp.array([[[.8],[-.3]]]);ls=jnp.array([[[-2.],[-1.5]]])
    targets=jnp.array([[[.9],[-.2]]]);w=jnp.array([[.4,.6]])
    fn=lambda m,l,t,ww:direct_gmm_nll(m,l,t,ww)[0]
    grads=jax.grad(fn,argnums=(0,1,2,3))(mu,ls,targets,w)
    assert all(np.isfinite(g).all() for g in grads)
    assert np.max(np.abs(grads[0]))>0 and np.max(np.abs(grads[1]))>0
    assert np.all(grads[2]==0) and np.all(grads[3]==0)
    for arg in [0,1]:
        vals=[mu,ls,targets,w];delta=jnp.zeros_like(vals[arg]).at[0,0,0].set(.001)
        plus=vals.copy();minus=vals.copy();plus[arg]+=delta;minus[arg]-=delta
        fd=(fn(*plus)-fn(*minus))/.002
        np.testing.assert_allclose(grads[arg][0,0,0],fd,atol=.003,rtol=.003)

def test_actor_and_policy_sampling():
    from flax.training.train_state import TrainState
    import optax
    model=SemiImplicitActor(2,(16,16),-5.,-1.,-1.)
    obs=jnp.ones((32,3));z=jnp.ones((32,2))
    params=model.init(jax.random.PRNGKey(4),obs,z)['params']
    state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(.0003))
    a=IBOLTPolicy.sample_action(state,obs,jax.random.PRNGKey(1))
    assert np.isfinite(a).all() and np.max(np.abs(a))<=1
    centers,_=model.apply({'params':params},obs,jnp.zeros_like(z))
    np.testing.assert_allclose(IBOLTPolicy.sample_action(state,obs,jax.random.PRNGKey(1),deterministic=True),centers,atol=1e-6)
    # Distinguish tanh centers from hard clipping.
    import copy
    altered=copy.deepcopy(params)
    altered['mu']['kernel']=jnp.zeros_like(altered['mu']['kernel'])
    altered['mu']['bias']=jnp.array([.5,2.])
    centers,_=model.apply({'params':altered},obs,z)
    np.testing.assert_allclose(centers,np.broadcast_to(np.tanh([.5,2.]),(32,2)),atol=1e-6)

@pytest.mark.parametrize('task',['humanoid','ant','halfcheetah','walker2d','hopper'])
def test_environment_update(task,tmp_path,monkeypatch):
    import copy
    from stable_baselines3.common.logger import configure
    import ibolt.algorithm as algorithm
    cfg=train.compose_config([f'benchmark={task}',f'output_root={tmp_path}',
        'alg.batch_size=4','alg.buffer_size=32','alg.learning_starts=2',
        'alg.actor.learning_starts=2','num_eval_episodes=1','eval_interval=4',
        'diagnostic_interval=4','checkpoint_interval=4','dacer.enabled=false'])
    assert not hasattr(algorithm,'sinkhorn')
    model,callbacks=train.runner.create_algorithm(cfg)
    from ibolt.runtime import StepLogger
    logger=configure(str(tmp_path/'logs'),['csv'])
    model.set_logger(StepLogger(logger.dir,logger.output_formats))
    cb=callbacks.callbacks[0]
    cb.eval_env.envs[0].env._max_episode_steps=2
    model.get_env().envs[0].env._max_episode_steps=2
    before=copy.deepcopy(model.policy.actor_state.params)
    try:
        model.learn(total_timesteps=6,callback=callbacks)
        assert model._n_updates==4 and model.backup_mode=='td'
        for head in ['mu','log_std']:
            assert not np.array_equal(before[head]['kernel'],model.policy.actor_state.params[head]['kernel'])
        assert all(np.isfinite(p).all() for p in jax.tree_util.tree_leaves(model.policy.actor_state.params))
        assert np.max(np.abs(model.replay_buffer.actions))<=1
        model.logger.dump(model.num_timesteps)
        import csv
        with (tmp_path/'logs'/'progress.csv').open() as stream:
            columns = set(csv.DictReader(stream).fieldnames)
        allowed = {'env_steps','train/n_updates','train/actor_loss','train/critic_loss',
                   'train/current_q_values','train/actor_std_mean','train/actor_std_min',
                   'train/actor_std_max','rollout/ep_rew_mean','rollout/ep_len_mean',
                   'time/fps','time/time_elapsed'}
        allowed.update(f'eval/{mode}/{metric}' for mode in ['zero_z','stochastic_z']
                       for metric in ['mean_reward','std_reward','mean_ep_length'])
        assert columns <= allowed
        assert {'train/actor_loss','train/actor_std_min','train/actor_std_max'} <= columns
        for mode in ['zero_z','stochastic_z']:
            with np.load(cb.directory/f'evaluations_{mode}.npz') as saved:
                assert set(saved.files)=={'timesteps','results','ep_lengths','env_seeds','policy_seeds'}
    finally:
        cb.eval_env.close();model.get_env().close();model.logger.close()
        algorithm.IBOLT.update_actor.clear_cache()
        algorithm.IBOLT.update_critic.clear_cache()
