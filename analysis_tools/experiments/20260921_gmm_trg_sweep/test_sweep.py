import copy
import importlib.util
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import train
from regulator import entropy_proxy, noisy_action, BehaviorRegulatedOptiQ

def test_config_only_requested_differences():
    from omegaconf import OmegaConf
    original=OmegaConf.to_container(train.base.compose_config(),resolve=True)
    cfg=train.compose_config(['benchmark=humanoid'])
    new=OmegaConf.to_container(cfg,resolve=True)
    assert new['alg']==original['alg']
    for t in [.05,.1,.25]:
        for b in [.5,.9,1.]:
            cfg=train.compose_config([f'alg.actor.temperature={t}',f'alg.actor.density_correction_beta={b}'])
            assert cfg.alg.actor.density_beta==b and cfg.alg.actor.temperature==t

def test_entropy_and_alpha_sign():
    samples=np.random.default_rng(1).normal(size=(2,200,3))*.2
    h,c=entropy_proxy(samples)
    assert np.isfinite(h) and 0<=c<=1
    for grad,direction in [(-2,1),(2,-1)]:
        a=jnp.array(np.log(.27));opt=optax.adam(.03)
        delta,_=opt.update(jnp.array(float(grad)),opt.init(a),a)
        assert np.sign(float(delta))==direction
    a=np.array([[.9,-.9]],dtype=np.float32)
    np.testing.assert_array_equal(noisy_action(a,np.array([[1.,-1.]]),.3),[[1.,-1.]])

def test_regulator_isolation(tmp_path,monkeypatch):
    from stable_baselines3.common.logger import configure
    import regulator
    called=[]
    original=regulator.entropy_proxy
    def record(a,*args):called.append(a.copy());return original(a,*args)
    monkeypatch.setattr(regulator,'entropy_proxy',record)
    cfg=train.compose_config(['benchmark=hopper','seed=0',f'output_root={tmp_path}',
       'alg.batch_size=4','alg.buffer_size=32','alg.learning_starts=2',
       'alg.actor.learning_starts=2','checkpoint_interval=0','diagnostic_interval=0',
       'dacer.enabled=true','dacer.interval_updates=2'])
    model,callbacks=train.runner.create_algorithm(cfg)
    model.model_save_path=None
    model.set_logger(configure(str(tmp_path/'logs'),[]))
    before=copy.deepcopy(model.policy.actor_state.params)
    try:
        model.learn(total_timesteps=8)
        assert model._n_updates==6 and model.regulator_count==3
        assert called[0].shape==(4,200,3)
        assert model.backup_mode=='td' and cfg.alg.ent_coef.init==0
        assert np.isfinite(float(model.regulator_log_alpha))
        assert np.max(np.abs(model.replay_buffer.actions))<=1
        assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(model.policy.actor_state.params))
        for head in ['mu','log_std']:
            assert not np.array_equal(before[head]['kernel'],model.policy.actor_state.params[head]['kernel'])
        assert (tmp_path/'dacer_regulator.json').exists()
    finally:
        callbacks.callbacks[0].eval_env.close();model.get_env().close();model.logger.close()

def test_final100k_selection(tmp_path):
    from campaign import outcome
    for key in ('actor','critic'):(tmp_path/f'{key}_state_1000000.msgpack').touch()
    steps=np.arange(0,1000001,5000)
    for mode in ('zero_z','stochastic_z'):
        vals=np.broadcast_to(steps[:,None],(len(steps),10))
        np.savez(tmp_path/f'evaluations_{mode}.npz',timesteps=steps,results=vals)
    assert outcome(tmp_path)['stochastic_z']==952500.

def test_campaign_counts_and_uniqueness():
    from coordinator import phase_jobs, initial_allocation
    best={'humanoid':.25,'ant':.25,'halfcheetah':.1,'walker2d':.1,'hopper':.05}
    phases=[phase_jobs(i,best) for i in range(1,5)]
    assert list(map(len,phases))==[40,30,50,10]
    ids=[j['id'] for phase in phases for j in phase]
    assert len(ids)==len(set(ids))==130
    initial=[j for jobs in initial_allocation().values() for j in jobs]
    assert len(initial)==len({j['id'] for j in initial})==28
    assert all(j['id'] in ids for j in initial)
