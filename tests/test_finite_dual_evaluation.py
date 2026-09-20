"""Evaluation must sample the trained finite support, with no Gaussian epsilon."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra import compose, initialize_config_dir
from pathlib import Path
from optiq_dime.latent import FiniteMixtureTrainState, finite_latent_codes
from optiq_dime.policy import OptiQPolicy
from optiq_dime.dual_evaluation import dual_mu_evaluation_spec
from run_optiq_dime import validate_config


def test_stochastic_finite_evaluation_uses_training_codes():
    # An identity mu makes any out-of-codebook evaluation action observable.
    # A huge sigma also makes accidental conditional noise easy to detect.
    def apply_fn(params, observations, z):
        return z, jnp.full_like(z, 10.)
    state = FiniteMixtureTrainState.create(apply_fn=apply_fn,
        params={"mu":{"bias":jnp.zeros(17)}}, tx=optax.adam(3e-4),
        latent_components=64, latent_codebook_seed=20260911)
    obs=jnp.zeros((4096, 376));key=jax.random.PRNGKey(41)
    action=np.asarray(OptiQPolicy.sample_action(state,obs,key,False,False))
    support=np.asarray(jnp.tanh(finite_latent_codes(state,17)))
    distance=np.max(np.abs(action[:,None,:]-support[None,:,:]),axis=-1)
    assert distance.min(axis=1).max()<1e-6
    counts=np.bincount(distance.argmin(axis=1),minlength=64)
    assert counts.min()>25 and counts.max()<110
    other=np.asarray(OptiQPolicy.sample_action(state,obs,jax.random.PRNGKey(42),False,False))
    assert not np.array_equal(action,other)
    fixed=np.asarray(OptiQPolicy.sample_action(state,obs,key,True,False))
    np.testing.assert_allclose(fixed,np.broadcast_to(support[0],fixed.shape),atol=1e-6)
    zero=np.asarray(OptiQPolicy.sample_action(state,obs,key,True,False,True))
    np.testing.assert_array_equal(zero,np.zeros_like(zero))
    assert not np.allclose(zero,fixed)  # codebook[0] must never be relabeled z0


def test_finite_and_continuous_evaluation_specs():
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        finite=compose(config_name='mujoco_v5_direct_gmm_humanoid_fixed64')
        continuous=compose(config_name='mujoco_v5_direct_gmm',overrides=['benchmark=humanoid'])
    validate_config(finite);validate_config(continuous)
    mode,spec=dual_mu_evaluation_spec(finite)
    assert mode=='zero_z' and spec['legacy_eval_alias']=='zero_z'
    assert spec['zero_z_role']=='out-of-support diagnostic' and spec['schema_version']==2
    assert spec['latent_components']==64 and spec['latent_codebook_seed']==20260911
    assert 'same fixed training codebook' in spec['stochastic_z']
    mode,spec=dual_mu_evaluation_spec(continuous)
    assert mode=='zero_z' and spec['legacy_eval_alias']=='zero_z'
    assert 'z~N(0,I)' in spec['stochastic_z']


def test_real_finite_eval_records_both_modes_and_restores_flags(tmp_path, monkeypatch):
    import copy
    import pytest
    from stable_baselines3.common.logger import configure
    from run_optiq_dime import create_algorithm
    from optiq_dime.runtime import WandbWriter
    import optiq_dime.dual_evaluation as evaluation

    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='mujoco_v5_direct_gmm_ant_fixed64',overrides=[
            f'output_root={tmp_path}', 'alg.batch_size=4','alg.buffer_size=32',
            'alg.learning_starts=2','alg.actor.learning_starts=2',
            'alg.actor.hidden_dims=[16,16]','alg.critic.hs=[16,16]',
            'num_eval_episodes=2','eval_interval=4','checkpoint_interval=4','diagnostic_interval=4'])
    validate_config(cfg)
    model,callbacks=create_algorithm(cfg);cb=callbacks.callbacks[0]
    class RecordingRun:
        def __init__(self):self.rows=[]
        def log(self,row):self.rows.append(dict(row))
    run=RecordingRun()
    model.set_logger(configure(str(tmp_path/'logs'),['csv']))
    model.logger.output_formats.append(WandbWriter(run))
    cb.eval_env.envs[0].env._max_episode_steps=2
    observed=[]
    original=evaluation.evaluate_policy
    def inspect_eval(*args,**kwargs):
        zero=kwargs['deterministic']
        assert model.policy.evaluation_zero_latent == zero
        assert model.policy.evaluation_mu_only
        observed.append(zero)
        return original(*args,**kwargs)
    monkeypatch.setattr(evaluation,'evaluate_policy',inspect_eval)
    try:
        model.learn(total_timesteps=8,callback=callbacks)
        assert model._n_updates==6
        assert observed==[True,True,False,False]*3
        assert not model.policy.evaluation_zero_latent and not model.policy.evaluation_mu_only
        rows=[row for row in run.rows if 'eval/zero_z/mean_reward' in row]
        assert [row['env_steps'] for row in rows]==[1,4,8]
        for i,row in enumerate(rows):
            assert row['eval/mean_reward']==row['eval/zero_z/mean_reward']
            assert not any('fixed_z' in key for key in row)
            for mode in ('zero_z','stochastic_z'):
                assert row[f'eval/{mode}/num_episodes']==2
                assert row[f'eval/{mode}/std_reward']>=0
                assert row[f'eval/{mode}/std_ep_length']>=0
                assert row[f'eval/{mode}/best_mean_reward']==max(x[f'eval/{mode}/mean_reward'] for x in rows[:i+1])
        for mode in cb.MODES:
            saved=np.load(cb.directory/f'evaluations_{mode}.npz')
            np.testing.assert_array_equal(saved['timesteps'],[1,4,8])
            assert saved['results'].shape==(3,2)
        for key in ('env_seeds','policy_seeds'):
            np.testing.assert_array_equal(cb.histories['zero_z'][key],cb.histories['stochastic_z'][key])
        key,noise=model.policy.key,model.policy.noise_key
        behavior=copy.deepcopy(model.behavior_rng.bit_generator.state)
        # Restore pre-existing flags as well as RNG when either mode raises.
        model.policy.evaluation_zero_latent=True
        model.policy.evaluation_mu_only=True
        def fail(*args,**kwargs):raise RuntimeError('intentional eval failure')
        monkeypatch.setattr(evaluation,'evaluate_policy',fail)
        cb.n_calls=12;cb.num_timesteps=12
        with pytest.raises(RuntimeError,match='intentional'):cb._on_step()
        np.testing.assert_array_equal(model.policy.key,key)
        np.testing.assert_array_equal(model.policy.noise_key,noise)
        assert model.policy.evaluation_zero_latent and model.policy.evaluation_mu_only
        assert model.behavior_rng.bit_generator.state==behavior
    finally:
        cb.eval_env.close();model.get_env().close();model.logger.close()
