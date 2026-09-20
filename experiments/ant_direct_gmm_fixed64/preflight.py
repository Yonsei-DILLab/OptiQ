"""Bounded validation, not one of the four requested 1M-step runs."""
import json,sys,copy
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3.common.logger import configure
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from run_optiq_dime import validate_config,create_algorithm
from optiq_dime.latent import FiniteMixtureTrainState,finite_latent_codes,sample_latents
import optiq_dime.algorithm as algorithm
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
 base=compose(config_name='mujoco_v5_direct_gmm',overrides=['benchmark=ant'])
 cfg=compose(config_name='mujoco_v5_direct_gmm_ant_fixed64')
validate_config(cfg)
b=OmegaConf.to_container(base.alg,resolve=True);c=OmegaConf.to_container(cfg.alg,resolve=True)
for k in ['latent_prior','latent_components','latent_codebook_seed']:c['actor'].pop(k)
c['actor']['num_policy_samples']=b['actor']['num_policy_samples'];c['actor']['proposals_per_policy_sample']=b['actor']['proposals_per_policy_sample']
assert b==c, 'Unexpected training setting change'
assert cfg.env_name=='Ant-v4' and cfg.alg.utd==1 and cfg.alg.batch_size==256
assert cfg.alg.actor.hidden_dims==[256,256] and cfg.alg.critic.hs==[256,256]
assert cfg.alg.actor.temperature==.25 and cfg.alg.actor.distillation_loss=='direct_gmm_nll'
assert cfg.alg.optimizer.lr_actor==cfg.alg.optimizer.lr_critic==3e-4
assert cfg.alg.actor.mean_output_init_scale==1e-4 and cfg.total_steps==1000000
for seed in range(4):
 cfg.seed=seed;(out/f'resolved-seed{seed}.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True),indent=2))
cfg.seed=0;cfg.output_root=str(out/'runtime');cfg.alg.buffer_size=512;cfg.alg.learning_starts=2;cfg.alg.actor.learning_starts=2
cfg.num_eval_episodes=1;cfg.eval_interval=4;cfg.diagnostic_interval=4;cfg.checkpoint_interval=4
validate_config(cfg)
def forbidden(*args,**kwargs):raise AssertionError('Direct GMM invoked Sinkhorn')
algorithm.sinkhorn=forbidden
model,callbacks=create_algorithm(cfg);model.set_logger(configure(str(out/'runtime'/'validation_logs'),['csv']))
cb=callbacks.callbacks[0];cb.eval_env.envs[0].env._max_episode_steps=2;model.get_env().envs[0].env._max_episode_steps=2
before=copy.deepcopy(model.policy.actor_state.params);codes=np.asarray(finite_latent_codes(model.policy.actor_state,8))
try:
 model.learn(total_timesteps=8,callback=callbacks)
 state=model.policy.actor_state
 assert isinstance(state,FiniteMixtureTrainState) and state.latent_components==64
 assert int(state.step)==model._n_updates==6
 assert model.backup_mode=='td' and model.soft_guard_attempts==0
 np.testing.assert_array_equal(codes,finite_latent_codes(state,8))
 draws=np.asarray(sample_latents(state,jax.random.PRNGKey(53),(256,8)))
 assert np.max(np.square(draws[:,None]-codes[None]).sum(-1).min(-1))==0
 for head in ['mu','log_std']:assert not np.array_equal(before[head]['kernel'],state.params[head]['kernel'])
 restored=serialization.from_bytes(state,serialization.to_bytes(state));np.testing.assert_array_equal(finite_latent_codes(restored,8),codes)
 assert cb.evaluations_timesteps==[1,4,8] and np.isfinite(cb.evaluations_results).all()
 result=dict(status='PASS',env='Ant-v4',batch=256,n=64,m=64,utd=1,actor_updates=6,critic_updates=model._n_updates,fixed_codebook_shape=list(codes.shape),no_sinkhorn=True,finite_evaluations=True,training_defaults_preserved=True)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
 cb.eval_env.close();model.get_env().close();model.logger.close()
