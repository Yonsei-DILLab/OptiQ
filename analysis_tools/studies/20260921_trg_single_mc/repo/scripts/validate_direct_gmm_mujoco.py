"""Short batch-256 training validation, separate from benchmark runs."""
import argparse,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import gymnasium as gym
from hydra import initialize_config_dir,compose
from stable_baselines3.common.logger import configure
from run_optiq_dime import create_algorithm,validate_config

p=argparse.ArgumentParser();p.add_argument('--env',choices=['ant','humanoid'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root.parent/'validation'/a.env;out.mkdir(parents=True,exist_ok=True)
with initialize_config_dir(version_base=None,config_dir=str(root/'configs')):
 cfg=compose(config_name='mujoco_direct_gmm',overrides=[f'benchmark={a.env}',
  'alg.learning_starts=32','alg.actor.learning_starts=32','total_steps=96',
  'eval_interval=1000','checkpoint_interval=64',f'output_root={out}','num_eval_episodes=1'])
validate_config(cfg);cfg.wandb.activate=False;assert jax.default_backend()=='gpu'
for name in ['Ant-v4','Humanoid-v4','Hopper-v4','Walker2d-v4','HalfCheetah-v4']:
 env=gym.make(name);obs,_=env.reset(seed=0);obs,*_=env.step(env.action_space.sample());assert np.isfinite(obs).all();env.close()
m,cb=create_algorithm(cfg);m.set_logger(configure(str(out),['csv']));start=time.time()
m.learn(total_timesteps=96,callback=cb,progress_bar=False)
assert m.num_timesteps==96 and int(m.policy.actor_state.step)==64
for tree in [m.policy.actor_state.params,m.policy.qf_state.params]:
 assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree))
assert list(out.rglob('actor_state_64.msgpack'))
record={'passed':True,'env':a.env,'steps':m.num_timesteps,'actor_updates':int(m.policy.actor_state.step),
 'batch_size':cfg.alg.batch_size,'seconds':time.time()-start,'devices':[str(x) for x in jax.devices()]}
(out/'PASSED.json').write_text(json.dumps(record,indent=2));print(json.dumps(record));m.get_env().close()
