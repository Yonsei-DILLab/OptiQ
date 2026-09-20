"""Full B256, real 5K warmup plus200updates; preflight only, no W&B run."""
import tempfile
import train
import jax
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

root=tempfile.mkdtemp(prefix='optiq-truncated-r2-smoke-')
cfg=train.compose_config(['benchmark=humanoid','seed=1','alg.buffer_size=10000',
    'checkpoint_interval=0','diagnostic_interval=0',f'output_root={root}'])
model,callbacks=train.runner.create_algorithm(cfg)
model.model_save_path=None
model.set_logger(configure(root,[]))
class Check(BaseCallback):
    def _on_step(self):
        if self.num_timesteps>5000:
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(self.model.policy.actor_state.params)),self.num_timesteps
            assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(self.model.policy.qf_state.params)),self.num_timesteps
        return True
try:
    model.learn(total_timesteps=5200,callback=Check())
    assert model._n_updates==200
    print('PASS B256 5200steps 200updates actor/critic finite',flush=True)
finally:
    callbacks.callbacks[0].eval_env.close();model.get_env().close();model.logger.close()
