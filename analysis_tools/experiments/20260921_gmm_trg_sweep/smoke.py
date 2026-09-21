"""Production batch/real warmup: DACER, beta=.5, no W&B production run."""
import tempfile
import jax
import numpy as np
import train
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

root=tempfile.mkdtemp(prefix='gmm-trg-dacer-smoke-')
cfg=train.compose_config(['benchmark=halfcheetah','seed=0','alg.buffer_size=10000',
    'alg.actor.density_correction_beta=0.5','dacer.enabled=true','dacer.noise_scale=0.15',
    'checkpoint_interval=0','diagnostic_interval=0',f'output_root={root}'])
model,callbacks=train.runner.create_algorithm(cfg)
model.model_save_path=None;model.set_logger(configure(root,[]))
class Check(BaseCallback):
    def _on_step(self):
        if self.num_timesteps>5000:
            for state in [self.model.policy.actor_state,self.model.policy.qf_state]:
                assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(state.params))
        return True
try:
    model.learn(total_timesteps=5200,callback=Check())
    assert model._n_updates==200 and model.regulator_count==1
    print('PASS full B256 5Kwarmup 200updates beta=.5 DACER behavior-only',flush=True)
finally:
    callbacks.callbacks[0].eval_env.close();model.get_env().close();model.logger.close()
