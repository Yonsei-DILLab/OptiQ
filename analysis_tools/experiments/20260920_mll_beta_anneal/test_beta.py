import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_beta import base,compose_config,beta_at_step,BetaAnnealedMLL
import inspect
import numpy as np
import pytest
import hydra
from stable_baselines3.common.logger import configure


@pytest.mark.parametrize('step,value',[(0,0),(5000,.05),(50000,.5),(100000,1),(1000000,1)])
def test_schedule(step,value):
    assert beta_at_step(step)==value


def test_beta_reaches_actual_jitted_learner_and_config_restored(tmp_path,monkeypatch):
    cfg=compose_config(['benchmark=ant',f'output_root={tmp_path}',
        'alg.batch_size=4','alg.buffer_size=32','alg.learning_starts=2',
        'alg.actor.learning_starts=2','num_eval_episodes=1','eval_interval=10000',
        'checkpoint_interval=0'])
    cfg=hydra.utils.instantiate(cfg)  # Match the production initialization path.
    monkeypatch.setattr(base.runner,'OptiQDIME',BetaAnnealedMLL)
    model,callbacks=base.runner.create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'logs'),['csv']))
    seen=[]; original=model._train; signature=inspect.signature(original)
    def record(*args,**kwargs):
        seen.append(float(signature.bind(*args,**kwargs).arguments['density_beta']))
        return original(*args,**kwargs)
    monkeypatch.setattr(model,'_train',record)
    try:
        model.learn(total_timesteps=6)
        np.testing.assert_allclose(seen,[.00003,.00004,.00005,.00006])
        for step in (50000,100000,110000):
            model.num_timesteps=step
            model.train(batch_size=4,gradient_steps=1)
            assert seen[-1]==beta_at_step(step)
            assert model.cfg.alg.actor.density_beta==1.0
            assert model.logger.name_to_value['train/density_beta']==beta_at_step(step)
    finally:
        for cb in callbacks.callbacks:cb.eval_env.close()
        model.get_env().close();model.logger.close()
