"""Validate the entire config matrix, selection window and optional GPU update."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from train_sweep import compose_config, base
from campaign import summarize_run, rank_temperature, TEMPERATURES


def checks():
    count=0
    for task,temps in TEMPERATURES.items():
        for temp in temps:
            for beta in (1.,.5,.9):
                for seed in range(5):
                    cfg=compose_config(task,temp,beta,seed,'temperature' if beta==1 else 'beta','/tmp/trg-validation')
                    assert cfg.alg.actor.density_beta==cfg.alg.actor.density_correction_beta==beta
                    assert cfg.alg.actor.temperature==temp and cfg.wandb.project=='gmm-trg'
                    count+=1
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)
        steps=np.r_[1,np.arange(5000,1000001,5000)]
        values=np.zeros((len(steps),10));values[steps==900000]=1e9;values[steps>900000]=7
        for mode in ('stochastic_z','zero_z'):
            np.savez(path/f'evaluations_{mode}.npz',timesteps=steps,results=values)
        metrics=summarize_run(path)
        assert metrics['stochastic_z_last_100k_mean']==7
        jobs=[dict(temperature=t,beta=1.,seed=s,status='completed',wandb_url='validation',
                   metrics=dict(stochastic_z_last_100k_mean=t+s,zero_z_last_100k_mean=999-t))
              for s in range(5) for t in (.05,.1)]
        assert rank_temperature(jobs)['selected_temperature']==.1
        try:
            rank_temperature(jobs[:-1])
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete five-seed set was accepted')
    print(json.dumps(dict(status='passed',configs=count,window='last100k exclusive start',missing_seed_gate=True)))


def gpu_check():
    import jax
    assert jax.default_backend()=='gpu'
    with tempfile.TemporaryDirectory(prefix='trg-beta-gpu-validation-') as folder:
        cfg=compose_config('ant',.1,.5,0,'beta',folder)
        model,callbacks=base.runner.create_algorithm(cfg)
        try:
            rng=np.random.default_rng(12)
            for _ in range(512):
                obs=rng.normal(size=(1,27)).astype(np.float32)
                model.replay_buffer.add(obs,obs+.01,rng.uniform(-1,1,size=(1,8)).astype(np.float32),
                    np.ones(1,dtype=np.float32),np.zeros(1),[{}])
            model.num_timesteps=10000
            model.logger.dump=lambda step:None
            policy=model.policy
            initial=(policy.qf_state,policy.actor_state,policy.target_actor_state,model.ent_coef_state,model.key)
            numpy_state=np.random.get_state()
            outcomes=[]
            for beta in (.5,.9):
                policy.qf_state,policy.actor_state,policy.target_actor_state,model.ent_coef_state,model.key=initial
                model._n_updates=0
                np.random.set_state(numpy_state)
                model.cfg.alg.actor.density_beta=beta
                model.cfg.alg.actor.density_correction_beta=beta
                model.train(batch_size=256,gradient_steps=1)
                logs=model.logger.name_to_value
                assert np.isfinite(logs['train/actor_loss']) and np.isfinite(logs['train/critic_loss'])
                outcomes.append((logs['train/actor_loss'],logs['train/source_ess_absolute']))
                print(json.dumps(dict(validation_only=True,beta=beta,actor_loss=logs['train/actor_loss'],
                                      critic_loss=logs['train/critic_loss'],source_ess=logs['train/source_ess_absolute'],
                                      actor_updates=int(model.policy.actor_state.step))))
            assert not np.allclose(outcomes[0],outcomes[1]), 'Beta did not change the matched-RNG objective/weights'
        finally:
            model.get_env().close();model.logger.close()
            for callback in callbacks.callbacks:
                if hasattr(callback,'eval_env'):callback.eval_env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    checks()
    if args.gpu:gpu_check()
