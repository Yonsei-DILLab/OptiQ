"""Run the unchanged MuJoCo learner with explicit logstd/UTD overrides and progress."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

TEMPERATURES = dict(humanoid=.5, walker2d=.1, hopper=.05, ant=.25, halfcheetah=.5)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def overrides(spec, smoke=False):
    result = [f'benchmark={spec["task"]}', '+critic=crossq', 'alg.critic.hs=[1024,1024]',
        '+actor=crossq', f'temperature={spec["temperature"]}', f'seed={spec["seed"]}',
        f'alg.utd={spec["utd"]}', 'alg.actor.custom_log_std_bounds=true',
        'alg.actor.log_std_min=-5.0', 'alg.actor.log_std_max=-3.5',
        'alg.actor.initial_log_std=-3.5', 'wandb.activate=false',
        f'run_name={spec["id"]}', f'output_root={Path(spec["output"])/"training"}']
    if smoke:
        result += ['total_steps=14', 'alg.learning_starts=8', 'alg.actor.learning_starts=8',
            'alg.buffer_size=1024', 'eval_interval=14', 'num_eval_episodes=1',
            'diagnostic_interval=14', 'checkpoint_interval=14']
    return result


def run(spec, smoke=False):
    branch = subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if not branch or 'anonymous' in branch.lower():
        raise RuntimeError('A named non-anonymous source worktree is required')
    if spec.get('source_commit') and revision != spec['source_commit']:
        raise RuntimeError('Source commit differs from the pinned campaign revision')
    output = Path(spec['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'spec.json', dict(spec, smoke=smoke, overrides=overrides(spec,smoke)))
    atomic_json(output/'progress.json',dict(status='initializing',steps=0,updates=0,
        actor_updates=0,observed_unix=time.time(),pid=os.getpid()))
    started=time.time()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import train
        import environment as environment_module
        from ibolt.policy import actor_variables
        from stable_baselines3.common.callbacks import BaseCallback
        cfg=train.compose_config(overrides(spec,smoke));train.validate_config(cfg)
        assert cfg.alg.actor.log_std_min == -5 and cfg.alg.actor.log_std_max == -3.5
        assert cfg.alg.actor.initial_log_std == -3.5 and cfg.alg.utd == spec['utd']
        assert cfg.alg.critic.type == 'crossq' and list(cfg.alg.critic.hs) == [1024,1024]
        assert cfg.alg.actor.batch_renorm and cfg.alg.actor.policy_delay == 3
        assert cfg.alg.batch_size == 256 and cfg.dacer.enabled
        assert jax.default_backend() == 'gpu'
        original_create=environment_module.create_algorithm
        holder={}

        class Progress(BaseCallback):
            def snapshot(self, status='running'):
                metrics={}
                for key,value in self.model.logger.name_to_value.items():
                    if isinstance(value,(int,float,np.number)):
                        value=float(value)
                        if key.startswith('train/') and not math.isfinite(value):
                            raise FloatingPointError('Nonfinite learner metric: '+key)
                        if math.isfinite(value):metrics[key]=value
                data=dict(status=status,steps=self.model.num_timesteps,updates=self.model._n_updates,
                    actor_updates=int(self.model.policy.actor_state.step),
                    critic_state_updates=int(self.model.policy.qf_state.step),
                    elapsed_seconds=time.time()-started,observed_unix=time.time(),backend=jax.default_backend(),
                    actual_run=str(cfg.output_root),metrics=metrics)
                atomic_json(output/'progress.json',data)
                return data

            def _on_step(self):
                if self.num_timesteps==1 or self.num_timesteps%1000==0 or self.num_timesteps==int(cfg.alg.learning_starts)+1:
                    self.snapshot()
                return True

            def _on_training_end(self):
                holder['progress']=self.snapshot('finishing')
                model=self.model
                for state in (model.policy.actor_state,model.policy.qf_state):
                    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
                replay=model.replay_buffer
                count=min(4,replay.size())
                obs=jnp.asarray(replay.observations[:count,0])
                z=jax.random.normal(jax.random.PRNGKey(9401),(count,model.action_space.shape[0]))
                _,ls=model.policy.actor_state.apply_fn(actor_variables(model.policy.actor_state),obs,z)
                ls=np.asarray(ls)
                assert np.isfinite(ls).all() and ls.min()>=-5.00001 and ls.max()<=-3.49999
                holder['logstd_probe']=dict(min=float(ls.min()),max=float(ls.max()),
                    actor_parameters_finite=True,critic_parameters_finite=True)

        def create(config):
            model,callbacks=original_create(config)
            callbacks.callbacks.append(Progress())
            holder['callbacks']=callbacks
            atomic_json(output/'actual-run.json',dict(path=str(config.output_root),source_commit=revision))
            return model,callbacks

        environment_module.create_algorithm=create
        train.run(cfg)
        done=json.loads((Path(cfg.output_root)/'completed.json').read_text())
        expected=(int(cfg.total_steps)-int(cfg.alg.learning_starts))*int(cfg.alg.utd)
        assert done['timesteps']==int(cfg.total_steps) and done['updates']==expected
        assert holder['progress']['actor_updates']==expected//3
        evaluations={}
        for mode in ('zero_z','stochastic_z','stochastic_z_sigma'):
            files=list((Path(cfg.output_root)/'eval').rglob('evaluations_'+mode+'.npz'))
            assert len(files)==1
            with np.load(files[0]) as data:
                assert int(data['timesteps'][-1])==int(cfg.total_steps)
                assert data['results'].shape[-1]==int(cfg.num_eval_episodes)
                evaluations[mode]=dict(episodes=int(cfg.num_eval_episodes),mean_return=float(np.mean(data['results'][-1])))
        result=dict(completed=True,steps=done['timesteps'],updates=done['updates'],
            actor_updates=holder['progress']['actor_updates'],backend='gpu',source_commit=revision,
            actual_run=str(cfg.output_root),elapsed_seconds=time.time()-started,finished_unix=time.time(),
            logstd_probe=holder['logstd_probe'],evaluations=evaluations,smoke=smoke)
        atomic_json(output/'completed.json',result)
        atomic_json(output/'progress.json',dict(result,status='completed',observed_unix=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),observed_unix=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
