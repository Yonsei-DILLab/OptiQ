"""Run the unchanged single-environment Fetch learner with authorized scale bounds."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


def atomic(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def run(spec,smoke=False):
    source=Path(spec['source']);sys.path.insert(0,str(source))
    assert 'anonymous' not in subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip().lower()
    for name,digest in json.loads(Path(spec['source_manifest']).read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    import numpy as np
    import train
    from omegaconf import OmegaConf
    from fetch_rebuttal import run as native
    from fetch_rebuttal.env import make_env
    original_validate=train.validate_config
    def validate_scales(cfg):
        # Preserve every original validation while allowing exactly the requested bounds.
        check=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        check.alg.actor.log_std_min=-5.;check.alg.actor.log_std_max=-1.;check.alg.actor.initial_log_std=-1.
        original_validate(check)
        cfg.alg.actor.log_std_min=-5.;cfg.alg.actor.log_std_max=1.;cfg.alg.actor.initial_log_std=-1.
    train.validate_config=validate_scales
    original_record=native.record_evaluation;cache={}
    def record(agent,cfg,output,step,updates,ibolt=False):
        result=original_record(agent,cfg,output,step,updates,ibolt)
        if cfg['task']=='pick-and-place':
            key=(cfg['seed'],cfg['final_episodes'])
            if key not in cache:
                import random
                python_state,numpy_state=random.getstate(),np.random.get_state()
                env=make_env(cfg['task'])
                try:initial=np.stack([env.reset(seed=700000+cfg['seed']*1000+i)[0] for i in range(cfg['final_episodes'])])
                finally:env.close();random.setstate(python_state);np.random.set_state(numpy_state)
                height=initial[:,-1]-initial[:,5]
                cache[key]=dict(ground=np.abs(height)<.001,low_air=(height>=.001)&(height<=.05),air_over5cm=height>.05)
            strata=dict(steps=step,updates=updates,modes={})
            for mode in ('stochastic_z','zero_z'):
                value=result[mode];count=value['episodes'];hits=np.asarray(value['final_success']);anyhits=np.asarray(value['any_success'])
                strata['modes'][mode]={name:dict(episodes=int(mask[:count].sum()),final_successes=int(hits[mask[:count]].sum()),
                    any_successes=int(anyhits[mask[:count]].sum())) for name,mask in cache[key].items()}
            with (Path(output)/'height-evaluations.jsonl').open('a') as stream:stream.write(json.dumps(strata)+'\n')
        return result
    native.record_evaluation=record
    output=Path(spec['output'])
    args=['--method','ibolt','--task',spec['task'],'--seed',str(spec['seed']),
        '--temperature',str(spec['temperature']),'--batch-size','256','--normalize-observations','--her','--output',str(output)]
    if smoke:args+=['--smoke']
    try:
        native.main(args)
        done=json.loads((output/'completed.json').read_text());cfg=json.loads((output/'config.json').read_text())
        learner=json.loads((output/'mujoco-config.json').read_text());a=learner['alg']['actor'];alg=learner['alg']
        assert (a['log_std_min'],a['log_std_max'],a['initial_log_std'])==(-5,1,-1)
        assert cfg['num_envs']==cfg['utd']==1 and cfg['batch_size']==256 and cfg['HER'] and cfg['normalize_observations']
        assert alg['optimizer']['lr_actor']==alg['optimizer']['lr_critic']==.0003
        assert alg['gamma']==.99 and alg['tau']==.005 and alg['utd']==1
        assert list(a['hidden_dims'])==list(alg['critic']['hs'])==[256,256]
        assert done['updates']==(4 if smoke else 995000) and done['runtime']['backend']=='gpu'
        import flax.serialization,jax
        for name in ('actor_state','critic_state'):
            values=flax.serialization.msgpack_restore((output/'checkpoints'/f'{name}_{cfg["steps"]}.msgpack').read_bytes())
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(values))
        her=done['runtime']['her_replay_last_sample'];assert her['relabeled_samples']==204 and her['sampled_future_within_episode']
        atomic(output/'verified-completion.json',dict(done,scale_override_verified=True,checkpoint_parameters_finite=True,
            log_std_min=-5,log_std_max=1,initial_log_std=-1))
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        atomic(output/'scale-worker-failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
