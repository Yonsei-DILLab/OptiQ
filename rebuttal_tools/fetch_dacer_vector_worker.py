"""DACER configuration overrides for the unchanged vec256 / UTD1/8 worker."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from . import fetch_sigma_vector_worker as baseline


def leaves(value, prefix=''):
    if not isinstance(value,dict):return {prefix:value}
    result={}
    for key,child in value.items():result.update(leaves(child,prefix+'.'+key if prefix else key))
    return result


def compare(reference,actual,allowed):
    before,after=leaves(reference),leaves(actual)
    changed={k:(before.get(k),after.get(k)) for k in before.keys()|after.keys()
             if before.get(k)!=after.get(k) and k not in allowed}
    assert not changed,changed


def verify_saved(spec,smoke):
    out=Path(spec['output']);cfg=json.loads((out/'mujoco-config.json').read_text())
    allowed={'seed','run_name','output_root','dacer.noise_scale','dacer.target_entropy_per_dim'}
    if smoke:allowed|={'total_steps','alg.buffer_size'}
    compare(json.loads(Path(spec['reference_learner_config']).read_text()),cfg,allowed)
    assert cfg['dacer']['noise_scale']==spec['dacer_noise_scale']
    assert cfg['dacer']['target_entropy_per_dim']==spec['dacer_target_entropy_per_dim']
    run=json.loads((out/'config.json').read_text());allowed={'seed','source_manifest','worker_cpus'}
    if smoke:allowed|={'smoke','steps','replay_capacity','eval_episodes','final_episodes','expected_updates'}
    compare(json.loads(Path(spec['reference_run_config']).read_text()),run,allowed)
    return cfg,run


def run(spec,smoke=False):
    assert spec['variant']=='wide'
    assert spec['dacer_noise_scale']==2. and spec['dacer_target_entropy_per_dim']==1.8
    sys.path.insert(0,spec['source']);import train
    original=train.validate_config;validated=[];out=Path(spec['output'])
    def validate(cfg):
        original(cfg)
        cfg.dacer.noise_scale=float(spec['dacer_noise_scale'])
        cfg.dacer.target_entropy_per_dim=float(spec['dacer_target_entropy_per_dim'])
        validated.append(True)
    train.validate_config=validate
    try:
        baseline.run(spec,smoke)
        assert len(validated)==1
        cfg,settings=verify_saved(spec,smoke)
        done=json.loads((out/'completed.json').read_text())
        regulator=json.loads((out/'dacer_regulator.json').read_text())
        assert all(math.isfinite(regulator[k]) for k in ('alpha','noise_std','entropy_proxy'))
        assert abs(regulator['noise_std']-2*regulator['alpha'])<1e-7
        if smoke:assert regulator['alpha']>.27
        baseline.atomic(out/'dacer-vector-verified-completion.json',dict(done,
            only_dacer_parameters_changed=True,reference_learner_config=spec['reference_learner_config'],
            reference_run_config=spec['reference_run_config'],dacer=cfg['dacer'],
            total_entropy_target=4*spec['dacer_target_entropy_per_dim'],regulator=regulator,
            dacer_controller_updates=int(regulator['learner_updates']//cfg['dacer']['interval_updates'])+1,
            alpha_update_direction_verified=bool(smoke)))
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True)
        baseline.atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:train.validate_config=original


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(json.loads(a.spec.read_text()),a.smoke)


if __name__=='__main__':main()
