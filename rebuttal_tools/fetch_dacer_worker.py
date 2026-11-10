"""Configuration-only DACER sweep around the frozen Fetch iBOLT learner."""
import argparse
import json
import math
from pathlib import Path
import traceback
from .fetch_wide_utd1_worker import atomic

TARGETS=(-.9,.5,.75,1.)
NOISE_SCALES=(.1,.3,1.)
ALLOWED_NOISE_SCALES=NOISE_SCALES+(0.,2.,3.,5.,10.)


def leaves(value,prefix=''):
    if not isinstance(value,dict):return {prefix:value}
    result={}
    for key,child in value.items():result.update(leaves(child,prefix+'.'+key if prefix else key))
    return result


def compare_configuration(reference,actual,allowed):
    before,after=leaves(reference),leaves(actual)
    changed={k:(before.get(k),after.get(k)) for k in before.keys()|after.keys()
             if before.get(k)!=after.get(k) and k not in allowed}
    assert not changed,changed


def verify_learner_configuration(spec,actual,smoke=False):
    reference=json.loads(Path(spec['reference_learner_config']).read_text())
    allowed={'seed','output_root','run_name','dacer.noise_scale','dacer.target_entropy_per_dim'}
    if smoke:allowed|={'total_steps','alg.learning_starts','alg.actor.learning_starts','alg.buffer_size','checkpoint_interval'}
    compare_configuration(reference,actual,allowed)
    assert actual['temperature']==actual['alg']['actor']['temperature']==.25
    actor=actual['alg']['actor'];assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'])==(-5,1,-1)
    assert actual['dacer']['target_entropy_per_dim']==spec['target_entropy_per_dim']
    assert actual['dacer']['noise_scale']==spec['noise_scale']


def verify_run_configuration(spec,cfg,smoke=False):
    reference=json.loads(Path(spec['reference_run_config']).read_text());allowed={'seed'}
    if smoke:allowed|={'smoke','steps','warmup','replay_capacity','eval_every','eval_episodes','final_episodes','checkpoint_every'}
    compare_configuration(reference,cfg,allowed)


def verify_saved_configuration(spec,smoke=False):
    output=Path(spec['output'])
    verify_learner_configuration(spec,json.loads((output/'mujoco-config.json').read_text()),smoke)
    verify_run_configuration(spec,json.loads((output/'config.json').read_text()),smoke)


def run(spec,smoke=False):
    assert spec['task']=='pick-and-place' and spec['method']=='ibolt' and spec['temperature']==.25
    assert spec['target_entropy_per_dim'] in TARGETS and spec['noise_scale'] in ALLOWED_NOISE_SCALES
    import train
    from omegaconf import OmegaConf
    from fetch_rebuttal import run as native
    from . import fetch_wide_utd1_worker as baseline
    original_main=native.main;validated=[];output=Path(spec['output'])
    def main_with_dacer_overrides(argv):
        # The baseline worker installs its existing scale validation before this
        # entry point. Apply only the two DACER overrides after that validation.
        previous_validate=train.validate_config
        def validate(cfg):
            previous_validate(cfg)
            cfg.dacer.target_entropy_per_dim=float(spec['target_entropy_per_dim'])
            cfg.dacer.noise_scale=float(spec['noise_scale'])
            actual=OmegaConf.to_container(cfg,resolve=True)
            verify_learner_configuration(spec,actual,smoke);validated.append(True)
            atomic(output/'dacer-search-config.json',dict(target_entropy_per_dim=spec['target_entropy_per_dim'],
                total_entropy_target=4*spec['target_entropy_per_dim'],noise_scale=spec['noise_scale'],
                only_dacer_parameters_changed=True,reference_learner_config=spec['reference_learner_config']))
        train.validate_config=validate
        try:return original_main(argv)
        finally:train.validate_config=previous_validate
    native.main=main_with_dacer_overrides
    try:
        baseline.run(spec,smoke);assert len(validated)==1
        verify_saved_configuration(spec,smoke)
        done=json.loads((output/'verified-completion.json').read_text())
        regulator=json.loads((output/'dacer_regulator.json').read_text())
        assert all(math.isfinite(regulator[k]) for k in ('alpha','noise_std','entropy_proxy'))
        assert abs(regulator['noise_std']-spec['noise_scale']*regulator['alpha'])<1e-7
        if smoke:
            error=regulator['entropy_proxy']-4*spec['target_entropy_per_dim']
            if error>1e-6:assert regulator['alpha']<.27
            if error< -1e-6:assert regulator['alpha']>.27
        atomic(output/'dacer-verified-completion.json',dict(done,only_dacer_parameters_changed=True,
            target_entropy_per_dim=spec['target_entropy_per_dim'],noise_scale=spec['noise_scale'],
            regulator=regulator,alpha_update_direction_verified=bool(smoke)))
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        atomic(output/'dacer-worker-failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:native.main=original_main


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
