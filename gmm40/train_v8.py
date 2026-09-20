"""Fresh-output GMM40 v8 training; preflight and checkpoint provenance required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

from .validate_v8 import ROOT, scientific_config, source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--chunk',type=int,default=100)
    p.add_argument('--evaluate-every',type=int,default=5000)
    p.add_argument('--eval-samples',type=int,default=10000)
    p.add_argument('--platform',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--max-iterations',type=int,default=500)
    p.add_argument('--min-iterations',type=int,default=10)
    p.add_argument('--relative-tolerance',type=float,default=1e-3)
    p.add_argument('--actor-max-grad-norm',type=float)
    args=p.parse_args()
    if min(args.steps,args.chunk,args.evaluate_every,args.eval_samples)<1:
        p.error('Positive step/chunk/evaluation counts required')
    if args.eval_samples<2:
        p.error('At least two evaluation samples required')
    os.environ['JAX_PLATFORMS']=args.platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import numpy as np
    from .validate_v8 import make_agent
    from .validate_v7 import checkpoint_audit
    from .evaluation import atomic_json,save_evaluation
    backend=jax.default_backend()  # Explicit CUDA failure cannot silently become CPU training.
    cfg=scientific_config(args.seed,max_iterations=args.max_iterations,min_iterations=args.min_iterations,
        relative_tolerance=args.relative_tolerance,actor_max_grad_norm=args.actor_max_grad_norm)
    hashes=source_hashes()
    preflight=json.loads(args.preflight.read_text())
    if preflight.get('status')!='passed' or preflight.get('source_sha256')!=hashes:
        raise ValueError('A passing preflight for these exact training sources is required')
    if args.platform=='cuda' and (preflight.get('backend')!='gpu' or not preflight.get('full_shape')):
        raise ValueError('Run a full-shape CUDA preflight on the destination GPU before training')
    for name in ('max_iterations','min_iterations','relative_tolerance','actor_max_grad_norm'):
        if preflight['config'][name]!=cfg[name]:
            raise ValueError(f'Preflight setting differs: {name}')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir()
    for directory in ('gmm40','optiq_dime','models','common','diffusion'):
        shutil.copytree(ROOT/directory,out/'source'/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'docs'/'v8',out/'source'/'docs'/'v8')
    config=dict(version=8,scientific=cfg,requested_updates=args.steps,scope='fixed_Q_training',
        Q='log p_GMM40(40*tanh(u))',alpha=1.,ot='fresh per-state Gaussian likelihood',
        source_importance_correction=False,persistent_dual=False,backend=backend,
        teacher_resampled_before_ot=True,source_sha256=hashes,preflight=str(args.preflight.resolve()),
        evaluation_samples=args.eval_samples,evaluation_seed=900000+args.seed,
        evaluation_reference_seed=20260917,started_unix=time.time(),pid=os.getpid(),
        resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
    atomic_json(out/'config.json',config)
    stop={'signal':None}
    def handler(number,_frame): stop['signal']=signal.Signals(number).name
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    agent=None;audits=[];total_seconds=0.;warm=[];sizes=set();start_step=0

    def runtime():
        return dict(training_seconds=total_seconds,actual_updates=agent.updates if agent else 0,
            starting_update=start_step,median_warm_update_seconds=float(np.median(warm)) if warm else None,
            timing_note='First block of each size excluded; evaluation/checkpoint time excluded')

    def save_checkpoint(label='step'):
        path=out/'checkpoints'/f'{label}_{agent.updates:07d}.bin'
        temporary=path.with_suffix('.tmp');agent.save(temporary)
        audit=checkpoint_audit(temporary,agent.updates);temporary.replace(path)
        audits.append(dict(step=agent.updates,path=str(path),**audit))
        atomic_json(out/'checkpoint_audits.json',audits)
        return path

    try:
        agent=make_agent(cfg)
        if args.resume_checkpoint:
            parent_config=json.loads((args.resume_checkpoint.parent.parent/'config.json').read_text())
            if parent_config.get('scientific')!=cfg or parent_config.get('source_sha256')!=hashes:
                raise ValueError('Resume must use the same scientific settings and numerical sources')
            agent.restore(args.resume_checkpoint)
            config['resume_audit']=checkpoint_audit(args.resume_checkpoint,agent.updates)
            config['resume_sha256']=hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest()
        start_step=agent.updates
        if args.steps<=start_step: raise ValueError('Final update must exceed current checkpoint step')
        config['starting_update']=start_step
        config['settings_signature']=agent.settings_signature;atomic_json(out/'config.json',config)
        reference=agent.target.sample(args.eval_samples,20260917)
        full_reference=agent.target.sample(args.eval_samples,20260917,bounded=False)
        goals=sorted({start_step,args.steps,*range(args.evaluate_every,args.steps+1,args.evaluate_every),
                      *[x for x in (1000,5000,10000,25000,50000,75000) if start_step<=x<=args.steps]})
        info={};result=None
        for goal in (x for x in goals if x>=start_step):
            while agent.updates<goal and stop['signal'] is None:
                n=min(args.chunk,goal-agent.updates)
                started=time.monotonic();info=agent.advance(n);elapsed=time.monotonic()-started
                total_seconds+=elapsed
                if n in sizes: warm.append(elapsed/n)
                sizes.add(n)
                row=dict(step=agent.updates,train_seconds=total_seconds,block_seconds=elapsed,metrics=info)
                with (out/'training.jsonl').open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(out/'runtime.json',runtime())
                atomic_json(out/'status.json',dict(status='training',**row))
            checkpoint=save_checkpoint()
            before=np.asarray(agent.key).copy()
            samples,extra=agent.evaluate_samples(args.eval_samples,900000+args.seed)
            np.testing.assert_array_equal(before,agent.key)
            result=save_evaluation(out,f'v8 Gaussian OT seed {args.seed}',agent.updates,samples,
                agent.target,reference,full_reference,dict(info,train_seconds=total_seconds),extra_samples=extra)
            print(json.dumps(dict(event='evaluation',step=agent.updates,coverage=result['mode_coverage'],
                near=result['high_density_fraction'],mmd2=result['mmd2'],checkpoint=str(checkpoint))),flush=True)
            if stop['signal']: break
        status='completed' if agent.updates==args.steps else 'stopped'
        atomic_json(out/'runtime.json',runtime())
        atomic_json(out/'completion.json',dict(status=status,actual_updates=agent.updates,
            requested_updates=args.steps,final_metrics=result,stop_signal=stop['signal']))
        atomic_json(out/'status.json',dict(status=status,step=agent.updates))
    except Exception as error:
        checkpoint=save_checkpoint('failure') if agent is not None else None
        atomic_json(out/'runtime.json',runtime())
        diagnostics={k:(v if np.isfinite(v) else str(v)) for k,v in (agent.last_metrics.items() if agent else [])}
        atomic_json(out/'status.json',dict(status='failed',error=repr(error),step=agent.updates if agent else 0,
            failure_checkpoint=str(checkpoint) if checkpoint else None,diagnostics=diagnostics))
        raise


if __name__=='__main__':
    main()
