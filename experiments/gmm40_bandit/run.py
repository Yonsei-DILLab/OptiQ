"""One durable run. Source SHA is mandatory; target sampling is evaluation-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from .evaluate import atomic_json,metrics,sample_actions,latency,plot_samples


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--steps',type=int);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--wandb',action='store_true')
    args=parser.parse_args()
    campaign=json.loads(Path(__file__).with_name('campaign.json').read_text())
    cfg=dict(next(c for c in campaign['conditions'] if c['name']==args.condition))
    cfg.update(seed=args.seed,steps=args.steps or campaign['steps'],alpha=campaign['alpha'],scale=campaign['scale'],
        source_commit=args.commit,campaign=campaign['campaign'],smoke=args.smoke)
    folder=args.output/(args.condition+f'_s{args.seed}')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'config.json').exists():
        previous=json.loads((folder/'config.json').read_text())
        if previous!=cfg:raise ValueError('Existing run config differs')
        if not args.resume:raise ValueError('Existing run; explicit --resume required')
    atomic_json(folder/'config.json',cfg)
    os.environ['GMM40_ACTION_SCALE']=str(cfg['scale'])
    os.environ['GMM40_RESULTS_ROOT']=str(args.output)
    from gmm40.target import Target
    from .adapters import EnergyOnly,make_agent
    target=Target();oracle=EnergyOnly(target)
    if hasattr(oracle,'sample') or hasattr(oracle,'means'):raise AssertionError('Energy interface leaked target')
    agent=make_agent(cfg,oracle,args.seed)
    checkpoint=folder/'checkpoint.bin';stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    history={};elapsed=0.
    if args.resume and checkpoint.exists():
        agent.restore(checkpoint)
        history=json.loads((folder/'checkpoint.json').read_text());elapsed=history['train_seconds']
    run=None
    if args.wandb:
        import wandb
        run=wandb.init(entity=os.environ.get('WANDB_ENTITY','OptiQ'),project='GMM40_heejoon',
            group=campaign['campaign'],name=folder.name,id=hashlib.sha256((args.commit+folder.name).encode()).hexdigest()[:12],
            resume='allow',config=cfg,mode='online',dir=str(folder))
    def save(status,info):
        temp=folder/'checkpoint.tmp';agent.save(temp);temp.replace(checkpoint)
        record=dict(status=status,updates=int(agent.updates),train_seconds=elapsed,metrics=info,source_commit=args.commit,pid=os.getpid())
        atomic_json(folder/'checkpoint.json',record);atomic_json(folder/'status.json',record)
    reference=target.sample(campaign['eval_samples'],20260922,bounded=False)
    goals=sorted(set([0,1000,5000,10000,20000,50000,cfg['steps']]))
    goals=[g for g in goals if agent.updates<=g<=cfg['steps']]
    try:
        info={}
        for goal in goals:
            while agent.updates<goal and not stop[0]:
                start=time.monotonic();info=agent.advance(min(50,goal-agent.updates));elapsed+=time.monotonic()-start
                if not all(np.isfinite(v) for v in info.values()):raise FloatingPointError(info)
                record=dict(status='training',updates=int(agent.updates),train_seconds=elapsed,metrics=info,pid=os.getpid())
                atomic_json(folder/'status.json',record)
                if agent.updates%1000==0:
                    save('training',info)
                    print(json.dumps(record),flush=True)
                    if run:run.log({'updates':agent.updates,'train_seconds':elapsed,**{'train/'+k:v for k,v in info.items()}})
            if stop[0]:save('paused',info);return
            save('evaluating',info)
            if args.smoke and goal==0:continue
            n=1024 if args.smoke else campaign['eval_samples']
            ref=reference[:n]
            samples=sample_actions(agent,n,900000+args.seed)
            result=metrics(samples,target,ref)
            result.update(updates=int(agent.updates),train_seconds=elapsed,Q_evaluations=info.get('Q_evaluations',0))
            if goal==cfg['steps']:
                result['latency']=latency(agent,cfg['method'],warmup=5 if args.smoke else 50,
                                          repetitions=10 if args.smoke else 200,blocks=2 if args.smoke else 5)
            atomic_json(folder/f'evaluation_{goal:07d}.json',result)
            np.save(folder/f'samples_{goal:07d}.npy',samples)
            if not args.smoke:plot_samples(folder/f'samples_{goal:07d}.png',samples,target,folder.name)
            atomic_json(folder/'latest.json',result)
            print(json.dumps(dict(event='evaluation',name=folder.name,**result)),flush=True)
            if run:run.log({'updates':agent.updates,**{'eval/'+k:v for k,v in result.items() if isinstance(v,(float,int))}})
        save('completed',info)
        if run:run.finish()
    except BaseException as exc:
        atomic_json(folder/'status.json',dict(status='failed',updates=int(agent.updates),error=repr(exc),pid=os.getpid()))
        if run:run.finish(exit_code=1)
        raise

if __name__=='__main__':main()
