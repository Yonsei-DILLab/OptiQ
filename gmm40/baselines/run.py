"""Run MEow or TD3 on GMM40: python -m gmm40.baselines.run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from gmm40.metrics import metrics
from gmm40.run import evaluation_steps, save_json
from gmm40.target import Target
from .meow_learner import MEowBanditLearner
from .td3_learner import TD3BanditLearner


def runtime(device):
    root = Path(__file__).resolve().parents[2]
    sources = sorted(Path(__file__).parent.rglob('*.py'))
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        revision = None
    return dict(revision=revision,device=str(device),cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                packages={k:importlib.metadata.version(k) for k in
                          ('torch','numpy','scipy','gymnasium','stable-baselines3')},
                baseline_source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sources})


def plot(path, samples, reference, target, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    for ax,x,label in zip(axes,[reference,samples],['Bounded GMM40 target',title]):
        ax.scatter(*x.T,s=1,alpha=.3)
        ax.scatter(*target.means.T,s=12,c='black',marker='+')
        ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',title=label)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',choices=['meow','td3'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=100000,help='Actor optimizer updates, not environment steps')
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--buffer-size',type=int,default=1000000)
    parser.add_argument('--eval-samples',type=int,default=10000)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.batch,args.buffer_size,args.threads)<1 or args.warmup<0 or args.eval_samples<40:
        parser.error('Invalid positive training/evaluation sizes or warmup')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Learning rate must be positive and finite')
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    target = Target()
    cls = MEowBanditLearner if args.baseline=='meow' else TD3BanditLearner
    agent = cls(target,seed=args.seed,batch=args.batch,learning_rate=args.learning_rate,
                warmup=args.warmup,buffer_size=args.buffer_size,device=str(device))
    config = dict(baseline=args.baseline,adaptation='terminal_one_step_bandit',
                  seed=args.seed,steps=args.steps,batch=args.batch,learning_rate=args.learning_rate,
                  warmup_transitions=args.warmup,buffer_size=args.buffer_size,
                  observation='constant zero, dimension 1',action_dimension=2,
                  physical_action_bounds=[-40.,40.],reward='log p_GMM40(40*a)',gamma=0.,
                  entropy_coefficient=1. if args.baseline=='meow' else 0.,
                  evaluation_policy='stochastic_flow' if args.baseline=='meow' else 'deterministic_actor',
                  eval_samples=args.eval_samples,evaluation_seed=900000+args.seed,
                  reference_seed=20260917,target=target.metadata,algorithm_settings=agent.settings,
                  runtime=runtime(device),training_uses_reference_samples=False)
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoints = args.output/'checkpoints';checkpoints.mkdir()
    save_json(args.output/'config.json',config)
    reference = target.sample(args.eval_samples,20260917,bounded=True)
    unbounded = target.sample(args.eval_samples,20260917,bounded=False)
    np.save(args.output/'reference.npy',reference)
    train_seconds,info = 0.,{}
    for goal in evaluation_steps(args.steps):
        while agent.updates<goal:
            start = time.monotonic()
            info = agent.advance(min(50,goal-agent.updates))
            train_seconds += time.monotonic()-start
            if not all(np.isfinite(v) for v in info.values()):
                raise FloatingPointError(info)
        if agent.updates!=goal:
            raise RuntimeError(f'Actor update count mismatch: {agent.updates} != {goal}')
        checkpoint = agent.save(checkpoints/f'step_{goal:07d}')
        samples = agent.evaluate_samples(args.eval_samples,900000+args.seed)
        if not np.isfinite(samples).all() or np.any(np.abs(samples)>40.00001):
            raise FloatingPointError('Invalid policy evaluation samples')
        folder = args.output/'evaluations'/f'step_{goal:07d}';folder.mkdir(parents=True)
        np.save(folder/'samples_full_policy.npy',samples)
        result = metrics(samples,target,reference,unbounded)
        result['mmd'] = math.sqrt(max(result['mmd2'],0.))
        report = dict(actor_updates=goal,Q_evaluations=agent.env.reward_evaluations,
                      train_seconds_this_process=train_seconds,training=info,
                      full_policy=result,checkpoint=str(checkpoint.relative_to(args.output)))
        save_json(folder/'metrics_full_policy.json',result)
        save_json(folder/'summary.json',report)
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(report,allow_nan=False)+'\n')
        if not args.no_plots:
            plot(folder/'samples.png',samples,reference,target,
                 'MEow stochastic policy' if args.baseline=='meow' else 'TD3 deterministic policy')
        print(json.dumps(dict(actor_updates=goal,Q_evaluations=agent.env.reward_evaluations,
                              mmd=result['mmd'],mode_coverage=result['mode_coverage'])),flush=True)
    save_json(args.output/'completed.json',dict(actor_updates=agent.updates,completed=True))


if __name__=='__main__':
    main()
