"""Run the paper fixed-Q experiment: python -m gmm40.run."""
import argparse
import json
import math
from pathlib import Path
import time
import jax
import numpy as np
from .learner import GMM40Learner
from .target import Target
from .metrics import metrics


def evaluation_steps(total):
    milestones = [0, 100, 500, 1000, 2500, 5000]
    milestones += list(range(10000, total+1, 10000))
    return sorted({s for s in milestones if s <= total} | {total})


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def plot_samples(path, full, centers, reference, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, samples, title in zip(axes, [reference, full, centers],
                                 ['Bounded GMM40', 'iBOLT: full policy', 'iBOLT: mean only']):
        ax.scatter(*samples.T, s=1, alpha=.3)
        ax.scatter(*target.means.T, s=12, c='black', marker='+')
        ax.set(xlim=(-40,40), ylim=(-40,40), aspect='equal', title=title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nm', type=int, choices=[64,128,256,512], default=256)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--eval-samples', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-mode', choices=['offline','online'], default='offline')
    parser.add_argument('--project', default='ibolt-gmm40')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1 or args.eval_samples < 40:
        parser.error('steps/batch must be positive; eval-samples must be at least 40')
    target = Target()
    config = dict(seed=args.seed, n=args.nm, m=args.nm, batch=args.batch,
                  hidden_dims=[256,256,256], activation='GELU', learning_rate=.0003,
                  temperature=1., density_beta=1., latent_dim=2,
                  mean_output_init_scale=16., log_std_min=-5., log_std_max=-3.5,
                  initial_log_std=-4., teacher_std_floor=.05,
                  steps=args.steps, eval_samples=args.eval_samples, dacer=False)
    if args.resume:
        previous = json.loads((args.resume.parent.parent/'config.json').read_text())
        for key in config.keys()-{'steps'}:
            if previous[key] != config[key]:
                parser.error(f'Resume configuration mismatch: {key}')
    agent = GMM40Learner(target, seed=args.seed, n=args.nm, m=args.nm, batch=args.batch)
    if args.resume:
        agent.restore(args.resume)
    if agent.updates > args.steps:
        parser.error('Checkpoint is beyond requested update budget')
    # Never overwrite an existing experiment directory.
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output/'checkpoints'
    checkpoints.mkdir()
    config['parameters'] = sum(p.size for p in jax.tree_util.tree_leaves(agent.state.params))
    config['backend'] = jax.default_backend()
    config['target'] = target.metadata
    save_json(args.output/'config.json', config)
    reference = target.sample(args.eval_samples, 20260917, bounded=True)
    unbounded_reference = target.sample(args.eval_samples, 20260917, bounded=False)
    np.save(args.output/'reference.npy', reference)
    tracking = None
    completed = False
    try:
        if args.wandb:
            import wandb
            tracking = wandb.init(project=args.project, mode=args.wandb_mode,
                                  name=f'ibolt-NM{args.nm}-seed{args.seed}', config=config,
                                  dir=str(args.output), save_code=False)
            tracking.define_metric('actor_updates')
            tracking.define_metric('*', step_metric='actor_updates')
        train_seconds, info = 0., {}
        goals = sorted(set([agent.updates]+[s for s in evaluation_steps(args.steps) if s>agent.updates]))
        for goal in goals:
            while agent.updates < goal:
                start = time.monotonic()
                info = agent.advance(min(50, goal-agent.updates))
                train_seconds += time.monotonic()-start
                if not all(np.isfinite(v) for v in info.values()):
                    raise FloatingPointError(info)
            assert int(agent.state.step) == agent.updates
            agent.save(checkpoints/f'step_{goal:07d}.bin')
            samples, _, extra = agent.evaluate_samples(args.eval_samples, 900000+args.seed)
            folder = args.output/'evaluations'/f'step_{goal:07d}'
            folder.mkdir(parents=True)
            report = dict(actor_updates=goal, Q_evaluations=goal*args.batch*args.nm,
                          train_seconds_this_process=train_seconds, training=info)
            for mode, points in [('full_policy',samples),('mu_only',extra['mu_only'])]:
                np.save(folder/f'samples_{mode}.npy',points)
                result = metrics(points,target,reference,unbounded_reference)
                result['mmd'] = math.sqrt(max(result['mmd2'],0.))
                save_json(folder/f'metrics_{mode}.json',result)
                report[mode] = result
            save_json(folder/'summary.json',report)
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(report,allow_nan=False)+'\n')
            if not args.no_plots:
                plot_samples(folder/'samples.png',samples,extra['mu_only'],reference,target)
            if tracking:
                values = {f'{mode}/{k}':v for mode in ['full_policy','mu_only']
                          for k,v in report[mode].items() if isinstance(v,(int,float))}
                tracking.log(dict(values,actor_updates=goal,Q_evaluations=report['Q_evaluations']))
            print(json.dumps({k:report[k] for k in ['actor_updates','Q_evaluations']}) ,flush=True)
        save_json(args.output/'completed.json',dict(actor_updates=agent.updates,completed=True))
        completed = True
    finally:
        if tracking:
            tracking.finish(exit_code=0 if completed else 1)


if __name__ == '__main__':
    main()
