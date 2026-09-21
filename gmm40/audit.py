"""Audit published intermediate artifacts without treating queued work as complete."""
import json
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from .target import RESULTS,Target
from .navigation import movement
from .evaluation import atomic_json


def main():
    queue=json.loads((RESULTS/'queue.json').read_text());target=Target();runs=[]
    for job in queue['jobs']:
        folder=RESULTS/job['name'];status_path=folder/'status.json'
        if not status_path.exists():
            runs.append(dict(name=job['name'],status='queued',complete=False,resolved=False));continue
        status=json.loads(status_path.read_text());cfg=json.loads((folder/'config.json').read_text())
        navigation='--navigation' in job.get('args',[])
        steps=[];errors=[]
        # metrics.jsonl is published only after a complete evaluation is durable.
        records=folder/'metrics.jsonl'
        published=[json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
        for row in published:
            step=row['updates'] if navigation else row['step'];steps.append(step)
            directory=folder/'evaluations'/f"{'update' if navigation else 'step'}_{step:07d}"
            required=['metrics.json','terminal_and_paths.png' if navigation else 'samples.png',
                      'environment_rollout.gif' if navigation else 'generation_rollout.gif']
            for name in required:
                file=directory/name
                if not file.exists() or file.stat().st_size==0:errors.append(f'{step}: missing/empty {name}')
            if navigation:
                with np.load(directory/'environment_rollout.npz') as data:
                    positions,actions,rewards=data['positions'],data['actions'],data['rewards']
                if positions.shape!=(101,cfg['eval_episodes'],2):errors.append(f'{step}: positions shape')
                if actions.shape!=(100,cfg['eval_episodes'],2):errors.append(f'{step}: actions shape')
                expected=movement(positions[:-1],actions)
                if not np.allclose(expected,positions[1:],atol=1e-6):errors.append(f'{step}: dynamics mismatch')
                if not np.allclose(target.log_prob(positions[1:]),rewards,atol=1e-6):errors.append(f'{step}: reward mismatch')
                if not np.isfinite(positions).all() or not np.isfinite(rewards).all():errors.append(f'{step}: nonfinite rollout')
                if not (directory/'learned_log_partition.npz').exists():errors.append(f'{step}: missing Q partition')
                checkpoint_files=list((folder/'checkpoints').glob(f'update_{step:07d}.*'))
            else:
                samples=np.load(directory/'samples.npy')
                with np.load(directory/'generation_rollout.npz') as data:path=data['positions']
                if samples.shape!=(cfg['eval_samples'],2) or not np.isfinite(samples).all():errors.append(f'{step}: invalid samples')
                if not np.array_equal(path[-1],samples[:len(path[-1])]):errors.append(f'{step}: rollout endpoint differs from samples')
                if row['n_samples']!=len(samples):errors.append(f'{step}: metric sample count mismatch')
                checkpoint_files=list((folder/'checkpoints').glob(f'step_{step:07d}.*'))
                if cfg['temperature']!=1:
                    job_args=job.get('args',[])
                    requested=float(job_args[job_args.index('--temperature')+1]) if '--temperature' in job_args else 1.0
                    valid_control=(cfg.get('method')=='optiq' and cfg.get('fixed_q_temperature_control')
                                   and cfg['temperature']==requested and np.isfinite(requested) and requested>0
                                   and cfg.get('evaluation_target_temperature')==1.0
                                   and np.isclose(cfg.get('teacher_boltzmann_power',0),1/requested))
                    if not valid_control:errors.append(f'{step}: fixed-Q temperature differs without a declared matching control')
            if not checkpoint_files or any(p.stat().st_size==0 for p in checkpoint_files):errors.append(f'{step}: missing/empty checkpoint')
            for file in directory.glob('*.png'):
                with Image.open(file) as picture:picture.verify()
            gif=directory/required[-1]
            if gif.exists():
                with Image.open(gif) as animation:
                    if animation.n_frames<2:errors.append(f'{step}: rollout GIF has fewer than 2 frames')
                    for frame in range(animation.n_frames):
                        animation.seek(frame)
                        if animation.info.get('duration',0)<=0:
                            errors.append(f'{step}: rollout GIF frame {frame} has no positive delay')
        finished=status.get('status')=='completed'
        stopped_early=status.get('status')=='stopped_early'
        if stopped_early and not (job.get('early_stop_authorized') and status.get('early_stop_authorized') and status.get('process_exit_verified')):
            errors.append('early-stop authorization or process exit verification is missing')
        if stopped_early and (status.get('saved_evaluated_updates') not in steps or max(steps,default=-1)!=status.get('saved_evaluated_updates')):
            errors.append('early-stop saved evaluation does not match published evaluations')
        if finished and (status.get('step',status.get('updates'))!=job['steps'] or job['steps'] not in steps):
            errors.append('completed status does not match requested update count/final evaluation')
        if finished or stopped_early:
            endpoint=status['saved_evaluated_updates'] if stopped_early else job['steps']
            start=cfg.get('resume')
            if navigation:expected=set([0]+list(range(10000,endpoint+1,10000))+[endpoint])
            else:
                lower=int(start.split('step_')[-1].split('.')[0]) if start else 0
                expected={lower,endpoint}|{s for s in [100,500,1000,2500,5000]+list(range(10000,endpoint+1,10000)) if lower<s<=endpoint}
            missing=expected-set(steps)
            if missing:errors.append(f'missing intermediate evaluation steps {sorted(missing)}')
        runs.append(dict(name=job['name'],status=status.get('status'),budget=job['steps'],evaluated_steps=steps,
                         errors=errors,artifact_checks_passed=not errors,complete=finished and not errors,
                         resolved=(finished or stopped_early) and not errors,
                         saved_evaluated_updates=status.get('saved_evaluated_updates'),
                         last_observed_updates=status.get('last_observed_updates'),
                         resolution='authorized_early_stop' if stopped_early else 'completed' if finished else 'pending'))
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),all_requested_runs_complete=all(r['complete'] for r in runs),
                all_runs_resolved=all(r['resolved'] for r in runs),
                scope='Published artifacts, array semantics, rollout endpoints/dynamics, image readability and positive GIF frame delays, checkpoints, temperature, budget and intermediate schedule; separate source/numerical audits establish algorithm semantics.',runs=runs)
    atomic_json(RESULTS/'artifact_audit.json',result)
    print(json.dumps(result,indent=2))
    if any(r.get('errors') for r in runs):raise SystemExit('Artifact audit found errors; inspect artifact_audit.json')


if __name__=='__main__':main()
