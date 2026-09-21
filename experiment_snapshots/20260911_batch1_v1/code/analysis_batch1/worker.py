"""Train one explicit case/version/seed and save paired, independent evaluations."""
import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
import optax
from flax import serialization
from scipy.stats import wasserstein_distance
from analysis_boltzmann.problems import make_problem
from .core import config_for, initialize, sampler, state_digest, update, write_json
from .reference import gmm_labels, gmm_logp


def evaluate(actor, draw, cfg, ref, step, output):
    final=step==cfg['updates']
    count=100000 if final else 20000
    samples=draw(actor.params,jax.random.PRNGKey(2000000+cfg['seed']+step),count)
    if not np.isfinite(samples).all(): raise FloatingPointError('Nonfinite actor samples')
    gmm=cfg['case']=='gmm40'
    if gmm:
        q=gmm_logp(samples,ref['centers'],ref['scales'])
        labels=gmm_labels(samples,ref['centers'])
    else:
        problem=make_problem(cfg['case']);q=problem.q(samples);labels=problem.labels(samples)
        assert np.max(abs(samples))<=1.
    occupancy=np.bincount(labels,minlength=len(ref['mode_mass']))/count
    positive=occupancy[occupancy>0]
    truth=float(ref['truth'])
    means=q[:(len(q)//50)*50].reshape(-1,50).mean(1)
    reference=ref['samples'][:count]
    directions=np.random.default_rng(888).normal(size=(samples.shape[1],32))
    directions/=np.linalg.norm(directions,axis=0)
    # Matched-size empirical reference; finite-sample SW2 is a diagnostic, not exact W2.
    sw2=np.sqrt(np.mean((np.sort(samples@directions,axis=0)-np.sort(reference@directions,axis=0))**2))
    row=dict(update=step,case=cfg['case'],version=cfg['version'],seed=cfg['seed'],
        sample_count=count,mean_q=float(q.mean()),reference_mean_q=truth,
        reference_mean_se=float(ref['truth_se']) if 'truth_se' in ref.files else 0.,
        mean_abs_error=float(abs(q.mean()-truth)),backup_k50_bias=float(means.mean()-truth),
        backup_k50_rmse=float(np.sqrt(np.mean((means-truth)**2))),
        backup_k50_std=float(means.std(ddof=1)),backup_repetitions=len(means),
        mode_bin_tv=float(.5*abs(occupancy-ref['mode_mass']).sum()),
        modes_observed=int((occupancy>0).sum()),modes_above_0p1pct=int((occupancy>=.001).sum()),
        effective_modes=float(np.exp(-np.sum(positive*np.log(positive)))),
        sliced_w2=float(sw2),std_per_coordinate=samples.std(0).tolist(),
        mean_per_coordinate=samples.mean(0).tolist(),
        evaluated_actor_queries=step*cfg['candidate_count'],
        proposed_random_actions=step*cfg['random_candidate_count'])
    if gmm: row['occupancy_tvd_uniform']=float(.5*abs(occupancy-1/40).sum())
    if samples.shape[1]==1:
        row['action_w1']=float(wasserstein_distance(samples[:,0],reference[:,0]))
    np.savez_compressed(output/('distribution_'+str(step)+'.npz'),samples=samples,
                        mode_mass=occupancy,reference_mode_mass=ref['mode_mass'])
    write_json(output/'latest_evaluation.json',row)
    with (output/'evaluations.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
    return row


def run(campaign,index):
    campaign=Path(campaign);task=json.loads((campaign/'tasks.json').read_text())[index]
    cfg=task['config'];assert cfg==config_for(cfg['case'],cfg['version'],cfg['seed'])
    output=campaign/'runs'/task['name'];output.mkdir(exist_ok=False)
    write_json(output/'config.json',cfg)
    if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
    reference_path=campaign/'references'/(cfg['case']+'.npz')
    ref=np.load(reference_path)
    actor,oracle,key=initialize(cfg)
    digest=state_digest(actor)
    write_json(output/'manifest.json',dict(initial_state_sha256=digest,
        job=os.environ.get('SLURM_JOB_ID'),devices=[str(d) for d in jax.devices()],
        source_hashes=json.loads((campaign/'source_hashes.json').read_text()),
        packages={p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','optax','numpy','scipy')},
        task=task,reference_metadata=json.loads(reference_path.with_suffix('.json').read_text())))
    draw=sampler(actor,cfg);start=time.monotonic();training_seconds=0.;step=0
    try:
        evaluate(actor,draw,cfg,ref,0,output)
        (output/'actor_0.msgpack').write_bytes(serialization.to_bytes(actor))
        window_start=time.monotonic()
        for step in range(1,cfg['updates']+1):
            if cfg['case']=='gmm40' and step==50001:
                actor=actor.replace(tx=optax.adam(cfg['learning_rate_after_50k'],b1=.9,b2=.999))
            actor,loss,key,metrics=update(actor,oracle,key,cfg,step-1)
            if step%100==0 or step in cfg['checkpoints']:
                loss=float(loss)
                if not np.isfinite(loss): raise FloatingPointError('Nonfinite actor loss')
                row=dict(update=step,loss=loss,elapsed_seconds=time.monotonic()-start,
                    **{k:float(metrics[k]) for k in ('source_ess_absolute','max_source_weight',
                       'policy_spread_l2','selected_delta_l2','source_q_std')})
                write_json(output/'progress.json',row)
                with (output/'training.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if step in cfg['checkpoints']:
                jax.block_until_ready(actor.params);training_seconds+=time.monotonic()-window_start
                final_metrics=evaluate(actor,draw,cfg,ref,step,output)
                if step in (100,1000,5000,20000,50000,75000,cfg['updates']):
                    (output/('actor_'+str(step)+'.msgpack')).write_bytes(serialization.to_bytes(actor))
                write_json(output/'timing.json',dict(update=step,training_seconds=training_seconds,
                    elapsed_seconds=time.monotonic()-start,includes_training_jit=True,
                    evaluated_actor_queries=step*cfg['candidate_count']))
                window_start=time.monotonic()
        write_json(output/'completed.json',dict(finished=True,updates=step,initial_state_sha256=digest,
            training_seconds=training_seconds,elapsed_seconds=time.monotonic()-start,final_metrics=final_metrics))
        (output/'COMPLETE').write_text('validated finite final metrics\n')
    except BaseException as exc:
        write_json(output/'failed.json',dict(update=step,error=repr(exc)))
        (output/'actor_interrupted.msgpack').write_bytes(serialization.to_bytes(actor))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--index',type=int,default=None);args=parser.parse_args()
    run(args.campaign,args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
