"""Audit sampler time across actual checkpoint ancestry; no training changes."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flax.serialization import msgpack_restore
import numpy as np
import wandb

from .accounting import density_evaluations

ROOT = Path(__file__).resolve().parents[2]


def segment_timing(rows, start, end):
    """Use steady training-log intervals containing no evaluation event.

    This estimates iteration time, not total wall time. The separate observed
    span includes evaluation, checkpoints, compilation and logging overhead.
    """
    selected = [row for row in rows if start <= row['update'] <= end]
    training = [(i,row) for i,row in enumerate(selected) if 'actor_loss' in row]
    rates = []
    # Exclude initial blocks, which may include JIT compilation and warmup.
    for (i,left),(j,right) in zip(training[2:],training[3:]):
        if any('eval/modes_covered' in row for row in selected[i+1:j]):
            continue
        updates = right['update']-left['update']
        seconds = right['elapsed_seconds']-left['elapsed_seconds']
        if updates > 0 and seconds > 0:
            rates.append(seconds/updates)
    if not rates:
        raise ValueError('Too few steady training intervals for timing audit')
    return dict(start_update=start,end_update=end,updates=end-start,
        steady_intervals=len(rates),
        seconds_per_update_p10_p50_p90=np.quantile(rates,[.1,.5,.9]).tolist(),
        estimated_training_seconds=float(np.median(rates)*(end-start)),
        observed_segment_span_seconds=selected[-1]['elapsed_seconds']-selected[0]['elapsed_seconds'],
        excluded='Initial3 training log entries and any interval crossing evaluation; '
                 'median also reduces asynchronous logging jitter. Estimate, not exact profiler time.')


@lru_cache(maxsize=None)
def checkpoint_lineage(checkpoint):
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint.parent/'config.json').read_text())
    end = int(msgpack_restore(checkpoint.read_bytes())['step'])
    if cfg.get('resume_checkpoint'):
        previous = checkpoint_lineage(str(Path(cfg['resume_checkpoint']).resolve()))
        start = previous[-1]['end_update']
    else:
        previous,start = [],0
    history = (checkpoint.parent/'history.jsonl').read_text()
    rows = [json.loads(line) for line in history.splitlines() if line.strip()]
    row = segment_timing(rows,start,end)
    timing_path = checkpoint.parent/'training_timing.jsonl'
    if timing_path.exists():
        measurements = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
        matches = [m for m in measurements if m['update'] == end and m['checkpoint'] == checkpoint.name]
        if matches:
            row['measured_timing'] = matches[-1]
    row.update(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        wandb_url=cfg.get('wandb_url'),K=cfg['num_policy_samples'],R=cfg['proposals_per_policy_sample'],
        batch_size=cfg['batch_size'],hidden_dims=cfg['hidden_dims'],training_seed=cfg['seed'],
        oracle_queries=(end-start)*cfg['batch_size']*cfg['num_policy_samples']*cfg['proposals_per_policy_sample'])
    return previous+[row]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--exports',type=Path,nargs='+',required=True,help='Export directories containing summary.json')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for file in [os.environ.get('OPTIQ_ENV_FILE'),ROOT/'.env',ROOT.parent/'.env']:
        if file:load_dotenv(file,override=False)
    result = dict(diagnostic_only=True,not_a_matched_hardware_win_claim=True,
        paper_reported_training_hours=.87,
        interpretation='Sum the exact ancestry used by each exported checkpoint. '
            'Report measured training time only when every ancestor has explicit '
            'evaluation-excluded checkpoint timing. Otherwise retain a missing measured '
            'total and the median-interval estimate, separately reporting observed run spans. '
            'Setup and initial evaluation are disclosed separately. Tuning and auxiliary CFM costs are not folded '
            'into per-candidate sampler time and must be reported separately. '
            'These3090 runs shared resources; the iDEM paper usedA100. '
            'Final confirmation needs standalone timing and all quality metrics.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidates=[])
    for export in args.exports:
        info=json.loads((export/'summary.json').read_text())
        stages=checkpoint_lineage(str(Path(info['checkpoint']).resolve()))
        assert stages[-1]['end_update']==info['actor_update']
        assert sum(s['oracle_queries'] for s in stages)==density_evaluations(info['parent_config'],info['actor_update'])
        measured_hours = (sum(s['measured_timing']['measured_training_seconds'] for s in stages)/3600
                          if all('measured_timing' in s for s in stages) else None)
        result['candidates'].append(dict(label=export.name,stages=stages,
            measured_training_hours=measured_hours,
            estimated_training_hours=sum(s['estimated_training_seconds'] for s in stages)/3600,
            observed_segment_span_hours=sum(s['observed_segment_span_seconds'] for s in stages)/3600,
            oracle_queries=sum(s['oracle_queries'] for s in stages)))
    run=wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'),mode='online',dir=str(args.output),
        group='gmm40-idem-protocol',job_type='timing-audit',name='GMM40-checkpoint-ancestry-timing',
        config={k:v for k,v in result.items() if k!='candidates'})
    result['wandb_url']=run.url
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# GMM40 sampler timing audit','',result['interpretation'],'',
           '| Candidate | Measured training h | Estimated training h | Observed stage spans h | Oracle queries |',
           '|---|---:|---:|---:|---:|']
    for row in result['candidates']:
        measured = '—' if row['measured_training_hours'] is None else f"{row['measured_training_hours']:.3f}"
        lines.append(f"| {row['label']} | {measured} | {row['estimated_training_hours']:.3f} | {row['observed_segment_span_hours']:.3f} | {row['oracle_queries']:,} |")
        run.log({row['label']+'/'+k:v for k,v in row.items() if isinstance(v,(int,float))})
        print(json.dumps({k:v for k,v in row.items() if k!='stages'}),flush=True)
    (args.output/'analysis.md').write_text('\n'.join(lines)+'\n')
    artifact=wandb.Artifact('gmm40-timing-audit-'+run.id,type='gmm40-evaluation')
    for name in ['summary.json','analysis.md']:artifact.add_file(str(args.output/name))
    run.log_artifact(artifact);run.finish()


if __name__=='__main__':main()
