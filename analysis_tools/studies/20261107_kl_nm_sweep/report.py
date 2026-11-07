"""CPU-only status, budget aggregation, and explicit preflight gate."""
import argparse
import csv
import json
import time
from pathlib import Path
from common import now, tasks, verify_source, write


def report(root, gate=False):
    commit = verify_source(root)
    counts = {}
    rows = []
    for phase in ['pilot','reuse','train']:
        planned = tasks(phase)
        complete = sum((root/'runtime'/phase/t['id']/'COMPLETE.json').exists() for t in planned)
        counts[phase] = dict(planned=len(planned), complete=complete)
    for begin in (root/'runtime').glob('*/*/attempts/*/BEGIN.json'):
        end = begin.with_name('END.json')
        latest = begin.with_name('CURRENT.json')
        d = json.loads((end if end.exists() else latest if latest.exists() else begin).read_text())
        timing = begin.parents[2]/f'timing_{d["attempt_id"]}.json'
        seconds = json.loads(timing.read_text())['seconds'] if timing.exists() else {}
        row = {k:d.get(k) for k in ['id','phase','target','method','n','m','L','seed','attempt_id','source_commit','job_id','gpu_index','start_utc','end_utc','exit_code','allocated_gpu_hours','peak_memory_mib','energy_kwh_estimate']}
        row.update(gpu_model=d['gpu'].get('model'), gpu_uuid=d['gpu'].get('uuid'),
                   timing_status='finished' if end.exists() else 'last_heartbeat_lower_bound',
                   synchronized_training_gpu_hours=seconds.get('training',0)/3600,
                   compile_seconds=seconds.get('compile',0), evaluation_seconds=seconds.get('evaluation',0))
        rows.append(row)
    if rows:
        tmp = root/'gpu_budget.csv.tmp'
        with tmp.open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(root/'gpu_budget.csv')
    allocations = [json.loads(p.read_text()) for p in (root/'allocations').glob('*.json')]
    reserved = sum(x.get('allocated_gpu_hours',(time.time()-x['start_epoch'])/3600) for x in allocations)
    attempt_hours = sum(x.get('allocated_gpu_hours') or 0 for x in rows)
    summary = dict(utc=now(), source_commit=commit, counts=counts,
                   worker_reserved_gpu_hours=reserved, attempt_gpu_hours=attempt_hours,
                   synchronized_training_gpu_hours=sum(x['synchronized_training_gpu_hours'] for x in rows),
                   note='Worker reservation includes between-task idle time; do not add it to overlapping attempt hours. Running attempts use last heartbeat. Rental charges outside worker reservations are not inferred.')
    write(root/'CAMPAIGN_STATUS.json', summary)
    if gate:
        assert counts['pilot']['complete']==12 and counts['reuse']['complete']==12, counts
        # Every successful full-shape pilot includes restore/RNG/finite checks.
        for task in tasks('pilot'):
            done=json.loads((root/'runtime/pilot'/task['id']/'COMPLETE.json').read_text())
            assert done['checkpoint_restore_passed'] and done['source_commit']==commit
        for task in tasks('reuse'):
            done=json.loads((root/'runtime/reuse'/task['id']/'COMPLETE.json').read_text())
            assert done['reuse_validated'] and done['sample_count']==1048576
        write(root/'VALIDATION.json', dict(status='passed',source_commit=commit,utc=now(),
              full_shape_pilots=12,restored_and_reevaluated_baselines=12,new_training_runs=96))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gate',action='store_true')
    a=p.parse_args();report(a.root.resolve(),a.gate)
