"""Persist live progress, sparse success curves and stalled/failed-run diagnostics."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import time
from .run import atomic


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    folder = a.plan.parent
    while True:
        plan = json.loads(a.plan.read_text())
        rows, curves, alerts = [], [], []
        for job in plan['jobs']:
            out = Path(job['output'])
            f = out/'progress.json'
            progress = json.loads(f.read_text()) if f.exists() else {}
            failure = out/'failure.json'
            row = dict(id=job['id'], task=job['task'], method=job['method'], seed=job['seed'],
                status=job['status'], steps=progress.get('steps', 0), updates=progress.get('updates', 0),
                progress_age_seconds=time.time()-f.stat().st_mtime if f.exists() else None,
                discovery=progress.get('discovery'), evaluations=[])
            if failure.exists():
                alerts.append(dict(id=job['id'], error=json.loads(failure.read_text())['error']))
            if job['status'] == 'running' and row['progress_age_seconds'] is not None and row['progress_age_seconds'] > 600:
                alerts.append(dict(id=job['id'], error='No progress update for more than 600 seconds; inspect worker and evaluation'))
            f = out/'evaluations.jsonl'
            if f.exists():
                for line in f.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for mode in ('zero_z', 'stochastic_z', 'stochastic_z_sigma', 'deterministic', 'stochastic', 'policy'):
                        if mode not in record:
                            continue
                        value = record[mode]
                        curve = dict(id=job['id'], task=job['task'], method=job['method'], seed=job['seed'],
                            steps=record['steps'], mode=mode, mean_return=value['mean_return'],
                            success_rate=value['success_rate'], final_success_rate=value['final_success_rate'],
                            success_25_steps_rate=value['success_25_steps_rate'], episodes=len(value['episodes']))
                        curves.append(curve)
                        row['evaluations'].append(curve)
            rows.append(row)
        state = dict(observed_unix=time.time(), campaign_status=plan['status'],
                     counts=dict(Counter(j['status'] for j in plan['jobs'])), runs=rows, alerts=alerts)
        atomic(folder/'monitor.json', state)
        with (folder/'monitor-history.jsonl').open('a') as out:
            out.write(json.dumps(dict(observed_unix=state['observed_unix'], counts=state['counts'],
                progress=[{k: r[k] for k in ('id', 'steps', 'updates', 'status')} for r in rows], alerts=alerts))+'\n')
        if curves:
            temp = folder/'evaluation-curves.tmp.csv'
            with temp.open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=list(curves[0]))
                writer.writeheader(); writer.writerows(curves)
            temp.replace(folder/'evaluation-curves.csv')
        print(json.dumps(dict(observed_unix=state['observed_unix'], counts=state['counts'],
            max_steps=max(r['steps'] for r in rows), alerts=alerts)), flush=True)
        if plan['status'] in ('completed', 'completed_with_failures', 'controller_error'):
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
