"""Summarize an execution root without touching learner processes."""
import argparse
import csv
import io
import json
from pathlib import Path
from .run import atomic


def summarize(root):
    rows = []
    for run in sorted((root / 'runs').glob('*')):
        try:
            spec = json.loads((run / 'spec.json').read_text())
            p = json.loads((run / 'progress.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        e = p.get('last_evaluation') or {}
        f = p.get('first_success') or {}
        rows.append(dict(run=run.name, method=spec['method'], task=spec['task'],
            seed=spec['seed'], profile=spec['profile'], status=p['status'], steps=p['steps'],
            updates=p['updates'], first_success=f.get('vector_step_end'),
            evaluation_step=e.get('step'), primary_mode=e.get('primary_mode', 'deterministic'),
            primary_success=e.get('success'),
            deterministic_success=e.get('success') if spec['method'] != 'ibolt' else e.get('zero_z_success'),
            success_25_steps=e.get('success_25_steps'), final_success=e.get('final_success'),
            stochastic_success=e.get('stochastic_success'),
            elapsed_seconds=round(p.get('elapsed_seconds', 0), 1),
            transitions_per_second=round(p.get('transitions_per_second', 0), 1),
            source_commit=spec['source_commit']))
    atomic(root / 'summary.json', rows)
    if rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        (root / 'summary.csv').write_text(buf.getvalue())
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); a = p.parse_args()
    print(json.dumps(summarize(a.root), indent=2))
