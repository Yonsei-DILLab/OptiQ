"""Fixed-alpha search, independent selection, and held-out-seed confirmation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .run import atomic, guard


def select(jobs):
    groups = {}
    for job in jobs:
        groups.setdefault((job['task'], float(job['sac_ent_coef'])), []).append(job)
    ranking = {}
    for (task, alpha), group in groups.items():
        assert sorted(j['seed'] for j in group) == [10, 11]
        values = []
        for j in group:
            if j['status'] != 'completed':
                break
            out = Path(j['output'])
            done = json.loads((out/'completed.json').read_text())
            assert done['completed'] and not done['smoke'] and done['steps'] == 5000192 and done['updates'] == 623424
            config = json.loads((out/'config.json').read_text())
            assert config['entropy_coefficient'] == alpha and not config['automatic_entropy_tuning']
            records = {r['steps']: r for r in map(json.loads, (out/'evaluations.jsonl').read_text().splitlines())}
            final = records[5000192]['deterministic']
            assert len(final['episodes']) == 100
            points = sorted((step, r['deterministic']['success_rate']) for step, r in records.items())
            auc = sum((b-a)*(x+y)/2 for (a, x), (b, y) in zip(points, points[1:]))/5000192
            values.append(dict(seed=j['seed'], success=final['success_rate'], auc=auc, mean_return=final['mean_return']))
        eligible = len(values) == 2
        row = dict(alpha=alpha, eligible=eligible, values=values)
        if eligible:
            row.update(mean_success=sum(v['success'] for v in values)/2,
                       mean_auc=sum(v['auc'] for v in values)/2,
                       mean_return=sum(v['mean_return'] for v in values)/2)
            row['rank'] = [row['mean_success'], row['mean_auc'], row['mean_return'], -alpha]
        ranking.setdefault(task, []).append(row)
    selected = {}
    for task, rows in ranking.items():
        rows.sort(key=lambda r: r.get('rank', [-1, -1, -1, -1]), reverse=True)
        if not rows[0]['eligible']:
            raise RuntimeError('No alpha has both search seeds completed: '+task)
        selected[task] = rows[0]['alpha']
    assert set(selected) == {'door', 'hammer', 'pen', 'relocate'}
    return dict(selected_alpha=selected, ranking=ranking, selection_seeds=[10, 11], confirmation_seeds=list(range(5)),
        rule='Mean final deterministic success (100 episodes/seed); ties: 0-5M success AUC, final mean return, lower alpha. Require both search seeds; retain failed candidates.')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    guard(); cfg = json.loads(a.config.read_text()); root = a.config.parent
    def state(status, **extra):
        atomic(root/'state.json', dict(status=status, observed_unix=time.time(), **extra))
    try:
        for phase in ('smoke', 'search'):
            plan = Path(cfg[phase]); data = json.loads(plan.read_text())
            assert data['status'] == 'prepared', 'No automatic replay resume or overwrite'
            state('running', phase=phase)
            subprocess.run([cfg['python'], '-u', '-m', 'adroit_rebuttal.sac_grid_campaign', '--plan', str(plan)],
                           cwd=cfg['source'], check=True)
            data = json.loads(plan.read_text())
            if phase == 'smoke':
                assert all(j['status'] == 'completed' for j in data['jobs'])
                hashes = {json.loads((Path(j['output'])/'config.json').read_text())['initial_actor_sha256'] for j in data['jobs']}
                assert len(hashes) == 1, 'Alpha variants must share paired initialization'
        search = json.loads(Path(cfg['search']).read_text())
        selection = select(search['jobs']); atomic(root/'selection.json', selection)
        confirmation = Path(cfg['confirmation']); data = json.loads(confirmation.read_text())
        assert data['status'] == 'prepared'
        for j in data['jobs']:
            j['sac_ent_coef'] = selection['selected_alpha'][j['task']]
        atomic(confirmation, data)
        state('running', phase='confirmation', selected_alpha=selection['selected_alpha'])
        subprocess.run([cfg['python'], '-u', '-m', 'adroit_rebuttal.sac_grid_campaign', '--plan', str(confirmation)],
                       cwd=cfg['source'], check=True)
        data = json.loads(confirmation.read_text())
        assert data['status'] == 'completed' and len(data['jobs']) == 20 and all(j['status'] == 'completed' for j in data['jobs'])
        for j in data['jobs']:
            r = json.loads((Path(j['output'])/'completed.json').read_text())
            assert r['steps'] == 5000192 and r['updates'] == 623424 and r['completed'] and not r['smoke']
        state('completed', selected_alpha=selection['selected_alpha'], confirmation_runs=20)
    except BaseException as error:
        state('held', reason=repr(error))
        raise


if __name__ == '__main__':main()
