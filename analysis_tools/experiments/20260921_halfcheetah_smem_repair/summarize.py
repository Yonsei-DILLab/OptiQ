"""Report actual evaluation steps, including stopped runs and tuning selection."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone


def collect(root):
    records = []
    for path in sorted(root.glob('outputs/*/config.json')):
        cfg = json.loads(path.read_text())
        csv_path = path.parent/'logs/progress.csv'
        if not csv_path.exists():
            continue
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        evaluations = [r for r in rows if r.get('eval/mean_reward') and r.get('time/total_timesteps')]
        method = cfg.get('experiment', {}).get('method', '')
        record = dict(
            directory=str(path.parent), method=method, seed=cfg['seed'],
            revision=cfg.get('experiment_revision', 'v1_stopped'),
            phase=cfg.get('comparison_phase', 'stopped_original'),
            steps=max((int(float(r['time/total_timesteps'])) for r in rows if r.get('time/total_timesteps')), default=0),
            evaluations=[dict(
                step=int(float(r['time/total_timesteps'])),
                zero_z=float(r['eval/mean_reward']),
                stochastic_z=float(r['eval/stochastic_z/mean_reward']),
            ) for r in evaluations],
            diagnostics={},
        )
        for key in ['actor_accepted_mean','actor_nll_gain_mean','actor_kl_bound_mean','smem_attempted_mean','smem_selected_mean','source_ess_absolute_mean']:
            values = [float(r['train/'+key]) for r in rows if r.get('train/'+key)]
            if values:
                record['diagnostics'][key] = values[-1]
        gate = path.parent/'learning_gate.json'
        if gate.exists():
            record['gate'] = json.loads(gate.read_text())
        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--original', type=Path)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    runs = collect(args.campaign)
    if args.original:
        runs = collect(args.original)+runs
    output = args.campaign/'analysis'
    output.mkdir(exist_ok=True)
    payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(),runs=runs)
    (output/'progress.json').write_text(json.dumps(payload,indent=2)+'\n')
    text = ['# HalfCheetah SMEM+TR repair progress', '',
            'Seed 0 was used for repair selection. Seeds 1/2 are subsequent fixed-method checks.',
            'Original runs were stopped by user request. Compare equal environment steps; latest points have different budgets.', '',
            '| Revision | Method | Seed | Eval step | Zero-z return | Stochastic-z return |',
            '|---|---|---:|---:|---:|---:|']
    for run in runs:
        if not run['evaluations']:
            continue
        ev = run['evaluations'][-1]
        text.append(f"| {run['revision']} | {run['method']} | {run['seed']} | {ev['step']} | {ev['zero_z']:.1f} | {ev['stochastic_z']:.1f} |")
        print(run['method'],run['seed'],run['steps'],ev,run['diagnostics'],run.get('gate'))
    groups = defaultdict(list)
    for run in runs:
        if run['revision']=='projection_repair_v2':
            groups[run['method']].append(run)
    text += ['', '## Exact 1M results (no extrapolation)', '']
    for method, group in groups.items():
        values = [e for r in group for e in r['evaluations'] if e['step']==1_000_000]
        if len(values) != 3:
            text.append(f'{method}: {len(values)}/3 seeds have an actual 1M evaluation.')
            continue
        import statistics
        for mode in ['zero_z','stochastic_z']:
            v = [e[mode] for e in values]
            text.append(f'{method} {mode}: {statistics.mean(v):.1f} ± {statistics.stdev(v):.1f} (sample SD, n=3).')
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
        colors = {'direct_gmm_trg':'#2878b5','state_conditioned_smem_tr':'#999999','smem_aux':'#e87500','direct':'#2878b5'}
        for r in runs:
            for ax,mode in zip(axes,['zero_z','stochastic_z']):
                ev=r['evaluations']
                label=f"{r['method']} s{r['seed']}"+(' (stopped)' if r['revision']=='v1_stopped' else '')
                ax.plot([e['step']/1000 for e in ev],[e[mode] for e in ev],color=colors.get(r['method']),
                        ls='--' if r['revision']=='v1_stopped' else '-',alpha=.7,label=label)
        for ax,title in zip(axes,['Zero-z evaluation','Stochastic-z evaluation']):
            ax.set(xlabel='Environment steps (thousands)',ylabel='Mean episode return',title=title)
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
        fig.savefig(output/'learning_curves.png',dpi=170)


if __name__=='__main__':
    main()
