"""Refresh the shared comparison from durable artifacts, including unfinished runs."""
import csv
import fcntl
import html
import json
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .target import RESULTS,Target


def sample_grid(rows,phase):
    """One current primary run per method; followups remain separate in the full table."""
    selected={}
    for row in rows:
        if (row['phase']==phase and row['budget']==100000 and row['mmd2'] is not None
                and (phase!='fixed' or row.get('temperature',1)==1)):
            selected.setdefault(row['method'],row)
    if not selected:return False
    from .evaluation import background
    target=Target();reference=target.sample(5000,20260917,bounded=phase=='fixed')
    fig,axes=plt.subplots(2,4,figsize=(20,10),constrained_layout=True)
    bounds=40 if phase=='fixed' else 50
    for ax in axes.ravel():background(ax,target);ax.set(xlim=(-bounds,bounds),ylim=(-bounds,bounds))
    axes[0,0].scatter(*reference.T,s=2,alpha=.3,color='#3676b9')
    axes[0,0].set_title('Ground truth | '+('bounded GMM40' if phase=='fixed' else 'original GMM40'))
    for ax,method in zip(axes.ravel()[1:],['optiq','direct_gmm','sac','dipo','meow','mfpo']):
        if method not in selected:
            ax.set_title(method.upper()+' | queued');continue
        row=selected[method];prefix='step' if phase=='fixed' else 'update'
        directory=RESULTS/row['name']/'evaluations'/f"{prefix}_{row['evaluated_updates']:07d}"
        if phase=='fixed':points=np.load(directory/'samples.npy')
        else:
            with np.load(directory/'environment_rollout.npz') as data:points=data['positions'][-1]
        ax.scatter(*points[:5000].T,s=2,alpha=.35,color='#dd7932')
        ax.set_title(f"{method.upper()} | {row['evaluated_updates']:,} updates | {row['status']}\nMMD² {row['mmd2']:.4g}, components {row['modes']}/40, precision {row['precision']:.1%}")
    axes.ravel()[-1].set_visible(False)
    fig.savefig(RESULTS/f'{phase}_samples_comparison.png',dpi=150);fig.savefig(RESULTS/f'{phase}_samples_comparison.pdf');plt.close(fig)
    return True


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def optiq_variant_grid(rows):
    selected=[row for row in rows if row['phase']=='fixed' and row['method'] in ('optiq','direct_gmm')
              and row.get('temperature',1)==1
              and (row['budget']==100000 or row.get('role')=='hparam_diagnostic')]
    if not selected:return False
    from .evaluation import background
    target=Target()
    fig,axes=plt.subplots(2,len(selected),figsize=(5*len(selected),9),squeeze=False,constrained_layout=True)
    for column,row in enumerate(selected):
        folder=RESULTS/row['name'];cfg=read(folder/'config.json',{})
        for ax in axes[:,column]:background(ax,target)
        if not (folder/'latest.json').exists():
            axes[0,column].set_title(row['name']+'\nqueued',fontsize=9)
            axes[1,column].set_title('Conditional means: queued');continue
        result=read(folder/'latest.json');step=result['step']
        directory=folder/'evaluations'/f'step_{step:07d}'
        full=np.load(directory/'samples.npy');mu=np.load(directory/'samples_mu_only.npy')
        means=read(directory/'metrics_mu_only.json')
        label=f"{row['method']} | {cfg.get('width',256)}×{cfg.get('depth',2)} | N={cfg['n']}, M={cfg['m']}"
        if row['method']=='optiq' and (cfg.get('epsilon',.1)!=.1 or cfg.get('sinkhorn_iterations',100)!=100):
            label+=f"\nOT ε={cfg.get('epsilon',.1):g}, iterations={cfg.get('sinkhorn_iterations',100)}"
        if cfg.get('actor_learning_rate',3e-4)!=3e-4:
            label+=f"\nActor LR={cfg['actor_learning_rate']:g}"
        if cfg.get('mean_output_init_scale',1e-4)!=1e-4:
            label+=f"\nMean-head init variance scale={cfg['mean_output_init_scale']:g}"
        axes[0,column].scatter(*full[:5000].T,s=2,alpha=.3,color='#dd7932')
        axes[1,column].scatter(*mu[:5000].T,s=2,alpha=.3,color='#8762a8')
        train=result.get('training',{})
        sigma=train.get('sigma_mean');ess=train.get('teacher_ess')
        stats='' if sigma is None or ess is None else f'\nσ={sigma:.3f}, teacher ESS={ess:.2f}'
        axes[0,column].set_title(f"{label} | {step:,} updates\nFull policy: {result['mode_coverage']}/40, precision {result['high_density_fraction']:.1%}{stats}",fontsize=11)
        axes[1,column].set_title(f"Conditional means only (diagnostic)\n{means['mode_coverage']}/40, precision {means['high_density_fraction']:.1%}",fontsize=11)
    fig.suptitle('OptiQ fixed-Q variants | T=1 | orange: full stochastic policy; purple: 40 tanh(μ) only\nGray contours and black crosses: target. The bottom row does not replace the policy evaluation.',fontsize=13)
    fig.savefig(RESULTS/'optiq_variants_comparison.png',dpi=140)
    fig.savefig(RESULTS/'optiq_variants_comparison.pdf');plt.close(fig)
    return True


def history(folder):
    cfg=read(folder/'config.json',{})
    rows=[]
    if cfg.get('resume'):
        parent=Path(cfg['resume']).parent.parent
        if parent!=folder:rows=history(parent)
    file=folder/'metrics.jsonl'
    if file.exists():rows += [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    by_step={row.get('step',row.get('updates')):row for row in rows}
    return [by_step[key] for key in sorted(by_step)]


def navigation_control_report(series_by_run):
    """Keep control performance alongside, rather than infer it from, density metrics."""
    if not series_by_run:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    lines = ['# Moving-Q control and temperature', '',
             f'Updated (UTC): {datetime.now(timezone.utc).isoformat()}', '',
             'Latest published evaluations; compare equal update counts before drawing conclusions.',
             'Return standard deviation is across evaluation episodes, not across training seeds.',
             'Terminal density matching is an empirical diagnostic, not a guaranteed control objective.', '',
             '| Run | Evaluated updates | Actual T / alpha | Return mean | Episode return std | Terminal reward mean | MMD² | Components /40 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row, series in series_by_run:
        if not series:
            continue
        result = series[-1]
        temperature = result.get('policy_temperature')
        temp_text = '—' if temperature is None else f'{temperature:.5g}'
        lines.append(f"| [{row['name']}]({row['name']}/) | {result['updates']:,} | {temp_text} | {result['mean_return']:.2f} | {result['return_std']:.2f} | {result['terminal_mean_reward']:.3f} | {result['mmd2']:.6f} | {result['mode_coverage']} |")
        xs = [item['updates'] for item in series]
        for ax, key in zip(axes[0], ['mean_return', 'terminal_mean_reward']):
            ax.plot(xs, [item[key] for item in series], marker='.', label=row['name'])
        later = [item for item in series if item['updates'] >= 10000]
        for ax, key in zip(axes[1], ['mean_return', 'terminal_mean_reward']):
            ax.plot([item['updates'] for item in later], [item[key] for item in later], marker='.', label=row['name'])
    for row_index, axis_row in enumerate(axes):
        for ax, title in zip(axis_row, ['Mean 100-step return ↑', 'Mean terminal reward ↑']):
            ax.set(xlabel='Actor updates', ylabel=title,
                   title='All published evaluations' if row_index == 0 else 'From 10K updates (expanded scale)')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='outside lower center', fontsize=8)
    fig.savefig(RESULTS/'navigation_control_curves.png', dpi=140)
    fig.savefig(RESULTS/'navigation_control_curves.pdf')
    plt.close(fig)
    lines.extend(['', '![Control learning curves](navigation_control_curves.png)', '',
                  'OptiQ uses a fixed teacher T per run. SAC and MFPO learn their entropy coefficient;',
                  'the table shows its value at the displayed evaluation. MEow uses fixed alpha=1.',
                  'DIPO has no entropy temperature. These parameters do not make the algorithms identical.', '',
                  'See [distribution metrics and curves](COMPARISON.md) and [OptiQ T=.25 analysis](NAVIGATION_OPTIQ_T025_100K_KO.md).'])
    (RESULTS/'NAVIGATION_CONTROL_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return True


def refresh():
    # Concurrent workers share summary plots; serialize their publication.
    with (RESULTS/'report.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        return _refresh_unlocked()


def _refresh_unlocked():
    jobs=read(RESULTS/'queue.json',{}).get('jobs',[])
    known={job['name'] for job in jobs}
    for path in sorted(RESULTS.glob('*/config.json')):
        name=path.parent.name
        if name in known or name.startswith('validation_'):
            continue
        cfg=read(path,{})
        if cfg.get('method') in ('optiq','direct_gmm','sac','dipo','meow','mfpo'):
            jobs.append(dict(name=name,method=cfg['method'],steps=cfg.get('steps',cfg.get('actor_updates'))))
    rows=[];groups={'fixed':[],'navigation':[]}
    for job in jobs:
        folder=RESULTS/job['name'];cfg=read(folder/'config.json',{})
        state=read(folder/'status.json',{})
        result=read(folder/'latest.json',{})
        sizes=read(folder/'model_sizes.json',{})
        navigation='--navigation' in job.get('args',[]) or cfg.get('Q')=='learned, not oracle'
        phase='navigation' if navigation else 'fixed'
        step=result.get('step',result.get('updates',0))
        training=result.get('training',{})
        row=dict(name=job['name'],phase=phase,method=job['method'],status=state.get('status','queued'),
                 role=job.get('comparison_role','baseline_or_prior_control'),
                 updates=state.get('step',state.get('updates',0)),evaluated_updates=step,budget=job['steps'],
                 temperature=cfg.get('temperature'),modes=result.get('mode_coverage'),mmd2=result.get('mmd2'),
                 sw2=result.get('sliced_wasserstein2'),precision=result.get('high_density_fraction'),
                 mean_return=result.get('mean_return'),parameters=sizes.get('total'),
                 training_seconds=training.get('train_seconds'),Q_evaluations=training.get('Q_evaluations'))
        rows.append(row)
        if result:groups[phase].append((row,history(folder)))
    stamp=datetime.now(timezone.utc).isoformat()
    lines=['# GMM40: live comparison','',f'Updated (UTC): {stamp}',
           '', 'Read the seed, budget and status of each run. Historical early-stop permissions do not authorize stopping new experiments early.',
           'For stopped_early runs, Updates / Last eval refer to the latest saved evaluated checkpoint; status.json also preserves the last observed live update count. These are not full-budget results.',
           'Earlier 10K runs are retained as historical stages; 100K continuations include their training history.',
           'Validation and invalid adapters are excluded from comparisons, but remain on disk.',
           '', 'Fixed-Q primary baselines use T=1. Explicit temperature controls target p_GMM^(1/T), but their evaluation reference remains the original T=1 GMM; they are excluded from the T=1 variant grid. Moving Q: OptiQ starts at T=.25 and is judged using Q spread, actual proposal ESS, and rollouts.',
           'Coverage alone is insufficient: inspect high-density fraction and MMD alongside it.',
           'Coverage counts 40 component centers, not mathematical density maxima. Numerical search found 36 strict local maxima; this auxiliary result does not replace the primary metric.',
           'Navigation terminal distributions are an empirical diagnostic, not a guaranteed Boltzmann target.',
           '']
    for phase in ('fixed','navigation'):
        lines.extend([f'## {phase}', '', '| Run | Status | Updates / budget | Last eval | Components /40 | MMD² ↓ | SW₂ ↓ | Precision ↑ | Network parameters |',
                      '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for row in rows:
            if row['phase']!=phase:continue
            def fmt(key):return '—' if row[key] is None else f'{row[key]:.5g}'
            name=row['name']
            lines.append(f"| [{name}]({name}/) | {row['status']} | {row['updates']:,} / {row['budget']:,} | {row['evaluated_updates']:,} | {fmt('modes')} | {fmt('mmd2')} | {fmt('sw2')} | {fmt('precision')} | {fmt('parameters')} |")
        lines.append('')
        if sample_grid(rows,phase):lines.append(f'![Primary sample comparison]({phase}_samples_comparison.png)\n')
        if groups[phase]:
            fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
            for row,series in groups[phase]:
                # Historical 10K-only stages remain in the table but not duplicate curve legends.
                if row['budget']<100000 and row.get('role')!='hparam_diagnostic':continue
                xs=[r.get('step',r.get('updates')) for r in series]
                for ax,key,title in zip(axes.ravel(),['mmd2','sliced_wasserstein2','mode_coverage','high_density_fraction'],['MMD² ↓','Sliced Wasserstein-2 ↓','Covered component centers /40 ↑','High-density fraction ↑']):
                    ax.plot(xs,[r[key] for r in series],marker='.',label=row['name'])
                    ax.set(xlabel='Actor updates',ylabel=title);ax.grid(alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels()
            if handles:fig.legend(handles,labels,loc='outside lower center',fontsize=8)
            fig.savefig(RESULTS/f'{phase}_curves.png',dpi=140);fig.savefig(RESULTS/f'{phase}_curves.pdf');plt.close(fig)
            lines.append(f'![Learning curves]({phase}_curves.png)\n')
    if navigation_control_report(groups['navigation']):
        lines.extend(['## Navigation control performance', '',
                      '[Return, terminal reward and actual temperature table](NAVIGATION_CONTROL_COMPARISON.md).',
                      '', '![Navigation control curves](navigation_control_curves.png)', ''])
    if optiq_variant_grid(rows):
        lines.extend(['## OptiQ variants','',
                      'Top: full stochastic policy. Bottom: conditional means only, a diagnostic that does not replace the actual policy distribution.',
                      'Each panel displays its evaluated update count; unfinished runs are not final results.',
                      '', '![OptiQ variants](optiq_variants_comparison.png)',''])
    lines.extend(['## Fixed-Q compute at the latest published evaluation','',
                  'Training seconds exclude evaluation, plots and checkpoint writing. They include compilation inside training calls.',
                  'Q evaluations count queried action points, not total neural-network operations. Equal update counts are not equal compute.',
                  '', '| Run | Evaluated updates | Training seconds | Q evaluations |', '|---|---:|---:|---:|'])
    for row in rows:
        if row['phase']!='fixed' or row['budget']!=100000 or row['training_seconds'] is None:continue
        query_count='—' if row['Q_evaluations'] is None else f"{int(row['Q_evaluations']):,}"
        lines.append(f"| {row['name']} | {row['evaluated_updates']:,} | {row['training_seconds']:.1f} | {query_count} |")
    lines.extend(['','## Latest artifacts',''])
    cards=[]
    for row in rows:
        folder=RESULTS/row['name']
        if not (folder/'latest.json').exists():continue
        prefix='update' if row['phase']=='navigation' else 'step'
        path=f"{row['name']}/evaluations/{prefix}_{row['evaluated_updates']:07d}"
        picture='terminal_and_paths.png' if row['phase']=='navigation' else 'samples.png'
        animation='environment_rollout.gif' if row['phase']=='navigation' else 'generation_rollout.gif'
        lines.append(f"- {row['name']}: [plot]({path}/{picture}), [rollout]({path}/{animation}), [metrics]({path}/metrics.json), [evolution]({row['name']}/training_evolution.gif)")
        cards.append(f'<article><h2>{html.escape(row["name"])}</h2><p>{row["status"]}; evaluated {row["evaluated_updates"]:,} updates</p><a href="{path}/{picture}"><img src="{path}/{picture}"></a><p><a href="{path}/{animation}">Rollout</a> · <a href="{path}/metrics.json">Metrics</a> · <a href="{row["name"]}/training_evolution.gif">Training evolution</a></p></article>')
    (RESULTS/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    if rows:
        with (RESULTS/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (RESULTS/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GMM40 experiments</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 16px}img{max-width:100%}article{border-top:1px solid #bbb;margin-top:35px}a{color:#165ba8}</style><h1>GMM40 experiments</h1><p>Updated '+html.escape(stamp)+'</p><p>Fixed Q: T=1. Moving Q: OptiQ T=.25 initially. Seed 0; 100K updates. Coverage must be read with precision and MMD. See <a href="COMPARISON.md">comparison</a>, <a href="comparison.csv">CSV</a>, and <a href="README_KO.md">methodology</a>.</p>'+''.join(cards))
    return rows


if __name__=='__main__':refresh()
