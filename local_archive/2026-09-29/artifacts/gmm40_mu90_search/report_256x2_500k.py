"""Final 256x2 comparison using all mu-only samples, exact metric pairing."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
campaign = root / 'gmm40-mu90-256x2-500k-20260921'
target = json.loads((campaign / 'results/target/definition.json').read_text())
means, std = np.array(target['means']), np.array(target['std'])
theta = np.linspace(0, 2*np.pi, 100)
fig, axes = plt.subplots(2, 2, figsize=(11, 12))
curve, ca = plt.subplots(1, 2, figsize=(11, 4.3))
records = []
for row, (cap, tag) in enumerate([(-4., '4p0'), (-4.5, '4p5')]):
    old = root / f'gmm40-mu90-repro-256x2-capm{tag}-20260921/results'
    old = next(old.glob('*nm64-s0'))
    new = campaign / f'results/mu90_256x2_capm{tag}_s0_500k'
    config = json.loads((new / 'config.json').read_text())
    audit = json.loads((new / 'update_count_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['actor_updates'] == 500000
    assert config['n'] == config['m'] == 64 and config['latent_mode'] == 'random'
    assert config['width'] == 256 and config['depth'] == 2
    assert config['trg_log_std_max'] == cap and config['initial_log_std'] == -4.75
    record = dict(cap=cap, config=config, audit=audit,
                  wandb=json.loads((new/'wandb_status.json').read_text())['url'])
    for col, (run, step) in enumerate([(old, 100000), (new, 500000)]):
        p = run / f'evaluations/step_{step:07d}'
        x = np.load(p/'samples_mu_only.npy')
        m = json.loads((p/'metrics_mu_only.json').read_text())
        near = (((x[:,None,:]-means)/std[None,:,None])**2).sum(-1).min(-1) <= 9
        assert len(x) == 10000 and np.isclose(near.mean(), m['high_density_fraction'])
        ax = axes[row, col]
        for mask, color in [(near, '#2381b4'), (~near, '#e9884f')]:
            ax.scatter(*x[mask].T, s=1.4, alpha=.45, c=color, rasterized=True)
        for center, s in zip(means, std):
            ax.plot(center[0]+3*s*np.cos(theta), center[1]+3*s*np.sin(theta),
                    '--', lw=.65, c='#838b94')
        ax.scatter(*means.T, marker='+', s=26, c='#17242e')
        ax.set(xlim=(-42,42), ylim=(-42,42), aspect='equal', xlabel='Action x₁', ylabel='Action x₂')
        ax.grid(alpha=.1)
        ax.set_title(f'log σ ∈ [−5, {cap:g}] · {step//1000}k updates\n'
                     f'near {near.mean():.2%} · coverage {m["mode_coverage"]}/40', fontsize=12)
        record[f'mu_{step}'] = m
    record['full_policy_500k'] = json.loads((new/'evaluations/step_0500000/metrics.json').read_text())
    validation = root/f'independent-256x2-500k-capm{tag}/validation.json'
    if validation.exists():
        record['independent_validation'] = json.loads(validation.read_text())
    points = []
    for run in (old, new):
        for p in run.glob('evaluations/*/metrics_mu_only.json'):
            step = int(p.parent.name.split('_')[1])
            m = json.loads(p.read_text())
            points.append((step, m['high_density_fraction']*100, m['mode_coverage']))
    points = np.array(sorted(set(points)))
    for ax, metric in zip(ca, [1,2]):
        ax.plot(points[:,0]/1000, points[:,metric], marker='o', ms=3, label=f'log σ cap {cap:g}')
    records.append(record)
fig.suptitle('256×2 · N=M64 · random latent · μ only', fontsize=18, y=.992)
fig.text(.5,.952,'Seed 0 · mean-head scale 16 · initial log σ −4.75 · teacher floor .05\n'
         'Batch 256 · Adam 3e−4 · T=1 · β=1 · exact continuation from 100k', ha='center', va='top', fontsize=10)
fig.subplots_adjust(top=.87,bottom=.08,hspace=.30,wspace=.20,left=.07,right=.98)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#2381b4',label='Within any GT 3σ boundary'),
                    Line2D([],[],marker='o',ls='',color='#e9884f',label='Outside all GT 3σ boundaries')],
           loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
for extension in ['png','pdf']:
    fig.savefig(root/f'reproduction_256x2_500k.{extension}',dpi=170)
ca[0].axhline(90,ls='--',color='gray',lw=1)
ca[0].set(ylabel='μ near (%)',ylim=(0,100))
ca[1].set(ylabel='Covered GT components',ylim=(0,41))
for ax in ca:
    ax.set(xlabel='Actor updates (thousands)')
    ax.grid(alpha=.2)
    ax.legend()
curve.suptitle('256×2 convergence · training seed 0 · random latent, N=M64')
curve.tight_layout()
curve.savefig(root/'reproduction_256x2_500k_curves.png',dpi=170)
(root/'reproduction_256x2_500k_summary.json').write_text(json.dumps(records,indent=2)+'\n')
print([(r['cap'],r['mu_500000']['high_density_fraction'],r['mu_500000']['mode_coverage']) for r in records])
