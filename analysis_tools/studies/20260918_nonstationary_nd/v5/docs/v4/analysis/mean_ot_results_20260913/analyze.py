from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/optiq-experiments/ant_v4_mean_ot_T025_20260912T235059Z')
sys.path.insert(0, str(CAMPAIGN))
from campaign_utils import read, save, now, completed_result
from run_queue import verify_inputs

manifest = read(CAMPAIGN / 'manifest.json')
verify_inputs(manifest)
state = read(CAMPAIGN / 'state.json')
assert state['phase'] == 'completed'
groups = {'baseline': manifest['baseline_records'], 'meanOT': list(state['jobs'].values())}
baseline_manifest = read(Path(manifest['baseline_campaign']) / 'manifest.json')
for seed in range(4):
    for label, m in [('baseline', baseline_manifest), ('meanOT', manifest)]:
        job = next(j for j in m['jobs'] if j['seed'] == seed and j['temperature'] == .25)
        result = completed_result(job, m)
        assert result['scores'] == groups[label][seed]['scores']

input_hashes, curves, logs, summary, metric_rows = {}, {}, {}, [], []
def digest(path):
    path = Path(path)
    input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

def sustained(x, y, target):
    match = np.where(np.convolve((y >= target).astype(int), np.ones(3, dtype=int), mode='valid') == 3)[0]
    return int(x[match[0]]) if len(match) else None

def rolling(y, n=5):
    return np.convolve(y, np.ones(n)/n, mode='valid')

keys = ['actor_latent_mean_variance_fraction', 'actor_std_mean', 'source_ess_absolute',
        'actor_loss', 'critic_loss', 'ot_row_marginal_error', 'ot_col_marginal_error',
        'current_q_values', 'backup_entropy_term', 'backup_discounted_entropy_term']
for label, records in groups.items():
    for seed, record in enumerate(records):
        directory = Path(record['output'])
        cfg = read(directory / 'config.json')
        assert cfg['alg']['behavior_uniform_probability'] == 0
        assert cfg['alg']['actor']['temperature'] == .25
        assert 'temperature_schedule' not in cfg['alg']['actor']
        digest(directory / 'config.json')
        for mode in ['zero_z', 'stochastic_z']:
            file = Path(record['evaluation_files'][mode]['path'])
            digest(file)
            with np.load(file) as data:
                x, rewards = data['timesteps'].copy(), data['results'].copy()
                env_seeds, policy_seeds = data['env_seeds'].copy(), data['policy_seeds'].copy()
            y = rewards.mean(axis=1)
            curves[label, seed, mode] = {'steps': x, 'mean': y, 'rewards': rewards,
                                         'env_seeds': env_seeds, 'policy_seeds': policy_seeds}
            tail = y[x >= 900000]
            smoothed = rolling(y)
            smoothed_x = x[4:]
            after300k = smoothed[smoothed_x >= 300000]
            drawdown = np.maximum.accumulate(after300k) - after300k
            summary.append({'variant': label, 'seed': seed, 'mode': mode,
                'final_return': float(y[-1]), 'tail_900k_1m': float(tail.mean()),
                'tail_min_5k_eval': float(tail.min()),
                'time_average_0_1m': float(np.trapz(y, x)/(x[-1]-x[0])),
                'first_three_consecutive_3k': sustained(x, y, 3000),
                'first_three_consecutive_4k': sustained(x, y, 4000),
                'first_three_consecutive_5k': sustained(x, y, 5000),
                'max_25k_rolling_drawdown_after_300k': float(drawdown.max()),
                'max_single_eval_drop_after_300k': float(-np.diff(y[x >= 300000]).min()),
                **{f'return_{int(step/1000)}k': float(y[np.where(x==step)[0][0]]) for step in (50000,100000,200000,300000,500000)}})
        file = next(directory.rglob('progress.csv'))
        digest(file)
        with file.open() as f:
            rows = list(csv.DictReader(f))
        logs[label, seed] = {}
        for k in keys:
            a = np.asarray([(float(r['train/n_updates'])+5000, float(r['train/'+k])) for r in rows
                            if r.get('train/n_updates') and r.get('train/'+k)], dtype=float)
            logs[label, seed][k] = a
            for start, end in [(5001,50000), (50000,200000), (200000,500000), (900000,1000000)]:
                v = a[(a[:,0] >= start) & (a[:,0] <= end),1]
                metric_rows.append({'variant': label, 'seed': seed, 'metric': k,
                                    'start': start, 'end': end, 'mean': float(v.mean()), 'count':len(v)})
        assert all(np.count_nonzero(logs[label, seed][k][:,1])==0
                   for k in ['backup_entropy_term','backup_discounted_entropy_term'])

for seed in range(4):
    for mode in ['zero_z','stochastic_z']:
        before, after = curves['baseline',seed,mode], curves['meanOT',seed,mode]
        for key in ('steps','env_seeds','policy_seeds'):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(before['rewards'][0], after['rewards'][0])

def write_csv(name, rows):
    with (ROOT/name).open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
write_csv('returns_by_seed.csv', summary)
write_csv('training_metrics.csv', metric_rows)
aggregates = []
for label in groups:
    for mode in ['zero_z','stochastic_z']:
        rows = [r for r in summary if r['variant']==label and r['mode']==mode]
        aggregates.append({'variant':label,'mode':mode,
            **{key:float(np.mean([r[key] for r in rows])) for key in ['tail_900k_1m','final_return','time_average_0_1m']},
            'tail_seed_sd':float(np.std([r['tail_900k_1m'] for r in rows],ddof=1)),
            'three_eval_3k_steps':[r['first_three_consecutive_3k'] for r in rows],
            'three_eval_4k_steps':[r['first_three_consecutive_4k'] for r in rows]})
save(ROOT/'summary.json',{'utc':now(),'seeds':summary,'aggregates':aggregates})
save(ROOT/'verification.json',{'passed':True,'utc':now(),'completed_1m_runs':8,
    'matched_eval_rng_across_variants':True,'matching_initial_evaluation':True,
    'zero_backup_entropy_all_logged_updates':True,'input_hashes':input_hashes})

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
colors={'baseline':'#63748c','meanOT':'#e47619'}
fig, axes = plt.subplots(2,4,figsize=(16,7.6),sharex=True,sharey=True,constrained_layout=True)
for row, mode in enumerate(['zero_z','stochastic_z']):
    for seed in range(4):
        ax=axes[row,seed]
        for label in groups:
            data=curves[label,seed,mode];x,y=data['steps']/1e6,data['mean']
            ax.plot(x,y,color=colors[label],alpha=.23,lw=.65)
            ax.plot(x[4:],rolling(y),color=colors[label],lw=1.8,label=label)
        ax.set_title(f'Seed {seed}');ax.set_ylim(-800,6400);ax.grid(alpha=.18)
        if row==1:ax.set_xlabel('Environment steps (millions)')
        if seed==0:ax.set_ylabel(('z = 0' if row==0 else 'sampled z')+'; epsilon = 0\nReturn')
        tail=[next(r['tail_900k_1m'] for r in summary if r['variant']==l and r['seed']==seed and r['mode']==mode) for l in groups]
        ax.text(.97,.04,f'900K-1M: {tail[0]:.0f} -> {tail[1]:.0f}',ha='right',va='bottom',transform=ax.transAxes,fontsize=9)
axes[0,0].legend(loc='upper left')
fig.suptitle('Ant-v4 | T = 0.25 fixed | no uniform exploration or annealing\nRaw evaluations every 5K (faint); trailing 5-evaluation mean (bold)',fontsize=13)
fig.savefig(ROOT/'returns_all_seeds.png',dpi=170);fig.savefig(ROOT/'returns_all_seeds.pdf');plt.close(fig)

fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for idx,(key,title) in enumerate([
    ('actor_latent_mean_variance_fraction','Fraction of pre-tanh variance explained by z-dependent means'),
    ('actor_std_mean','Conditional Gaussian sigma'),
    ('source_ess_absolute','Teacher effective sample size (out of 64)'),
    ('actor_loss','Conditional Gaussian NLL (targets differ)')]):
    ax=axes.flat[idx]
    for label in groups:
        for seed in range(4):
            arr=logs[label,seed][key]
            ax.plot(arr[:,0]/1e6,arr[:,1]*(100 if idx==0 else 1),color=colors[label],alpha=.25,lw=.7,label=label if seed==0 else None)
    if idx==0:ax.set_yscale('log');ax.set_ylabel('Percent (log scale)')
    ax.set_title(title,fontsize=10);ax.set_xlabel('Environment steps (millions)');ax.grid(alpha=.18)
axes[0,0].legend()
fig.savefig(ROOT/'training_diagnostics.png',dpi=170);plt.close(fig)
print(json.dumps(aggregates,indent=2))
print('Seed zero-z details:')
for row in summary:
    if row['mode']=='zero_z':print(json.dumps(row))
