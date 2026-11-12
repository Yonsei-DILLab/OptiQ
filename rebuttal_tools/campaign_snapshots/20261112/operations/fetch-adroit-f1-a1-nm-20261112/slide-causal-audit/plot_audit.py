from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
series = json.loads((P/'training-series.json').read_text())
probe = json.loads((P/'probe-results.json').read_text())
colors = {64:'#2474AB', 256:'#D75532'}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 3, figsize=(14.4, 8), sharex=True)
for row, seed in enumerate([5,6]):
    a, b, c = axes[row]
    for m in [64,256]:
        logs = [x for x in series[f'{seed}-{m}'] if x['steps'] <= 1500160]
        x = np.array([v['steps'] for v in logs])/1e6
        a.plot(x,[v['train/current_q_values'] for v in logs],color=colors[m],lw=1.3,label=f'M={m}')
        rows = sorted([r for r in probe['rows'] if r['seed']==seed and r['trained_M']==m],key=lambda r:r['step'])
        x = np.array([r['step'] for r in rows])/1e6
        z = [r['mean'][str(m)] for r in rows]
        b.plot(x,[v['ess_mean'] for v in z],'-o',color=colors[m],ms=3,label=f'Trained M={m}; probe M={m}')
        c.plot(x,[v['q_std_over_temperature'] for v in z],'-o',color=colors[m],ms=3,label=f'M={m}: SD(Q)/T')
        c.plot(x,[v['log_density_std'] for v in z],'--',color=colors[m],lw=1.5,label=f'M={m}: SD(log q)')
    a.axhline(-100,color='#6B7280',ls='--',lw=1)
    a.axhline(0,color='#6B7280',ls=':',lw=.8)
    onset={5:611072,6:533248}[seed]/1e6
    a.axvline(onset,color='#555555',ls=':',lw=1)
    a.set_yscale('symlog',linthresh=20)
    a.set_ylabel(f'Seed {seed}\nReplay min-Q mean')
    a.set_title('Actual training Q (unsmoothed)')
    a.legend(fontsize=8,loc='lower left')
    b.set_yscale('log'); b.set_ylim(1,256); b.set_ylabel('Effective candidates (ESS)')
    b.set_title('Matched-checkpoint weight diagnostic')
    b.legend(fontsize=8)
    c.set_yscale('log'); c.set_ylabel('Across-candidate standard deviation')
    c.set_title('Q term versus density term')
    c.legend(fontsize=7.5)
    for ax in (a,b,c):
        ax.grid(alpha=.18); ax.set_xlim(0,1.52)
        if row==1:ax.set_xlabel('Environment steps (million)')
fig.suptitle('FetchSlide 20 cm | N=64, T=0.1, DACER H/d=-0.9 | paired training seeds',fontsize=15,y=.99)
fig.text(.5,.015,'Left: historical training logs; dotted vertical = sustained |Q256 - Q64| > 10.\nMiddle/right: 128 shared raw states, own frozen normalizers, 4 proposal draws; reconstructed diagnostics, not historical replay batches.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.955))
fig.savefig(P/'slide-q-importance-audit.png',dpi=180)
fig.savefig(P/'slide-q-importance-audit.pdf')

summary={'comparison':'Slide20cm; N64; M64 vs M256; T=.1; H/d=-.9; paired seeds5/6',
 'onset_definition':'3 consecutive exact common training-log steps',
 'onsets':{'5':{'q_difference_gt10':611072,'max_weight_gt0p5':1045248,'ess_lt4':1082112,'q_below_minus100':1266432},
           '6':{'q_difference_gt10':533248,'max_weight_gt0p5':725760,'ess_lt4':926464,'q_below_minus100':979712}},
 'probe_method':probe['method'], 'checkpoint_groups':len(probe['rows']),
 'limitations':['Baseline historical ESS was not logged; checkpoint diagnostics cannot reconstruct historical minibatches.',
                'Two training seeds on different hosts do not isolate the entire causal effect of M.',
                'Raw shared state bank uses equally weighted 1.5M evaluation rollouts from all four policies, not the historical replay distribution.',
                'M64 seed6 also becomes unstable later.'],
 'findings':['Configured scientific difference is proposals_per_policy_sample 1 versus4.',
             'Q trajectories separate before sustained severe weight concentration and large negative Q drift.',
             'At fixed checkpoint/state, increasing candidate count increases ESS; there is no automatic fourfold loss scale.',
             'At late M256 checkpoints, Q-only weights nearly reproduce the low ESS; density-only ESS is substantially higher.',
             'Both critic and actor feedback may be involved; M256 amplifying critic exploitation is a mechanism hypothesis, not proven by these two trajectories.']}
(P/'analysis-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(P/'slide-q-importance-audit.png')
