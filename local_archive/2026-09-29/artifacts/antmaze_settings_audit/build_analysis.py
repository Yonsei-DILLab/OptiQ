"""Summarize saved histories and checkpoint diagnostics; generate static figure."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
history = {}
for p in root.glob('history-*.json'):
    history.update(json.loads(p.read_text()))
forward = {p.name.removesuffix('-forward.json'):json.loads(p.read_text())
           for p in root.glob('*-forward.json')}
noveld = {}
for p in root.glob('noveld-*.json'):
    noveld.update(json.loads(p.read_text()))
last = {}
for name,run in history.items():
    rows = [r for r in run['history'] if 'updates' in r and 'step' in r]
    maximum = max(r['step'] for r in rows)
    selected = [r for r in rows if maximum-100000 < r['step'] <= maximum]
    keys = ['noveld_mean','train/current_q_values','train/critic_loss','exploration/noise_std',
            'exploration/entropy_proxy','temperature','entropy','next_logps','q_mean_1','q_std_1']
    values = {}
    for k in keys:
        v = [r[k] for r in selected if k in r]
        if v: values[k] = dict(n=len(v),mean=float(np.mean(v)),minimum=min(v),maximum=max(v))
    last[name] = dict(from_exclusive=maximum-100000,through_inclusive=maximum,
                     samples=len(selected),values=values)

result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    source_commit='0ebd8d26c711d79723f457343b36eb8788bd87b8',
    upstream_commit='7edd06c4799abbab0f8fa534c21deb56253b018e',
    mfpo_commit='d8b3977d29d4ef2d315e871337e5826f2eb79eb2',
    scope='Old native-budget sparse + NovelD 0.01 campaign; seed 0 only.',
    method='Read existing histories; CPU-only final-checkpoint forward diagnostics; no new training.',
    forward=forward,noveld_forward=noveld,last_100k_logged_batches=last,
    reference=dict(uniform_action_coordinate_std=float(np.sqrt(1/3)),
                   ddiffpg_mixed_external_noise_rms=float(np.sqrt(np.mean(np.linspace(.05,.6,256)**2)))),
    limitations=['Each policy uses 128 states sampled from its own final replay; not identical states across methods.',
                 'Q-only ESS for MFPO is a common diagnostic, not its actual kernel/density-corrected actor weights.',
                 'Forward temperature reweighting is not a training ablation or an environment performance result.',
                 'All policies are single training seed 0; final replay retains the most recent 1M transitions.'])
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','figure.dpi':140})
fig,axes = plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
op = [forward[f'v{i}-optiq-s0'] for i in [1,2]]
mf = [forward[f'v{i}-mfpo-s0'] for i in [1,2]]
x=np.arange(2); w=.32
ax=axes[0,0]
ax.bar(x-w/2,[r['q']['logit_std_mean'] for r in op],w,label='Q / T',color='#2864a5')
ax.bar(x+w/2,[r['density_logit_std'] for r in op],w,label='- log proposal density',color='#d57d28')
ax.set_yscale('log');ax.set_ylim(.001,2.5);ax.set_xticks(x,['v1','v2'])
ax.set_ylabel('Across-candidate standard deviation')
ax.set_title('OptiQ: Q term is much smaller at T = 0.25')
ax.legend(loc='upper right',fontsize=10)
for i,r in enumerate(op):ax.text(i,.02,f"density / Q: {r['density_to_q_logit_std']:.0f}x",ha='center')

ax=axes[0,1]
tv=[r['same_checkpoint_temperature_reweighting'][0]['weight_tv_from_density_only']*100 for r in op]
ax.bar(x,tv,.5,color='#2864a5');ax.set_xticks(x,['v1','v2']);ax.set_ylim(0,1)
ax.set_ylabel('Total variation distance (%)')
ax.set_title('OptiQ: removing Q barely changes teacher weights')
for i,v in enumerate(tv):ax.text(i,v+.03,f'{v:.3f}%',ha='center')
ax.text(.5,.9,'Same candidates and density; only Q term removed',ha='center',transform=ax.transAxes,fontsize=10)

ax=axes[1,0]
for dx,items,label,color in [(-w/2,op,'OptiQ full policy','#2864a5'),(w/2,mf,'MFPO direct policy','#2a8b74')]:
    values=[r['policy_spread']['rms_coordinate_std'] for r in items]
    ax.bar(x+dx,values,w,label=label,color=color)
    for i,v in enumerate(values):ax.text(i+dx,v+.012,f'{v:.3f}',ha='center',fontsize=10)
ax.axhline(np.sqrt(1/3),color='#777777',ls='--',label='Uniform [-1, 1] reference')
ax.set_ylim(0,.8);ax.set_xticks(x,['v1','v2']);ax.set_ylabel('RMS action-coordinate standard deviation')
ax.set_title('OptiQ actions are already widely dispersed')
ax.legend(loc='upper right',fontsize=9)

ax=axes[1,1]
values=[r['same_checkpoint_temperature_reweighting'][0]['q_only_ess'] for r in op]+[r['q_only_ess'] for r in mf]
labels=['OptiQ v1','OptiQ v2','MFPO v1','MFPO v2']
ax.bar(np.arange(4),values,color=['#2864a5']*2+['#2a8b74']*2,width=.6)
ax.set_xticks(np.arange(4),labels);ax.set_ylim(0,82);ax.set_ylabel('Effective candidates out of 64')
ax.set_title('Opposite Q-only weighting regimes')
for i,v in enumerate(values):ax.text(i,v+1.5,f'{v:.2f}',ha='center')
ax.text(.5,.94,'Diagnostic softmax(Q / temperature), both methods',ha='center',transform=ax.transAxes,fontsize=9)
fig.suptitle('Sparse + NovelD 0.01: scale audit of preserved checkpoints\n3,008,256 interactions · training seed 0 · 128 replay states × 64 candidates',fontsize=15)
fig.supxlabel('Forward diagnostics only. MFPO actual actor weights also contain kernel and density terms.\nAction spread is not spatial coverage or successful locomotion.',fontsize=10)
fig.savefig(root/'scale_diagnostics.png',dpi=170)
plt.close(fig)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*') if p.is_file() and p.name not in ['manifest.json']}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'runs':len(forward),'noveld_runs':len(noveld),'figure':str(root/'scale_diagnostics.png')}))
