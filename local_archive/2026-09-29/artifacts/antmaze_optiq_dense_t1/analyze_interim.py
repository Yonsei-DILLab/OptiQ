"""Compare immutable saved T=1 and T=.01 policy rollouts at equal budgets."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('route_audit',REPO/'artifacts/antmaze_dense_multimodality_audit_20260924/analyze.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
CAMPAIGNS={'T1':'antmaze-optiq-dense-off-T1-s0-20260924','T001':'antmaze-dense-off-16-current-s0-20260924'}
SOURCES={'T1':'7af193833466f5bc41853948d6f417fc6aaa6803','T001':'a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3'}
load=lambda p:json.loads(p.read_text())


def main():
    snapshot=Path((ROOT/'latest-interim-path.txt').read_text().strip())
    rows={};raw={};configs={}
    for label,campaign in CAMPAIGNS.items():
        for host in ('vast-heechan-180','vast-heechan-199'):
            base=snapshot/host/campaign
            for run in sorted((base/'runs').glob('*-optiq-s0')):
                cfg=load(run/'config.json');task=cfg['task'];configs[label,task]=cfg
                assert cfg['source_commit']==SOURCES[label]
                assert cfg['reward_profile']=='dense' and cfg['noveld_enabled'] is False
                assert cfg['eval_starts']=='random' and cfg['interim_eval_episodes']==40
                progress=load(run/'progress.json');assert progress['rnd_updates']==0
                row=dict(task=task,host=host,budget=cfg['steps'],progress=progress,
                    completed=(run/'result.json').exists(),evaluations=[],wandb=load(run/'wandb.json'))
                for folder in sorted((run/'evaluations').glob('*/*-natural')):
                    if not (folder/'summary.json').exists():continue
                    summary=load(folder/'summary.json');mode=summary['mode'];step=summary['step']
                    d,m=audit.read_rollout(folder/'rollouts.npz',task)
                    assert m['episodes']==summary['episodes'] and abs(m['successes']/m['episodes']-summary['success_rate'])<1e-10
                    assert int(d['env_steps'])==step and str(d['mode'])==mode and not bool(d['fixed'])
                    starts=d['initial_full_state'][:,:2]
                    assert len(np.unique(starts,axis=0))==m['episodes'] and np.abs(starts).max()<=2.00001
                    np.testing.assert_allclose(d['xy'][:,0],starts,atol=1e-6)
                    distance=np.linalg.norm(d['xy'][:,:,None,:]-np.asarray(audit.GOALS[task])[None,None,:,:],axis=-1)
                    points=d['xy'][np.isfinite(d['xy']).all(-1)]
                    m.update(step=step,mode=mode,mean_return=float(d['returns'].mean()),
                        closest_distance_per_goal=np.nanmin(distance,axis=(0,1)).tolist(),
                        xy_min=points.min(0).tolist(),xy_max=points.max(0).tolist(),
                        evaluation_bins_05m=len(np.unique(np.floor(points/.5).astype(int),axis=0)),
                        left_entries=int(np.any(d['xy'][:,:,0]<-4,axis=1).sum()),
                        right_entries=int(np.any(d['xy'][:,:,0]>4,axis=1).sum()),
                        upper_entries=int(np.any(d['xy'][:,:,1]>4,axis=1).sum()),
                        lower_entries=int(np.any(d['xy'][:,:,1]<-4,axis=1).sum()))
                    row['evaluations'].append(m);raw[label,task,step,mode]=d
                row['latest']={mode:max([e for e in row['evaluations'] if e['mode']==mode],key=lambda e:e['step'])
                    for mode in {e['mode'] for e in row['evaluations']}}
                rows[label,task]=row
    pairs={}
    for task in ('v1','v2','v3','v4'):
        old=configs['T001',task]['native']['alg'];new=configs['T1',task]['native']['alg']
        old=json.loads(json.dumps(old));old['actor']['temperature']=1.
        assert old==new,task
        assert configs['T001',task]['native']['dacer']==configs['T1',task]['native']['dacer']
        cur=rows['T1',task]['latest']['policy'];step=cur['step']
        baseline=next(e for e in rows['T001',task]['evaluations'] if e['step']==step and e['mode']=='policy')
        assert cur['episodes']==baseline['episodes']
        pairs[task]=dict(step=step,T1=cur,T001=baseline)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,4,figsize=(17,11))
    fig.subplots_adjust(top=.88,bottom=.13,left=.045,right=.99,wspace=.22,hspace=.36)
    for col,task in enumerate(('v1','v2','v3','v4')):
        for row,label in enumerate(('T001','T1')):
            m=pairs[task][label];step=m['step'];ax=axes[row,col]
            audit.draw(ax,task,raw[label,task,step,'policy'],m)
            ax.set_title(f'{task.upper()} | T={"0.01" if label=="T001" else "1"} | {step/1e6:.2f}M\n'
                f'Success {m["successes"]}/{m["episodes"]}; routes {len(m["successful_routes"])}',fontsize=11)
    fig.suptitle('OptiQ temperature comparison | same environment-step budget per column\nDense reward, NovelD OFF | random-start direct policy | training seed 0',fontsize=16,y=.98)
    fig.legend(handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Successful route'),
        Line2D([0],[0],color=audit.NEG,lw=2,label='v1 successful lower route'),
        Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure (red endpoint)')],loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
    fig.text(.5,.02,'40 episodes per panel; all failures included. Random z + conditional sigma; no external DACER noise.\n'
        'Different maze progress; compare rows within each column. One training seed; rare routes may be missed.',ha='center',fontsize=10)
    fig.savefig(snapshot/'matched_trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    for ax,task in zip(axes,('v1','v2','v3','v4')):
        for label,color in [('T001','#8b949e'),('T1','#1679aa')]:
            es=sorted([e for e in rows[label,task]['evaluations'] if e['mode']=='policy'],key=lambda e:e['step'])
            ax.plot([e['step']/1e6 for e in es],[100*e['successes']/e['episodes'] for e in es],'.-',color=color,label='T=0.01' if label=='T001' else 'T=1')
        ax.set(title=task.upper(),xlabel='Environment transitions (M)',ylim=(-3,103));ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Direct-policy success (%)')
    fig.suptitle('Dense + NovelD OFF | random-start direct-policy learning curves | seed 0',fontsize=14)
    fig.text(.5,.02,'T=1 is still training. 40 episodes per interim evaluation; 100 at final. T=0.01 continued beyond the matched budgets.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(snapshot/'success_curves.png',dpi=170);plt.close(fig)
    output=dict(snapshot=load(snapshot/'collection-verification.json'),sources=SOURCES,
        same_learning_configuration_except_temperature=True,raw_validation_passed=True,
        input_sha256=audit.INPUTS,comparison=pairs,
        runs={label:{task:rows[label,task] for task in ('v1','v2','v3','v4')} for label in CAMPAIGNS})
    (snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(snapshot),comparison=pairs,
        progress={t:rows['T1',t]['progress'] for t in pairs},native={t:rows['T1',t]['latest'].get('native') for t in pairs}),indent=2))


if __name__=='__main__':main()
