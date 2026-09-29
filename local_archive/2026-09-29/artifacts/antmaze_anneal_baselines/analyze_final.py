"""Audit saved final evaluations and summarize annealing versus fixed temperature."""
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
load=lambda p:json.loads(p.read_text())
CAMPAIGNS={'anneal':'antmaze-dense-anneal-baselines-s0-20260924','T1':'antmaze-optiq-dense-off-T1-s0-20260924','previous':'antmaze-dense-off-16-current-s0-20260924'}
TASKS=['v1','v2','v3','v4']
LABELS=['T=0.01','T=1','10->1','10->0.25','10->0.5']

def main():
    snapshot=Path((ROOT/'latest-report-path.txt').read_text().strip())
    rows={};raw={}
    for cohort,campaign in CAMPAIGNS.items():
        for host in ['vast-heechan-180','vast-heechan-199']:
            for run in sorted((snapshot/host/campaign/'runs').glob('*')):
                if not (run/'result.json').is_file():continue
                cfg=load(run/'config.json');result=load(run/'result.json')
                if not result.get('completed'):continue
                task=cfg['task'];method=cfg['method']
                label=('10->'+format(cfg['temperature_schedule']['final_temperature'],'g') if cohort=='anneal' and method=='optiq'
                       else 'T=1' if cohort=='T1' else 'T=0.01' if method=='optiq' else method.upper())
                assert (label,task) not in rows,(label,task)
                assert cfg['reward_profile']=='dense' and not cfg['noveld_enabled'] and cfg['eval_starts']=='random'
                assert result['steps']==cfg['steps'] and result['rnd_updates']==0
                checkpoint=load(run/'checkpoint-verification.json')
                assert checkpoint['steps']==result['steps'] and checkpoint['readback_verified'] and checkpoint['dense_replay_verified']
                row=dict(label=label,task=task,method=method,run_id=run.name,host=host,source=cfg['source_commit'],steps=result['steps'],updates=result['updates'],modes={},history=[],checkpoint_verification=checkpoint)
                for mode in ['policy-natural','policy-fixed','native-natural']:
                    folder=run/'evaluations'/f"{result['steps']:010d}"/mode
                    d,m=audit.read_rollout(folder/'rollouts.npz',task,fixed=mode.endswith('fixed'))
                    summary=load(folder/'summary.json')
                    assert m['episodes']==100 and m['successes']==round(summary['success_rate']*100)
                    assert int(d['env_steps'])==result['steps'] and str(d['mode'])==mode.split('-')[0]
                    assert bool(d['fixed'])==mode.endswith('fixed')
                    starts=d['initial_full_state'][:,:2]
                    if mode.endswith('natural'):
                        assert len(np.unique(starts,axis=0))==100 and np.abs(starts).max()<=2.00001
                    np.testing.assert_allclose(d['xy'][:,0],starts,atol=1e-6)
                    m['mean_return']=float(d['returns'].mean())
                    m['mean_length']=float(d['lengths'].mean())
                    if task=='v1':
                        m['initial_y_by_route']={r:dict(count=sum(x==r for x in d['route_labels']),positive=int(sum(x==r and y>0 for x,y in zip(d['route_labels'],starts[:,1]))),negative=int(sum(x==r and y<0 for x,y in zip(d['route_labels'],starts[:,1])))) for r in set(d['route_labels'])}
                    row['modes'][mode]=m;raw[label,task,mode]=d
                for p in sorted((run/'evaluations').glob('*/policy-natural/summary.json')):
                    row['history'].append(load(p))
                rows[label,task]=row
    assert len(rows)==32,len(rows)
    assert all((label,task) in rows for label in LABELS+['SAC','DIPO','MFPO'] for task in TASKS)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    fig,axes=plt.subplots(3,4,figsize=(16,14))
    fig.subplots_adjust(top=.9,bottom=.10,left=.05,right=.99,wspace=.25,hspace=.38)
    for i,label in enumerate(LABELS[2:]):
        for j,task in enumerate(TASKS):
            m=rows[label,task]['modes']['policy-natural'];ax=axes[i,j]
            audit.draw(ax,task,raw[label,task,'policy-natural'],m)
            ax.set_title(f'{task.upper()} | T {label}\nSuccess {m["successes"]}/100 | observed routes {len(m["successful_routes"])}',fontsize=10)
    fig.suptitle('OptiQ annealing | final direct-policy trajectories\nDense reward, NovelD OFF | 100 random starts per policy | training seed 0',fontsize=16,y=.97)
    fig.legend(handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Success'),Line2D([0],[0],color=audit.NEG,lw=2,label='v1 lower-route success'),Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure; red endpoint')],loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.text(.5,.015,'Warmup excluded: linear decay over 1M transitions, then hold. Native budgets: v1/v2 3M, v3 4M, v4 5M.\nRandom z + conditional sigma; no external DACER noise. All failures included. Multiple starts are not same-state multimodality.',ha='center',fontsize=9)
    fig.savefig(snapshot/'annealing_final_trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4.5),sharey=True)
    colors=['#9b9ea3','#30343b','#1975ae','#dd8434','#31916c']
    for ax,task in zip(axes,TASKS):
        for label,color in zip(LABELS,colors):
            h=rows[label,task]['history']
            ax.plot([x['step']/1e6 for x in h],[100*x['success_rate'] for x in h],label=label,color=color,lw=1.6)
        ax.set(title=task.upper(),xlabel='Environment transitions (M)',ylim=(-3,103));ax.grid(alpha=.18)
    axes[0].set_ylabel('Direct-policy success (%)')
    fig.suptitle('OptiQ | fixed temperature versus annealing | dense + NovelD OFF',fontsize=15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,frameon=False)
    fig.text(.5,.01,'Training seed 0 only. 40 random-start episodes per interim evaluation; 100 at final.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.16,1,.92));fig.savefig(snapshot/'success_curves.png',dpi=170);plt.close(fig)
    output=dict(snapshot=load(snapshot/'collection-verification.json'),single_training_seed=0,raw_validation_passed=True,input_sha256=audit.INPUTS,runs={label:{task:rows[label,task] for task in TASKS} for label in LABELS+['SAC','DIPO','MFPO']})
    (snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# AntMaze annealing 최종 결과','', '모두 dense reward + NovelD OFF, training seed 0. 성공률은 최종 100회 random-start direct-policy 평가다. OptiQ는 random latent + conditional sigma를 포함하며 외부 DACER 잡음은 없다.','', '| 설정 | v1 | v2 | v3 | v4 |','|---|---:|---:|---:|---:|']
    for label in LABELS+['SAC','DIPO','MFPO']:
        lines.append('| '+label+' | '+' | '.join(f"{rows[label,t]['modes']['policy-natural']['successes']}%" for t in TASKS)+' |')
    lines+=['','Warmup 8,192 transition 이후 1M 동안 선형 감소하고 최종 온도를 유지했다. 전체 예산은 기존 v1/v2 3M, v3 4M, v4 5M (+warmup 및 vector step 반올림).','', '## Annealing 경로 비율','', '| 설정 | 환경 | random-start 성공 경로 | 동일 full-state 성공 경로 |','|---|---|---|---|']
    for label in LABELS[2:]:
        for task in TASKS:
            r=rows[label,task]
            lines.append('| '+label+' | '+task+' | '+json.dumps(r['modes']['policy-natural']['routes'])+' | '+json.dumps(r['modes']['policy-fixed']['routes'])+' |')
    lines+=['','성공 경로는 기하학적 통로 교차로 분류했다. 랜덤 시작점에 따른 경로 차이를 동일 상태에서의 정책 multimodality로 해석하지 않는다. 단일 학습 seed이며, 100회 평가에 관찰되지 않은 드문 경로의 부재를 증명하지 않는다.','', '원시 궤적의 유한성·padding, 목표 반경 진입, dense return 합, 초기 상태, SHA256를 검증했다. 설정·평가 NPZ·검증 sidecar를 로컬에 보관했고 전체 모델/replay checkpoint는 서버에 남아 있다.','', '학습 source 및 각 결과 출처는 analysis.json에 기록했다. W&B 199 서버의 offline 자료는 네트워크 복구 후 동기화 대기 중이다.']
    (snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(snapshot=str(snapshot),success_table={label:{t:rows[label,t]['modes']['policy-natural']['successes'] for t in TASKS} for label in LABELS+['SAC','DIPO','MFPO']},anneal_routes={label:{t:{mode:rows[label,t]['modes'][mode]['routes'] for mode in ['policy-natural','policy-fixed']} for t in TASKS} for label in LABELS[2:]}),indent=2))

if __name__=='__main__':main()
