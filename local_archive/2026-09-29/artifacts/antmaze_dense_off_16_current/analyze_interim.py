"""Validate saved current-campaign evaluations and report routes per checkpoint."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('route_audit',REPO/'artifacts/antmaze_dense_multimodality_audit_20260924/analyze.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
SOURCE='a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3'

def load(path):return json.loads(path.read_text())

def main():
    snapshot=Path((ROOT/'latest-interim-path.txt').read_text().strip())
    rows={};raw={};states={}
    for host in ['vast-heechan-180','vast-heechan-199']:
        base=snapshot/host;status=load(base/'status.json');states[host]=status
        assert status['source_commit']==SOURCE and not status['failed']
        for run in sorted((base/'runs').iterdir()):
            cfg=load(run/'config.json');task=cfg['task'];method=cfg['method']
            assert cfg['source_commit']==SOURCE and cfg['reward_profile']=='dense'
            assert not cfg['noveld_enabled'] and cfg['noveld_coefficient']==0
            assert cfg['eval_starts']=='random' and cfg['interim_eval_episodes']==40
            progress=load(run/'progress.json')
            assert progress['rnd_updates']==0
            row=dict(task=task,method=method,host=host,progress=progress,
                     budget=cfg['steps'],completed=run.name in status['completed'],evaluations=[])
            for folder in sorted((run/'evaluations').glob('*/*-natural')):
                summary=load(folder/'summary.json');mode=summary['mode'];step=summary['step']
                d,m=audit.read_rollout(folder/'rollouts.npz',task)
                assert m['episodes']==summary['episodes']
                assert abs(m['successes']/m['episodes']-summary['success_rate'])<1e-10
                assert int(d['env_steps'])==step and str(d['mode'])==mode and not bool(d['fixed'])
                starts=d['initial_full_state'][:,:2]
                assert len(np.unique(starts,axis=0))==m['episodes']
                assert np.abs(starts).max()<=2.00001
                np.testing.assert_allclose(d['xy'][:,0],starts,atol=1e-6)
                distance=np.linalg.norm(d['xy'][:,:,None,:]-np.asarray(audit.GOALS[task])[None,None,:,:],axis=-1)
                m.update(step=step,mode=mode,mean_return=float(d['returns'].mean()),
                         closest_distance_per_goal=np.nanmin(distance,axis=(0,1)).tolist(),
                         unique_initial_positions=len(np.unique(starts,axis=0)))
                row['evaluations'].append(m);raw[(run.name,step,mode)]=d
            row['latest']={mode:max((e for e in row['evaluations'] if e['mode']==mode),key=lambda e:e['step'])
                           for mode in sorted({e['mode'] for e in row['evaluations']})}
            rows[run.name]=row
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    for mode in ['policy','native']:
        fig,axes=plt.subplots(1,4,figsize=(17.5,6.2))
        fig.subplots_adjust(left=.04,right=.99,bottom=.20,top=.78,wspace=.23)
        for ax,task in zip(axes,['v1','v2','v3','v4']):
            key=f'{task}-optiq-s0';m=rows[key]['latest'][mode];d=raw[(key,m['step'],mode)]
            audit.draw(ax,task,d,m)
            routes=len(m['successful_routes'])
            ax.set_title(f'{task.upper()} | {m["step"]/1e6:.3f}M steps\n'
                         f'Success {m["successes"]}/{m["episodes"]}; successful routes {routes}',fontsize=11)
        desc='Direct policy: random z + conditional sigma' if mode=='policy' else 'Mu-only: random z, no conditional sigma'
        fig.suptitle(f'CURRENT OptiQ | dense reward + NovelD OFF | {desc}\n'
                     'T=0.01, actor/critic 256x3, 256 environments; seed 0',fontsize=15,y=.985)
        handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Successful route'),
                 Line2D([0],[0],color=audit.NEG,lw=2,label='v1 successful lower route'),
                 Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False)
        fig.text(.5,.028,'All saved episodes shown, including failures. Random initial xy in [-2,2] x [-2,2]. No external DACER noise.\n'
                 'v1: final 100 episodes; v2-v4: latest completed interim evaluation, 40 episodes each. Different training budgets/progress.',ha='center',fontsize=9)
        fig.savefig(snapshot/f'optiq_latest_{mode}.png',dpi=175);plt.close(fig)
    colors=dict(optiq='#197ab3',dipo='#178970',sac='#b77924',mfpo='#9654aa')
    fig,axes=plt.subplots(1,4,figsize=(15,4.4),sharey=True)
    for ax,task in zip(axes,['v1','v2','v3','v4']):
        for method,color in colors.items():
            row=rows.get(f'{task}-{method}-s0')
            if not row:continue
            es=sorted((e for e in row['evaluations'] if e['mode']=='policy'),key=lambda e:e['step'])
            ax.plot([e['step']/1e6 for e in es],[100*e['successes']/e['episodes'] for e in es],'.-',color=color,label=method.upper())
        ax.set(title=task.upper(),xlabel='Environment steps (M)',ylim=(-3,103));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Direct-policy success (%)')
    fig.suptitle('Current dense / NovelD OFF campaign | random-start saved evaluations | seed 0')
    fig.text(.5,.02,'40 episodes per interim evaluation; 100 at completion. Evaluation variability is not training-seed uncertainty.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(snapshot/'success_curves.png',dpi=175);plt.close(fig)
    output=dict(collection=load(snapshot/'collection-verification.json'),source_commit=SOURCE,
                status=states,raw_trajectory_validation_passed=True,input_sha256=audit.INPUTS,runs=rows)
    (snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# 현재 dense + NovelD OFF 실험 중간 평가','',
           f'학습 source `{SOURCE}`. 단일 seed0. 기존 과거 모델의 재평가와 구분한다.',
           '저장된 random-start 평가만 분석. 새 평가/학습 작업을 실행하지 않았다. 모든 실패 궤적을 포함했다.',
           'OptiQ T=.01, actor/critic256x3, actorLR3e-4/criticLR5e-4, batch4096,256env,8updates/256transitions.',
           '정책 직접 샘플은 conditional sigma 포함, mu-only는 별도 보조그림. 평가 외부 DACER 잡음 없음.', '',
           '|작업|직접정책 평가 step|직접정책 성공|성공 경로|mu-only/native 평가 step|mu-only/native 성공|','|---|---:|---:|---|---:|---:|']
    for key,row in rows.items():
        p=row['latest'].get('policy');n=row['latest'].get('native')
        if not p:continue
        routes=', '.join(f'{k}:{v}' for k,v in p['successful_routes'].items()) or '없음'
        lines.append(f'|{key}|{p["step"]:,}|{p["successes"]}/{p["episodes"]}|{routes}|{n["step"]:,}|{n["successes"]}/{n["episodes"]}|')
    lines+=['','v1 최종은100회, 중간은40회이므로 표본 수가 다르다. 해당 정책·체크포인트의 관찰이며 드문 경로의 부재를 증명하지 않는다.',
            '원시 궤적의 finite/padding, 실제 goal 반경 도달, dense 거리합 reward, goal ID, initial state random 여부, 전송SHA256를 검증했다.',
            '학습진행률과 마지막 평가 step은 다를 수 있다. 경로 분류는 성공 episode의 미로 통로 횡단 위치에 따른다.',
            '그림: optiq_latest_policy.png / optiq_latest_native.png / success_curves.png. 전체 체크포인트별 지표: analysis.json.']
    (snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(snapshot),'runs':{k:dict(progress=r['progress']['step'],budget=r['budget'],completed=r['completed'],latest={m:dict(step=e['step'],success=e['successes'],episodes=e['episodes'],routes=e['routes'],closest=e['closest_distance_per_goal']) for m,e in r['latest'].items()}) for k,r in rows.items()}},indent=2))

if __name__=='__main__':main()
