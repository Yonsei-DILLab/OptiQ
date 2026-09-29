"""Report all dense DIPO runs without treating incomplete or initial evals as final."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('route_audit',REPO/'artifacts/antmaze_dense_multimodality_audit_20260924/analyze.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
GROUPS={
 'antmaze-multimodal-100k-20260921':('legacy-off','이전 자체 환경 OFF'),
 'antmaze-v1-dipo-noveld001-1m-s0-20260922':('legacy-on','이전 자체 환경 ON'),
 'antmaze-v234-noveld01-1m-s0-20260922':('legacy-on','이전 자체 환경 ON'),
 'antmaze-upstream-dense-nativebudget-64env-s0-20260923':('upstream-on','공식 환경 ON, 64env'),
 'antmaze-dense-noveld-off-probe-s0-20260923':('off-probe','공식 환경 OFF, 10k-update probe'),
 'antmaze-dense-noveld-off-main-s0-20260923':('off-previous','공식 환경 OFF, 이전 중단'),
 'antmaze-dense-off-16-current-s0-20260924':('off-current','공식 환경 OFF, 현재'),
}

def load(path):return json.loads(path.read_text())

def main():
 snapshot=Path((ROOT/'latest.txt').read_text().strip());runs={};arrays={};index={};warnings=[]
 for host in ['vast-heechan-180','vast-heechan-199']:
  folder=snapshot/host
  for rec in load(folder/'inventory.json')['runs']:
   run=folder/rec['run'];campaign=run.parent.parent.name
   group,label=GROUPS[campaign];cfg=load(run/'config.json');task=cfg['task']
   progress=load(run/'progress.json');key=f'{group}/{task}'
   assert key not in runs and cfg['method']=='dipo' and cfg['seed']==0
   coefficient=cfg.get('noveld_coefficient',cfg.get('intrinsic',{}).get('coefficient',0))
   status=load(run.parent.parent/'status.json')
   completed=bool((load(run/'result.json') if (run/'result.json').exists() else {}).get('completed'))
   row=dict(group=group,label=label,task=task,host=host,source=cfg['source_commit'],
            coefficient=coefficient,num_envs=cfg.get('num_envs'),batch=cfg['batch_size'],
            progress_step=progress.get('step',progress.get('env_steps')),
            updates=progress['updates'],rnd_updates=progress.get('rnd_updates'),
            training_successes=progress.get('successes',progress.get('training_successes')),
            completed=completed,state='running' if group=='off-current' else 'completed probe' if completed else 'stopped',
            config=str((run/'config.json').relative_to(REPO)),evaluations=[],history={})
   if group=='off-current':
    assert run.name in {r['id'] for r in status['running']} or run.name in status['completed']
   for path in sorted((run/'evaluations').glob('*/*/rollouts.npz')):
    summary=load(path.with_name('summary.json'));mode=summary['mode'];step=summary['step']
    d,m=audit.read_rollout(path,task,fixed=summary['fixed'])
    assert m['episodes']==summary['episodes'] and np.isclose(m['successes']/m['episodes'],summary['success_rate'])
    dist=np.linalg.norm(d['xy'][:,:,None,:]-np.asarray(audit.GOALS[task])[None,None,:,:],axis=-1)
    m.update(step=step,mode=mode,fixed=summary['fixed'],mean_return=summary['mean_return'],
             initial_positions_distinct=len(np.unique(d['initial_full_state'][:,:2],axis=0)),
             closest_goal_m=float(np.nanmin(dist)),closest_goal_per_goal=np.nanmin(dist,axis=(0,1)).tolist(),
             raw_verified=True)
    row['evaluations'].append(m);arrays[(key,step,mode,summary['fixed'])]=d
   for mode in ['policy','native']:
    path=run/f'history-{mode}-natural.json'
    if path.exists():
     history=load(path)
     for e in history:
      assert len(e['routes'])==e['episodes']
      assert np.isclose(sum(r!='failure' for r in e['routes'])/e['episodes'],e['success_rate'])
     row['history'][mode]=history
   row['latest']={mode:max((e for e in row['evaluations'] if e['mode']==mode and not e['fixed']),key=lambda e:e['step'])
                  for mode in {e['mode'] for e in row['evaluations'] if not e['fixed']}}
   row['best_policy']=max((e for e in row['evaluations'] if e['mode']=='policy' and not e['fixed']),
                          key=lambda e:(e['successes']/e['episodes'],e['step']),default=None)
   row['latest_post_training_history']={mode:next((e for e in reversed(hist) if e['step']>0),None)
                                        for mode,hist in row['history'].items()}
   runs[key]=row;index[key]=run
  for campaign in folder.glob('*'):
   if campaign.is_dir() and (campaign/'failure.json').exists():
    warnings.append(dict(host=host,campaign=campaign.name,failure=load(campaign/'failure.json')))
 assert len(runs)==19,len(runs)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
 fig,axes=plt.subplots(2,4,figsize=(17,10.8))
 fig.subplots_adjust(left=.045,right=.99,bottom=.15,top=.86,hspace=.43,wspace=.22)
 for row,group in enumerate(['upstream-on','off-current']):
  for ax,task in zip(axes[row],['v1','v2','v3','v4']):
   key=f'{group}/{task}';r=runs[key]
   mode='native' if group=='upstream-on' and task=='v3' else 'policy'
   m=r['latest'][mode];d=arrays[(key,m['step'],mode,False)]
   audit.draw(ax,task,d,m)
   reset='random starts' if m['initial_positions_distinct']>1 else 'fixed start'
   ax.set_title(f'{task.upper()} | {m["step"]/1000:,.0f}k | {mode} | {reset}\n'
                f'Success {m["successes"]}/{m["episodes"]}; routes {len(m["successful_routes"])}',fontsize=10)
 fig.suptitle('DIPO dense-reward experiments | NovelD ON and OFF\nLatest saved trajectories; training seed 0; every failure retained',fontsize=15,y=.985)
 fig.text(.5,.90,'NovelD 0.01 ON | earlier official environment run | 64 environments | stopped',ha='center',fontsize=12,weight='bold')
 fig.text(.5,.49,'NovelD OFF | current official environment run | 256 environments | in progress',ha='center',fontsize=12,weight='bold')
 handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Successful route'),
          Line2D([0],[0],color=audit.NEG,lw=2,label='v1 lower route'),
          Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
 fig.text(.5,.015,'ON v3: 500k NATIVE evaluation only; latest saved direct-policy evaluation was 250k (0/10).\n'
          'Different budgets, environment counts, initial-state distributions and numerical safeguards: not a controlled NovelD ablation.',ha='center',fontsize=9)
 fig.savefig(snapshot/'dipo_dense_on_off_trajectories.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(16,8),sharex='col')
 styles={'upstream-on':('#c77523','NovelD .01 ON / 64env'),
         'off-previous':('#87929c','Previous OFF / 256env'),
         'off-current':('#197eb2','Current OFF / 256env'),
         'off-probe':('#746198','OFF 328k probe')}
 for col,task in enumerate(['v1','v2','v3','v4']):
  for group,(color,label) in styles.items():
   row=runs.get(f'{group}/{task}')
   if not row:continue
   es=sorted((e for e in row['evaluations'] if e['mode']=='policy' and not e['fixed']),key=lambda e:e['step'])
   axes[0,col].plot([e['step']/1e6 for e in es],[100*e['successes']/e['episodes'] for e in es],'.-',color=color,label=label)
   axes[1,col].plot([e['step']/1e6 for e in es],[e['mean_return'] for e in es],'.-',color=color)
  if task=='v3':
   e=runs['upstream-on/v3']['latest']['native']
   axes[0,col].scatter([e['step']/1e6],[100*e['successes']/e['episodes']],marker='x',s=70,color='#c77523',label='ON native-only 500k')
  axes[0,col].set(title=task.upper(),ylim=(-3,103));axes[1,col].set_xlabel('Environment transitions (M)')
  for ax in axes[:,col]:ax.grid(alpha=.2)
 axes[0,0].set_ylabel('Policy success (%)');axes[1,0].set_ylabel('Mean dense return')
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
 fig.suptitle('DIPO saved direct-policy evaluations | dense reward | one seed; unequal budgets and reset distributions')
 fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(snapshot/'dipo_dense_learning_curves.png',dpi=170);plt.close(fig)
 output=dict(collection=load(snapshot/'collection-verification.json'),runs=runs,
             raw_rollout_sha256=audit.INPUTS,raw_evaluations_verified=len(audit.INPUTS),other_campaign_failures=warnings,
             limitations=['all training seed0','different budgets/environment counts/reset distributions/numerical safeguards',
                          'legacy NovelD ON runs stopped before first post-training evaluation',
                          'ON v3 500k native result is not a saved 500k direct-policy result'])
 (snapshot/'results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
 lines=['# DIPO dense reward: 전체 NovelD ON/OFF 기록','',
        '두 서버의 실제 dense DIPO 학습19개를 수집했다. preflight, sparse 실험, 실행되지 않은 큐는 결과에서 제외했다.',
        '모든 실험 seed0. 초기화 전 평가(step0)는 학습된 정책 결과로 표시하지 않는다.',
        '모든 원시 궤적의 goal 도달, dense 거리합 return, padding/finite, fixed 상태 여부, 전송SHA256를 검증했다. 초기 자체 환경의 일부 기록은 요약 history만 남아 있어 별도 표시한다.','',
        '|구분|환경|NovelD|학습 step|업데이트|학습 성공|마지막 direct-policy 평가|성공 경로|상태|',
        '|---|---|---:|---:|---:|---:|---|---|---|']
 order=['legacy-off','legacy-on','upstream-on','off-probe','off-previous','off-current']
 for group in order:
  for task in ['v1','v2','v3','v4']:
   row=runs.get(f'{group}/{task}')
   if not row:continue
   p=row['latest'].get('policy');hist=row['latest_post_training_history'].get('policy')
   if p:
    summary=f'{p["step"]:,}: {p["successes"]}/{p["episodes"]}'
    routes=', '.join(f'{k}={v}' for k,v in p['successful_routes'].items()) or '없음'
   elif hist:
    summary=f'{hist["step"]:,}: {int(round(hist["success_rate"]*hist["episodes"]))}/{hist["episodes"]} (history)';routes='없음'
   else:summary='학습 후 평가 없음(step0만 존재)';routes='판정 불가'
   lines.append(f'|{row["label"]}|{task}|{row["coefficient"]}|{row["progress_step"]:,}|{row["updates"]:,}|{row["training_successes"]}|{summary}|{routes}|{row["state"]}|')
 lines+=['','## 해석','',
         '- 현재 OFF v1은2.5M평가39/40, 아래쪽 한경로. v2는1.25M평가36/40, 오른쪽 목표 한경로. v3/v4는같은최신평가각0/40이다.',
         '- ON0.01 공식64env에서는v1 500k정책2/10, v2 250k정책8/10. v3는500k native4/10(오른쪽아래목표), 같은500k direct-policy는저장되지않았다. 마지막direct는250k0/10. v4 250k0/10.',
         '- ONv3의성공경험은실제로있었으나이를NovelD의단독효과라고확정할수없다. ON은64env/2updates, OFF는256env/8updates이며per-transition업데이트비율은둘다1/32다. OFF는수치안정성보정이추가됐다.',
         '- ONv2/v3/v4의저장평가는원본고정시작,현재OFF는모두랜덤시작이다. 평가10회와40회,학습예산도다르다.',
         '- 자체환경ON실험(v1=.01,v2-v4=.1)은학습중성공0이고250k정기평가전에사용자지시로중단됐다. 이를학습된정책평가0%라고치환하지않는다.',
         '- 328,192step probe는v1/v3각100회성공0이며본실험완료와구분한다. 이전OFFmain은중단결과다.',
         '- 모든그림은실패포함. DIPO의자체diffusion샘플링을사용하며OptiQ conditional sigma를가져와섞지않았다.',
         '', '## 실행 중 별도 문제','',
         '현재180서버MFPO v1이사전검사에서ModuleNotFoundError: configs.mfpo_config로종료되었다. 해당shard의대기v3MFPO가보류중이며실행중DIPO는보존된다. 이보고작업은학습/큐를변경하지않았다.',
         '', '## 원자료','', 'results.json에모든평가시점/모드별경로와return, source/config경로를보관했다. 호스트별하위폴더에는원격원본캠페인경로를유지했다.']
 (snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(output=str(snapshot),runs=len(runs),raw_evaluations_verified=len(audit.INPUTS),current={task:runs[f'off-current/{task}']['latest']['policy'] for task in ['v1','v2','v3','v4']}),indent=2))

if __name__=='__main__':main()
