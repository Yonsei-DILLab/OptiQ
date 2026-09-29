"""Validate saved DACER OFF rollouts and summarize the current four policies."""
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_route_tools',ROOT.parent/'antmaze_dacer_entropy_t1/analyze_rollouts.py')
tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
audit=tools.audit
SOURCE='484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
snapshot=Path((ROOT/'latest-report-path.txt').read_text().strip())
load=lambda p:json.loads(p.read_text())
runs=[];arrays={};states={}
for host in ('vast-heechan-180','vast-heechan-199'):
    base=snapshot/host;states[host]=load(base/'status.json')
    assert not states[host]['failed'] and not states[host]['pending_held']
    for folder in sorted((base/'runs').glob('*')):
        cfg=load(folder/'config.json');progress=load(folder/'progress.json')
        assert cfg['source_commit']==SOURCE and cfg['temperature']==1 and cfg['seed']==0
        assert not cfg['dacer_enabled'] and not cfg['noveld_enabled'] and cfg['reward_profile']=='dense'
        assert progress['dacer_updates']==progress['dacer_noise_std']==progress['rnd_updates']==0
        history=[]
        for raw in sorted((folder/'evaluations').glob('*/policy-natural/rollouts.npz')):
            d,m=tools.read_mode(raw.parent,cfg['task']);history.append(m)
        latest=history[-1];arrays[cfg['task'],'policy-natural']=d
        modes={'policy-natural':latest}
        for path in sorted(raw.parent.parent.glob('*/rollouts.npz')):
            if path.parent.name=='policy-natural':continue
            dm,mm=tools.read_mode(path.parent,cfg['task']);modes[path.parent.name]=mm
            arrays[cfg['task'],path.parent.name]=dm
        result=load(folder/'result.json') if (folder/'result.json').exists() else {}
        if result.get('completed'):
            proof=load(folder/'checkpoint-verification.json')
            assert proof['readback_verified'] and proof['dense_replay_verified']
            assert result['steps']==cfg['steps'] and not result['dacer_enabled']
        runs.append(dict(task=cfg['task'],id=folder.name,host=host,source=SOURCE,
            status='completed' if result.get('completed') else 'training',
            step=progress['step'],budget=cfg['steps'],percent=100*progress['step']/cfg['steps'],
            current_average_eta_seconds=(cfg['steps']-progress['step'])*progress['seconds']/progress['step'],
            training_successes=progress['successes'],modes=modes,history=history))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11.8,10.8))
fig.subplots_adjust(top=.87,bottom=.12,left=.07,right=.98,hspace=.35,wspace=.23)
for ax,row in zip(axes.flat,sorted(runs,key=lambda r:r['task'])):
    m=row['modes']['policy-natural'];audit.draw(ax,row['task'],arrays[row['task'],'policy-natural'],m)
    ax.set_title(f'{row["task"].upper()} | evaluation {m["step"]/1e6:.3f}M\nSuccess {m["successes"]}/{m["episodes"]} | observed successful routes {len(m["successful_routes"])}')
fig.suptitle('OptiQ DACER OFF | latest saved direct-policy trajectories\nT=1, dense reward, NovelD OFF, seed 0',fontsize=17,y=.97)
fig.legend(handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Success'),
    Line2D([0],[0],color=audit.NEG,lw=2,label='v1 lower-route success'),
    Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure (red endpoint)')],loc='lower center',bbox_to_anchor=(.5,.053),ncol=3,frameon=False)
fig.text(.5,.015,'Random starts; 100 episodes at final, 40 at intermediate checkpoints. All failures included.\nRandom z + conditional sigma; external noise OFF. Different budgets; one training seed.',ha='center',fontsize=10)
fig.savefig(snapshot/'latest_trajectories.png',dpi=165);plt.close(fig)

v1=next(r for r in runs if r['task']=='v1')
if 'policy-fixed' in v1['modes']:
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,mode in zip(axes,('policy-natural','policy-fixed')):
        m=v1['modes'][mode];audit.draw(ax,'v1',arrays['v1',mode],m)
        ax.set_title(('Random starts' if mode=='policy-natural' else 'Identical full start state')+f'\nUpper {m["routes"].get("G1/upper",0)} | lower {m["routes"].get("G1/lower",0)} | fail {m["failures"]}')
    fig.suptitle('v1 DACER OFF | same final policy, 100 rollouts per panel',fontsize=15)
    fig.savefig(snapshot/'v1_start_comparison.png',dpi=165);plt.close(fig)

output=dict(snapshot=str(snapshot),source=SOURCE,raw_validation_passed=True,single_training_seed=0,
    collection=load(snapshot/'collection-verification.json'),status_counts={h:{k:len(s[k]) for k in ('completed','running','pending','failed')} for h,s in states.items()},
    input_sha256=audit.INPUTS,runs=sorted(runs,key=lambda r:r['task']))
(snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
lines=['# DACER OFF 진행 보고','',
    'T=1, dense reward, NovelD OFF, 각 환경 seed0. 원시 direct-policy rollout을 검증해 성공 및 통과 경로를 다시 집계했다.',
    '','|환경|진행|최신 평가|성공|경로별 성공|동일 상태 평가|','|---|---|---|---|---|---|']
for r in output['runs']:
    m=r['modes']['policy-natural'];fixed=r['modes'].get('policy-fixed')
    lines.append(f'|{r["task"]}|{r["step"]}/{r["budget"]} ({r["percent"]:.1f}%)|{m["step"]}|{m["successes"]}/{m["episodes"]}|{json.dumps(m["successful_routes"])}|'+(json.dumps(fixed['routes']) if fixed else '최종 평가 전')+'|')
lines += ['', '학습 seed는 하나다. 랜덤 시작점에서 여러 경로를 쓴 것과 동일 상태에서 여러 경로를 선택한 것을 구분한다. 성공이 없거나 드문 중간 시점으로 최종 경로 붕괴를 단정하지 않는다.',
    'raw 좌표와 dense distance 보상 합, 목표 도달, 시작 상태, 유한성, padding, SHA256 검증을 통과했다. 학습 및 checkpoint를 수정하거나 평가 rollout을 새로 실행하지 않았다.']
(snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(snapshot=str(snapshot),runs=[{k:v for k,v in r.items() if k not in ('history',)} for r in output['runs']]),indent=2))
