import json,sys,importlib.util,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('detail',REPO/'artifacts/antmaze_dacer_off_t1/report_completed.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=json.load(open('/tmp/antmaze-1m-comparison.json'));selected=[];proofs={}
for task in ['v3','v4']:
 old=next(r for r in rows if r['task']==task and r['reward']=='dense' and r['T']==1)
 new=next(r for r in rows if r['task']==task and r['reward']=='progress_euclidean')
 assert old['starts_sha']==new['starts_sha']
 for k in ('alg','dacer'):assert old['config']['native'][k]==new['config']['native'][k]
 assert old['step']==new['step']==1000192
 assert old['episodes']==new['episodes']==40
 proofs[task]=dict(same_start_states=True,same_native_algorithm_config=True,same_dacer_config=True,step=1000192,episodes=40)
 selected.extend([old,new])
fig,axs=plt.subplots(2,2,figsize=(11.5,11.5));fig.subplots_adjust(top=.84,bottom=.12,wspace=.25,hspace=.44)
for ax,r in zip(axs.flat,selected):
 d=np.load(r['input']);m.audit.decorate(ax,r['task'])
 for xy,n,g in zip(d['xy'],d['lengths'],d['goals']):
  p=xy[:int(n)+1];fam=m.family(r['task'],p);color=m.COLORS[fam]
  ax.plot(p[:,0],p[:,1],c=color,alpha=.32,lw=.7)
  ax.scatter(*p[0],c=color,marker='^',s=10,alpha=.5);ax.scatter(*p[-1],c='#bd2942',marker='x',s=13,alpha=.6)
 label='Previous: -distance' if r['reward']=='dense' else 'Current: progress -0.01 + bonus'
 ax.set_title(r['task'].upper()+' | '+label+'\n'+f'1.00M | success {r["success"]}/40 | mean max travel {r["mean_max_displacement"]:.2f}m\n'+', '.join(f'{k}:{v}' for k,v in r['corridors'].items()),fontsize=10)
fig.suptitle('Same 1M budget and identical evaluation starts\nOptiQ T=1, DACER OFF, NovelD OFF; 40 direct-policy rollouts, seed0',fontsize=14,y=.97)
fig.text(.5,.025,'Blue: upper/left; orange: lower/right; gray: no corridor gate. All failures included.\nRandom starts; random z + conditional sigma. Corridor use is not successful goal arrival.',ha='center',fontsize=10)
fig.savefig(ROOT/'comparison_1m.png',dpi=160);plt.close(fig)
result=dict(verified_comparison=proofs,runs=rows,input_sha256={r['input']:hashlib.sha256(Path(r['input']).read_bytes()).hexdigest() for r in rows})
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 동일 1M 시점 보상 비교','','T1, DACER OFF, NovelD OFF. 1,000,192 transitions, 랜덤 시작 direct-policy 40회, seed0. 시작 full-state와 native 알고리즘 설정 일치 확인.','', '|미로|보상|성공|통로 사용|평균 최대 출발점 이격(m)|평균 목표 최근접 거리(m)|','|---|---|---|---|---:|---:|']
for r in selected:lines.append(f'|{r["task"]}|{r["reward"]}|{r["success"]}/40|{r["corridors"]}|{r["mean_max_displacement"]:.2f}|{r["mean_closest_goal"]:.2f}|')
lines+=['','v3 통로 기준 x<-8 / x>8, v4 x=-4 통과 시 y>2 / y<-2. 통로 통과와 성공을 구분. v3 새 Geodesic은 수집 당시 1M 평가 미완료라 동일 1M 표에 포함하지 않음. 새 성공보상 ON/OFF는 1M 전 목표 도달이 없어 동일 궤적을 생성했다. v4 Geodesic은 1M 성공0/40, 통로0/40, 평균최대이격2.54m.','', '원시 return은 보상 정의가 달라 직접 비교하지 않음. 한 시드 결과이며 최종 성능은 미확정.','', '![비교](comparison_1m.png)']
(ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(ROOT/'comparison_1m.png')
