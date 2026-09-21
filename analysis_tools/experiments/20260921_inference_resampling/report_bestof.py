"""Join the completed three-way evaluation with its paired best-of64 follow-up."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
r=p.parse_args().root;base=r.parent
m=json.loads((r/'manifest.json').read_text());oldm=json.loads((base/'manifest.json').read_text())
assert m['jobs']==oldm['jobs'] and m['seed_base']==oldm['seed_base'] and m['episodes']==oldm['episodes']==20
allmodes=['mu_one','mu_q64','mu_best64','mu_kde_is64']
combined=[]
for job in m['jobs']:
 old=json.loads((base/'results'/(job['name']+'.json')).read_text())
 new=json.loads((r/'results'/(job['name']+'.json')).read_text())
 assert old['complete'] and new['complete'] and new['checkpoint_state_unchanged'] and new['preflight_passed']
 assert old['job']['input_sha256']==new['job']['input_sha256']
 assert new['modes']['mu_best64']['reset_seeds']==old['modes']['mu_one']['reset_seeds']
 assert new['modes']['mu_best64']['policy_seeds']==old['modes']['mu_one']['policy_seeds']
 assert len(new['modes']['mu_best64']['returns'])==20
 combined.append(dict(job=job,modes={**old['modes'],**new['modes']}))
summary={};fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,task in zip(axs,['halfcheetah','ant']):
 rows=sorted([d for d in combined if d['job']['task']==task],key=lambda d:d['job']['seed'])
 vals=np.array([[d['modes'][mode]['mean_return'] for mode in allmodes] for d in rows])
 details={}
 for j,mode in enumerate(allmodes):
  delta=vals[:,j]-vals[:,0]
  details[mode]=dict(mean=float(vals[:,j].mean()),seed_sd=float(vals[:,j].std(ddof=1)),
    mean_delta=float(delta.mean()),delta_seed_sd=float(delta.std(ddof=1)),
    per_seed=vals[:,j].tolist(),per_seed_delta=delta.tolist(),
    percent_delta=float((vals[:,j].mean()/vals[:,0].mean()-1)*100),
    improved_seeds=int((delta>0).sum()))
 delta_q=vals[:,2]-vals[:,1]
 details['best_vs_q']=dict(mean_delta=float(delta_q.mean()),seed_sd=float(delta_q.std(ddof=1)),
    per_seed_delta=delta_q.tolist(),improved_seeds=int((delta_q>0).sum()))
 summary[task]=dict(seeds=[d['job']['seed'] for d in rows],methods=details)
 for row,v in zip(rows,vals):
  ax.plot(range(3),v[:3],'o-',lw=1,alpha=.45,label=f'seed {row["job"]["seed"]}')
 ax.errorbar(range(3),vals[:,:3].mean(0),yerr=vals[:,:3].std(0,ddof=1),fmt='o-',c='#173349',lw=2.5,capsize=5,label='Mean ± seed SD')
 ax.set(xticks=range(3),xticklabels=['One μ','Q-weighted 64 μ','Best-of-64 μ'],ylabel='Episode return')
 ax.set_title(task.title()+f' · {len(rows)} training seeds × 20 episodes')
 ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Inference-only best-of-64 · final 1M checkpoint · T=0.25 DACER training',fontsize=14)
fig.text(.5,.01,'Same actor, critic and reset seeds · no conditional σ / DACER noise · argmax of twin-critic mean Q',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ['png','pdf']:fig.savefig(r/f'comparison.{ext}',dpi=180)
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(r/'combined_results.json').write_text(json.dumps(combined,indent=2)+'\n')
lines=['최종 1M 학습 완료 모델에서 추론만 best-of-64로 바꾸었다. 새 학습이나 checkpoint 선택은 하지 않았다. 기존 비교와 같은 9개 actor·critic, 같은 20개 reset/policy seed를 사용했다. 모든 방식은 μ-only이며 σ·DACER 행동잡음은 없다.\n',
'매 step z 64개를 새로 뽑고 a_i=μ(s,z_i)를 계산한 뒤, current critic 두 개의 평균 Q가 최대인 후보 하나를 실행한다. Q 가중 방식과 후보 생성 코드가 같고 선택만 categorical softmax에서 argmax로 바뀌었다. 학습 T=.25는 유지되며 argmax 순위에는 temperature가 영향을 주지 않는다.\n',
'| 환경 | 방식 | return 평균 ± 학습 seed SD | 기존 대비 Δ | 개선 seed |',
'|---|---|---:|---:|---:|']
for task,d in summary.items():
 for mode in allmodes:
  v=d['methods'][mode]
  lines.append(f'| {task} | {mode} | {v["mean"]:.1f} ± {v["seed_sd"]:.1f} | {v["mean_delta"]:+.1f} ({v["percent_delta"]:+.2f}%) | {v["improved_seeds"]}/{len(d["seeds"])} |')
lines += ['\n![비교](comparison.png)\n',
'이는 마지막100k 학습 평균이 아닌 최종 체크포인트의 새 episode 평가다. 학습 seed 간 편차와 episode 간 편차를 구분했다. 실제 환경 reward로 비교했으며, critic 예측 Q가 최대라는 사실만으로 성능 향상을 가정하지 않았다.\n',
'Best-of-64는 Boltzmann 분포 샘플링이나 importance correction이 아니다. 랜덤 후보 집합 안에서 Q 최대 행동을 고르는 정책이다. 정확한 q_mu 밀도를 사용할 수 없어 앞선 IS 비교는 독립 μ 256개로 추정한 KDE 보정이었다. 해당 한 가지 추정법의 결과를 모든 importance sampling 방법으로 일반화하지 않는다.\n',
'체크포인트/config SHA256, actor·critic 불변성, 기존 μ-only sample_action 일치, σ head 변경 불변성, best-of 선택 ESS=1과 Q gain≥0 검증을 통과했다. 원본 결과와 frozen source는 보존했다.\n',
'원본 평가 commit: `'+m['base_evaluation_commit']+'`. Best-of 평가 commit: `'+m['evaluation_commit']+'`.\n',
'| 환경 | seed | 기존 μ | Q64 | Best64 | Δ Best−기존 | Δ Best−Q64 |',
'|---|---:|---:|---:|---:|---:|---:|']
for row in combined:
 v=[row['modes'][k]['mean_return'] for k in allmodes]
 lines.append(f'| {row["job"]["task"]} | {row["job"]["seed"]} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[2]-v[0]:+.1f} | {v[2]-v[1]:+.1f} |')
(r/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
