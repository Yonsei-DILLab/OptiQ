from pathlib import Path
import json
r=Path('/home/heechan/optiq-experiments/antmaze-v1-noveld-100k-s0-20260921')
rows=[]
for method in ['optiq','sac','mfpo','meow']:
 d=r/'runs'/f'v1-{method}-s0';result=json.loads((d/'result.json').read_text());progress=json.loads((d/'progress.json').read_text())
 row=dict(method=method,steps=result['steps'],updates=result['updates'],wall_seconds=result['seconds'],training_successes=progress['training_successes'])
 for label in ['policy-fixed','policy-natural']:
  s=result['summaries'][label]
  row[label]=dict(success_rate=s['success_rate'],mean_return=s['mean_return'],mean_min_distance=s['mean_min_distance'],mean_final_distance=s['mean_final_distance'],routes=s['successful_routes'])
 row['final_intrinsic']=json.loads((d/'intrinsic-audit.json').read_text())['metrics']
 rows.append(row)
(r/'report/pilot-summary.json').write_text(json.dumps(rows,indent=2))
p=r/'report/REPORT_KO.md'
with p.open('a') as f:
 f.write('\n## 이번 파일럿 설정과 해석\n\nDDiffPG NovelD와 환경256개, batch4096, 수집 round당8updates를 적용했습니다. Warmup8192를 포함한 정확100k interactions, 실제 learner/RND 업데이트2872회입니다. 각 방법 고유 모델/LR/tau는 유지했습니다. 보상은 dense 최근접목표 거리 penalty이며 평가에는 NovelD와 DACER 외부 잡음을 넣지 않았습니다.\n\n이전 dense-only100k와 비교하면 NovelD, batch, 병렬 수집, 업데이트 빈도와 warmup이 함께 달라 순수 intrinsic reward ablation이 아닙니다. 각 방법 training seed0 하나의 결과이므로 seed간 우위는 판정하지 않습니다.\n\n')
 f.write('| 방법 | 본실험 시간(평가 포함, 분) | fixed 성공/100 | natural 성공/100 | fixed 평균 최단거리 |\n|---|---:|---:|---:|---:|\n')
 for x in rows:
  f.write(f"| {x['method']} | {x['wall_seconds']/60:.2f} | {x['policy-fixed']['success_rate']*100:.0f} | {x['policy-natural']['success_rate']*100:.0f} | {x['policy-fixed']['mean_min_distance']:.3f} |\n")
print(json.dumps(rows,indent=2))
