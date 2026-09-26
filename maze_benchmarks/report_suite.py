"""Korean result index for the original33 policies and six temperature additions."""
import argparse
import json
from pathlib import Path
from .run_nway_job import atomic_json


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--temperatures',type=Path,required=True)
    a=p.parse_args()
    main=json.loads((a.root/'archive-manifest.json').read_text())
    temp=json.loads((a.temperatures/'archive-manifest.json').read_text())
    original_plan=json.loads(Path(__file__).with_name('DEADLINE_1M_PLAN.json').read_text())
    extra_plan=json.loads(Path(__file__).with_name('POINTMAZE_T5_T10_PLAN.json').read_text())
    original_names={j['name'] for j in original_plan['jobs']+original_plan['reused']}
    extra_names={j['name'] for j in extra_plan['jobs']}
    if original_names & extra_names or len(original_names|extra_names)!=39:
        raise ValueError('unexpected experiment scope')
    if not set(main['runs'])<=original_names or not set(temp['runs'])<=extra_names:
        raise ValueError('unrequested or duplicate result')
    merged={**main['runs'],**temp['runs']}
    rows=[]
    for name,result in sorted(merged.items()):
        mode='mu_only' if 'optiq' in name else 'policy'
        data=result['record'][mode]
        counts=data['goals']
        rows.append(dict(name=name,mode=mode,steps=result['steps'],updates=result['updates'],
            reused=result.get('reused',False),source=result['training_source'],
            episodes=data['episodes'],success=data['success'],counts=counts,
            reached=sum(n>0 for n in counts),goals=len(counts),failure=data['failure']))
    atomic_json(a.root/'suite-results.json',dict(expected=39,verified=len(rows),rows=rows,
        missing=sorted((original_names|extra_names)-set(merged))))
    lines=['# 1M 실험 결과', '',f'원자료·체크포인트 보관 완료: **{len(rows)}/39**.', '',
      '- OptiQ는 매 행동마다 새 정규분포 latent를 뽑는 **μ-only** 평가입니다. Conditional σ는 없습니다.',
      '- 나머지 방법은 native sampler를 사용합니다. 모든 실험은 학습 seed0 한 개입니다.',
      '- 성공률은 어떤 목표든 도달한 비율이며, 모든 목표 방문이나 균형 잡힌 방문을 뜻하지 않습니다.',
      '- 신규 실험은 256환경·batch4096·16updates/256transitions, 1,000,192전이·62,000업데이트입니다.',
      '- 재사용한 8/16-Way T1은 1,000,000전이·998,976업데이트·16환경·batch256입니다. 새 T3/5/10과 temperature 단독 효과로 해석할 수 없습니다.',
      '- PointMaze는 원래 DrAC 맵과 sparse +100 보상입니다. 과거 AntMaze dense 실험과 구분합니다.',
      '- T3는 재사용했고 T5/T10 여섯 개만 추가했습니다. 학습 소스와 보고 소스는 각각 보관합니다.', '',
      '|실험|평가 수|성공률|방문 목표|목표별 횟수|실패|', '|---|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['name']}{' (재사용)' if r['reused'] else ''}|{r['episodes']}|{r['success']:.1%}|{r['reached']}/{r['goals']}|{r['counts']}|{r['failure']}|")
    missing=sorted((original_names|extra_names)-set(merged))
    if missing:lines += ['', '아직 완료·보관되지 않은 실험: '+', '.join(missing)]
    lines += ['', '주 그림: `figures/way_trajectories_mu_only.png`, `figures/pointmaze_medium_hard.png`, `figures/pointmaze_learning_curves.png`. 추가 온도 비교는 별도 temperature artifacts의 `figures/trajectories_mu_only.png`에 있습니다.', '',
      '4-Way Q 그림은 `figures/4way_mu_only_and_learned_q.png`입니다(추가 평가 완료 후). 이는 학습된 Q를 μ-only action에 대해 평균낸 값이며 실제 μ-only return의 정확한 값이라는 뜻은 아닙니다.', '',
      'σ 포함 이전 그림은 `figures/supplementary_sigma/`에 보존합니다. Removal SR5는 원시 goal IDs로 경계 조건을 교정했으며 학습 데이터나 궤적을 바꾸지 않았습니다.']
    (a.root/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified=len(rows),expected=39,missing=missing),ensure_ascii=False))


if __name__=='__main__':main()
