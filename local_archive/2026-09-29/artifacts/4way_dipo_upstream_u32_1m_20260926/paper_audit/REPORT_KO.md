# DIPO 원논문과 현재 4-Way 비교 — 2026-09-26

실행 중인 실험과 학습 코드는 변경하지 않았다. 완료된 seed 0의 저장 평가와 코드, 원논문 및 공식 저장소를 읽었다.

## 확인된 결론

원논문은 4목표 multi-goal에서 diffusion policy가 여러 방향을 표현하는 결과를 보고한다. 그러나 논문 G.2.1의 그림은 서로 다른 49개 상태 {-3,…,3}²에서 각각 뽑은 행동이다. 동일한 원점에서 반복 rollout한 목표 선택 비율과는 다른 검사다. 현재 체크포인트도 동일한 49개 상태 형식으로 그리면 네 방향의 행동이 보인다. 원점의 현재 평가 샘플러에서는 1,024회 모두 남쪽에 도달한다. 두 결과는 모순되지 않는다.

- 실제 run source: 5f4b809d0e95899eef2435dd80cac2da48058931
- 공식 DIPO pinned source: c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc
- 최종 전이 수: 1,001,472 / learner updates: 15,520
- 목표 E/W/N/S: 0/0/0/1024, 성공률 100%, 평균 길이 10
- 저장 probe 원점 8개 행동 평균: (0.00943914, -0.82281685)
- 해당 8개 행동 표준편차: (1.86e-7, 7.25e-7). 이 8개만으로 전체 샘플러의 분포를 확정하지 않는다.
- 49-state 그림은 저장된 각 상태의 첫 샘플을 사용했다. 새 inference 또는 학습은 하지 않았다. 화살표 길이는 가독성을 위해 1.5배 표시한다.

## 원본 확인 범위

DIPO 논문 §3.2 및 Appendix G는 목표 (±5,0),(0,±5), 행동 제곱 비용, 가장 가까운 목표까지의 제곱거리 비용, 성공 +10을 설명한다. 10,000 iterations 학습을 보고하지만 G의 iteration을 환경 전이 수 또는 optimizer 횟수와 동일시할 충분한 실행 설정은 공개 본문에서 확인되지 않는다. 행동 비용의 정확한 계수, G 전용 warmup/배치/horizon/시작 분포/평가 noise 설정까지 확정할 수 없다.

공식 저장소 pinned commit의 전체 tree는 11개 파일로 MuJoCo 학습기/실행 스크립트를 제공한다. G 전용 multi-goal 환경과 실행 스크립트는 없다. 따라서 실제 공식 DIPO learner를 사용한다는 것과 논문의 multi-goal 실험을 완전히 재현한다는 것은 구분해야 한다.

논문이 인용하는 SQL 공식 MultiGoalEnv 기본값은 action cost 30, distance cost 1, success +10, 성공 반경 1, action [-1,1], 시작 N(0,0.1²I)다. 현재 비용 수식은 이 기본값과 일치한다. 현재 시작은 정확한 원점이고 horizon은 20이다. SQL 환경 클래스 자체에는 horizon이 없으며 외부 실험의 시간 제한은 별도 확인이 필요하다. 이 SQL 기본 시작 분포가 DIPO G 실험에서도 그대로 사용됐는지는 미확인이다.

## 현재 설정과 공개 MuJoCo 실행 코드 비교

| 항목 | 공식 main.py | 현재 4-Way |
|---|---|---|
| 환경 수 | 1 | 2048 |
| 배치 | 256 | 4096 |
| gradient updates / transition | 1 | 32/2048 = 1/64 |
| random warmup | 10000 전이 | 8192 전이 = 환경당 4걸음 |
| 약 1M 전이의 updates | 약 990000 | 15520 |
| diffusion | 100 cosine steps | 동일 |
| action improvement | 기본 20 steps, LR .03 | 동일 |
| actor/critic LR | 3e-4 | 동일 |
| gamma / tau / replay | .99 / .005 / 1M | 동일 |
| 공식 실제 MLP | 256×3 hidden, Mish | 동일 |
| 평가 | initial Gaussian ON, reverse noise OFF | 동일 |

이 표의 원본 열은 MuJoCo 실행 코드이며, 공개되지 않은 G 전용 설정을 대신 확정하는 표가 아니다. 배치가 16배 커졌으므로 optimizer 횟수 64배 차이를 그대로 총 데이터 사용량 차이로 읽으면 안 된다. 대략 minibatch sample 사용량은 원본의 1/4이다. 이것이 붕괴의 원인인지는 통제 실험이 필요하다.

원점에서 action이 각 좌표 ±1로 제한되고 성공 반경이 엄격히 1 미만이므로 현재 네 번의 warmup 동안 어느 환경도 목표에 도달할 수 없다. Dense reward 자체는 학습 신호를 주지만, 모든 환경이 전체 경로를 경험하기 전에 초기 actor가 수집을 맡는 설정이다. 이는 검증할 가치가 있는 실제 차이다.

배치 변경 시 주의할 추가 구현: 공식 action-gradient clip은 개별 action이 아니라 배치 전체 action tensor의 norm을 제한한다. 같은 threshold .2와 batch4096을 쓰는 것은 batch256과 동일한 수치 조건이 아니다. Adam의 정규화 때문에 실제 action 변화가 단순히 배치 제곱근 비율로 줄어든다고 단정할 수 없다.

## 평가 해석과 다음 검증 순서

현재 DIPO는 매 행동에서 fresh initial Gaussian을 사용하지만 reverse denoising 단계 noise는 eval=True로 꺼진다. 공식 MuJoCo 평가와 일치하므로 이것만으로 구현 오류라 할 수 없다. 다만 논문 G에서 같은 모드를 썼는지 확인되지 않았고, initial noise가 출력에서 거의 사라지는 현상이 현재 저장 probe에 나타난다.

우선 같은 checkpoint에서 reverse noise ON/OFF를 비교하여 평가 샘플러의 영향과 학습된 분포를 분리해야 한다. 그 다음 배치256/업데이트 비율/충분한 전체 에피소드 warmup을 갖춘 작은 재현 대조군이 적절하다. 시작점 미세 랜덤화는 인용 환경과의 차이를 확인하는 별도 대조군이지, 동일 상태의 multimodality를 입증하는 대체 지표가 아니다. 이 조사에서는 추가 학습이나 평가 샘플러 변경을 실행하지 않았다.

“엔트로피 항이 없으므로 현재 단일 경로가 당연하다” 또는 “업데이트를 늘리면 반드시 네 모드를 찾는다”는 결론은 이 근거로 성립하지 않는다. 확정된 사실은 현재 평가의 원점 경로 집중, 서로 다른 상태에서 네 방향 행동, 그리고 원본 실행 설정과의 차이다.

## 근거

- DIPO §3.2, Appendix G/H: https://arxiv.org/html/2305.13122#S3.SS2 , https://arxiv.org/html/2305.13122#A7
- 공식 repository: https://github.com/BellmanTimeHut/DIPO/tree/c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc
- main: https://github.com/BellmanTimeHut/DIPO/blob/c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc/main.py
- sampler: https://github.com/BellmanTimeHut/DIPO/blob/c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc/agent/diffusion.py
- 引用 SQL environment: https://github.com/haarnoja/softqlearning/blob/master/softqlearning/environments/multigoal.py
- 현재 config/progress 및 원시 NPZ: 상위 폴더
- 입력 SHA256 및 수치: probe_comparison.json
- 그림: paper_grid_vs_origin_rollouts.png
