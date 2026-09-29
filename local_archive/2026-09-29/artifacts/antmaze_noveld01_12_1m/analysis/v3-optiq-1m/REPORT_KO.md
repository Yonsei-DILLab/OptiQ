# v3 OptiQ 1M 결과 분석

NovelD 0.1, seed0, 1M environment interactions / 995k learner updates. 학습 source `0d23377f31bdd5c360e9c79d44bb9f4b1341012b`.

## 최종 정책

- 직접 샘플링(random z + conditional sigma): 99/100 성공, 모두 G1/passage-y+4. 실패 1회도 그림에 포함.
- random z mu-only: 100/100 성공, 모두 같은 통로와 G1.
- z=0 mu-only: 100/100 성공, 동일 초기 상태에서 동일한 경로 반복.
- 목표 도달 성능은 높지만 여러 목표·통로를 사용하는 정책은 관찰되지 않았다. action distribution 자체가 단봉이라는 결론은 낼 수 없다.
- 실제 원시 좌표를 그렸다. 750k 도식과 달리 1M raw rollout이 보존되어 있다.

## 학습 중 탐색

- 최초 성공: 220,136 steps. 전체 학습 성공: 4,161회, 목표별 {'1': 4161}.
- 학습 성공 경로 분류: {'G1/passage-y+4': 4161}.
- 누적 방문 0.5m 격자: 491개. 이전 0.01은 679개이며 현재가 27.7% 적다.
- G2 최단 접근 거리: 현재 14.14m / 이전 6.20m. 성공 기준은 0.5m.
- G2쪽 영역(x>2, y<-2) 방문 비율: 현재 0.000%, 이전 0.778%. 이는 이 영역의 시간 비율이며 전체 우측 공간 탐색이나 성공 경로 수와 동일하지 않다.
- 마지막 100k 방문 격자: 현재 184, 이전 199.

## 해석

이 한 seed에서는 계수 0.01에서 0.1로 높여도 최종 목표·경로 다양성이 늘지 않았고, 전체 공간 탐색 범위는 줄었다. 직접 샘플링 최종 성공률은 두 조건 모두 99%였다.
G1 또는 G2 중 가까운 목표에 도달하면 되는 보상이며, NovelD는 목표별 방문 균형을 직접 최적화하지 않는다. 관찰 결과는 G1 통로에 집중된 학습과 일치하지만, 원인을 계수 하나로 확정하려면 추가 seed/통제 실험이 필요하다. CPU/GPU 병렬화는 속도 개선이며 다중 경로를 자동 보장하지 않는다.

## 비교·검증 한계

이전 학습 source `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`. 모델 초기 actor/critic hash, 환경, 나머지 학습 config는 일치한다. 계수 외에도 평가 주기(25k 대 250k), full checkpoint 주기(100k 대 최종만), source commit이 다르다. 이번 최종100k에는 정기 평가가 1M 한 점뿐이므로 촘촘한 마지막100k 평가 평균을 주장하지 않는다.
두 reset 모드의 초기 상태 및 rollout 배열이 동일하여 natural/fixed를 합쳐 200회로 세지 않는다. 결과는 각각 단일 training seed0 정책의 100회 평가다.
기존 archive SHA256/full-state 검증을 확인하고, 1M replay 보상을 목표 거리와 재대조했다. Coverage는 float64 시뮬레이터 좌표, replay는 float32 관측이므로 각 조건에서 경계상의 1개 표본이 인접 격자로 반올림된다. 모든 차이가 float32 반올림 구간으로 설명됨을 검증했고 방문 격자 집합은 완전히 일치했다. 방문 지도는 원래 저장 coverage를 사용한다. Raw rollout에서 성공률·경로·보상을 재계산하여 저장 summary와 대조했다. 입력 rollout hash는 results.json에 기록했다.

그림: final-trajectories.png, coefficient-trajectories.png, training-coverage.png, learning-and-exploration.png. PDF도 보관.
