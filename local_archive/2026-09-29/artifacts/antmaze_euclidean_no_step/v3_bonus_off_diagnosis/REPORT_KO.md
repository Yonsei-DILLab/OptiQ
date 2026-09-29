# v3 — 성공 보너스 없는 경로 집중 점검

실험 source: f953d28456d3800860dddb9b9cb91b6bd520ae00.
보상은100*(d_current-d_next), nearest-goal Euclidean. 성공 보너스0, step penalty0,
NovelD OFF, DACER OFF. success done은 유지하지만 bonus는 보상에 더하지 않는다.
환경 wrapper와 저장 raw rollout의 누적보상 재계산으로 확인했다.

250112step에서 같은 시작점의 왼쪽12episode와 오른쪽23episode를 비교:
- 700step 할인 누적보상(gamma=.99): 왼쪽546.0069 ±49.6937, 오른쪽550.5367 ±65.3330 (sampleSD).
- 비할인 누적보상: 왼쪽1242.5348, 오른쪽1321.2310.
- 최종 최근접 목표거리: 왼쪽4.5452m, 오른쪽3.7583m.
- 양쪽 모두 성공0. 이 자료만으로 오른쪽의 큰 보상 우위가 붕괴를 일으켰다고 결론내릴 수 없다.
이 수치는 유한 rollout의 실제 return이며 학습 critic의 Q값과 동일하다고 간주하지 않는다.

수학적으로 G_L=100*[d0-(1-gamma)*sum(t=1..L-1,gamma^(t-1)*d_t)-gamma^(L-1)*d_L].
gamma=1이면 종료거리 차이만 남지만 .99에서는 보상이 발생하는 시점도 영향을 준다.
보너스 제거는 경로 사용 비율을 보존하는 목적을 추가하지 않는다. 단일 경로로 잘 가는
정책도 높은 return을 얻을 수 있다. Critic TD target은 현재/target policy의 후속행동을
평가하므로 이상적으로 가능하지만 아직 배우지 못한 경로를 자동으로 높게 평가하지 않는다.

실제 로그 160k..750k: temperature1 고정, 미래 엔트로피 backup항0,
actor 전체평균sigma 약.367 유지. Source ESS는 대략18.2/64→20.2/64이며 최고가중치
평균은.192→.176. 따라서 전체sigma 붕괴나 전체teacher 가중치의 갑작스러운
단일후보 집중이 확인된 것은 아니다. 이 값은 replay전체 평균으로 분기 상태별 차이는 배제하지 못한다.

구현상 actor teacher는 softmax(Q/T - beta*log proposal_density) 가중치를 쓰며,
TD critic은미래엔트로피를 backup하지 않는다. 밀도보정 및 stochastic action 분포가
있어도 서로 다른 장기 경로 사용비율을 직접 보존하는 제약은 없다. 작은 방향별 Q오차/
학습진도 차이→그 방향 actor/방문 증가→반대방향 학습 약화라는 자기강화는 가능한
설명이다. 이 실험에서 실제 인과를 확정하려면 동일한 분기상태에서 좌우 action Q와
후속 return을 짝지어 비교해야 한다. 현재자료로 reward버그/숨은성공bonus를 원인으로
주장하지 않는다. 학습/설정/평가는 변경하지 않았다.
