# 실제 데이터로 본 Q 가중치·GMM responsibility·경로 붕괴

## 판정

**초기 Q 가중치 편중이 teacher 분포를 한쪽으로 미는 경로는 v3에서 관측된다. 하지만 responsibility 자체가 critic Q를 직접 무너뜨린다는 주장은 구현상 맞지 않고, responsibility 붕괴가 경로 소실의 원인이라는 증거도 현재 자료에는 없다.** 오히려 같은 v3 run의 집계 책임도 지표는 250k→750k 동안 나빠지지 않았다. 확인된 흐름은 `critic Q → teacher 후보 가중치 → responsibility를 통한 actor fitting → 다음 행동/방문 분포`이며, 그 뒤 replay 편중이 후속 Q 학습을 되먹임할 수 있다.

## 순서와 실제 값

Direct-GMM actor update는 먼저 Q로 각 teacher action의 source weight를 계산하고, 그 다음 conditional Gaussian 성분의 posterior responsibility를 계산해 NLL 기울기를 만든다. Critic은 replay의 TD loss만으로 갱신되며, 그 minibatch에서 계산한 actor responsibility가 같은 스텝의 critic target이나 loss에 직접 들어가지는 않는다. Responsibility는 actor의 다음 행동 분포를 바꾸고, 그 행동이 수집되어 replay에 들어간 뒤에야 Q 데이터 분포에 영향을 준다.

고정 시작 v3, source `f953d28456d3800860dddb9b9cb91b6bd520ae00`, 250,112-step 체크포인트의 64개 teacher 후보에서 후보 자체의 후속 경로 분포는 왼쪽 23.0%, 오른쪽 66.8%였다. Q만으로 softmax 가중하면 왼쪽은 3.5%, 오른쪽은 84.7%로 바뀌었다. `-log q` density correction을 함께 적용한 실제 teacher는 왼쪽 15.7%, 오른쪽 75.7%였다. 즉 이 한 후보 cloud에서는 **Q reweighting이 소수 경로를 강하게 억제했고, density correction이 일부 복구했지만 완전히 되돌리진 못했다.**

같은 후보 cloud에서 전체 후보의 route-conditional Q 평균은 왼쪽 143.35, 오른쪽 145.27이었다. 작은 평균 차이만으로 설명하기엔 Q-weighted route mass의 이동이 컸다. 후보 점수 분포와 softmax 꼬리 가중이 평균보다 중요하다는 뜻이다. 다만 이는 단일 cloud와 continuation 4회 기반이어서 일반적인 확률의 정밀 추정은 아니다.

그 후 실제 고정 시작 정책 평가에서 v3 경로는 250k의 `왼쪽12/오른쪽23/미진입5`에서 500k의 `왼쪽0/오른쪽39/미진입1`로 바뀌었고 750k에도 거의 같았다. 이 시간 순서는 Q 가중치 편중이 actor teacher의 초기 불균형과 함께 나타나고, 이후 경로 샘플링이 한쪽으로 집중된 설명과 일치한다.

그러나 GMM posterior responsibility 자체가 단일 성분으로 붕괴했다는 자료는 없다. 250k teacher의 실제 GMM responsibility를 각 route로 조건화했을 때 유효 성분 수(64개 중)는 v3 왼쪽 21.5, 오른쪽 14.4였고, 가장 큰 단일 성분 비중도 왼쪽 9.7%, 오른쪽 21.7%였다. 저빈도 왼쪽 teacher 질량은 여러 성분에 계속 배정되어 있었다.

같은 v3 run의 actor training log를 시간 구간별로 보면, 64개 GMM 성분 중 유효 성분 수는 약 `26.3 → 27.3 → 28.3`으로 250k, 500k, 750k 구간에서 증가했다. 사용량이 기준보다 낮은 성분 비율도 `10.7% → 9.7% → 8.3%`로 감소했다. 반면 평가상 왼쪽 경로는 500k에서 이미 사라졌다. 이 집계 지표는 모든 replay state를 평균한 값이라 분기점에서의 책임도 붕괴를 배제하지는 않지만, **전역적인 성분 responsibility collapse가 경로 소실을 선행했다는 가설은 지지하지 않는다.**

v4에서는 초기 250k cloud에서 lower route mass가 제안분포 36.7%에서 Q-only 35.2%, 실제 Q+density teacher 30.9%로만 낮아졌다. 그런데 평가 경로는 lower 9회에서 500k upper 40회로 바뀌었다. 그러므로 v4 경로 소실을 초기 Q-only 편향만으로 설명하기 어렵다. 표본운·후속 actor의 연속 행동 변화, 장기 TD/replay 피드백이 더 필요하다.

## 결론과 남은 한계

- **직접 확인:** v3에서 Q weighting이 teacher route 질량을 왼쪽 23.0%에서 Q-only 3.5%까지 누르고, density correction 후에도 15.7%에 머물렀다. 이후 500k 평가에서 왼쪽 경로가 0/40이 되었다.
- **확인되지 않음:** GMM responsibility가 Q를 같은 업데이트에서 직접 바꿨다는 주장. 코드 경로상 그런 직접 연결은 없다.
- **오히려 반대 증거:** 집계 GMM 유효 성분 수와 사용률은 route collapse 때 하락하지 않았다. 250k 국소 responsibility도 소수 경로에 여러 성분이 남아 있었다.
- **가능한 간접 루프:** Q-weighted teacher가 actor의 경로 확률을 기울임 → 수집 행동과 replay 방문이 한쪽으로 쏠림 → 반대 경로의 TD 감독이 줄고 분포 밖 critic 추정이 틀어짐 → 다음 actor 업데이트가 다시 영향을 받음.

따라서 지금 증거로는 **Q→teacher 샘플 가중→actor 분포 편향**의 초기 연결은 v3에서 지지되지만, responsibility가 Q 불균형을 만들었다거나 그것만으로 최종 붕괴를 유발했다고 판정할 수 없다. 특히 500k/750k 같은 시작 상태에서 route-conditioned responsibility를 반복 측정하지 않아 collapse 전후의 latent-component별 변화는 미확인이다.

![Q 가중치, responsibility, 경로 비율](responsibility_causal_check.png)

재현 자료는 `analysis.py`와 `analysis.json`이다. 원학습 체크포인트·코드·설정은 수정하지 않았고, 저장된 배열 및 학습 로그만 읽었다.
