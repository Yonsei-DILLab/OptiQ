# OptiQ teacher의 best-of-k 분포를 직접 계산

Profile: `mujoco_v5_bestk_boltzmann` / `v5/bestk_boltzmann`, branch `v5_bestk`.
2026-09-15 사용자 지시: winner 중심 Gaussian과 임의의 50:50 혼합을 제거한다.
이 프로필은 원래 `v5/final`에서 **수집 K=8과 teacher 변환 K=8만** 추가한다.
현재의 가설이며, 학습 성능이 입증된 기본 알고리즘이 아니다.

## 목표와 실제 계산

원래 OptiQ대로 student latent 16개와 conditional Gaussian proposal `q`를 만들고,
그 proposal에서 teacher 행동 64개 `a_j=tanh(u_j)`를 샘플링한다. 원래의
sigma floor .05, tanh Jacobian을 포함한 proposal 밀도, twin-mean Q를 사용한다.

먼저 기존 OptiQ와 동일한 질량을 계산한다.

\[
w_j=\operatorname{softmax}_j\left(Q_{\rm mean}(s,a_j)/T-\log q(a_j\mid s)\right),
\qquad \widehat p_T=\sum_j w_j\delta_{a_j}.
\]

이제 **이 유한 Boltzmann teacher에서 독립적으로 K번 뽑고, Q가 가장 높은
행동을 반환하는 분포**를 계산한다. Q가 서로 다를 때, Q 오름차순으로 정렬한
질량의 누적합을 `C_j`라 하면 다음과 같다.

\[
v_j=C_j^K-C_{j-1}^K,\qquad C_0=0.
\]

최댓값이 j 이하일 확률이 `C_j^K`이므로 이 식은 근사적인 선별 규칙이 아니다.
현재 후보 집합과 가중치에 조건부로 **정확한 categorical winner law**다.
실제로 재샘플링할 필요 없이 원래 64개 행동 전체를 보존하고 OT의 열 질량만
`w`에서 `v`로 바꾼다. 추가 pilot, Gaussian 중심 추정, proposal 혼합 비율,
CDF 추정용 별도 샘플, teacher 재샘플링 RNG는 없다. K는 512개 새 행동을
평가한다는 뜻이 아니라, 기존 유한 teacher에서의 8회 추출을 뜻한다.

동점은 추출 순서의 첫 winner를 택하는 규칙에 대응한다. 동일 Q 그룹의
누적확률 하한/상한을 L/U라 할 때 그룹 내부 질량은 원래 w에 비례한다.
구현은 차의 상쇄와 0 나누기를 피하는 동일한 다항식을 사용한다.

\[
v_i=w_i\sum_{r=0}^{K-1}U^{K-1-r}L^r.
\]

Q가 모두 같으면 v=w이며 K=1이면 기존 경로를 그대로 실행한다.
최종 정규화는 부동소수점 반올림 오차 보정이다. 원래 Sinkhorn의 극소
질량 floor와 유한 반복 오차는 기존과 동일하게 남는다.

## 보존되는 부분과 달라지는 부분

| 부분 | 새 프로필 |
|---|---|
| Teacher proposal / 밀도 | 기존 conditional Gaussian mixture q 그대로 |
| T / 밀도보정 | Ant .25, Hopper .01; exp(Q/T)/q, beta=1 |
| Teacher 목표 | **기존 Boltzmann teacher의 best-of-8 winner** |
| Actor 학습 | 기존 16 mean-action OT, epsilon=.1/100회, full-row Gaussian NLL |
| 행동 수집 | warmup5K 뒤 actor full Gaussian 8개 중 live twin-min 최대 |
| TD target | 기존 full Gaussian 1개, target twin-min, entropy 없음 |
| 평가 | 기존 zero-z / stochastic-z, epsilon=0, best-k 없음 |
| 부가 휴리스틱 | guided proposal, 50:50 수집, 지연 활성화 없음 |

**기존 Boltzmann 목표 자체와 같다고 주장하지 않는다.** 연속 목표로 쓰면
`B_K[p_T]`이며, Q에 동점 확률이 없을 때 그 밀도는
`K p_T(a) F_{p_T}(Q(a))^(K-1)`이다. 실제 구현은 기존 SNIS 근사
`p_hat_T` 위에 이 연산을 정확히 적용한다. 따라서 유한 후보 집합이 연속
Boltzmann 분포를 근사하는 오차는 남고, 이를 정확한 연속 sampling으로
표현해서는 안 된다.

이것은 **actor에서 직접 뽑은 winner `B_K[pi]`의 증류도 아니다.** 수집은
`B_8[pi]`, teacher는 `B_8[p_hat_T]`다. 둘을 같다고 가정하거나 replay와
teacher가 일치한다고 주장하지 않는다. 기존 OptiQ도 behavior와 Boltzmann
teacher가 다른 off-policy 학습이다. 이 차이만으로 버그라고 할 수 없지만,
exploration 감소와 critic 오류는 실제 성능 검증이 필요한 문제다.

`1/q`는 **선별 전** 기존 Boltzmann teacher를 구성하는 단계에 정확히 한 번
쓴다. 변환한 winner 질량에 다시 `exp(Q/T)/q`를 곱하지 않는다. 이렇게
T와 밀도보정은 남지만, actor winner를 기존 q로 잘못 보정하지 않는다.

최댓값의 누적확률 법칙을 사용한 직접 유도다. 관련 best-of-N의 유도 분포와
일반적인 reward-tilted 분포가 다르다는 논의는
[Variational Best-of-N Alignment, §3](https://arxiv.org/html/2407.06057v1#S3)를
참고한다. 해당 논문의 RLHF 실험이 이 OptiQ 변형의 학습 성공을 입증하지는 않는다.

## 검증과 실험 해석

`tests/test_bestk_teacher.py`는 작은 categorical 분포의 모든 K회 추출을
열거하여 질량을 비교하고, 동점·영질량·아주 작은 최고점 질량·순열을 검사한다.
K=1 업데이트/키 보존, 두 actor head의 학습, teacher와 critic 방향의 gradient
차단, T/밀도보정의 실질적 영향, 원래 profile 대비 두 설정만의 차이를 검사한다.
Ant/Hopper의 batch256 실제 짧은 학습으로 guided teacher 미호출, 수집/TD/평가
분리, replay 행동을 확인한다. 짧은 검사는 성능 실험과 구분한다.

검증 결과: 새 테스트 24개와 기존 v5 / guided proposal / actor-winner 회귀를
포함한 **69개 테스트 통과**. 실제 학습 검사는 CPU에서 Ant/Hopper 각각
batch256, 4 updates이며, 두 평가 모드를 실행했다. 학습 로그에서도
teacher K=8, T=.25/.01, beta=1과 pilot 미사용을 확인했다. 아직 새 프로필의
GPU 성능 실험을 실행한 결과는 없다.

T가 이미 낮아 원래 질량이 하나에 집중되면 best-k의 추가 효과가 거의 없을
수 있다. Q가 잘못되면 집중을 강화할 수도 있다. `teacher_base_ess`,
`teacher_base_max_weight`, 실제 `source_ess_absolute`, `teacher_bestk_q_gain`을
함께 기록한다. 마지막 지표는 현재 critic이 예측한 차이이며 실제 return
향상의 증거가 아니다. `q_only_*`, `density_only_*`, logit 관련 기존 진단은
변환 전 teacher 구성요소의 진단이다. 교차 critic/가상 T별 ESS는 변환 후
teacher를 기준으로 계산한다.

기존 proposal-only/delayed/mixed 실행과 소스는 과거 실험으로 보존한다.
이 문서나 새 profile을 추가하는 것만으로 실행 중인 worker가 바뀌지 않는다.
새 성능 실험은 별도 run ID와 처음부터의 학습으로 구분한다.
