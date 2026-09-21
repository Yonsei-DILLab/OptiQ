# Backup 정확도가 개선되지 않은 이유: 저장 actor 진단

2026-09-11. 현재 batch1 캠페인의 seed0,20k 체크포인트를 읽기 전용으로 분석했다. 진행 중인 학습이나 설정을 변경하지 않았다. 이는5-seed 전체에 대한 인과적 결론이 아니라, 완료된 actor에서 직접 확인한 분포·업데이트 단계별 현상이다.

**Ver2는 고차원의 mode 비중과 가중 후보의 Q 평균을 개선했지만, 각 mode 안의 밀도가 목표보다 뾰족해졌고 그 분포가 actor에 전달되는 과정에서 Q 평균이 다시 높아졌다.** 2D에서는 row-argmax 단계의 차이가 크고, 8D에서는 랜덤 target을 MSE로 회귀하는 과정의 평균화와도 강하게 일치한다.

## 1. 측정 대상과 오차

이 실험의 기준은 고정된 정확한 Q의 Boltzmann 기대값이다.

\[
B=\mathbb E_{a\sim\pi_B}[Q(a)],\quad
A=\mathbb E_{a\sim\pi_\theta}[Q(a)],\quad
\pi_B(a)\propto e^{Q(a)/T},\quad T=1.
\]

K개의 독립 actor action으로 평가하는 backup에 대해

\[
\mathbb E[(\widehat B_K-B)^2]
=(A-B)^2+\operatorname{Var}_{\pi_\theta}(Q)/K.
\]

두 버전의 평가 K는50으로 같다. 늘어난2048은 학습 OT의 actor sample 수다. 현재8D ver2의 A=1.762254,B=−.180362이며, 평균 오차1.942616은100k 표본 평균의 표준오차.002126보다 훨씬 크다. 주된 문제는 actor의 기대값 편향이다. 이는 learned critic의 overestimation을 측정한 것이 아니다.

## 2. Mode 비중과 mode 내부를 분리

Mode bin을 m, actor/reference의 mode 비중을 a_m,b_m, 각 bin 안의 조건부 Q 평균을 μ_m^A,μ_m^B라 놓으면

\[
A-B=\underbrace{\sum_m(a_m-b_m)\mu_m^B}_{\text{mode 비중 차이}}
+\underbrace{\sum_m a_m(\mu_m^A-\mu_m^B)}_{\text{mode 내부 분포 차이}}.
\]

Reference 조건부 평균은 원래 T1 수치 적분으로 계산했다. Separable 문제에서는 각 좌표의 조건부 평균을 합해 다차원 값을 정확하게 구성한다. 아래 두 항의 합이 실제 bias와 일치하는지 수치 검증했다.

| 문제 / 버전 | Mode 비중 항 | Mode 내부 항 | 총 bias |
|---|---:|---:|---:|
| 4D ver1 | .061786 | .367437 | .429223 |
| 4D ver2 | .000803 | .452300 | .453103 |
| 8D ver1 | .265931 | 1.331887 | 1.597818 |
| 8D ver2 | −.006692 | 1.949308 | 1.942616 |
| 2D8 ver1 | .004098 | .213332 | .217431 |
| 2D8 ver2 | .008087 | .301888 | .309974 |

8D ver2의 mode 비중을 유지하고 mode 내부만 Boltzmann의 조건부 분포로 바꾸면, 계산상 평균 Q는1.762254→−.187053이 된다. 기준−.180362와 거의 같다. 4D ver2도 같은 계산에서.362922→−.089378로 바뀌며 기준−.090181에 가깝다.

8D에서 각 좌표를 왼쪽/오른쪽 mode로 나눈 조건부 표준편차:

| 좌표별 mode bin | Boltzmann T1 | ver2 |
|---|---:|---:|
| 왼쪽 | .194916 | .074258 |
| 오른쪽 | .279906 | .119216 |

중앙 영역 |a_i|<.2의 확률은 reference8.953% 대 ver2 2.131%다. Ver2는 여러 mode를 방문하지만 그 안에서는 높은 Q 부분에 지나치게 집중한다. 4D의 일부 bin은 표준편차만 비슷해도 밀도가 여러 좁은 peak로 갈라져 있으므로, 분산 하나만으로 내부 분포의 일치를 주장할 수 없다.

![저장된8D actor와 Boltzmann 분포](results/8d_density.png)

왼쪽은8개 좌표를 모은 marginal density, 오른쪽은 Q(a)의 분포다. 검은 점선이 기준 분포다. 그림은 전체8D joint density의 검증을 대신하지 않는다.

## 3. 가중 후보 → OT → row-argmax → actor

동일한 저장 actor에서 새 후보를8회 뽑아 실제 update를 재구성했다. 재구성 loss와 source ESS가 실제 update 호출과 일치하는 것을 확인했으며, 그 update 결과는 버렸다. 훈련 파라미터는 변경하지 않았다.

| 문제 / 버전 | Boltzmann 기준 | 가중 후보 평균 Q | OT row 기대값 | Row-argmax target Q | Actor Q |
|---|---:|---:|---:|---:|---:|
| 8D ver1 | −.180362 | 1.071790 | 1.065165 | 1.592246 | 1.471345 |
| 8D ver2 | −.180362 | −.002850 | −.002844 | .464575 | 1.754940 |
| 4D ver1 | −.090181 | .295420 | .288287 | .531209 | .392731 |
| 4D ver2 | −.090181 | −.090163 | −.090161 | .139724 | .360272 |
| 2D8 ver1 | −.013407 | .085037 | .088671 | .226631 | .230631 |
| 2D8 ver2 | −.013407 | −.010096 | −.010025 | .284780 | .295757 |

Actor 열은 이8회 진단에서 뽑은 source samples의 평균이므로,100k 최종 평가 평균과 작은 차이가 있다. 특히ver1은16×8개 샘플만 포함하므로 변동이 크다. 반복별 표준편차는 원자료에 보존했다.

Ver2의 가중 후보 평균은 실제로 크게 개선됐다. 그럼에도 weighted mass를 row-argmax로 단일 target으로 바꾸면 같은 질량 배분을 유지한다는 보장이 없다. 2D8에서는 이 단계만으로 Q 평균이−.0101→.2848로 변한다.

Ver2의 Sinkhorn 반복을 학습 없이30→300으로 바꾸어 재계산한 hard target Q는8D .464575→.464689,4D .139724→.139673,2D8 .284780→.284816이다. 이 checkpoint들에서는 단순 Sinkhorn 미수렴이 주요 설명과 맞지 않는다. 이는 epsilon을 바꾼 실험이나300회로 재학습한 결과가 아니다.

8D ver2는 후보 평균도 아직 기준과.1775 정도 차이가 있으므로 proposal support/finite-sample 오차가 완전히 해결됐다고 할 수 없다. Ver1과ver2는 N뿐 아니라 anchor도 다르므로 이 개선을 N 하나의 인과 효과로 단정하지 않는다.

## 4. 랜덤 target의 평균화 진단

Ver2의 actor와2048개 latent 전체를 고정하고, proposal cloud만64번 다시 뽑았다. 각 latent에는 서로 다른 row-argmax target64개가 생긴다. 개별 target을 Q로 평가한 평균과, 각 latent별 target 행동을 먼저 평균낸 뒤 Q로 평가한 값을 비교했다.

| 문제 | 랜덤 target들의 평균 Q | 각 latent의 평균 target을 평가한 Q 평균 | 실제 actor Q |
|---|---:|---:|---:|
| 8D | .481674 | 1.777068 | 1.763852 |
| 4D | .134836 | .345602 | .353655 |
| 2D8 | .288525 | .297146 | .295901 |

**8D에서 actor의 Q 평균은 개별 랜덤 target보다 평균 target의 Q와 거의 같다.** 고정된 actor/latent cloud에 대해 MSE를 다음처럼 분해했다.

\[
\mathbb E_\xi\|g_\theta(z)-Y(z,\xi)\|^2
=\|g_\theta(z)-\mathbb E_\xi Y(z,\xi)\|^2
+\mathbb E_\xi\|Y(z,\xi)-\mathbb E_\xi Y(z,\xi)\|^2.
\]

8D의 전체 MSE .504173 중 target 조건부 변동이 .476612(94.53%), 평균 target과의 잔차가 .027562다. 좌표당 평균 target 대비 RMSE는.058696이다. 4D에서도 target 변동이 MSE의89.19%다.

높은 Q를 가진 봉우리 주변으로 랜덤 target이 흩어져 있을 때, 그 행동들을 평균내면 봉우리 중심에 가까워질 수 있다. Q가 비선형이므로 **Q의 평균과 평균 행동의 Q는 다르다.** Stop-gradient target을 MSE로 회귀하는 현재 update는 각 latent에서 조건부 평균 방향으로 학습한다. 위 수치는 이러한 평균화가 mode 내부 분포를 좁히는 설명과 강하게 일치한다.

다만 이 진단은 전체 source cloud까지 고정하고 proposal randomness만 분리했다. 실제 학습은 source cloud와 actor도 변한다. 따라서 고정 checkpoint에서의 평균화 현상은 확인했지만, 학습 경로 전체의 유일한 원인을 증명한 것은 아니다. MSE나 row-argmax가 언제나 실패한다는 주장도 아니다. GMM40 성공 설정처럼 target 대응이 더 안정적인 조건에서는 같은 구조가 잘 작동할 수 있다.

## 자료

- [Mode 오차 분해](results/decomposition.json), [단계별 평균과 표준편차·parity](results/stages.json), [64회 조건부 target 진단](results/conditional_targets.json).
- [4D/8D 전체 그림](results/within_mode.png), [8D 그림 PDF](results/8d_density.pdf).
- [샘플 분해 코드](decompose.py), [단계별 진단 코드](checkpoint_stages.py), [조건부 target 진단 코드](conditional_targets.py).
- 원격 CPU jobs2204057/2204100. 진행 중인GPU 학습과 분리해 실행했다.
