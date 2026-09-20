# GMM40 v7: 100K 실행과 epsilon 비교

사용자 요청에 따라 T=alpha=1로 seed 0·1·2·3을 각각 GPU 0·1·2·3에서
처음부터 100,000 actor 업데이트했고 네 seed 모두 완료했다. 이후 epsilon 0.01/0.001을
각각 seed 0/1로 100K 실행했으며 이 네 실험도 완료했다. 초기 fresh-Sinkhorn 실행은 solver 복원 요청에
따라 checkpoint를 남기고 중단했고, 현재 기본값은 persistent dual 1회 갱신이다.
각 seed는 독립 실험이며 GPU 4개를
하나의 학습 배치로 합치지 않는다. 고정 Q는 원래 GMM40의 log density다.
표기는 [공통 표기](NOTATION_KO.md)를 따른다. $P_{ij}$는 OT 행렬,
$W_j$는 teacher 가중치, $R_{ij}$는 행 조건부다.

## 공통 설정

| 항목 | 값 |
|---|---|
| Latent 적분점 | H=4096, seed별 고정 표준정규 좌표 |
| Teacher 생성 | 새 latent 256개, 각 Gaussian에서 행동 1개 |
| 밀도 보정 | 256개 전체 혼합물 q로 $W_j=\operatorname{softmax}_j(Q(s,b_j)/T-\log q(b_j\mid s))$ |
| OT 전 중요도 재표집 | 256→16, 중복 유지 |
| OT | 상태별 4096×16, epsilon=.1, persistent dual Adam 1회 |
| Dual | ReLU 256×2 → 4096, 0 출력 초기화, Adam 1e-4 |
| Actor 학습 | 선택된 latent 16개에서 새 행동 생성, conditional SAC loss |
| Teacher / actor 온도 | T=alpha=1, 고정 |
| Batch / actor MLP / Adam | 256 / 256×2 / 3e-4 |
| Actor sigma | 초기 .5, log sigma 범위 [-5,1] |
| Teacher sigma 하한 | .05 |
| 업데이트 예산 | seed마다 100,000회 |

이론과 전체 의사코드는 [ALGORITHM_KO.md](ALGORITHM_KO.md)에 있다.
GMM40은 고정 Q 검증이라 critic/TD 학습을 하지 않는다. T=alpha=1은 teacher
Boltzmann 가중치와 actor의 조건부 entropy·assignment 항에 함께 적용한다.

## 비교군과 중단 기록

초기 fresh-Sinkhorn 캠페인은
`/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-100k-20260919`다.
중단 이유는 `solver_revert_requested`이며 100K 완주로 표시하지 않는다.
seed0/1/2/3의 최종 저장 step은 각각 6500/7100/7300/5200이고 actor·Adam·RNG를
보존했다. 해당 디렉터리의 `stop_audit.json`에서 PID 종료와 저장 카운터를 확인한다.

복원한 새 캠페인은
`/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-persistent-100k-20260919`다.
seed마다 초기 상태에서 다시 시작하여 solver를 섞지 않는다. Teacher와 actor는
같은 pre-update potential을 사용하고 dual parameter·Adam만 다음 step으로 이어간다.

## 기존 OptiQ의 속도 기준

기존 실험의 저장된 runtime을 사용한다. Baseline 코드는 복사만 하며 새 baseline
학습을 실행하지 않는다. 아래 시간은 compilation을 제외한 warm update 중앙값이며,
평가·그림·checkpoint 저장 시간도 제외한다. 모두 batch256, MLP256×2였다.

| 구현 | OT와 학습 차이 | ms/update | 100K 학습 시간, compilation 포함 |
|---|---|---:|---:|
| 이전 latent OptiQ | H4096·(256→16), joint NLL, epsilon .001, 지속 dual NN 1회 업데이트 | 2.038 | 210.5초 |
| v7, 사전 200회 검증 | H4096·(256→16), conditional SAC, epsilon .1, fresh Sinkhorn 100회 | 18.995 | 8.63초, 200회만 |
| v7, GPU3 warm 재측정 | 같은 v7 설정, 50-update block 중앙값 | 19.289 | 약32.1분 예상, 평가·저장 제외 |
| v7, persistent 복원 GPU 사전 측정 | H4096·(256→16), dual Adam 1회, conditional SAC | 1.897 | 약3.2분 예상, 평가·저장 제외 |

참조 파일:

- `/root/OptiQ-v6/gmm40-results/v7_joint_nn_preot_m256_k16_h4096_e0001_s0_100k/runtime.json`
- `/root/optiq-experiments/v7/gmm40-validation/oneperlatent256-h4096-k16-s0-200/runtime.json`
- `/root/optiq-experiments/v7/speed-audit/current_stages.json`

비교 기준은 이전의 **같은 4096×16 OT-NLL**이다. 초기 fresh-Sinkhorn v7은 약9.3–9.5배 느렸다.
OT 행렬 크기와 teacher 생성 개수는 같아서 이것으로 차이를 설명할 수 없다.
이전에는 한 GMM 상태에 대한 dual network·Adam을 유지하고 analytical mass
gradient로 한 번 갱신했다. 초기 v7은 배치의 각 lane에서 매번 초기화하여
Sinkhorn을 100회 반복했다. Teacher는 두 구현 모두 매번 새로 만든다.
이 solver 변경이 주요 비용 차이다. 따라서 actor loss 하나만 바꾼 비교는 아니다.

초기 fresh-Sinkhorn v7을 GPU3에서 warm-up 후 각 stage를 20회 측정했다. Stage별 JIT fusion 경계와
동기화가 다르므로 아래 합이 통합 update 시간과 정확히 같지는 않다.

| 초기 fresh-Sinkhorn v7의 단계 | 중앙값 ms |
|---|---:|
| 후보256개 teacher 생성·전체 mixture 밀도·중요도 재표집 | 0.868 |
| 4096×16 OT 비용·Sinkhorn 100회 | 15.586 |
| Teacher·OT·source pair 구성 전체 | 17.979 |
| 새 행동의 actor gradient·Adam | 1.065 |
| 통합 update, 50회 block 측정 | 19.289 |

분리 측정상 Sinkhorn 비용이 통합 update의 약81%에 해당한다. Conditional SAC
loss가 필연적으로 9배 무거운 것이 아니라 v7 구현에서 solver를 바꾼 영향이 크다.
측정 스크립트는 `/root/optiq-experiments/v7/speed-audit/profile_current.py`다.
Stage 측정은 별도의 임시 actor를 사용했으며 캠페인 모델·RNG·업데이트 수에 포함되지 않는다.
4-GPU 본 학습의 warm timing과 진행 상황은 별도 campaign runtime에 기록한다.

## Persistent 복원 검증과 관측된 주의사항

CPU에서 adapter와 공유 actor·dual core가 동일하게 갱신되고 actor/dual/각 Adam/RNG의
checkpoint 재개 오차가 0임을 확인했다. GPU에서는 batch256으로 50회 compile block과
50회 warm block을 실행했다. Actor·dual·두 Adam count=100, 모든 기록된 값이
유한했고 warm 시간은1.897ms/update였다. 이 사전 측정은 캠페인 학습에 섞지 않는다.
당시 비교는 약10배의 속도 회복이며 여러 block의 본 학습 중앙값으로 재확인한다.

속도 회복과 별개로 source importance 분산은 중요하다. GPU 사전 검증100회 후
per-lane source TV는 .418, GMM batch 평균 TV는 약.051이었다. 마지막 별도 map
진단의 선택 weight 최대는 약175.6, full-map log 2차 모멘트는83.67이었다.
Block 기준 ESS fraction은 약.56이었다. Finite라는 사실만으로 안정적 estimator라고
판정하면 안 된다. Persistent dual은 매16개 teacher의 균형을 보장하지 않으며,
일부 source의 작은 행 질량이 $(1/H)/(\sum_jP_{ij})$를 크게 만든다. Clipping이나 자기 정규화는 추가하지
않았고, source TV·최소 log-mass·importance 2차 모멘트와 학습 결과를 함께 기록한다.

GPU preflight와 hash 기록은 새 캠페인의 `preflight_gpu.json`에 있다.
현재 수치는 GMM의 단일 상태를 공유한 경우다. RL은 상태별 dual forward가 필요하므로
GMM의1.9ms를 그대로 RL 전체 업데이트 속도로 해석하면 안 된다.

## 본 실행 시작 확인

GPU0/1/2/3에서 각각 seed0/1/2/3 worker를 시작했다. 최초 시작 확인 시
네 run 모두10K 업데이트 이상 진행했고 각 runtime의 warm 중앙값은
1.792/1.788/1.823/1.828ms/update였다. 초기 fresh-Sinkhorn 실행의
19.015/18.234/18.780/18.857ms 대비 약10배 빠르다. 이전4096×16 OT-NLL의
2.04ms와 같은 수준이며, 실행 시점 차이가 있어 작은 차이의 우열은 주장하지 않는다.

최신 상태는 새 campaign의 `jobs.json`, 각 `seed0`–`seed3`의 `status.json`,
`runtime.json`, `training.jsonl`에서 확인한다. 이 문단은 시작 확인 당시 기록이며
100K 완료 보고가 아니다. 학습 중간·최종 checkpoint와 그림은 run별로 저장한다.

## 100K 완료 결과와 epsilon 비교

아래는 모든 run의 저장된 100K 평가다. 온도 $T=\alpha=1$, H4096·M256→K16,
actor 16쌍, persistent dual과 추가 student 표집 보정을 유지했다. 동일한 seed에서
바뀐 scientific 설정은 OT epsilon뿐이며 학습 core source hash도 같다.
평가 표본은 run마다 10,000개이며 참조 분포와 평가 기준을 바꾸지 않았다.

| epsilon | seed | 모드 근처 비율 | 복원 component /40 | MMD²↓ | SW₂↓ |
|---:|---:|---:|---:|---:|---:|
|0.1|0|77.56%|27|0.02849|9.243|
|0.1|1|87.00%|22|0.03785|9.803|
|0.01|0|70.92%|20|0.04423|10.653|
|0.01|1|74.34%|18|0.06515|14.649|
|0.001|0|74.63%|15|0.11620|18.091|
|0.001|1|73.98%|13|0.09685|12.936|

epsilon 0.1의 추가 seed 2/3은 각각 near 81.27%/81.64%, coverage 27/22,
MMD² 0.02773/0.05276이다. epsilon 비교의 집계는 양쪽 모두 존재하는 seed 0/1로 제한한다.
Coverage는 40개 GMM component center에 대한 충분한 3σ 질량의 개수이며,
연속 density의 정확한 local maximum 개수는 아니다. Near만으로 전체 분포 복원
성공을 판정하지 않는다. 이 두 seed에서는 작은epsilon이 coverage와MMD²를 모두 악화시켰다.

추가 실험의 경로는
`/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-eps001-eps0001-20260919T224120Z`다.
GPU 0/1은 epsilon 0.01의 seed 0/1, GPU 2/3은 epsilon 0.001의 seed 0/1을 담당했다.
네 run 모두 100K actor·dual·각 Adam을 저장했다. Warm 속도는 약 1.79–1.83ms/update,
compile 포함 순수 학습 시간은 약 185–189초다. 평가·그림·저장 시간은 이 속도에서 제외된다.

[학습곡선](/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-eps001-eps0001-20260919T224120Z/comparison100k/learning_curves.png),
[100K 동일 축 샘플 비교](/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-eps001-eps0001-20260919T224120Z/comparison100k/samples_100k.png),
[상세 지표와 재현 기록](/root/optiq-experiments/v7/gmm40-training/P256-H4096-K16-T1-eps001-eps0001-20260919T224120Z/comparison100k/REPORT.md)을 저장했다.

## 작은 epsilon의 보정 진단

70K 초과 90K 이하 20K 업데이트에서 추가 student 표집 보정계수
$(1/H)/(\sum_jP_{ij})$의 관측 평균은 다음과 같다. 기대값 항등식의 이상적 평균은 1이다.

| epsilon | seed0 | seed1 |
|---:|---:|---:|
|0.1|0.9729|0.9745|
|0.01|0.8185|0.7815|
|0.001|0.2386|0.2252|

현재 persistent potential은 전체 teacher 모집단의 균형을 추적하며 이번 16개
teacher의 모든 행 질량을 매번 맞추지 않는다. 작은 epsilon은 배정을 집중시켜
많은 $\sum_jP_{ij}$를 극도로 작게 만든다. 드물게 선택되는 행을 큰 inverse weight로
보상해야 하므로 수식이 맞아도 유한 표본에서 심한 분산이 발생한다.
90K 체크포인트의 별도 CPU 진단 32개 lane에서 epsilon 0.001은 prior 질량의 52% 이상이
$\sum_jP_{ij}<10^{-20}$인 행에 있었다. 어떤 행이 작은지는 teacher마다 바뀔 수 있다.
관측 평균만으로 이론적 estimator의 bias를 증명하는 것은 아니다.

작은 epsilon에서는 학습된 sigma가 감소했지만 coverage가 감소했다.
마지막 100개 업데이트에서 관측한 pre-tanh sigma의 두 seed 평균은
epsilon 0.1/0.01/0.001에서 각각 0.07242/0.03156/0.01041이다.
이는 학습에 선택된 latent의 평균이며, 전체 prior의 policy 분산이나 물리 좌표의
표준편차가 아니다. 하한은 모든 실험에서 $\exp(-5)=0.006738$로 같다.

조건부를 선명하게 만드는 효과만으로 전체 분포 복원이 개선되지 않았다.
표집 보정의 분산, population 배정 오차, 가파른 조건부 목표를 단일 Gaussian으로
학습하는 문제가 함께 바뀌므로 한 요인만의 인과 효과로 확정하지 않는다.
기존 teacher $W_j$의 proposal density 계산 오류가 발견된 것은 아니다.

기존 OT-NLL과 현재 actor에서 epsilon이 작용하는 경로도 다르다. 이전에는
$P_{ij}$로 매칭한 teacher 좌표 자체를 회귀했다. 현재는 새 actor 행동의
$-T\log\Pr(i\mid a,s)$까지 미분하므로, epsilon이 배정뿐 아니라 actor가 받는
영역 제약의 가파름에도 영향을 준다. 이전 NLL에서 좋았던 epsilon0.001이
현재 conditional SAC에서도 같은 효과를 낸다고 가정할 수 없다.

학습곡선에서도 단순히 학습이 느린 것만으로는 설명하기 어렵다.
epsilon0.001 seed0은25K에 coverage26/40, near51.31%였지만100K에는15/40,
74.63%가 됐다. 이미 잡은 일부 중심에 집중하는 동안 다른 component를 놓쳤다.
epsilon0.01 seed1도50K의coverage24/40·MMD²0.02700에서100K의18/40·0.06515로
악화했다. 이는 작은epsilon이 모든시점·모든seed에서 단조롭게나쁘다는뜻은 아니다.

재현 자료는 위 캠페인의 `density_correction_audit/summary.json`,
`support_probe.py`, `support_probe.json`이며 학습 체크포인트와 RNG를 바꾸지 않았다.
