# Legacy row-argmax와 1D empirical Monge 비교

## 질문과 정확한 크기

기존 actor는 `a=clip(G(s,z),-1,1)`인 deterministic implicit actor다.
Row별 최대 OT 항목을 독립적으로 선택하면 teacher의 질량을 잃을 수 있다.
이 선택을 전역적으로 최적인, 각 source particle당 하나의 target을 지정하는
matching으로 바꾸면 teacher 질량과 학습 분포를 더 잘 보존하는지 확인한다.

**Source 256 / proposal candidates 1,024 / Monge target representatives 256.**
1:1 Monge 문제 자체는 **256×256**이다. `256×1024 Monge`라고 부르지 않는다.
현재 pilot은 1D 두 문제 × 세 방법 × seed 0,1 = **12 runs**다.
기존 대규모 non-stationary 큐와 독립적으로 실행한다.

## 원래 weighted teacher에 정확한 Monge를 적용하지 못하는 이유

Source는 각 질량이 1/N인 N개의 표본이고, teacher는 다음과 같다.

\[
\mu_N=\frac1N\sum_i\delta_{x_i},\qquad
\widehat\nu_M=\sum_j w_j\delta_{b_j},\qquad
w_j=\operatorname{softmax}_j(Q(b_j)/0.25-\log q_F(b_j)).
\]

Source 원자 하나를 쪼개지 않는 map이면 도착 질량은 1/N의 정수배다.
일반적인 importance weights는 이 조건을 만족하지 않는다. M>N인 teacher의
모든 candidate가 양의 질량을 가지는 경우에는 더욱 불가능하다.
N=M이라도 arbitrary weights 그대로라면 단순 bijection으로는 맞출 수 없다.

따라서 weighted teacher의 CDF F에서 midpoint quantile 대표점을 추출한다.

\[
\widetilde b_k=F^{-1}((k-1/2)/N),\quad
\widetilde\nu_N=\frac1N\sum_{k=1}^N\delta_{\widetilde b_k}.
\]

대표점은 기존 candidate 값이며 중복될 수 있다. 원래 teacher와 대표점
teacher의 CDF 최대 오차는 정확한 산술에서 1/(2N) 이하이다. 이는 원자별
TV가 작다는 뜻이 아니다. 원자별 TV, CDF 오차, Wasserstein-1, mode 질량
오차를 따로 기록한다. N=256이면 CDF 한도는 약 0.001953이다.

각 source particle과 대표점을 정렬해 같은 순위끼리 짝짓는다.

\[
\sigma^*\in\arg\min_{\sigma\in S_N}
\frac1N\sum_i|x_i-\widetilde b_{\sigma(i)}|^2.
\]

이는 1D squared cost에서 **대표점 teacher에 대한 정확한 N×N assignment**다.
원래 weighted teacher나 population actor distribution에 대한 정확한 Monge
solver라고 주장하지 않는다. Action clipping으로 source 값이 겹치면 latent
표본의 index로 particle을 구분하고 stable sorting으로 tie를 처리한다.
그 경우 순수 action 값만의 single-valued map과 구분하며 tie 비율을 기록한다.
다차원에 단순 정렬을 적용하지 않는다.

## 세 조건

| 이름 | Plan | 회귀 target |
|---|---|---|
| Legacy Sinkhorn row-argmax | 원래 weighted teacher, epsilon .05, 30 iterations | 각 row의 argmax candidate |
| Exact OT row-argmax | 원래 weighted teacher, 1D exact Kantorovich coupling | 각 row의 argmax candidate |
| Quantile Monge | 256개 equal-weight 대표점에 대한 exact bijection | 매칭된 대표점 |

Exact OT는 source mass splitting을 허용한다. 이후 argmax를 취하면 column
marginal 보장은 사라진다. 이 조건을 넣어 Sinkhorn 근사와 hard selection을
구분한다. 세 번째 조건은 teacher 양자화까지 포함한 알고리즘 변경이다.
Solver만 바꾼 완벽한 단일 요인 비교라고 부르지 않는다.

별도로 한 번 뽑은 **동일 source/candidate/weight**에서 세 assignment와
`같은 양자화 teacher에 Sinkhorn을 적용한 row-argmax`를 모두 계산한다.
이를 통해 teacher를 양자화한 효과와 matching 선택의 차이를 구분한다.
Baseline action cloud를 사용하는 counterfactual 그림은 그 출처를 명시한다.

## 유지한 코드와 설정

읽기 전용 base: `a08517ef9ed5fb8743252132997638b00feb5d75`, source code ID
`2c5908035f33763f536109bf34158f7413c34a69088eb3a30abf3f86ba91ecf8`.
`legacy_optiq.algorithm.OptiQDIME.update_actor`를 그대로 복사하고 assignment
분기 두 개만 추가했다. v5 Gaussian actor에 끼워 넣은 버전이 아니다.

| 항목 | 값 |
|---|---|
| Actor | legacy implicit, hidden 256×3, GELU, 1D standard-normal latent |
| State | 항상 0, actor state batch 1 |
| Source N | 256 |
| Proposal M | 1,024: center당 random 4, anchor 없음 |
| KDE | 기존 truncated Gaussian, std .2, local clip .5, action bounds [-1,1] |
| Importance correction | 기존 KDE density, beta 1, adaptive beta off |
| Temperature | .25 |
| Cost | 원래 mean-normalized squared action distance |
| Sinkhorn | epsilon .05, 30 iterations; convergence residual 명시 |
| Loss | raw G(s,z)와 선택한 action target 사이의 MSE |
| Optimizer | Adam 3e-4, EMA/history 추가 없음 |
| Duration | 35,000 actor updates, 처음 20,000은 stationary prefix |
| Seeds | 0,1: 탐색적 pilot이며 유의성 주장 없음 |

## 두 Q 환경

\[
Q_t(a)=0.25\log\sum_k\rho_{kt}\mathcal N(a;c_{kt},h^2),
\quad a\in[-1,1].
\]

정답은 이 mixture를 [-1,1]에서 정규화한 분포다. 기존 schedule을 재사용한다.

- **double_mass:** centers (-.65,.65), h=.12. 20K까지 50:50 → 25K까지
  80:20 → 30K까지 20:80 → 35K까지 50:50. Mode 위치는 변하지 않는다.
- **tri_split:** centers (-.6,0,.6), h=.1. 처음20K 세 mode;20K–22K에
  각 center를 ±.15까지 분리;25K까지 여섯 mode 유지;27K에 다시 합침;
  35K까지 세 mode 유지. Transition과 critic 학습은 없는 frozen analytic Q다.

각 조건은 같은 seed의 초기 actor/optimizer/RNG에서 출발한다. 첫 update의
source/candidate/weights가 같음을 검증한다. 이후에는 actor 변화로 cloud도
달라지는 실제 feedback을 허용한다. 초기 fitting 실패도 숨기지 않는다.

## 그림과 지표

- Actor density는 **32,768개 실제 action histogram**, 512 bins. KDE나
  Gaussian 적분으로 actor density를 만들지 않는다. Exact target만 CDF로 계산.
- Histogram TV: 같은 bins에서 target과 actor의 질량 차이 절댓값 합의 절반.
- Basin TV: 정답 density의 valley로 나눈 각 mode 영역의 질량 차이 절반.
  Mode 안의 폭 차이에 덜 민감하며 histogram TV와 다른 지표다.
- Teacher → selected target → actor의 단계별 mode 질량/CDF/W1 오차;
  backup bias = actor 평균 Q - 정답 Boltzmann 평균 Q.
- Change 전후 분포와 시간별 TV. 업데이트 수를 주축으로 한다. Wall clock은
  GPU 모델, compilation, 진단 비용을 함께 명시하고 speedup을 혼동하지 않는다.
- Plan 원본/정렬 heatmap. Monge는 실제 **256×256 bijection**을 별도 표시.
  비교용 256×1024 그림은 중복 대표점을 원래 candidate column으로 합친
  표현이며 원래 w를 만족하는 OT 행렬이 아님을 명시한다.
- Evaluation의 teacher는 해당 update 전, actor는 update 후이다. 독립 action
  표본으로 평가하며 proposal sample 수는 1,024임을 분명히 한다.

기본 density 평가는500updates마다, change 주변±100에서는20updates마다;
plan audit는1,20000,20001,22000,25001,27000,30001,35000에 저장한다.
Full checkpoint/optimizer/RNG는200updates마다 저장한다.

## 검증·실행·보관

Commit/push 후 immutable snapshot에서 검증한다. 독립 SciPy assignment와
Monge cost를 비교하고, exact OT는 작은 문제의 독립 linear program과 대조한다.
CDF bound, marginals, duplicate source/one-point teacher, baseline production
update parity, 초기 teacher 동일성, 실제20updates, checkpoint 재개 일치를 확인.
GPU validation 통과 후 12개 independent run을 최대4개 GPU에서 실행한다.
기존 대형 큐를 변경하거나 중단하지 않는다. 2시간 Slurm slice 내 checkpoint
종료를 지원하며, 필요시 같은 commit으로 재개한다.

Source·manifest·DEPLOYMENT·SUBMISSIONS·results 보관:
`login4:/lustre/hobbit9882/OptiQ-nonstationary-q-20260917/extensions/20260918_nd/monge_REV/`
→ 기존 read-only backup 경로를 통해
`dildata:/data1/heejoonorm/OptiQ/studies/20260918_nonstationary_nd/monge_REV/`.
실제 REV와 commit/job ID는 별도 실행 기록에 남긴다. Checkpoint와 sample은 Git에 넣지 않는다.

참고: [POT의 Monge/Kantorovich 및 1D 정렬 설명](https://pythonot.github.io/master/user_guide.html),
[SciPy independent assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
