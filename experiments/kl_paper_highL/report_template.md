# Forward KL vs. Reverse KL: 두 target의 최종 결과와 score 수렴

작성일: 2026-09-26. 두 환경 모두 100K updates가 끝난 checkpoint를 사용했다. **Reverse 학습에서 score 추정에 사용하는 독립 latent MC sample 수는 $L=2^{20}$이며, 최종 density 평가도 별도로 seed당 $2^{20}$개 action을 사용한다.** 이 두 숫자는 역할이 다르다. Forward에는 이 추가 MC sample 수 $L$이 적용되지 않는다.

## 1. 최종 비교 그림

![최종 L20 비교](forward_vs_reverse_1x4_L20.png)

- (a,b): 중심 $(-4.25,0,4.25)$, 표준편차 $0.5$, 동일 질량의 세 Gaussian.
- (c,d): 왼쪽의 좁은 logistic spike와 오른쪽으로 올라가는 비대칭 ramp.
- 검정 점선: 정규화된 target density. 파란색: Forward KL. 주황색: Reverse KL.
- 굵은 곡선: seed 0–3의 histogram density 평균. 옅은 가는 선: 개별 seed. 채움은 density 아래 면적이며 신뢰구간이 아니다.
- 각 seed에서 실제 action **1,048,576개**, **512개의 동일 폭 bin**, KDE smoothing 없음. 패널마다 y축 범위가 다르다.

이 그림은 앞서 만든 $L=2^{10}$ 그림의 이름만 바꾼 것이 아니다. Reverse는 $L=2^{20}$으로 100K 학습한 checkpoint이다. Forward는 $L$을 사용하지 않으므로 기존 100K checkpoint를 재사용하고 동일한 $2^{20}$개 action으로 다시 평가했다. 각 환경·seed의 Forward와 Reverse 초기 actor parameter hash는 일치한다.

| Target | 방법 | Histogram TV, mean ± seed SD | Wasserstein-1, mean ± seed SD | Seed 0–3의 missing modes |
|---|---|---:|---:|---|
{{DENSITY_TABLE}}

TV와 $W_1$은 각각 **seed별 오차를 계산한 뒤 평균**했다. 평균 density의 오차를 계산한 값이 아니다. SD는 네 training seed 사이의 sample standard deviation이며 confidence interval은 아니다.

네 seed 모두에서 Forward는 두 target의 mode들을 복구했다. Reverse는 Gaussian의 양쪽 mode 두 개, spike+ramp의 왼쪽 spike를 놓쳤다. Forward에도 작은 fitting 오차와 골짜기의 과잉 질량이 남으므로 “완벽한 복구”라고 부르지는 않는다. 두 target은 이전 screening에서 선정한 설명용 사례이며, 모든 target이나 초기화에서의 우열을 의미하지 않는다.

## 2. 실험은 무엇을 통제했는가?

이 실험은 **1D frozen-Q actor fitting**이다. Critic을 학습하거나 환경 데이터를 수집하는 RL 실험은 아니다. 두 방법에 동일한 $Q(a)$와 동일한 actor 구조를 주고 actor objective를 바꾼다.

$$
Q(a)=\alpha\log f(a),\qquad
p^\star(a)=\frac{f(a)}{\int_{-10}^{10}f(x)\,dx},\qquad
\alpha=0.25.
$$

따라서 Boltzmann target은 $p^\star$이다. 학습에는 target sample이나 정답 mode label을 주지 않는다. Forward는 $Q(a)$, Reverse는 $Q(a)$와 그 action gradient를 사용한다. 정답 density와 mode 경계는 평가용이다.

### 2.1 Three Gaussian modes

$$
f_G(a)=\frac13\sum_{c\in\{-4.25,0,4.25\}}
\mathcal N(a;c,0.5^2),\qquad -10\le a\le10.
$$

이를 action 구간에서 정규화한다. 각 Gaussian의 **표준편차**가 $0.5$이며 variance는 $0.25$다. 이 설정은 이전의 중심 $(-5,0,5)$, 폭 1 실험과 다르다.

### 2.2 Non-Gaussian spike + ramp

$S(x)=(1+e^{-x})^{-1}$로 두면,

$$
u(a)=\frac1{0.1}
S\!\left(\frac{a+4.25}{0.1}\right)
S\!\left(-\frac{a+4.25}{0.1}\right),
$$

$$
v(a)=
S\!\left(\frac{a-0.5}{0.6}\right)
S\!\left(\frac{8-a}{0.08}\right)
\exp\!\left[0.2(a-4.25)\right].
$$

각 항을 $[-10,10]$에서 정규화하여 결합한다.

$$
Z_u=\int_{-10}^{10}u(a)\,da,\qquad
Z_v=\int_{-10}^{10}v(a)\,da,
$$

$$
p^\star(a)=0.2\frac{u(a)}{Z_u}+0.8\frac{v(a)}{Z_v}.
$$

왼쪽 spike는 중심 $-4.25$, logistic scale $0.1$이며 target 질량의 20%를 가진다. 오른쪽 ramp는 완만하게 올라오는 왼쪽 경계 $0.5$, 빠르게 내려가는 오른쪽 경계 $8$, 상승 기울기 계수 $0.2$를 가지며 질량은 80%다. $0.6$과 $0.08$은 두 경계의 부드러움을 정한다. 이 target은 GMM이 아니다. $Z_u,Z_v$는 알려진 target 함수를 적분한 값이며 actor marginal을 수치적분한 것이 아니다.

## 3. Actor와 공통 hyperparameter

$$
z\sim\mathcal N(0,1),\qquad
(\mu_\theta(z),\log\sigma_\theta(z))=G_\theta(0,z),
$$

$$
k_\theta(a\mid z)=
\frac{\phi((a-\mu_\theta(z))/\sigma_\theta(z))}
{\sigma_\theta(z)\left[
\Phi((10-\mu_\theta(z))/\sigma_\theta(z))-
\Phi((-10-\mu_\theta(z))/\sigma_\theta(z))
\right]},\qquad -10\le a\le10,
$$

$$
q_\theta(a)=\mathbb E_z[k_\theta(a\mid z)].
$$

$\phi,\Phi$는 표준정규분포의 density와 CDF다. Conditional distribution은 **box-truncated Gaussian**이다. Mean head에 tanh를 사용하지만, Gaussian action 자체에 tanh를 적용하는 squashed Gaussian은 아니다.

| 항목 | 실제 설정 |
|---|---|
| Action 범위 | $[-10,10]$ |
| State / latent | 고정 state 0 / 1D standard Normal latent |
| Hidden network | 256–256, GELU |
| Mean 출력 | $\mu=10\tanh(h_\mu)$ |
| Mean-head 초기화 | `variance_scaling(scale=3.0, mode=fan_avg, distribution=uniform)` |
| Log-sigma 출력 범위 | $[-5,-1]$, hard clip |
| Sigma 범위 | $[e^{-5},e^{-1}]\approx[0.006738,0.367879]$ |
| 초기 log-sigma | $-1$; log-sigma head kernel은 0으로 초기화 |
| Student latent 수 $N$ | 128 / group |
| Candidate 또는 action 수 $M$ | 128 / group |
| Batch | 독립 group 32개 / optimizer update |
| Temperature | $\alpha=0.25$ |
| Optimizer | Adam, learning rate $3\times10^{-4}$, 기본 momentum/epsilon |
| Updates | 100,000 |
| Seeds | 0, 1, 2, 3 |
| Score 추정용 MC sample 수 $L$ | $2^{20}=1,048,576$, group별 독립 latent samples |
| MC sample 처리 | chunk 4096; 매 update 독립 latent resampling |
| Forward proposal sigma floor | $e^{-5}$, actor sigma 하한과 동일 |
| 최종 평가 | seed당 $2^{20}$ action, 512 histogram bins |

여기서 batch 32는 동일 state에서 독립적인 candidate/latent 집합을 32번 생성한 뒤 gradient를 평균한다는 뜻이다. `mean_output_init_scale=3.0`은 **초기 weight 분산을 정하는 initializer 인자**이며 학습 내내 mean에 3을 곱한다는 뜻이 아니다.

## 4. 두 objective와 실제 gradient

### 4.1 Forward KL: SNIS + finite-mixture marginal NLL

이상적인 objective는

$$
\mathcal J_F(\theta)=\mathrm{KL}(p^\star\Vert q_\theta)
=-\mathbb E_{a\sim p^\star}\log q_\theta(a)+C.
$$

실제 group에서는 새 latent $z_{1:N}$를 뽑고 finite mixture를 만든다.

$$
q_{\theta,N}(a)=\frac1N\sum_{i=1}^{N}k_\theta(a\mid z_i).
$$

현재 actor의 parameter를 detach한 mixture $\tilde q_N$를 proposal로 사용한다. $M$개 candidate 각각에 대해 component index를 균등하게 뽑고 해당 truncated Gaussian에서 샘플링한다.

$$
b_j\sim\tilde q_N,\qquad
w_j=\frac{\exp[Q(b_j)/\alpha-\log\tilde q_N(b_j)]}
{\sum_{m=1}^{M}\exp[Q(b_m)/\alpha-\log\tilde q_N(b_m)]}.
$$

$$
\widehat{\mathcal L}_F(\theta)
=-\sum_{j=1}^{M}\operatorname{sg}(w_j)
\log q_{\theta,N}(\operatorname{sg}(b_j)).
$$

$\operatorname{sg}$는 stop-gradient다. Candidate, proposal, importance weight는 고정하고 student의 $\mu,\sigma$에 gradient를 보낸다. Proposal과 student는 해당 group에서 같은 latent 집합을 사용한다. 그룹 loss 32개를 평균한 뒤 한 번 update한다. 이는 SNIS와 finite-$N$ mixture를 이용한 근사이며 exact infinite-mixture Forward KL을 직접 적분하는 것이 아니다. OT/Sinkhorn은 사용하지 않는다.

### 4.2 Reverse KL: 독립 latent MC samples로 action score 추정

$$
\mathcal J_R(\theta)=\mathrm{KL}(q_\theta\Vert p^\star)
=\mathbb E_{a\sim q_\theta}\left[\log q_\theta(a)-Q(a)/\alpha\right]+C.
$$

Policy action은 group의 $N$개 conditional Gaussian에서 $M$개를 reparameterized sampling하여 얻는다. 별도의 독립 latent $\tilde z_{1:L}$로 score를 계산한다.

$$
\hat q_L(a)=\frac1L\sum_{\ell=1}^{L}k_\theta(a\mid\tilde z_\ell),
\qquad
\hat s_L(a)=\nabla_a\log\hat q_L(a).
$$

구현에서 backpropagation하는 surrogate는

$$
\widehat{\mathcal L}^{\mathrm{sur}}_R
=\frac1M\sum_{j=1}^{M}
a_{\theta,j}\operatorname{sg}\!\left[
\hat s_L(a_{\theta,j})-\frac1\alpha\nabla_a Q(a_{\theta,j})
\right].
$$

이 surrogate의 값 자체가 Reverse KL은 아니다. Action sampling 경로를 통해 다음 gradient를 만들기 위한 식이다.

$$
\widehat{\nabla_\theta\mathcal J_R}
=\frac1M\sum_j\left(\hat s_L(a_j)-\nabla_a\log p^\star(a_j)\right)
\nabla_\theta a_{\theta,j}.
$$

Score 추정에 사용한 conditional Gaussian parameter, score 계산에 들어간 action, bracket 전체는 detach한다. 미분은 reparameterized $a_{\theta,j}$를 통해서만 흐른다. Exact marginal density에서는 직접 parameter score 항의 expectation이 0인 score identity를 이용한 형태이며, 유한 $L$에서는 marginal score를 MC로 근사한 gradient이다. Group마다 독립적인 latent MC samples를 사용하므로 $L$을 키우면 추가 계산량이 증가한다. **같은 $N,M$이 같은 총 계산량을 뜻하지는 않는다.**

## 5. 왜 score 수렴을 따로 측정하는가?

$N,M$을 128로 고정해도 reverse gradient에 필요한 $\nabla_a\log q_\theta(a)$를 정확히 알 수는 없다. $L$은 학습 가능한 component 개수를 추가하는 값이 아니라, **같은 frozen neural actor가 만드는 marginal density의 score를 계산하기 위한 독립 latent MC sample 수**다.

Truncated Gaussian의 정규화 상수는 $\mu,\sigma$에는 의존하지만 구간 내부 action $a$에는 의존하지 않는다. 따라서

$$
\nabla_a\log k_\theta(a\mid z_\ell)
=\frac{\mu_\ell-a}{\sigma_\ell^2},
$$

$$
\boxed{
\hat s_L(a)=
\frac{\sum_{\ell=1}^{L}k_\theta(a\mid z_\ell)
(\mu_\ell-a)/\sigma_\ell^2}
{\sum_{\ell=1}^{L}k_\theta(a\mid z_\ell)}.
}
$$

이번에는 두 target의 **최종 Reverse checkpoint 8개를 고정**했다. Actor parameter 및 checkpoint hash가 측정 전후 동일함을 확인했다.

| 측정 항목 | 설정 |
|---|---|
| $L$ | $2^7,2^8,\ldots,2^{24}$ |
| 반복 | 독립 MC 반복 16회 |
| $L$ 사이 관계 | 각 반복 안에서는 같은 latent sequence의 prefix를 사용; 반복끼리는 독립 |
| 기준 score | 추가 독립 MC 추정 4회, 각각 latent sample $2^{24}$개; 네 score 추정값의 평균 |
| 주 그림 action | seed 0 policy sample의 10%, 50%, 90% 분위수, $L$과 무관하게 고정 |
| 정량 평가 action | 각 seed의 저장된 policy sample에서 등간격 index로 고른 128개 action |
| 추가 probe | Target 영역의 고정 action 9개, policy-action 통계와 분리 |
| Actor / latent | 학습과 같은 float32; MC latent의 network matmul은 highest precision |
| Ratio 누산 | log-weight 최대값을 빼서 안정화하고 float64로 누산 |

기준선은 exact score가 아니다. 각각 $2^{24}$개 latent samples를 사용하는 독립 MC 추정 네 회가 주는 **경험적 기준값**이다. Conditional Gaussian의 score 식은 정확하지만, latent expectation을 유한한 MC samples로 대체하는 오차는 남는다.

### 5.1 Score 수렴 그림: action별 확대

Figure의 x축은 **Number of MC samples $L$**로 표기한다. 이는 score 추정용 latent sample 수이며, objective gradient용 action 수 $M$이나 독립 반복 횟수 16과는 다른 값이다.

![Score 수렴 확대](score_convergence_zoom.png)

위 행은 Three Gaussian modes, 아래 행은 spike+ramp다. 모두 최종 Reverse actor의 seed 0을 사용한다. Forward score는 그리지 않는다.

파란 실선은 16회 MC 평균, **파란 음영은 반복값의 10–90% 구간**이다. Mean의 confidence interval은 아니다. 검정 점선은 독립 $2^{24}$-sample 기준 score, 세로 회색 파선은 실제 학습의 $L=2^{20}$이다. **회색 reference band는 그리지 않았다.** 각 패널의 y축은 해당 score 변화가 보이도록 독립적으로 확대했다.

| Target | Policy quantile | 고정 action | $L=2^{20}$ 평균 score | $2^{24}$-sample 기준 score | $L=2^{20}$ MC SD |
|---|---:|---:|---:|---:|---:|
{{QUANTILE_TABLE}}

### 5.2 공통 y축 범위 버전

![Score 수렴 공통 y축](score_convergence_common_y.png)

한 target의 세 action에 같은 y축 범위를 사용했다. 서로 다른 target 행끼리는 범위가 다르다. 확대 버전과 원자료는 동일하다.

### 5.3 네 seed의 정량 결과

128개의 고정 policy action에서 반복 $r$의 오차를

$$
E_{L,r}=
\sqrt{\frac1{128}\sum_{j=1}^{128}
\left(\hat s_{L,r}(a_j)-\bar s_{\mathrm{ref}}(a_j)\right)^2},
\qquad
S_{\mathrm{ref}}=
\sqrt{\frac1{128}\sum_j\bar s_{\mathrm{ref}}(a_j)^2}
$$

로 계산했다. 아래 relative RMSE는 $\frac1{16}\sum_r E_{L,r}/S_{\mathrm{ref}}$이다. Score가 0에 가까운 개별 action으로 나누는 pointwise relative error는 사용하지 않는다.

| Target | Seed | $L=2^{10}$ relative RMSE | $L=2^{20}$ relative RMSE | 기준 score의 SE RMS |
|---|---:|---:|---:|---:|
{{SCORE_TABLE}}

Gaussian에서는 $L=2^{20}$ 오차가 약 **0.11–0.19%**, spike+ramp에서는 약 **0.34–2.80%**였다. 따라서 Gaussian의 policy가 실제로 샘플링하는 영역에서는 매우 안정적이며, spike+ramp는 큰 $L$의 오차가 줄어들어도 일부 seed에서 수 %의 잔여 오차가 남는다. 두 환경 모두 “정확한 infinite-mixture gradient가 보장됐다”는 표현은 쓰지 않는다.

이 검사는 최종 checkpoint의 실제 policy 영역에서 score 근사 오차를 정량화한다. 학습 중 모든 시점이나 policy가 거의 방문하지 않는 영역까지 정확함을 보장하지 않는다. 특히 **사라진 mode의 낮은 policy density 영역은 더 큰 $L$에도 수렴이 느릴 수 있다.** $L$이 충분히 커 보인다는 사실만으로 mode missing의 원인을 KL objective 하나로 완전히 분리했다고 주장할 수는 없다.

## 6. 평가 지표와 추가 수치 검증

512 bin에서 empirical mass를 $\hat p_b$, target 적분 mass를 $p_b^\star$라 두면

$$
\widehat{\mathrm{TV}}_{512}=\frac12\sum_{b=1}^{512}|\hat p_b-p_b^\star|.
$$

이는 binning한 분포의 TV이며 continuous-density TV 자체를 정확히 계산한 값은 아니다. $W_1$은 1D CDF 차이로 계산했다.

$$
W_1=\int_{-10}^{10}|\hat F(a)-F^\star(a)|\,da.
$$

131,073개 grid point로 적분하고, 65,537개 grid 결과와 비교한 차이를 각 `metrics_100000.json`에 기록했다. Mode missing은 target valley로 나눈 basin 및 사전에 정한 core 양쪽에서 actor 질량이 target 질량의 25% 미만인 경우다. Gaussian core는 각 중심의 ±0.5, spike+ramp core는 $[-4.5,-4]$와 $[3,7.7]$이다.

Score 구현은 같은 유한 latent MC samples를 사용한 autodiff gradient와 일치함을 확인했고, near-boundary Gaussian을 포함해 truncation normalization도 검사했다. Float32/float64 ratio 계산의 차이도 같은 latent MC samples에서 확인했다. 다음 표에서 target probe 오차는 policy-action 오차와 **분리해서** 읽어야 한다.

| Target | Seed | 9개 target probe의 최대 절대 평균-score 편차, $L=2^{20}$ | 128 policy action의 최대 float32–float64 score 차이 |
|---|---:|---:|---:|
{{PROBE_TABLE}}

Gaussian target probe는 $(-4.75,-4.25,-3.75,-0.5,0,0.5,3.75,4.25,4.75)$, spike+ramp probe는 $(-4.45,-4.25,-4.05,0.5,2,4,6,7.5,8)$이다. Target probe의 기준 역시 exact 값이 아닌 $2^{24}$-sample MC 추정값이다.

## 7. 파일과 재현 정보

### 그림

- 최종 1×4: [PNG](forward_vs_reverse_1x4_L20.png) · [PDF](forward_vs_reverse_1x4_L20.pdf) · [SVG](forward_vs_reverse_1x4_L20.svg)
- Score 합본 확대: [PNG](score_convergence_zoom.png) · [PDF](score_convergence_zoom.pdf)
- Score 합본 공통 y축: [PNG](score_convergence_common_y.png) · [PDF](score_convergence_common_y.pdf)
- Gaussian만, 1×3: [확대 PDF](score_convergence_gaussian_zoom.pdf) · [공통 y축 PDF](score_convergence_gaussian_common_y.pdf)
- Spike+ramp만, 1×3: [확대 PDF](score_convergence_spike_ramp_zoom.pdf) · [공통 y축 PDF](score_convergence_spike_ramp_common_y.pdf)
- 모든 seed: [Gaussian](score_all_seeds_gaussian.png) · [Spike+ramp](score_all_seeds_spike_ramp.png)
- [논문용 Experiment details appendix](appendix.md) · [영문 figure captions](captions.md)
- [Density 원수치](density_metrics.csv) · [모든 L의 score 수치](score_metrics.csv) · [분위수 score 수치](score_quantiles.csv) · [전체 provenance](PROVENANCE.json)

### 코드와 저장 위치

- Git branch: `Yonsei-DILLab/OptiQ:heejoon`.
- Reverse high-L training commit: `41a020577dbf8a87d8f96d714ab3888a4bb82260`.
- Forward Gaussian training commit: `e89d570108decc3d40d44c9a8b23a6cd8debc49c`.
- Forward spike+ramp training commit: `55da33adca2d213b7847dd8eb6c5fa30825155ab`.
- 이번 score 측정 commit: `{{MEASUREMENT_COMMIT}}`; Slurm array `2336406`, index 0–7.
- 그림·보고서 생성 commit: `{{RENDER_COMMIT}}`.
- 로컬 보고서: `/Users/heejoon/Documents/ChatGPT/OptiQ/reports/20260926_kl_paper_highL/`.
- 로컬 입력·score 원자료: `/Users/heejoon/Documents/ChatGPT/OptiQ/studies/20260926_kl_paper_highL/`.
- 학습 원본: `login4:/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/kl_six_highL_20260925/runtime/confirm/`.
- Score 실행 원본: `login4:/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/kl_paper_highL_20260926/`.
- 보관 경로: `dildata:/data1/heejoonorm/OptiQ/studies/20260926_kl_paper_highL/`, `dildata:/data1/heejoonorm/OptiQ/reports/20260926_kl_paper_highL/`.

측정은 고정 checkpoint를 읽기만 했으며 원래 학습 코드·parameter·optimizer를 바꾸지 않았다. Source manifest, checkpoint SHA256, job ID, raw score와 figure input hash는 provenance와 study 폴더에 보관한다.
