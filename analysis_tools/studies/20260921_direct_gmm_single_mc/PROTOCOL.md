# Direct GMM: single critic + 64-action Monte Carlo backup

2026-09-21 사용자 요청. Humanoid → HalfCheetah → Ant 우선순위, 각 seed0–3.
새로운 12개 run, 각1M environment steps. 기존 run은 재개·덮어쓰기하지 않는다.

## 정확한 변경

기존0917 DirectGMM의 v5 actor / teacher / marginal GMM NLL을 유지한다.
Critic의 수만1개로 하고, next-state policy action64개의 Q를 평균한다.
N=64 student latent, M=64 teacher candidate, K=64 TD action sample이다.
세 숫자가 같아도 student/teacher/TD 표본을 재사용하지 않는다.

\[
z'_{ik},\epsilon'_{ik}\overset{iid}{\sim}\mathcal N(0,I),\quad
a'_{ik}=\tanh(\mu_\theta(s'_i,z'_{ik})+
\sigma_\theta(s'_i,z'_{ik})\odot\epsilon'_{ik}),
\]
\[
y_i=\operatorname{sg}\left[r_i+\gamma(1-d_i)
\frac1{64}\sum_{k=1}^{64}Q_{\bar\phi}(s'_i,a'_{ik})\right],\qquad
L_Q=\frac1B\sum_i(Q_\phi(s_i,a_i)-y_i)^2.
\]

Single live critic과 single target critic (Polyak copy)이다. Twin min/max/평균,
entropy bonus, action maximization, TD smoothing은 사용하지 않는다.
Next action은 현재 actor의 full Gaussian policy다. Mean-only action이나
teacher proposal floor를 TD에 넣지 않는다. True terminal만 bootstrap을 끄며
기존 replay의 time-limit truncation 처리를 유지한다.

Actor teacher는 single live Q로 계산한다:
\[
w_j=\operatorname{softmax}_j(Q_\phi(s,b_j)/0.25-\log q_F(b_j\mid s)),\quad
L_\pi=-\sum_j w_j\log\left[\frac1{64}\sum_i k_\theta(b_j\mid s,z_i)\right].
\]
Teacher / weights / candidates는 stop-gradient. M=64 후보는 기존처럼64개
conditional Gaussian mixture에서 IID로 뽑으며 component당1개를 강제하지 않는다.

## Boltzmann 연결과 비교의 한계

\[
\mathbb E[\hat V(s')\mid\theta,\bar\phi]
=\mathbb E_{a\sim\pi_\theta}[Q_{\bar\phi}(s',a)].
\]
Actor가 \(\pi_{Q_\phi}^{\tau}\propto e^{Q_\phi/\tau}\)에 정확히 맞고
\(\bar\phi=\phi\)이면 Boltzmann expectation의 unbiased MC 추정이다.
실제로는 actor extraction error, live/target Q lag, 유한K sampling error가 남는다.
K=64는 sampling variance를 줄이나 actor bias나 target lag를 제거하지 않는다.
독립 표본의 조건부 variance는 \(\operatorname{Var}_{\pi}(Q)/K\)다.
기존 twin-min/K1 대비 critic 수와 K가 함께 바뀌므로 각각의 인과 기여를
분리하는 ablation으로 해석하지 않는다. Return 향상을 사전에 가정하지 않는다.

## 유지한 설정

| 항목 | 설정 |
|---|---|
| 환경 | Humanoid-v4, HalfCheetah-v4, Ant-v4 |
| Seeds / steps | 0,1,2,3 / 각1M, 총12개 |
| Actor / critic | 256×2 GELU / single256×2 GELU |
| Sigma | learned, initial0.5, log sigma[-5,1] |
| Proposal | conditional mixture, teacher-only sigma floor0.05, density beta1 |
| N / M / K / T | 64 /64 /64 /0.25 |
| Critic | scalar reward-only, gamma0.99, target Polyak0.005 |
| Update | batch256, UTD1, actor delay1, Adam3e-4, clipping 없음 |
| Warmup | 5K uniform, 이후 full stochastic actor |
| Eval | 기존 dual mean-only: zero-z / sampled-z, epsilon0, 5K마다 각10episodes |
| Checkpoint | 기존50K actor/critic/optimizer, replay/RNG 완전재개는 미지원 |
| W&B | OptiQ/DirectGMM_heejoon, 20260921_DirectGMM_SingleQ_MC64_N64_M64_T025 |

## 검증·실행

원본source ID fd918514023ae5ce20a91db53f19801549fd9d249e46e963b643d498834f5f28을
hash 검사하고 prepare.py가 별도 snapshot을 생성한다. 원본372개 파일은 수정하지 않는다.
새 snapshot의 단일 critic 지원과 MC update만 변경하며 manifest를 생성한다.
소스·prepare·config·test·launcher·본protocol을 heejoon에 commit/push한 뒤 실행한다.
실행commit, source manifest, 실제command, job ID, W&B ID를 각run에 기록한다.

K1/K64 analytic target, 실제optimizer gradient, terminal mask, actor/target Q
gradient 차단, full Gaussian draw, Polyak, 실제 _train 호출 경로를 검증한다.
세 환경의 batch256/N64/M64/K64 128-step GPU smoke와 finite actor/critic loss,
W&B online 재조회가 통과해야 본실험이 시작된다. Smoke는 별도validation group.

세 환경은 서로 의존하지 않는다. 공통 검증 gate 뒤 모든seed가 동시에 eligible이며
환경 우선순위만 적용한다. 중단/보류된 이전실험은 재개하지 않는다.
사용자가 재가동한 Vast 51591913 (31.148.50.247:11717)의 RTX3090 네 장을 사용한다.
기존 취소된 MuJoCo queue는 재개하지 않는다. GPU당2run, run당CPU2개를
배정하여 최대8run을 동시에 실행한다. 실제VRAM과초기진행을 확인한다.
추가 지표: backup sample Q std/MC standard error, K, critic_count, actor sigma,
teacher ESS, critic/actor loss와 기존return. Mean-only 평가와 full policy TD를 구분한다.

데이터·checkpoint·로그·credential은Git에 넣지 않으며 dildata중앙보관한다.
Remote: /home/heejoonorm/OptiQ/legacy_monge/direct_gmm_single_mc/COMMIT/.
기존 dildata collector의 legacy_monge 상대경로에서 중앙 보관한다.
