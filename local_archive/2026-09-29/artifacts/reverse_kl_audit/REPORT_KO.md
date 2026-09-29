# IDAC 및 과거 reverse-KL 비교 실험 감사

확인일: 2026-09-22. 원격 origin 전체 브랜치의 관련 파일명과 git 기록을 검색하고, 특히 v2–v5, v8, v9, heejoon, direct-gmm-trg를 확인했다. 학습을 새로 실행하지 않았다.

## 결론과 증거 수준

- Semi-implicit actor의 one-step 행동 생성 자체는 IDAC와 공유한다. IDAC 원문 §3.1(a)는 평가 시 latent를 뽑고 조건부 평균을 출력한다고 명시한다. §3.1(b), Figure 2에는 multimodal action 분포 시각화도 있다. 따라서 IDAC가 이를 못 한다거나 관련 실험이 없다는 설명은 수정해야 한다. 동일한 구조적 계산 단계와 실제 측정 latency의 동일성은 별개다.
- `origin/v8`의 commit `e08003e4ba52be76e0110d38f48bd9bf44d6eb83`에는 서로 다른 두 실험의 기록이 함께 있다. 현재 소스는 conditional SAC+OT이고, marginal SAC는 중단·복원된 과거 실험이다.
- 100K에서 coverage 15/40, 13/40을 기록한 실험은 **conditional SAC+OT, epsilon=0.001**이다. marginal reverse-KL 또는 IDAC의 결과로 바꿔 부르면 안 된다. 이 수치가 사용자가 언급한 실험과 같은지는 아직 확인되지 않았다.
- marginal SAC의 구현 설명과 검증 기록은 찾았지만 당시 실행 소스 archive와 결과 원자료는 접근한 두 현재 서버의 기록된 경로에서 찾지 못했다. 따라서 문서에 적힌 구현을 설명할 수는 있지만, 실제 해당 run의 coverage를 확인했다고 주장할 수 없다.

## 1. IDAC의 entropy

정책은 pi(a|s)=E_z[pi(a|s,z)]. 행동 생성에 사용한 z0와 독립 보조 latent z1,...,zL을 포함하여

    log_pi_est = log[(pi(a|s,z0) + sum_{l=1}^L pi(a|s,zl))/(L+1)]
    a ~ pi(.|s,z0)
    loss = E[alpha * log_pi_est - Q(s,a)]

로 학습한다. 단일 conditional entropy가 아니다. 생성 component를 포함하는 유한-mixture 추정량이며, 기대 negative entropy의 상한, 즉 entropy의 하한을 제공한다. 연속-mixture entropy의 유한 표본 불편 추정량이라고 표현하면 안 된다.

출처: [IDAC 원문](https://papers.nips.cc/paper/2020/file/4f20f7f5d2e7a1b640ebc8244428558c-Paper.pdf), §2.4–2.5, Eq. (11)–(16), Algorithm 1, §3.1.

우리 저장소의 `idac_action_and_log_density`라는 이름만으로 IDAC actor 학습을 했다고 판단할 수 없다. 해당 함수가 critic soft-TD나 진단용으로 쓰이는 경로도 있기 때문이다.

## 2. 코드까지 확인한 conditional SAC+OT

파일: `v8_e08003e/optiq_dime/conditional_sac.py`.

- 31–35행: 생성에 사용한 단일 Gaussian의 log density에 tanh Jacobian과 물리 좌표 action_scale Jacobian을 반영.
- 148행: source 표집 보정계수는 stop-gradient.
- 166–188행: 현재 actor에서 새 행동을 재매개화하고, 조건부 density, Q, OT 배정함수를 새 행동에서 계산.
- 231행: loss에 autodiff를 적용한다. OT potential은 고정하지만 Q와 배정함수의 action 경로는 미분한다.

    c_i = (1/H) / sum_j P_ij
    loss = mean(c_i * [T log pi_theta(a|s,z_i) - Q(s,a) - T log r_OT(i|a,s)])

entropy 항은 conditional entropy다. `r_OT`는 현재 actor의 Bayes posterior와 동일하다고 보장되지 않는다. 따라서 assignment 항을 더했다고 marginal entropy가 정확히 되는 것은 아니다.

uniform finite-bank 정책에 대한 기대 목적함수는

    J_conditional = J_marginal_SAC
                    + T E_{a~pi_H} KL(actor_posterior(i|a) || r_OT(i|a))
                    + T log H

로 분해된다. 즉 pure marginal reverse KL에 추가 조건이 붙는다. `docs/v7/ALGORITHM_KO.md` 277행부터 동일한 분해가 기록되어 있다.

`gmm40/v7.py`에서 actor는 hidden 256×2, initial sigma .5, log-sigma [-5,1], mean_output_init_scale=1e-4다. H4096, teacher256→16, actor16쌍, batch256, T=1이다. 평가 표본은 fresh continuous latent와 Gaussian noise로 생성한다.

`docs/v7/GMM100K_KO.md`의 저장된 결과 표:

| OT epsilon | seed 0 coverage | seed 1 coverage |
|---|---:|---:|
| 0.1 | 27/40 | 22/40 |
| 0.01 | 20/40 | 18/40 |
| 0.001 | 15/40 | 13/40 |

이는 문서에 기록된 100K 결과이며 이번 감사에서 checkpoint를 다시 평가하지 않았다. Coverage는 40개 target component 중심의 3-sigma 영역에 충분한 표본 질량이 있는지 세는 기준이다. 연속 density의 엄밀한 local maximum 개수가 아니다.

같은 문서에는 작은 epsilon에서 source 보정의 심한 분산이 기록되어 있다. epsilon=.001에서 보정계수의 관측 평균은 약 .239/.225이고, 별도 진단에서는 prior 질량의 52% 이상이 empirical source mass <1e-20인 행에 있었다. 따라서 모드 손실을 KL 방향의 효과 하나로 귀속할 수 없다.

## 3. 문서로 확인한 marginal SAC 실험

출처: `v8_e08003e/docs/v7/MARGINAL_SAC_VALIDATION_KO.md`.

    pi_H(a|s) = (1/4096) sum_i pi_theta(a|s,z_i)
    entropy_est = -mean(c_i * log pi_H(a|s))
    actor_loss = mean(c_i * [T log pi_H(a|s) - Q(s,a)])

고정 normal latent bank의 4096개 current actor density를 모두 합산한다. OT로 선택된 16개만 합산하는 방식이 아니다. 별도의 assignment loss는 없다. Teacher sigma floor를 actor density에 적용하지 않는다. Tanh와 action scale Jacobian을 포함하고 행동 생성 및 전체 density에 autodiff를 적용했다고 기록되어 있다.

이 밀도는 고정 finite-bank 정책에 대해서는 명시적 밀도다. Entropy의 행동 적분은 여전히 표본 평균과 source 보정으로 추정한다. Continuous latent 정책의 정확한 entropy를 얻었다는 뜻은 아니다.

Teacher256→16, H4096×16 OT, actor16쌍, batch256, T1, epsilon.1. Seed0은83K, seed1–3은85K에서 중단했다고 기록되어 있다. 100K 완료 결과나 IDAC 공식 재현이라고 부를 수 없다.

## 4. 교수님께 설명할 차이와 남은 검증

One-step semi-implicit architecture는 기존과 공유한다고 인정해야 한다. 현재 주장 가능한 연구 방향은 같은 expressive actor를 importance-corrected Boltzmann teacher에 forward-KL 방식으로 적합하여 학습의 mode coverage를 개선하는 것이다. 이 학습법의 유효성은 검증할 가설이며 architecture 자체의 신규성 또는 모든 모드의 보장이 아니다.

Reverse KL은 현재 정책이 뽑는 행동을 중심으로 Q와 log density의 action gradient를 사용한다. Forward fitting은 proposal이 발견한 여러 target 영역의 행동에 보정된 가중치를 주고 log-likelihood를 학습시킨다. Proposal이 모드를 발견하지 못하거나 중요도 가중치가 퇴화하면 forward 방식도 모드를 놓칠 수 있다. 충분히 표현 가능한 이상적 모델에서 두 KL의 전역 최적점은 같은 target이다.

비교 실험에는 동일 actor, 초기화, conditional noise, 평가 정책(full policy 또는 mu-only), Q/온도, 학습 예산을 맞춘 reverse-KL 대조군이 필요하다. 특히 OT assignment/source correction을 포함한 과거 결과를 순수 KL 방향 ablation으로 해석하지 않는다. IDAC 공식 전체 알고리즘 성능 비교와 같은 actor의 학습목적 ablation도 구분한다.

GMM40에서 DIPO보다 좋은 특정 지표가 나오는 것은 유용한 empirical evidence지만, 그 하나로 diffusion보다 보편적으로 더 expressive하거나 multimodality가 우월하다고 결론내리지는 않는다.
