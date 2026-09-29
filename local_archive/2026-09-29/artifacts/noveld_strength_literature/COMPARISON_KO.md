# Dense AntMaze: OptiQ, DDiffPG, MaxEntDP 보상/탐색 비교

2026-09-22 조사. 실행 중 NovelD 계수 실험은 변경하지 않았다.

## 확인한 소스

- 우리 학습: direct-gmm-trg frozen `262a10d6280eb6e79eec24e6170541a7b0f58e72`.
- DDiffPG 공개 코드: `7edd06c4799abbab0f8fa534c21deb56253b018e`.
- MaxEntDP 공개 코드: `8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077`.
- MaxEntDP 논문 https://arxiv.org/html/2502.11612v3 : Eq.1–3, §5.3, Appendix D.2.
- DDiffPG 논문 https://arxiv.org/html/2406.00681v1 : §4.1–4.3.
- NovelD 원논문 부록 https://openreview.net/attachment?id=CYUzpnOkFJp&name=supplementary_material.

## 현재 우리 설정

환경 보상은 `r_dense=-min_goal ||xy_next-goal||_2`.
Replay는 환경 보상만 저장하고 minibatch 추출 시 `lambda*max(n(next)-.5*n(current),0)`를 더한다.
기존 lambda=.01; 신규 네 실행만 .1,1,5,10. RND L2 error, normalize=false.
우리 Q는 clipped double Q지만 두 critic 모두 같은 합산 보상 목적을 학습한다.
경로별 Q나 NovelD 전용 Q는 없다.

현재 `backup_mode=td`, `n_atoms=1`: target은
`r_dense + r_intrinsic + gamma*(1-done)*min(Q_target(s_next,a_next))`.
`algorithm.py`의 entropy_adjustment는 이 분기에서 0이다.
Actor T=.25는 actor 목표 분포 쪽에 사용된다.
DACER는 행동 잡음만 조절하고 teacher/Bellman target에 entropy 항을 추가하지 않는다.
우리 density correction beta=1과 논문에서 temperature를 표기한 beta는 서로 다른 변수다.

## DDiffPG 구조를 dense로 옮길 경우

원래 논문 실험은 sparse이므로 아래는 dense 보상으로 바꾼 구조적 적용안이다.
공개 `ddiffpg.py:update_net`은 탐색 mode(i=0)에 intrinsic reward만 사용하고,
나머지 mode에는 환경 보상+intrinsic reward를 사용한다.
따라서 dense로 바꾸면 탐색 Q는 `r_intrinsic`, 각 경로 Q는 `r_dense+r_intrinsic`를 학습한다.
성공 궤적을 DTW 거리로 군집화하고 경로별 Q/target action을 유지하며 각 mode의 데이터를 나눈 minibatch로 정책을 학습한다.
탐색 embedding을 가진 행동을 수집하고 평가에서는 탐색 mode를 제외한다.
현재 우리 방식은 이 저장소 SAC baseline의 보상 합산 방식과 가깝다.
계수만 키워서는 DDiffPG의 mode 보존/탐색 분리 구조가 생기지 않는다.
원본 distributional Q의 양수 support를 dense 음수 보상에 그대로 가져오는 것도 부적절하다.

공식 코드:
https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/ddiffpg.py
https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/replay/diffusion_replay.py

## MaxEntDP와의 핵심 차이

Appendix D.2는 nearest-goal distance penalty를 dense 환경 보상으로 쓰고 1M 환경 interaction 뒤 궤적/coverage를 보고한다고 명시한다.
현재 공개 repo의 82개 Python/YAML/Markdown 파일 검색과 학습 경로 확인에서 NovelD/RND 보너스 구현을 찾지 못했다.
AntMaze 문자열은 offline D4RL/IQL 코드에만 있고, 논문의 online dense AntMaze 환경/전용 설정은 공개 트리에서 찾지 못했다.
따라서 논문의 AntMaze에 숨겨진 설정이 전혀 없었다고 단정하지 않는다. 공개 구현상 확인되는 탐색 기제는 MaxEnt RL이다.

공개 `MaxEntropyLearner.update_q`는
`y = r + gamma*mask*(min Q_target - temp*log_pi(a_next|s_next))`를 사용한다.
설정의 `backup_entropy=True`. 같은 temp가 actor의 Q-weighted Noise Estimation에도 사용된다.
현재 OptiQ는 actor에 temperature가 있어도 critic backup의 entropy 항은 빠져 있으므로 MaxEntDP와 같은 목적이 아니다.
논문 §5.3은 Q의 entropy를 제거하는 ablation에서 성능/안정성 저하를 보고하지만,
이것만으로 우리 v3 단일 경로의 원인이 입증되는 것은 아니다.

공식 코드:
https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/jaxrl5/agents/score_matching/max_entropy_learner.py
https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/configs/max_entropy_learner_config.py

## OptiQ를 유지하며 우선 고려할 비교

현재 승인된 NovelD 네 실험은 그대로 끝낸다.
이후 별도 사용자 승인 실험에서 dense 보상/actor/seed/예산 등을 맞추고 TD 대 soft TD를 먼저 비교하는 것이 MaxEntDP 목적에 더 직접적인 점검이다.
Soft backup의 log_pi는 latent 조건부 한 Gaussian의 log probability가 아니라 random-z marginal policy density여야 한다.
고정 codebook의 finite mixture이면 그 혼합밀도를, fresh latent이면 별도 latent Monte Carlo marginal 추정과 bias/분산 점검이 필요하다.
기존 `soft_td` 분기가 있어도 현재 TRG 프로파일 validator가 TD를 강제하므로 단순한 옵션 한 줄 변경으로 검증이 완료되는 것은 아니다.
NovelD 강도와 entropy backup을 동시에 바꾼 결과만으로 각 효과를 분리해서 설명하면 안 된다.
MaxEntDP처럼 목표거리 환경 보상과 MaxEnt 항만 사용하는 대조군은 NovelD/DACER 외부 잡음을 제외한 별도 설정이어야 한다.
DDiffPG처럼 mode-specific Q/cluster까지 추가하는 것은 더 큰 알고리즘 변경이다.

## NovelD 계수 참고

DDiffPG .01; NovelD 원논문 MiniGrid ObstructedMaze .05, 나머지 MiniGrid .1, NetHack100.
계수는 RND 오차 정의/정규화/환경 보상 규모/episodic 조건에 따라 의미가 달라 그대로 크기만 비교하지 않는다.
NovelD 원논문의 .5는 novelty 차분 계수이며 전체 intrinsic reward multiplier와 구분한다.
