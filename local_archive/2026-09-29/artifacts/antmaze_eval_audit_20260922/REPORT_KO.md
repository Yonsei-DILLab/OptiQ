# AntMaze 학습 예산·평가 방식 점검

2026-09-22. 학습 또는 알고리즘 수정 없이 원본과 현재 코드의 호출 경로를 대조했다.

## 중단

SAC dense v1 seed0 500k 캠페인은 사용자 요청으로 취소했다. supervisor 중지 후 controller와 learner에 SIGINT를 보내 종료했고, 해당 캠페인 프로세스가 없음을 확인했다. 자동화도 PAUSED다. 마지막 기록은 47,000 env steps, 41,999 updates, 학습 성공 0/94 episodes, 최소 목표 거리 5.83147이다. 실제 종료 직전 step은 마지막 로그보다 조금 클 수 있다. KeyboardInterrupt는 사용자 취소에 따른 기록이며 수치 실패가 아니다. 원본 로그와 취소 전 status를 보존했다. W&B 취소 태그 추가는 별도 로그인 셸의 인증 정보 부재로 실행되지 않았으며, 로컬·서버 cancellation.json이 취소의 근거다.

## 원본 예산

DDiffPG commit 7edd06c4799abbab0f8fa534c21deb56253b018e의 preprocess_cfg:

| 과제 | max_step | horizon |
|---|---:|---:|
| AntMaze-v1 | 3,000,000 | 500 |
| AntMaze-v3 | 4,000,000 | 700 |
| AntMaze-v4 | 5,000,000 | 700 |

이는 알고리즘·seed 하나당 환경 interaction 예산이다. 병렬 환경 수만큼 이미 합산하며 gradient update 횟수가 아니다. baselines_main.py도 같은 preprocess_cfg를 호출한다. 원본 global_steps는 warmup 이후부터 세며 > max_step에서 종료하므로 실제 총 interaction은 warmup과 마지막 병렬 batch만큼 더 크다. SAC 기본 warmup32×256=8192 interactions.

원본 기본은 sparse reward + NovelD다. SAC 원본도 critic update에 NovelD를 더한다. 원본 SAC는 256환경, batch4096, 256 transitions마다8 learner updates, actor LR3e-4/critic LR5e-4, tau.05다. 현재 중단한 실험은 dense reward, NovelD 없음, 1환경, batch256, UTD1, LR3e-4, tau.005였다. 따라서 3/4/5M만 맞춰도 원본 재현은 아니다.

Dense reward MaxEntDP 논문 Appendix D.2 Figure8/9는 1M interactions 결과다. 이와 DDiffPG의 sparse 기본 예산을 구분해야 한다. MFPO Figure10의100k는 별도 논문의 보고 시점이며 범용 AntMaze 표준 예산이 아니다.

## 현재 평가와 공개 기본값 비교

현재 경로: antmaze.multimodal.run → Evaluator → PolicyView.act. antmaze/agents.py의 native act만 보면 실제 평가 동작을 오판할 수 있다. PolicyView가 이를 직접 우회한다.

| 방법 | 현재 AntMaze | 원본/기존 성능평가 | 판정 |
|---|---|---|---|
| SAC | predict(deterministic=False), tanh Gaussian draw | DDiffPG actor.forward(sample=False), SB3 evaluate_policy 기본 deterministic=True | stochastic 평가로 유효하나 원본 성능 평가와 다름 |
| MFPO | eval_actions_sample_batch, fresh Gaussian latent, flow1회 | 공개 기본 eval_action_selection=True, 후보10개 중 target Q 최대 선택 | 원본 성능 평가와 다름 |
| MEOW | policy.eval() 후 sample(deterministic=False) | 공개 continuous-control evaluate 기본 deterministic=True | 원본 성능 평가와 다름 |
| OptiQ | random z + box-truncated conditional Gaussian; 추가 DACER noise 없음 | 우리 기존 RL 기준 random-z mu-only 및 zero-z 별도 | 주 평가가 기존 μ-only 규칙과 다름; final μ-only 보조 결과 별도 |

MEOW에서 .eval()은 dropout 등을 끄는 동작이며 action sampling 자체를 deterministic으로 바꾸지 않는다. deterministic 분기는 Gaussian prior 평균을 flow에 통과시키는 방식으로 전체 action 분포의 기댓값이라고 단정하면 안 된다. SAC도 deterministic action은 tanh(mu)이며 squashed 분포의 정확한 기댓값과 일반적으로 다르다.

MFPO config의 eval_action_selection=True 및 MEOW config의 deterministic_action=True는 native 설정으로 저장되지만 PolicyView는 이를 우회한다. config의 evaluation 문자열에는 직접 정책 샘플링으로 적혀 있으나, native 필드만 보면 오해하기 쉽다. 실제 적용 모드를 명시적으로 별도 기록할 필요가 있다.

## 구현 점검 범위와 결론

코드 수준에서 fresh RNG를 매 행동마다 전진시키고 batch shape를 검증하는 것을 확인했다. 학습과 평가 환경은 분리되고 Python/NumPy/Torch RNG는 저장·복원한다. JAX OptiQ RNG도 복원하며 MFPO는 eval_agent의 별도 RNG를 쓴다. 성공/timeout 후 그 에피소드의 추가 step을 하지 않는다. 자연 reset과 전체 simulator state 고정 reset은 별도로 저장하며 고정 초기 simulator state는 일치 assertion이 있다. 평가 방문은 학습 coverage에 들어가지 않는다.

원치 않는 외부 Gaussian/DACER noise를 덧붙이는 경로는 현재 네 방법의 PolicyView에 없다. 하지만 learned sigma 또는 flow latent를 샘플링하는 것은 실제로 포함한다. 이는 완전한 확률 정책의 경로 다양성 평가로 해석할 수 있지만, native benchmark 성능 평가와 같다고 보고하면 안 된다. MFPO 논문 AntMaze 전용 평가 코드가 확인되지 않았으므로 Figure10과 동일한 샘플링이라고 단정할 수 없다.

성능 확인은 SAC deterministic, MEOW native deterministic, MFPO native best-of10, OptiQ 기존 random-z μ-only/zero-z로 별도 기록하고, 정책 다양성은 별도 direct stochastic rollout으로 기록하는 구성이 적절하다. 각 결과에 mode를 표시하고 섞어 평균내지 않는다. 이번 요청에서는 평가 코드를 바꾸거나 새 학습을 시작하지 않았다.

## 근거

- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/utils/common.py#L36
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/scripts/baselines_main.py#L73
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/models/mlp.py#L73
- https://arxiv.org/html/2502.11612v3#A4.SS2
- https://raw.githubusercontent.com/dongxiaoyi-xyz/MFPO/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/configs/mfpo_config.py
- https://raw.githubusercontent.com/dongxiaoyi-xyz/MFPO/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/jaxrl5/agents/mean_flow_learner.py
- https://github.com/ChienFeng-hub/meow/blob/main/cleanrl/cleanrl/meow_continuous_action.py
- https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
- Current frozen source79766ba4856b6f4092e4d5277f62296d6194361c: antmaze/multimodal/policy.py, evaluation.py, run.py, env.py; antmaze/agents.py and evaluation.py.
