# Humanoid-v4 고정 latent 64개 · Direct GMM NLL

요청: seeds 0–3, temperature 0.1, N=M=64, fixed latent, W&B `OptiQ/v5-heechan-gmm`.

Actor/Critic 256×2, replay batch 256, UTD 1, policy delay 1, Adam 학습률 3e-4, warmup 5,000, 목표 1M 환경 step/seed. 기존 초기화(mean scale 1e-4, sigma 0.5), plain TD 등 학습 기본값 유지.

## 평가 수정

고정 latent 정책도 v5 paired dual evaluation을 지원한다. `eval/stochastic_z/mean_reward`는 학습에 사용하는 64×17 codebook에서 매 행동마다 균등 선택한 z로 `tanh(mu(s,z))`를 평가한다. codebook seed는 20260911이고 epsilon=0이다. `eval/fixed_z/mean_reward`는 같은 codebook의 첫 성분을 사용한다. `eval/mean_reward`는 고정 latent 정책에서 stochastic_z 결과를 가리킨다.

기존 연속 Gaussian latent 정책의 zero_z/stochastic_z 동작과 zero_z alias는 유지한다. 두 모드의 환경 seed는 paired이며 평가 RNG는 collection RNG와 분리하고 예외 발생 때도 복원한다. 평가 주기 5,000, 모드별 10 에피소드.

실제 Humanoid full-size 사전 검증 PASS: batch256/N64/M64, actor/critic 각 6회 업데이트, fixed64×17 codebook 유지, Sinkhorn 미호출, 두 평가 모드 및 RNG 복원 확인. 사전 검증의 2-step 에피소드는 기능 점검용이며 학습 성능 결과가 아니다.

기존 Ant 실행과 frozen source는 유지한다. Humanoid는 vast-heechan-180 GPU0–3에 새 source snapshot과 별도 supervisor worker로 실행한다.

## 실행

Source commit: `9ce812054e9e271915143c095b541a6421e76305` (두 서버의 v5-direct-gmm branch 반영, push 없음).

Campaign: `/home/heechan/optiq-experiments/humanoid-direct-gmm-fixed64-20260920T153146Z`

Frozen source: `/home/heechan/OptiQ-ops/sources/9ce812054e9e271915143c095b541a6421e76305`

| Seed | GPU | W&B |
|---:|---:|---|
| 0 | 0 | [run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/90ex6i6o) |
| 1 | 1 | [run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/knhv43d1) |
| 2 | 2 | [run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/o5zw6z38) |
| 3 | 3 | [run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/no5fvkc6) |

회귀 테스트 37개 PASS (64.70초): finite evaluation, v4/v5 continuous evaluation, Direct GMM objective/training.
