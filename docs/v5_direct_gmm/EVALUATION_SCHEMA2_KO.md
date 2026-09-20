# Evaluation schema 2: zero-z / stochastic-z 기록 복구

## 비교한 브랜치

- `v5`: `a5d5e281bcb10b5c31557073d95e03a1c5f2af4f`
- `origin/v4`: `f73c96b4457bef12f921f3509b41a2694bf751d0`
- `origin/v8`: `e08003e4ba52be76e0110d38f48bd9bf44d6eb83`
- `origin/v9`: `703a61b671f27295b3de65ace7daab290565889f`
- `origin/v5_bestk`: `5ac2be86a7b41d2eee728f7fe4438edb4062d306`

위 브랜치의 paired evaluator는 실제 z=0과 stochastic-z를 분리하고 `eval/mean_reward`를 zero_z에 연결한다. 기존 fixed64 추가 커밋 9ce8120은 finite 정책의 zero-z를 codebook[0]인 fixed_z로 대체하고 기본 alias를 stochastic_z로 바꿨다. 이 둘은 같은 평가가 아니므로 다음 실행부터 의미를 복구한다.

## 다음 실행의 계약

- `eval/zero_z/mean_reward`: 실제 영벡터 z=0, epsilon=0. finite 정책에서는 학습 support 밖의 diagnostic임을 config metadata에 명시.
- `eval/stochastic_z/mean_reward`: finite는 학습 codebook에서 균등 샘플, continuous는 기존 Gaussian prior. epsilon=0.
- `eval/mean_reward` / `final_eval_return`: 다른 브랜치와 동일하게 zero_z alias.
- 각 mode별 std_reward, best_mean_reward, mean/std_ep_length, num_episodes 및 최종 return/std/best summary를 기록.
- 모드별 NPZ에 episode reward/length, environment seeds, policy seeds, evaluation timesteps 보존.
- 환경 reset seed를 mode 간 짝지으며 평가 RNG/zero-latent flag/mu-only flag는 성공 및 예외 시 복원.
- W&B는 기존 직접 scalar writer로 두 모드를 같은 env_steps에 한 번에 기록. 평가와 학습이 같은 env step에 flush해도 W&B internal step은 따로 증가.
- Ant/Humanoid fixed64 profile 모두 dual_mu_eval=true; W&B OptiQ/v5-heechan-gmm.

## 현재 실험

실행 중 Ant snapshot 25a6923 및 Humanoid snapshot 9ce8120, supervisor worker, W&B run에는 변경하지 않는다. 기존 fixed_z 기록을 zero_z로 재명명하지 않는다. 변경된 브랜치로 시작하는 다음 실험에만 schema_version=2가 적용된다.

## 검증

Identity-mu finite actor로 literal zero action과 codebook[0] action을 구별하고 stochastic action의 support를 확인한다. 실제 Ant 짧은 CPU integration에서 두 mode의 per-episode 결과, paired seeds, CSV/W&B payload의 동일 env_steps, mode별 best 통계, 예외 후 RNG/flag 복원을 검사한다. 기존 v4/v5/Direct GMM 회귀도 함께 실행한다.

검증 결과: CPU에서 `tests/test_finite_dual_evaluation.py tests/test_v4.py tests/test_v5.py tests/test_direct_gmm.py` 38 passed, 13 dependency deprecation warnings, 55.24초. GPU 실험 worker를 중단/재시작하지 않고 검증했다.
