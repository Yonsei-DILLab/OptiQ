# 앞으로 적용할 기본 범위

사용자 요청: 기본 `log σ` 범위를 `[-5, -1]`로 고정.

- `direct-gmm-trg`의 기본 TRG entry에서 `log_std_min=-5.0`, `log_std_max=-1.0`을 명시.
- 이후 기본값을 실험 결과에 따라 바꾸지 않도록 `AGENTS.md`에 기록.
- 기본 config 해석 결과 검증 완료. 초기 log σ 등 다른 기본값은 변경하지 않음.
- 커밋 및 push: `c1a73ec41a50ffddc4b205c402f12e9767d8f121`.

현재 요청된 Ant/Humanoid 상한 0 대 −2, 20k 비교는 명시적 실험용 override입니다.
이 비교는 앞서 고정한 `d01fc50c004f8bd0f7f9a10a8a5dac63d4e97654` 소스로 계속 실행했습니다.
기본값 명시 커밋은 실험 시작 후 작성됐으며, 실행 중인 frozen source를 수정하지 않았습니다.
