# Original Gaussian 수집을 보존하는 best-of-8 혼합

`mujoco_v5_bestk_mixed` / `v5/bestk_mixed`는 기존 보정 proposal 결합에
수집 확률 `alg.behavior_best_of_k_probability=.5`를 적용하는 후속 가설이다.
성능이 검증된 기본값이 아니며, 프로필 추가가 기존 실행을 바꾸지는 않는다.

상태 s에서 원래 full Gaussian actor를 π, live twin-min Q로 선택하는
best-of-8 분포를 π⁸라고 하면, 수집 분포는 다음과 같다.

`bρ(a|s) = (1−ρ) π(a|s) + ρ π⁸(a|s), ρ=.5`.

5K uniform warmup 이후 매 환경마다 독립 Bernoulli 선택을 한다. 절반은
기존 Gaussian 행동을 그대로 실행하며, 나머지는 그 행동을 후보0으로 삼고
독립 z·Gaussian epsilon으로 만든 추가7개와 Q를 비교한다. 두 경우 모두
실제 실행한 행동을 replay에 저장한다. 기본 혼합 프로필에는250K 지연이 없다.

고정된 상태·actor·critic에서 `bρ ≥ (1−ρ)π`이므로 `π/bρ ≤ 1/(1−ρ)`이다.
ρ=.5에서는 이 비율이2를 넘지 않는다. 이것은 원래 actor 행동의 확률 질량을
보존한다는 성질이다. 과거 정책이 섞인 replay 전체의 밀도비, 상태 방문 범위,
critic 정확도 또는 성능 개선을 보장하지 않는다. 실패한 run을 보며 즉시
집계 방식만 바꾸는 것과는 별도의 탐색 가설이다.

Teacher는 기존처럼 best8 pilot으로 만든 명시적 Gaussian 혼합 proposal에서
독립적인64개 행동을 새로 뽑고 `exp(Q_mean/T)/q_mix`를 적용한다. 수집의
`bρ`와 teacher의 `q_mix`는 서로 다른 분포다. T(Ant .25/Hopper .01), beta1,
sigma floor, mean OT, full NLL, plain TD/Kt1, 기존 두 epsilon=0 평가는 유지한다.
수집 혼합 때문에 teacher의 밀도 보정을 제거하거나 교체하지 않는다.

선택용 NumPy RNG는 `(seed,580050)`으로 분리하며 학습·actor·replay·평가
RNG를 소비하지 않는다. Warmup/지연 전에는 혼합 RNG도 사용하지 않는다.
확률1은 원래 best8 경로, 확률0은 원래 Gaussian 수집 경로를 보존한다.
혼합 프로필은0<ρ<1만 허용해 실험 이름을 명확히 한다. 추가 uniform 교체와
혼합하는 설정은 거부한다.

로그의 `behavior_best_of_k=8`과 `behavior_best_of_k_active=1`은 선별 기능이
활성화됐다는 의미다. 매 행동이 선별됐다는 뜻은 아니다.
`behavior_best_of_k_probability`, `best_of_k_applied_fraction`(현재 batch),
`best_of_k_fraction`(누적), `best_of_k_opportunity_count`를 함께 확인해야 한다.
선별에서 후보0이 우승해도 best8을 적용한 행동으로 센다.

검증: 상태별 분기·행동 scaling/replay,0/1 경계의 실제 Ant/Hopper 학습
parameter·optimizer·RNG·평가 일치, TD/평가에서 selector 미호출을 검사한다.
두 환경의 batch256·40회 update 검사에서도 T/beta1/plainTD/paired 평가와
예상한 Bernoulli 선택 횟수(해당 seed에서17/40)를 확인했다. 초기 경계값
테스트4개는 Hydra struct에 필드를 추가하는 테스트 설정 오류였으며 수정 후
재검증했다. 구현 검증은 학습 성능 검증을 대신하지 않는다.
