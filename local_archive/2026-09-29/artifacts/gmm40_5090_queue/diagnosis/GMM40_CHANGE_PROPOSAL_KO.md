# GMM40에 맞춘 수정 전 조사 및 제안

학습 코드와 실행 중인 큐는 수정하지 않았다. 근거는 frozen source 87d5d8ff210569ace7e59bd8a53ad02141b67f0a, 이전 fixed/fresh 실험 184bd7e26736a3af136f23b028f52168a34e80e0, 저장된 체크포인트와 샘플이다. 초기화 점검은 CPU forward sampling만 수행했으며 optimizer update는 0회다.

## 1. 좌표와 범위

- 현재 모든 fixed-Q adapter는 normalized action a를 physical x=40a로 바꾸어 원본 GMM log density를 질의한다. box는 [-40,40]^2, 정규화 box는 [-1,1]^2다.
- 타깃 component sigma는 physical 1.31326163, normalized 0.03283154, log sigma -3.41636562다.
- 현재 TRG log sigma [-5,-1]은 physical Gaussian scale parameter [0.2695,14.7152]에 해당한다. 필요한 타깃 scale은 이 구간 안에 있다. 표현 불가능을 이유로 상한을 늘릴 근거는 없다.
- 물리 좌표 변환의 density 상수 2 log 40은 self-normalized importance weights에서 상쇄된다. 기존 tanh-Gaussian의 Jacobian과 현재 truncated Gaussian의 normalization은 서로 다르므로 혼용하면 안 된다.
- Q=log p, T=1을 유지해야 원본 bounded GMM을 평가 대상으로 삼는 실험과 일치한다. T를 바꾸면 목표 분포도 달라진다.

## 2. 실제 문제와 수정 우선순위

### 가장 먼저: sigma 상한 포화와 초기화

현행 actor는 initial_log_std=-1로 상한에서 시작하고 raw output을 hard clip한다. 저장된 100k actor의 CPU 점검에서 raw>-1인 좌표가 seed0 35.715%, seed1 12.94%다. hard clip 초과 영역의 직접 미분이 0인 것은 수치적으로 확인했다. 이 현상과 넓은 conditional 분포가 남는 것은 확인됐지만, 모든 성능 하락을 이 원인 하나에 귀속하지 않는다.

검토할 수정은 hard clip 대신 범위가 제한된 smooth parameterization을 선택할 수 있게 하고 초기값을 상한보다 안쪽에 두는 것이다. 예: [-5,-1] 안에서 sigmoid/tanh mapping과 initial_log_std=-1.1. smooth mapping 역시 경계 근처 gradient가 약해질 수 있으므로 raw quantile, saturation, sigma gradient를 함께 측정해야 한다. -1.1은 초기 탐색폭을 크게 줄이지 않는 시험 후보이지 검증된 최적값이 아니다.

상한이나 초기 sigma를 무조건 줄이면 원거리 탐색을 해칠 수 있다. 10,000개 초기 prior sample에서 target component별 3σ 안에 최소 10개가 들어온 개수는 다음과 같다. 이것은 원인 점검용 count이며 정식 coverage threshold와 다르다.

| mean-head scale | initial log sigma | seed0 | seed1 |
|---|---:|---:|---:|
| 1e-4 | -1 | 28 | 31 |
| 1e-4 | -1.1 | 24 | 26 |
| 1e-4 | -2 | 8 | 7 |
| 1e-4 | -3 | 2 | 2 |
| 1 | -1 | 28 | 28 |
| 1 | -1.1 | 24 | 26 |
| 1 | -2 | 8 | 8 |
| 1 | -3 | 2 | 3 |

### 다음: mean-head initialization

예전 성공 실험 scale=1, 현재는 Actor 기본값 1e-4다. 현재 초기 component center의 physical 좌표 표준편차(축별 std 평균)는 seed0 약0.00724, seed1 약0.02522이며 거의 동일한 위치에서 시작한다. 동일 RNG에서 scale=1로 forward만 바꾸면 약0.7232, 2.5091이 된다. 초기 대칭을 줄이는 후보로 scale=1 복원 비교가 타당하다. 다만 이 변화만으로 초기 full-policy coverage가 크게 좋아지지는 않았고 수렴 개선은 학습으로 검증해야 한다.

### teacher proposal와 latent는 분리 비교

- 예전 teacher std floor 0.05, 현재 exp(-5)=0.00673795다. 현재 box 좌표에서 타깃 sigma 0.03283 대비 각각 약1.52배와0.205배다. 예전 floor는 pre-tanh 좌표 값이므로 0.05를 동일한 행동 공간 폭으로 해석하지 않는다. 0.03/0.05를 별도 시험 후보로 둘 수 있지만 actor sigma의 최솟값과 묶어 변경하지 않는다.
- 현재 proposal floor는 actor 최솟값과 같아 추가 탐색폭을 제공하지 않는다. 실제 sampling과 logq에 동일한 effective sigma를 써야 한다.
- fixed64 prior와 fresh64 continuous prior는 서로 다른 정책이다. 예전 fixed near93~97%, fresh76~79%이고 현재 fresh55~72%다. fixed로 바꾸는 것만으로 원인 진단을 끝낼 수 없다. fixed 실험은 학습·평가 모두 동일 codebook을 써야 한다.
- N=M64 유지가 먼저다. M 증가를 시험하면 finite-sample teacher ESS 개선 여부와 Q 질의량 증가를 함께 보고해야 한다. 완벽한 균등40모드에서 독립64회 추출해도 기대 고유 모드 수는 약32.09지만, 이 계산이 실제 policy capacity의 상한이라는 뜻은 아니다.

## 3. 필요한 인터페이스/기록 변경

현재 adapter는 log sigma bounds/init, mean-head init, latent mode, teacher floor가 코드에 고정돼 있다. CLI mean_output_init_scale 옵션은 optiq_trg에 대해 허용되지 않으며 config의 None은 실제 사용값 1e-4를 드러내지 않는다.

GMM40 전용 profile에서 위 값을 명시적으로 선택·기록하도록 하는 것이 우선적인 수정 대상이다. RL 기본값을 바꾸지 않고 profile을 분리한다. policy_family(tanh Gaussian/box truncated), sigma coordinate system, resolved init scale, bound mapping, prior, codebook hash, proposal floor를 결과에 기록한다. 진단에는 raw sigma quantile/상한초과 비율, component mean spread, teacher ESS, sigma gradient를 추가한다.

## 4. 다른 알고리즘에 대한 조사

- SAC: 현재 state가 상수이고 단일 diagonal tanh-Gaussian actor다. 40개 군집을 표현하는 능력이 OptiQ/flow와 같지 않다. 범위만 조절해 40-mode model로 만들 수 없으며 GMM actor로 교체하면 별도 SAC variant가 된다. T=1은 이미 objective의 entropy coefficient와 일치한다.
- SQL: local SVGD particle16을 fixed8/update8로 나눈다. seed0 100k의 평균 kernel h≈0.0658(커널 exp(-distance²/h))이며 normalized target sigma²≈0.00108이다. 입자수64/128, bandwidth와 초기 분산을 분리 점검할 후보가 있다. h를 target sigma²로 고정하면 개선된다는 결론이나 16particle이16mode 표현 한계라는 결론은 내리지 않는다.
- DIPO: action improvement가 log p의 gradient ascent이며 명시적 T=1 entropy objective는 없다. diffusion step100, action optimizer lr0.03, inner20steps, action grad clip0.2다. density matching 알고리즘과 목표가 완전히 같다고 소개하면 안 된다. action lr0.03은 normalized 좌표 값이고 단순 위치 step으로 보면 physical1.2에 해당하지만 Adam 실제 이동량은 gradient history에 의존한다.
- MEOW: flow Q regression과 native parameter ranges를 사용한다. seed0 50k saved evaluation의 boundary fraction은7.08%이고 회귀loss는21.53이다. Q regression residual, replay support와 boundary saturation을 먼저 점검해야 하며 TRG sigma 범위를 그대로 적용하면 안 된다.
- MFPO: native16 policy +32 Gaussian candidates, sampler2steps다. seed0 100k의 ESS는 policy 약3.56, Gaussian 약4.16이다. 후보 수/샘플러 적분오차의 영향을 별도 비교할 수 있다. TRG sigma 범위와 대응되는 동일 파라미터가 있는 것은 아니다.

## 5. 제안하는 검증 순서

1. 이전 성공 소스/설정을 그대로 사용하는 재현 대조군을 준비한다. 이는 현재 TRG 개선과 별도의 재현 실험이다.
2. 현재 TRG의 sigma mapping과 초기값을 조사한다. 초기값만 바꾼 hard-clip 대조군과 동일 초기값의 smooth-bound 대조군으로 두 효과를 분리한다.
3. mean-head scale1 대조, teacher floor 대조, fixed/fresh 대조를 각각 한 요인씩 진행한다.
4. 공통 target/T/평가 seed/샘플 수를 고정하고 near, coverage, MMD², mass TV, boundary fraction을 함께 본다. 같은 actor updates 외에 Q 질의량·시간도 기록한다.

256×2, batch256, Adam3e-4, N=M64, T1, 100k를 동시에 바꾸는 sweep은 권하지 않는다. 이번 조사로 최적 hyperparameter가 확정된 것은 아니다.
