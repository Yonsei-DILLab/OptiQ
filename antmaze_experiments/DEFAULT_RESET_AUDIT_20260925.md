# AntMaze 실험 중단 및 OptiQ 기본 설정 점검 (2026-09-25)

이 문서는 새 실험을 등록하지 않는다. `direct-gmm-trg-antmaze`의 기존 학습 소스,
캠페인 설정, 체크포인트와 결과는 보존한다. 이후의 AntMaze 실험에서 "기본
OptiQ"은 `analysis_tools/experiments/20260921_gmm_trg_sweep/train.py`의
`compose_config(['benchmark=ant', 'seed=0'])`으로 해석한다. 이는 **Direct
GMM/TRG** 기준이다. 초기 OptiQ OT 구현이나 DDiffPG의 DACER/NovelD 설정과
혼용하지 않는다. 이 해석은 frozen source `b111a993b1e1bf895966dac4469abdfdb2c37c05`의
구성 함수를 실제로 호출해 확인했다.

## 중단 확인

5090 서버 `vast-heechan-180`과 `vast-heechan-199`에서 AntMaze 학습/controller
supervisor의 RUNNING 항목은 없고 GPU compute process도 없었다. 남아 있던
W&B 동기화 감시 서비스 4개(상한 실험 양 서버, annealing 199, dense T1 199)를
`supervisorctl stop`으로 중지했다. 해당 supervisor 파일은 모두
`autostart=false`, `autorestart=false`다. 과거 캠페인의 `status.json`에는
중단 당시의 `pending`/`running` 목록이 남아 있지만 실제 프로세스 상태가
아니다. controller는 기존 `status.json`을 감지하면 재시작을 거부한다.
이 파일들을 완료/실패로 재분류하거나 frozen 로그를 덮어쓰지 않는다.
4090 `vast1`은 GMM40 전용으로 유지했다.

## Direct GMM/TRG 기본 알고리즘

| 항목 | 기본값 | 최근 v3/v4 상한 실험의 값 |
|---|---|---|
| actor/쌍둥이 critic | 각각 256×2 GELU, scalar TD critic | 각각 256×3 |
| actor 목표 | direct marginal GMM NLL, anchor 없음, 밀도 보정 β=1 | 동일 |
| latent/teacher | 매 행동 독립 정규분포 z, N=M=64, exact proposal choice | 동일 |
| teacher 온도 | 0.25, 고정 | v3 3, v4 1 |
| actor log σ | [−5,−1], 초기 −1 | 상한 0/1/2/3 또는 미실행 무상한 |
| 평균 head 초기화 scale | 1e−4 | 1 |
| teacher σ 하한 | exp(−5)≈0.006738 | v3 1, v4 0.5 |
| DACER 행동 잡음 | 꺼짐 | 켜짐, 목표 엔트로피/차원 +0.7, 500 update 간격 |
| γ / critic EMA τ / actor target τ | .99 / .005 / 1 | .999 / .005 / 1 |
| actor/critic Adam 학습률 | 각각 3e−4 | 3e−4 / 5e−4 |
| gradient clipping | 없음 | 없음 |
| replay / warmup | 1M / 5000 transitions | 1M / 8192 transitions |
| batch / UTD | 256 / 수집 transition당 learner update 1회 | 4096 / 256 transitions당 8회 (=1/32) |

기본 teacher σ 하한은 actor가 낼 수 있는 최소 σ와 같아서 실제 proposal을
추가로 넓히지 않는다. 최근 0.5/1.0 설정은 teacher proposal 분포를
변경하는 실험값이다. DACER의 목표 엔트로피 −0.9/차원은 DACER가 꺼진
기본 설정에서 **사용되지 않는 설정 필드**다. 기본 latent는 고정 codebook이
아니고 상태/행동 시점마다 새 `N(0,I)`를 뽑는다.

## AntMaze 환경·학습과 OptiQ 기본값의 경계

OptiQ 자체에는 AntMaze 보상 함수가 없다. 추가 shaping을 걷어낸 환경
기준은 vendored DDiffPG의 원본 sparse reward(목표 전 0, 성공 시 10;
v2 첫 목표는 20)이며 목표 반경 0.5에서 종료한다. v1은 시작 xy가
`[−2,2]²`에서 무작위이고 500 step, v2는 원본 고정 시작/500 step,
v3·v4는 원본 고정 시작/700 step이다. 상태 29차원, 행동 8차원
`[−1,1]`; goal 좌표는 관측에 추가하지 않는다. NovelD는 DDiffPG의
탐색 보너스이지 OptiQ의 일부가 아니므로 기본 OptiQ 비교에서는 꺼진다.
Geodesic/progress 보상, 성공 bonus 제거, step cost, γ=.999은 최근
실험 프로필로만 남는다.

기존 AntMaze 어댑터 `learners.py`는 256 병렬 환경, batch4096,
8 updates/256 transitions, warmup8192, 256×3, mean-init1, DACER on,
critic LR5e−4를 명시적으로 적용한다. 따라서 **현재 `run.py` 기본 인자로
새 학습을 실행하면 위 OptiQ 기본 설정이 재현되지 않는다.** 또한
`run.py`의 기본 NovelD는 on이다. 이 실행 경로는 이전 캠페인의 frozen
재현을 위해 남기며, 기본값으로 표시하거나 재사용하지 않는다. 새 기본
AntMaze 학습을 등록하려면 독립적인 profile과 실제 batch256·UTD1
계산(256-env 수집 블록당 256 update)을 구현·검증하고 commit을 동결해야
한다. 이는 기존 8/256 프로토콜보다 update가 32배 많으며 실행시간도
다르다. 환경별 native 3M/3M/4M/5M 예산과 vector 수 역시 AntMaze
실험 설계로서 별도 기록할 항목이다. 이 문서는 학습을 시작하지 않는다.

## 현재 평가 코드의 실제 동작과 다음 기본 보고 기준

`run.py`의 기본 평가는 250k transitions마다 모드별 20 episodes,
최종 모드별 100 episodes이며 평가 슬롯은 20개다. 최근 상한 실험은
50k마다 40 episodes로 명시적으로 바꿨다. 중간 평가는 `native`와
`policy`, 최종은 OptiQ의 `native`/`policy`/`zero_z`를 별도 저장한다.

- `policy`: 각 상태에서 새 정규 z와 conditional truncated Gaussian σ를
  모두 샘플링하는 실제 확률 정책. 행동은 `[−1,1]`에 있다.
- `native`: 매 행동 새 z를 뽑되 조건부 Gaussian 잡음은 끈 μ-only 출력.
- `zero_z`: z=0과 μ-only 출력. 별도 대조군이다.

DACER의 외부 행동 잡음과 NovelD 보상은 평가에 넣지 않는다. 평가 RNG는
학습 RNG로부터 격리하고, 원시 xy 경로·return·목표·길이·초기 full state를
NPZ에 저장한다. `eval_starts=upstream`에서는 v1이 학습과 같은 무작위
시작, v2/v3/v4가 학습과 같은 원본 고정 시작이다. 그림의 `fixed`는
초기 **시뮬레이터 full state**를 뜻하며 latent를 고정한다는 뜻이
아니다. 여러 성공 경로 주장은 동일 checkpoint·같은 시작 상태에서
`policy` rollout의 목표별 성공 경로 수로 판단한다. 통로 진입만으로
성공이나 멀티모달 성능을 선언하지 않는다. v1 무작위 시작의 다양성과
한 시작 상태에서의 정책 다양성은 따로 표시한다. 최종 평가의 100회는
단일 학습 seed를 대체하지 않는다.

후속 신규 학습은 이 감사에서 구분한 기본 알고리즘·환경·수집/평가
프로필을 별도 manifest에 모두 적고, 실제 초기 config/optimizer/update
횟수 및 샘플링을 preflight로 확인한 뒤 시작한다. 이전 캠페인은 자동
재개하지 않는다.
