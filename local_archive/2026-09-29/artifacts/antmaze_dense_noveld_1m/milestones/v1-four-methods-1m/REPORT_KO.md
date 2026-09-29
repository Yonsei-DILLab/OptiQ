# AntMaze v1, 네 방법의 1M 비교

OptiQ, SAC, MFPO, MEOW 모두 training seed 0으로 1M 환경 상호작용을 완료했다. 네 정책의 최종 체크포인트·전체 리플레이·평가 파일은 로컬 저장 검증을 통과했다. 그림의 모든 rollout은 동일한 전체 초기 시뮬레이터 상태에서 시작했다.

## 직접 stochastic 정책을 샘플링한 결과

| 방법 | 성공/100 | 위 경로 | 아래 경로 | 실패 |
|---|---:|---:|---:|---:|
| OptiQ | 100 | 71 | 29 | 0 |
| SAC | 0 | 0 | 0 | 100 |
| MFPO | 100 | 0 | 100 | 0 |
| MEOW | 95 | 95 | 0 | 5 |

이 seed와 고정 출발 상태에서는 OptiQ만 두 성공 경로를 모두 사용했다. MFPO와 MEOW도 목적지에 도달하지만 각각 한 경로에 집중했다. SAC는 유효한 성공 경로를 학습하지 못했다. 이 결과는 한 seed에서 관찰한 경로 다양성이며, 일반적인 알고리즘 순위나 고정 state에서 action density의 다봉성을 증명하지 않는다.

OptiQ의 μ-only 평가에서도 위 75회·아래 21회·실패 4회가 나왔다. 따라서 두 경로가 conditional sigma noise를 추가했을 때만 나타나는 현상은 아니다.

## 평가 방식과 출발 상태에 따른 차이

각 값은 서로 다른 평가 조건의 100회 중 성공 횟수다. 표의 열들을 합산하지 않는다.

| 방법 | 정책 샘플링, 고정 출발 | 정책 샘플링, 원래 출발 분포 | Native, 고정 출발 | Native, 원래 출발 분포 |
|---|---:|---:|---:|---:|
| OptiQ | 100 | 99 | 96 | 96 |
| SAC | 0 | 0 | 0 | 0 |
| MFPO | 100 | 98 | 99 | 98 |
| MEOW | 95 | 93 | 100 | 18 |

Native는 OptiQ random-z μ-only, SAC tanh(μ), MFPO Q-best-of-10, MEOW center-prior를 뜻한다. MEOW native의 고정 출발 성공 100%를 원래 출발 분포의 강건성으로 해석하면 안 된다. 원래 분포의 native 성공은 18%이며 직접 정책 샘플링에서는 93%다.

직접 정책 샘플링 그림은 각 방법의 정책에서 행동을 직접 뽑고 외부 DACER 행동 잡음이나 NovelD 보상을 평가에 추가하지 않았다. OptiQ policy에는 학습된 conditional sigma가 포함된다. 학습은 공통 dense+NovelD를 사용하지만 모델, optimizer, entropy backup, warmup 등은 각 방법의 기존 설정을 유지했으므로 정책 구조만 바꾼 단일 요인 ablation은 아니다.

## 보관 및 검증

학습 소스는 `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`이며 저장 검증 모듈은 `dd107f29e10adf6a7297c4603aac07a781252292`이다. 각 `runs/v1-<method>-s0/archive-verification.json`은 실제 replay reward 1M건, 전체 checkpoint digest, 평가 원자료와 메트릭의 일치를 검증한다. 그림 생성 전에 각 rollout NPZ의 SHA256과 전체 초기 시뮬레이터 상태를 대조했다.

주 그림은 `policy-fixed-comparison.png/.pdf`, native 평가까지 포함한 보조 그림은 `fixed-state-comparison.png/.pdf`다. 해당 원자료 해시와 집계는 각 verification JSON에 기록했다. 다른 미로의 학습은 계속 진행 중이다.

![동일 초기 상태에서 직접 정책을 100회 평가](policy-fixed-comparison.png)
