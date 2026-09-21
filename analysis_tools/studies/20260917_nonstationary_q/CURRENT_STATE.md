# 1D non-stationary Q 실행 기록

2026-09-17 16:32 KST 확인. 구현·검증 후 본 실험을 등록했으며 초기 학습이 진행 중이다.

- 최종 비교 궤적448개: mass / split / 공통 learned-Q replay / closed actor–critic.
- 각 종류는7방법 × 4크기 × 4seeds. Temperature0.25.
- 공통20K prefix112개를 복사해 mass와 split에 사용한다. Critic source4개를 포함한 실제 실행 노드는564개다.
- 확인 시점: prefix22개 완료,6개 GPU 작업 실행 중, 학습 실패 기록0개. 나머지는 자원·단계 의존성 대기다. 본 비교의 non-stationary 구간은 아직 시작 전이다.
- 최대16 GPU 병렬 요청, GPU당 CPU2개. 실제 동시 GPU 수는 SLURM 가용 자원과 우선순위에 따른다.

## 작업 ID

| 단계 | Array job | 개수 | 시작 조건 |
|---|---:|---:|---|
| 공통 prefix | 2273894 | 112 | 즉시 등록 |
| mass / split | 2273895 | 224 | 모든 prefix 완료 |
| 실제 critic Q source | 2273896 | 4 | mass / split 완료 |
| 공통 learned-Q replay | 2273897 | 112 | source 완료 |
| closed actor–critic | 2273898 | 112 | replay 완료 |
| MD·HTML 보고서 및 그림 | 2274033 | CPU 작업1개 | closed array 종료 |

정확한 제출 명령은 [SUBMISSION.json](SUBMISSION.json), 보고서 작업은 [REPORT_SUBMISSION.json](REPORT_SUBMISSION.json)에 있다. 일부 수치 실험이 실패하면 다음 단계는 성공 의존성 때문에 시작하지 않는다. 실패를 숨기거나 해당 seed를 제외하지 말고 로그를 확인한다.

## 완료한 검증

네 N×M 크기에서7방법 모두 finite update를 확인했다. 3→6→3 peak와 정규화, Exact OT의 독립 LP 대비 비용·marginal, GMM likelihood·gradient·OT 미호출, fixed sigma 불변성, v5 actor update와 reward-only twin TD/target update의 일치를 검사했다. Prefix 복사, Q 궤적 저장·재생, 중단 후 actor·critic·replay·RNG의 동일한 재개까지 통과했다.

- Exact OT vs 독립 LP 비용 차이 최대 약1.24e−8.
- 원래 v5 actor update와 parameter L∞ 차이: Sinkhorn9.62e−6 이내, GMM8.33e−6 이내. 부동소수점 연산 순서 차이를 허용한 수치 비교이며 bitwise 동일하다고 주장하지 않는다.
- 수치 코드384개 파일의 동일 hash를 login4·dildata에서 확인했다.
- Code ID: `3f57d312b62a555ba0e1c0776031d55337856542cf929a65c9276a3e0345ea94`.
- 검증 근거: `VALIDATION_0.json`–`VALIDATION_3.json`.
- 처음 발견한 grid/TF32 문제는 reference 평가만 highest matmul precision을 적용해 수정했다. 학습 경로는 v5 정밀도를 유지한다. 자세한 내용은 [PROTOCOL.md](PROTOCOL.md).

보고서 생성기는 실제 완료 prefix의 저장 표본으로 시험했다. Actor는32768개 action의 histogram이며 별도 KDE smoothing을 하지 않는다. Teacher의M개 표본 histogram과 actor histogram은 패널을 나누어 표시한다.

## 보관 위치와 재개

- 계산: `login4:/lustre/hobbit9882/OptiQ-nonstationary-q-20260917`
- 중앙 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260917_nonstationary_q`
- 자동 생성 최종 보고서: 위 두 실험 폴더의 `report/report.md`, `report/report.html`.
- 로컬 보고서 틀: `/Users/heejoon/Documents/ChatGPT/OptiQ/reports/20260917_nonstationary_q/`.

dildata tmux `optiq-nonstationary-sync`가3분마다 직접 수집하며 완료 run은 hash 검증 후 `BACKUP_VERIFIED.json`을 기록한다. 확인 시점에는19개 prefix가 hash 검증까지 끝났고, 이후 완료된 run도 다음 주기에 수집한다. 전체 run과 최종 보고서가 보관될 때까지 collector를 유지한다. 상태 파일은 스토리지의 `STORAGE_SYNC_STATUS.json`이다.

Checkpoint는200 updates마다 actor·optimizer, critic·target·optimizer, 학습/행동수집 RNG, 전체 유효 replay·sampling RNG와 환경 위치까지 저장한다. 동일 task index로 재실행하면 해당 checkpoint에서 이어간다. 시간 제한 예고 시 저장 후 requeue하도록 설정했다. 이미 등록된 전체 array를 새로 중복 제출하지 않는다.

기존 Vast MuJoCo 및 다른 login4 실험은 변경하지 않았다. SLURM accounting DB가 응답하지 않아 `sacct` 조회는 실패했으며, 위 상태는 `squeue`, 실행 로그, `progress.json`, `COMPLETE.json`으로 확인했다.
