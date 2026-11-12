# Repository instructions

## Heejoon pure-online Adroit pilot — user instruction, 2026-11-12

- This study uses branch `heejoon-rebuttal`, based on `heechan-rebuttal`.
- Run only SAC and TD3 from scratch on native Adroit Sparse-v2, seeds4/5,
  approximately3M interactions. Vary only environment count and UTD as recorded
  in `adroit_online/PROTOCOL.md`; no iBOLT or alpha/noise sweep, no confirmation
  seed0-3 jobs yet. Existing F1/A1 experiments are unrelated and remain untouched.
- Use the two explicitly supplied servers `175.155.64.205:19159` and
  `179.255.127.236:31020`, user `heejoonorm`, four GPUs each. The original
  `203.75.167.56:42913` failed GPU device access and is excluded from this study.
- Commit exact source/configuration/protocol/launch scripts before training;
  record the full SHA and source manifest with run IDs. Archive data outside Git
  to `dildata:/data1/heejoonorm/OptiQ/runs/adroit-online-20261112`.
- New SAC uses the shared verified JAX implementation. Preserve the anonymous
  separation rules below. Do not restart or alter other users' campaigns.

## 실험 전체 중지 및 프로토콜 확정 대기 — 사용자 지시, 2026-11-12

- Fetch는 F1, Adroit는 A1을 기준으로 정리한다. 모든 신규 SAC는 검증된 JAX SAC를 사용한다.
- 사용자가 먼저 서버의 본인 실험을 모두 내리도록 지시했다. 본인 소유가 실행 경로·설정·배치 기록으로 확인되는 학습과 자동 후속 큐만 중지하며, 다른 사용자의 작업은 건드리지 않는다.
- 기존 결과·로그·체크포인트·실행 소스는 보존한다. 중지된 작업을 장애로 간주해 자동 복구하거나 구형 예약을 재개하지 않는다.
- DACER on/off의 기존 비교 증거를 확인하고 최종 설정을 정리하는 단계다. 부족한 과제를 seed 0~4, 5M으로 채우는 후속 학습은 현재 시작하지 않는다. 추가 실행 지시 전까지 새 학습·smoke training·자동 큐 등록을 보류한다.
- 이 최신 중지 지시는 이전의 계속 실행·후속 실행 지시보다 우선한다. 서버1 신규 배치 제한과 anonymous 분리 규칙은 유지한다.
- 서버2 (`ibolt-server-2`, `114.32.64.6:41442`)는 사용자가 "앞으로 버려"라고 명시했다. 향후 배치·자동 큐·점검 대상에서 제외한다. 접속 불가 상태에서 중지 완료로 간주하지 않는다.

### 후속 비교 범위 변경 — 사용자 지시, 2026-11-12

- 사용자가 DACER on/off 비교 대신 **DACER on을 기본으로 고정하고 iBOLT actor temperature {0.05, 0.1, 0.25, 0.5, 1.0}만 sweep**하도록 변경했다. DACER-off 실험이나 예약은 재개하지 않는다.
- Fetch F1 및 Adroit A1의 나머지 설정은 고정한다. SAC의 자동 entropy 조절은 유지하고, iBOLT temperature sweep을 SAC 고정 alpha sweep으로 확대하지 않는다.
- 이전 큐를 일괄 재개하지 않는다. 새 sweep의 명시된 범위와 일치하는 작업만 별도 실행 기록으로 준비한다.
- 사용자가 탐색 seed **5·6, 각 5M**으로 온도 비교 후 과제별 최적 온도를 **seed 0~4, 각 5M**으로 확인하는 범위를 선택했다. Fetch·Adroit 8개 과제, 온도 5개로 탐색 대상은 80개 run이다. 이 새 sweep 및 후속 확인은 실행 가능하며 이전의 전체 보류에서 이 범위만 해제한다.
- 온도 선택은 탐색 seed 두 개의 마지막 5회 공통 정기 평가 성공률 평균으로 고정한다. 확인 seed 0~4의 성능으로 온도를 다시 선택하거나 좋은 반복만 추려내지 않는다. 동률이면 사전에 고정한 온도 순서(0.05, 0.1, 0.25, 0.5, 1.0)의 첫 값을 사용한다.
- 사용자가 **JAX SAC와 iBOLT 온도 sweep을 모두 현재 정한 설정으로 실행**하도록 최종 지시했다. JAX SAC는 8개 과제 × seed 0~4, 각 5M 대조군이다. iBOLT는 탐색 80개 조합 후 선택된 온도의 확인 40개다. 기존 F1과 정확히 일치하는 완료된 Slide 탐색 6개(T=0.1/0.25/0.5, seed 5·6)는 결과에 따른 재선별 없이 재사용한다.
- 신규 배치는 서버3·4·5의 여유 GPU로 제한한다. 다른 사용자 프로세스가 사용하는 GPU는 제외한다. 서버1은 새 배치에서 제외하고 서버2는 폐기 대상으로 유지한다.
- 중앙 저장소는 `heechan@dil.yonsei.ac.kr:2200`의 `/data1/heechan/iBOLT/20261112-f1-a1-temp`이다. 학습 서버의 `/workspace/heechan-unified-f1-a1-20261112`를 자동 수집한다. 중앙 수집기의 전용 SSH 키는 허용된 본인 실험 경로에 대한 제한된 수집·해시 확인·확인 실험 선택 전달에만 사용한다.
- 사용자가 디스크가 부족한 서버는 공간을 먼저 확보하도록 지시했다. 기존 결과는 중앙 저장 후 SHA256 일치가 확인된 체크포인트 사본만 학습 서버에서 정리한다. 원본 로그·설정·소스 및 각 run의 최신 두 체크포인트 step은 보존한다. 다른 사용자 파일·프로세스는 정리 대상이 아니다.

### 이번 F1/A1 실행에 서버1 추가 — 사용자 지시, 2026-11-12

- 사용자가 **“서버1도 함께 사용하도록해”**라고 명시했다. 이번 Fetch F1 / Adroit A1 JAX SAC 및 iBOLT 온도 탐색·후속 확인은 서버1·3·4·5에 배치할 수 있다. 이 범위에 한해 이전 서버1 제외 규칙을 해제한다. 서버2 제외는 유지한다.
- 실행 중인 학습은 보존하고, 대기 작업을 중복 없이 재배치한다. 서버1도 동일한 고정 소스·데이터·설정을 사용하며 기존 중앙 수집·해시 검증·체크포인트 보존 정책에 포함한다.
- 다른 실험의 서버1 사용 제한이나 준비 전용 CrossQ 실행 보류를 해제하지 않는다.

### F1/A1 GPU별 병렬도 확대 — 사용자 지시, 2026-11-12

- 사용자가 **“한 gpu 당 5개씩 올릴 것”**이라고 명시했다. 이번 F1/A1의 사용 가능한 GPU마다 최대 5개 학습을 배치한다. 다른 사용자가 사용하는 GPU는 계속 제외한다.
- CPU·메모리·디스크를 확인하고 실행 중인 learner는 유지한 채 CPU affinity와 운영 큐만 조정한다. vec256·수집 worker 수·UTD·batch·알고리즘·평가·데이터는 바꾸지 않는다. 대기 작업의 서버 간 이동은 중복 없이 기록한다.

### FetchSlide 성공 허용반경 20cm — 사용자 지시, 2026-11-12

- 사용자가 **“slide 허용반경 20cm로 다시 돌려. slide 실험들 모두 종료. 다시 올릴 것. 앞으로 기본 세팅도 20cm로해”**라고 명시했다. 앞으로 FetchSlide의 기본 `distance_threshold`는 **0.20m**이다. 이전의 Slide 환경 무변경 지시는 이 거리 기준에 한해 대체한다.
- iBOLT와 JAX SAC 모두 학습 환경·HER 보상 재계산·평가·성공 통계에 동일한 20cm 기준을 사용한다. 물리·목표 분포·horizon50 및 나머지 F1 설정은 유지한다. 논문·그림·설정에는 **Slide 20cm**를 명시한다.
- 기존 Slide 학습·예약만 종료·교체한다. 기존 로그·체크포인트·소스는 보존하며 다른 과제와 다른 사용자의 작업은 유지한다. 이전 5cm Slide 탐색 6개도 새 조건에 재사용하지 않는다.
- 새 Slide는 별도 run ID·소스 스냅샷으로 초기화한다. iBOLT 5개 온도 × 탐색 seed5·6의 10개와 JAX SAC seed0~4의 5개를 각각 5M까지 실행한다. 탐색 완료 후 기존 선택 규칙으로 고른 온도의 독립 seed0~4 확인 5개를 해제한다. 확인 seed를 미리 실행하거나 구형 큐를 재개하지 않는다.

### F1/A1 DACER 목표 엔트로피 후속 탐색 — 사용자 지시, 2026-11-12

- 기존 온도 탐색·JAX SAC·실행 중인 학습은 유지한다. 과제별 기존 탐색 seed 5·6의 사전 정의된 기준으로 선택된 actor temperature를 고정한 후, DACER 목표 엔트로피를 차원당 {-0.9, 0.0, 0.25, 0.5, 0.75}에서 seed 5·6, 각각 5M 탐색한다. 온도 선택 전에는 대기한다.
- -0.9 탐색은 동일 과제·선택 T·seed의 기존 결과를 재사용한다. 새 4개 목표 × 8과제 × 2seed = 64개를 처음부터 실행한다. 각 과제의 10개 결과가 완료된 뒤 마지막 5회 공통 정기 primary 성공률의 두 seed 평균으로 목표를 선택하며, 동률이면 낮은 목표를 선택한다.
- 선택한 T·엔트로피 목표의 독립 seed 0~4를 각각 5M 실행한다. 목표가 -0.9이면 기존 동일 설정의 확인 실행을 재사용하며 중복 실행하지 않는다. 확인 결과로 T·엔트로피를 재선택하지 않는다.
- 차원당 목표에는 각 환경의 실제 행동 차원을 곱한다. Fetch noise scale=2, Adroit=0.1, 초기 alpha=0.27, alpha 학습률=0.03, 업데이트 간격=10000 및 F1/A1의 나머지 설정은 유지한다. 새로운 알고리즘·보상·환경 보정을 추가하지 않는다.
- 서버1·3·4·5의 기존 F1/A1 큐와 중앙 저장소에 별도 entropy_pilot/entropy_confirm 단계 및 동결 소스로 등록한다. GPU당 기존 실행을 포함하여 최대 5개, 검증된 런타임 범위와 다른 사용자 보호·디스크/해시 검증 보존 규칙을 유지한다. 서버2는 제외한다.
- 실행 기록은 outputs/fetch-adroit-f1-a1-entropy-20261112/protocol.json을 기준으로 한다. 기존 온도 비교 결과와 새로운 목표 엔트로피 탐색 결과를 구분해 보고한다.

### 서버2 새 주소 재등록 — 사용자 지시, 2026-11-12

- 사용자가 `ssh -p 41188 heechan@114.32.64.6`을 목록에 추가하고 사용하도록 명시했다. 이 새 인스턴스를 `ibolt-server-2`로 등록하며 이전 주소 `:41442`의 폐기·접속 금지 지시는 새 `:41188`에 적용하지 않는다.
- 서버2의 런타임·환경·시연·GPU·디스크를 검증한 뒤 이번 F1/A1 및 DACER 목표 엔트로피 탐색·확인 큐를 서버1~5에 분배할 수 있다. 아직 시작하지 않은 작업만 중복 없이 옮기며, 기존 learner와 다른 사용자의 작업은 보존한다.

### 엔트로피 탐색 즉시 병렬 실행 — 사용자 지시, 2026-11-12

- 사용자가 GPU 대기를 줄이기 위해 **“현재 공통 step 최선 T로 바로 시작”**을 선택했다. 엔트로피 탐색에 한해 모든 온도 탐색의 5M 완료를 기다리는 조건을 해제한다.
- 각 과제의 5온도 × seed5·6 전부에 존재하는 마지막 5회 공통 정기 평가 성공률 평균으로 현재 T를 고정한다. 동률은 낮은 T다. 고정값·공통 step·근거는 `outputs/fetch-adroit-f1-a1-entropy-20261112/capacity-audit/current-step-temperature-proposal.json`과 배포된 `entropy-temperature-selections/`에 기록한다.
- 기존 온도 탐색·최종 T 확인·JAX SAC는 유지한다. 엔트로피 탐색에 고정한 T는 이후 원래 온도 탐색 결과나 확인 seed 성능에 따라 바꾸지 않는다. 현재 step 고정과 기존 5M 온도 선택을 보고서에서 구분한다.
- 고정 T에서 차원당 H={-0.9,0,0.25,0.5,0.75}, seed5·6 각각5M 비교 후 기존 기준으로 H를 선택하고 seed0~4 확인한다. H=-0.9의 탐색은 같은 T의 기존 pilot을 재사용한다. H=-0.9 확인은 원래 최종 T와 같을 때만 재사용하고, 다르면 새 고정 T 확인을 실행한다.
- 실행 중 learner PID는 보존하고, 미시작 탐색을 서버1~5의 사용 가능한 GPU에 균등 분배한다. GPU당 최대5개·서버5 GPU3 제외는 유지한다. 기존 알고리즘·데이터·학습·평가 조건은 유지한다.

### F1/A1 운영 점검 기록 — 2026-11-12 12시

- 서버3의 디스크 시작 기준 때문에 대기하던 미시작 작업 15개를 서버4(Fetch 11개)·서버5(Adroit 4개)로 이동했다. 기존 learner와 과학 설정은 유지했다. 실제 배치는 각 서버 `plan.json`과 `outputs/fetch-adroit-f1-a1-entropy-20261112/status-rebalance-20261112T1210/movement.json`을 따른다. 이전 호스트에서 중복 재개하지 않는다.
- DACER H 선택기의 부동소수점 동률 오류를 발견해 성공 에피소드의 정수 개수로 비교하도록 수정했다. Pen의 H/차원=-0.9, 0.25, 0.5는 동일한 탐색 seed·마지막 5회 공통 평가에서 모두 94/100이었다. 사전 규칙에 따른 선택은 **T=0.1, H/차원=-0.9**이며, 기존 동일 조건의 `adroit-pen-ibolt-s{0..4}-t0p1-confirm`을 사용한다. 확인 seed 성적에 따른 재선택이 아니다.
- 잘못 선택되어 완료된 Pen H=0.25의 seed 0·1·2·4는 보조 실험으로 보존하고 주 비교에서 제외한다. 미시작 seed3 예약은 `not_selected`로 해제했으며 재개하지 않는다. 선택 파일은 중앙 및 서버1~5에 정정했다. 기존·정정 선택과 동일한 탐색 증거는 `outputs/fetch-adroit-f1-a1-entropy-20261112/selection-tie-audit/` 및 중앙 `administration/entropy-tie-correction-20261112T1220/`에 보존했다. 나머지 7개 H 선택과 기존 8개 최종 T 선택은 재검증 결과 변경이 없다.
- 서버1·2의 로컬 SSH 점검은 `KexAlgorithms=curve25519-sha256`으로 연결한다. 접속 오류를 학습 실패로 간주하지 말고 중앙 수집 시각과 learner 상태를 확인한다.

### F1/A1 N·M 비교 및 DACER 목표 복원 — 사용자 지시, 2026-11-12

- 사용자가 “dacer는 되돌리고 n=64 m=256 n=16 m=64 에 대해 지금 빈 gpu들에 fetch adroit 실험 나눠서 올려봐”라고 지시했다. 새 비교는 DACER **on, 차원당 목표 -0.9**를 사용한다. 기존 실행의 설정을 중간에 바꾸거나 결과를 덮어쓰지 않는다.
- Fetch·Adroit 8과제 × (N=64/M=256, N=16/M=64) × 탐색 seed5·6, 각 5,000,192 step의 새 초기화 **32개**를 실행한다. 두 조합 모두 `proposals_per_policy_sample=4`이며 M은 실제 총 후보 행동 수다.
- Temperature는 원래 5M 온도 탐색의 사전 선택 규칙으로 확정된 값: Fetch Reach=.05, Push=.05, Pick-and-Place=.1, Slide=.1; Adroit Door=.1, Hammer=1, Pen=.1, Relocate=.25다. 엔트로피 탐색을 조기 시작할 때 고정했던 T와 구분한다.
- F1/A1의 데이터·시연·actor/critic256×2·LN·UTD·vec256·worker4·평가·Slide20cm는 유지한다. 기존 같은 T/H의 N=64/M=64 seed5·6 결과를 비교 대조군으로 사용한다. SAC와 탐색 seed 집합이 다르면 짝지은 5시드 검증 결과로 표현하지 않는다.
- 실험 기록은 `outputs/fetch-adroit-f1-a1-nm-20261112/protocol.json`이다. 기존 허용 범위의 서버1~5 중 빈 GPU를 우선하며 GPU당 최대5개, 타사용자 보호, 중앙 수집·SHA256·최신2step 보존을 유지한다. 최초 배치는 서버1 Fetch8, 서버2 Fetch8+Adroit4, 서버3 Adroit12다.
- ESS·최대 중요도 가중치·후보 Q/로그밀도 표준편차를 추가 기록하되 난수 추출·손실·업데이트 식은 변경하지 않는다. N/M의 최적값 자동 선택이나 후속 seed0~4 실행은 아직 추가하지 않는다. 기존 249개 캠페인과 새32개를 구분하며 기존 controller를 종료할 때 새 실행까지 끝났는지 확인한다.

### SAC alpha=0 탐색 — 사용자 지시, 2026-11-12

- 사용자가 Fetch·Adroit 8과제 모두에 JAX SAC의 고정 alpha=0을 seed 0·1, 각 5M으로 시험하도록 선택했다. 자동 alpha 조절을 끄며 기존 자동-alpha SAC의 동일 seed·step과 비교한다. 신규 16개는 별도 탐색이며 기존 iBOLT·SAC 실행과 결과를 보존한다.
- F1/A1의 네트워크·LayerNorm·동일 expert 시연·HER·정규화·UTD·vec256·batch256·평가·Slide20cm는 그대로 둔다. 정책의 확률적 수집은 유지하며 alpha=0을 결정론적 알고리즘으로 바꾸지 않는다.
- `baselines/jax_sac.py`에 고정 0 지원을 확장하고 손실·기울기·Adam·target·저장/복원을 기존 SAC와 대조한 후 실행한다. 프로토콜은 `outputs/fetch-adroit-sac-alpha0-20261112/protocol.json`이다. 기존 승인 서버의 남는 슬롯과 중앙 수집·해시 보존 정책을 사용한다.

## 최우선 규칙: PointMaze와 anonymous 브랜치의 완전한 분리

**사용자의 명시적 금지 사항이다. PointMaze 코드나 설정에 관한 모든 작업과
논의 내용은 anonymous 브랜치에 절대로 섞어 넣지 않는다. 사용자 본인이
이 금지 규칙을 명시적으로 변경하지 않는 한 예외를 두지 않는다.**

### 보호 대상

- `codex/ibolt-anonymous`를 포함하여 이름에 `anonymous`가 들어가는 모든
  로컬·원격 브랜치와 익명 배포용 스냅샷·미러·내보내기를 보호한다.
- PointMaze 전용 브랜치, 별도 worktree 또는 현재 rebuttal 작업에서 얻은
  내용이라는 이유로 익명 배포 경계를 넘어 반영해서는 안 된다.

### 금지 범위

PointMaze와 관련된 다음 내용을 보호 대상에 추가·복원·수정·이식하거나,
그 대상으로 커밋·merge·cherry-pick·rebase·push·복사·export하지 않는다.

- 환경·학습·평가·agent·vector env/collector 구현과 관련 코드.
- 하이퍼파라미터, seed, N/M, temperature, batch, UTD, warmup, 병렬 환경 수,
  보상, 미로, horizon, 평가 방식 및 실행·스케줄링 설정.
- README, 프로토콜, 보고서, 주석, 커밋 메시지, PR 설명 등 PointMaze 논의의
  문서화 내용과 과거 실험 복원·비교·분석 결과.
- 테스트, 지도·asset, 스크립트, dependency 변경, 설정 파일, 체크포인트,
  로그, 그림, 수치, 원시 데이터 및 결과·출처 링크.
- `pointmaze/`뿐 아니라 `maze_benchmarks/`, `pointmaze_experiments/`,
  `analysis_tools/`, `configs/` 등 다른 경로와 공용 파일에 들어가는 관련 변경.

파일명이나 경로를 바꾸거나 공용 기능으로 표현해도 PointMaze 목적 또는
PointMaze 논의에서 파생된 내용이면 같은 금지 규칙을 적용한다. 관련 변경이
섞인 커밋을 통째로 익명 브랜치에 옮기는 것도 금지한다.

### 작업 절차

1. 변경·커밋·브랜치 간 이식·배포 전에 현재 브랜치와 최종 반영 대상을 확인한다.
2. PointMaze 작업은 전용 브랜치/worktree 또는 `heechan-rebuttal` 같은
   비익명 작업 영역에서만 수행한다. anonymous 브랜치를 PointMaze 작업의
   중간 저장소나 임시 작업 공간으로도 사용하지 않는다.
3. 익명 브랜치로 옮길 변경은 커밋 메시지·전체 diff·새 파일·공용 설정·문서·
   테스트·의존성까지 확인하여 PointMaze 관련 내용이 없음을 검토한다.
4. 혼합 변경이 발견되면 비익명 작업 영역에서 먼저 분리하고, PointMaze 관련
   부분은 익명 반영 대상에서 완전히 제외한다. 관련 변경 여부가 불분명한
   항목도 보호 대상으로 반영하지 않는다.
5. 이미 익명 스냅샷에 PointMaze 내용이 존재하더라도 추가 반영의 근거로 삼지
   않는다. 기존 익명 스냅샷 정리는 별도 작업으로 다루며 PointMaze 개발 변경과
   섞지 않는다.

**이 규칙은 PointMaze 작업의 기본 제약이다. 실험 준비, 호환성 수정, 코드 정리,
문서 보완, 재현성 확보 또는 편의를 이유로 완화하지 않는다.**

## 서버 배치 규칙 — 사용자 지시, 2026-11-10

- 서버1 (`ibolt-server-1`, `heechan@122.117.118.68:41261`)은 사용자가 별도
  실험에 사용할 예정이므로 신규 실험을 추가로 배치하지 않는다. 자동 큐와
  후속 seed 실행 대상에서도 제외한다. 사용자가 명시적으로 변경하기 전까지 유지한다.
- 신규 실험은 서버2~4 위주로 배치한다. 이 지시만으로 새 실험을 시작하지 않는다.
- 이 배치 규칙을 이유로 기존 실행 중인 다른 실험을 중지하지 않는다.
- 사용자가 중지한 Fetch 10M 탐색과 후속 자동 실행 큐는 다시 시작하지 않는다.
  기존 SAC·iBOLT 5M 비교 실험은 유지한다.

### 이번 검증에 대한 서버1 사용 허용 — 사용자 지시, 2026-11-11

- 사용자가 iBOLT 후보 Q 평가의 정규화 통계 검증에 서버1~4를 모두 활용하도록
  명시했다. 이 검증 작업에는 서버1도 사용할 수 있다. 다른 신규 실험에 대한
  서버1 배치 제한은 유지한다.
- 기존 CrossQ 실행을 재개하는 대신 별도 검증 경로와 실행 기록을 사용한다.

### 추가 서버 배치 대상 — 사용자 지시, 2026-11-11

- `ibolt-server-3` (`heechan@211.72.13.201:42866`)과 `ibolt-server-5`
  (`heechan@211.72.13.201:42757`)를 새 실험 배치 후보에 포함한다.
  두 주소는 포트가 다른 별도 인스턴스다. 배치 전 GPU와 디스크 여유를 확인한다.

### 서버1 CrossQ 표준 설정 재검증 준비 — 사용자 지시, 2026-11-11

- Humanoid(T=0.5)·Hopper(T=0.05)의 iBOLT+CrossQ, mean Q, actor BRN,
  logstd [-5,-1], UTD1, batch256, 각 seed 0–4 실험에 서버1 사용을 허용했다.
- 이후 사용자가 **“준비만 하고 일단 돌리진 마”**로 범위를 제한했다.
  현재는 코드·설정·실행 스크립트 준비 및 정적 검증만 허용한다.
  추가 실행 지시 전에는 학습·smoke training·자동 큐 등록을 하지 않는다.
- 이 허용은 다른 실험의 서버1 배치 제한을 해제하거나 기존 큐를 변경하지 않는다.

## Fetch 논문 실험: 국소적 보정 금지 — 사용자 지시, 2026-11-11

- 사용자는 **“논문에 써야돼서 국소적인 조정같은거 하지마”**라고 명시했다.
  특정 환경이나 관찰된 실패 사례에 맞춘 임의 보정으로 성능을 올리지 않는다.
- 원 알고리즘의 업데이트 규칙과 공통 실험 프로토콜을 유지한다. 행동의 특정
  차원을 강제로 고정·마스킹하거나, 보상·종료 조건·할인율을 바꾸거나, 특정
  방법의 수집·업데이트 시점만 바꾸는 방식을 성능 보정 수단으로 쓰지 않는다.
- 로그·체크포인트·시뮬레이터를 읽는 진단은 가능하지만, 진단에서 발견한
  패턴을 근거로 환경별 학습 보정을 임의 추가하지 않는다. 진단용 변형 결과를
  공통 설정의 논문 비교 결과에 섞지 않는다.
- 사용자가 명시적으로 요청한 현재 구조 비교는 **FetchSlide의 iBOLT만,
  actor와 critic 모두 256×3, 5M, seeds 0–4**이다. 기존 256×2 대비 깊이만
  바꾸며 T=0.5, logstd [-5,1], vec256, UTD=1/8, batch256,
  수정된 HER 정규화, LayerNorm off를 유지한다.
- 사용자는 이어서 **“기본환경을 써”**라고 명시했다. Farama의 기본
  `FetchSlide-v4` 환경을 사용하며, 보상·물리·목표 분포·행동 공간·에피소드
  길이를 환경별로 덮어쓰지 않는다. 기존에 요청된 HER와 관측 정규화는
  공통 학습 처리로 유지하며 환경 자체를 변형하는 근거로 삼지 않는다.

### FetchSlide 256×3 비교에 SAC 추가 — 사용자 지시, 2026-11-12

- SAC도 actor·critic 모두 256×3으로, seeds 0–4를 각각 5M까지 병렬 실행한다.
  위의 iBOLT 한정 범위는 이 구조 비교에 대해 SAC까지 확장되었다.
- 기존 SAC의 나머지 설정(기본 FetchSlide-v4, 자동 entropy 조절, gamma 0.99,
  vec256, UTD 1/8, batch256, 수정된 HER 정규화, LayerNorm off)을 유지한다.
- 서버1을 제외하고 여유 있는 서버에 배치한다. CrossQ의 불필요한 체크포인트
  정리는 허용되었지만, 다른 실험의 체크포인트 삭제로 범위를 넓히지 않는다.

### FetchSlide SAC를 JAX로 전환 — 사용자 지시, 2026-11-12

- 기존 SAC와 동작을 검증한 JAX 구현으로 actor·critic 256×2 및 256×3을
  각각 5M, seeds 0–4 병렬 실행한다. 나머지 SAC·환경·HER·평가 설정은 유지한다.
- 기존 PyTorch SAC 실행은 JAX 구현과 검증을 완료하고 새 실행의 정상 업데이트를
  확인한 뒤 종료한다. 기존 iBOLT나 별도의 진단 실험은 중지 대상으로 확장하지 않는다.
- 기존 결과와 실행 소스는 보존하고, JAX 실행은 별도 소스 스냅샷과 결과 경로에
  기록한다. 프레임워크 간 난수열과 부동소수점 반올림의 차이를 구분해서 보고한다.

## 모든 향후 SAC 실험은 검증된 JAX SAC 사용 — 사용자 지시, 2026-11-12

**앞으로 모든 벤치마크의 신규 SAC 학습·재실험·자동 큐는 아래 JAX SAC를
사용한다. Fetch에만 한정된 규칙이 아니다. 사용자의 명시적 변경 지시 없이
PyTorch SAC나 다른 SAC 구현으로 되돌리거나 대체하지 않는다.**

- 기준 구현: `baselines/jax_sac.py`의 `JaxSAC`.
  학습과 행동 추론은 JAX로 수행한다. GPU 실험은 실제 JAX GPU backend를 확인한다.
- 기존 SAC와의 수치 검증: `rebuttal_tools/check_fetch_jax_sac.py`.
  Fetch 실행 연결 예시: `rebuttal_tools/fetch_jax_sac_worker.py`.
- 이 구현의 PyTorch/SB3 사용은 기존 seed별 초기 가중치 보존, 공통 수집·replay
  인터페이스, 호환 체크포인트 및 수치 비교용 참조 코드에 해당한다.
  이를 PyTorch 학습 루프를 신규 실험에 사용하는 근거로 해석하지 않는다.
- 현재 구현에는 Fetch에서 검증한 관측·행동 및 학습 설정에 대한 제약이 있다.
  다른 벤치마크나 미지원 설정은 **해당 실험의 기존 알고리즘·하이퍼파라미터·
  수집·평가 조건을 유지하면서 JAX 지원을 확장하고, 동작 검증 후 실행한다.**
  구현 제약에 맞추려고 실험 설정을 바꾸거나 검증 없이 기존 SAC로 우회하지 않는다.
- 확장 시 동일한 가중치·배치·잡음으로 손실, 기울기, entropy 자동 조절,
  Adam, target 업데이트와 저장·복원을 대조하고 실제 환경에서 연결을 검증한다.
  프레임워크 간 난수열 차이를 학습 궤적의 완전한 일치로 보고하지 않는다.
- 기존 결과·체크포인트·실행 소스는 보존한다. 이 규칙 자체는 실행 중인 작업을
  일괄 중지하거나 새 실험을 시작하라는 지시가 아니다.
- anonymous 분리 규칙은 그대로 적용한다. 이 비익명 구현이나 관련 변경을
  anonymous 브랜치로 복사·이식·배포할 권한으로 해석하지 않는다.
