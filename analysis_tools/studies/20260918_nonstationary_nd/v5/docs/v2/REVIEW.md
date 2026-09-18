# Final v2 review — 2026-09-11

**완료된 checked K64 구성을 최종 v2로 확정하고, 기본 경로와 상세 명세를 검토했다.**
기준 학습 소스는 `8cb4f237aacb61113f65e693e23236f17d77f978`이다.
이번 정리에서 공통 학습 소스 39개의 SHA-256은 변경 전과 모두 같다.
최종 구성은 연속 latent, mean-Q teacher, min-Q TD/guard, T=.1, beta1,
조건부 Gaussian proposal, 16×64 full OT NLL, IDAC M16, sampled guard를 사용한다.

## 검토 결과

| 검토 | 결과 | 근거 |
|---|---|---|
| 최종 설정 ↔ 완료된 run 설정 | `alg` 전체 값 동일 | `REFERENCE_CONFIG.json`, `test_v2_final.py` |
| 기본/호환/직접 설정 | `mujoco_v2`, `mujoco_v2_checked`, `v2/final` 동일 | Hydra compose 및 실행 launcher 확인 |
| 구버전 보존 | 5개 archive 설정의 전체 resolved 값 동일 | `tests/data/v2_pre_final_configs.json` |
| 기존 구버전 명령 | conditional/guarded/finite/proximal 이름의 값 동일 | 호환 alias 회귀검사 |
| 환경 선택 | Humanoid/Ant/HalfCheetah/Walker2d/Hopper 정상 compose | 각 환경 validate_config 통과 |
| 의사코드의 밀도/OT/NLL | 독립 NumPy 계산과 JAX 결과·gradient 일치 | `reference_math.py`, final 수치 테스트 |
| 원본 업데이트와 현재 코드 | actor/critic/guard 상태·기존 지표·반환 RNG array-equal | `review/historical_update_parity.json` |
| 실제 공통 학습 경로 | 짧은 Humanoid 실행·timeout bootstrap·checkpoint 복원 통과 | final integration test |
| 기존 후처리 | 12개 기준 모델 검증 및 설정 이동 검증 통과 | `review/legacy_readiness.json` |
| 통합 검증 | **92 tests passed** | `review/validation.txt` |

## 수치 검토의 구체적인 범위

밀도는 action 차원을 먼저 합한 뒤 component logsumexp를 계산하고, 안정적인
tanh Jacobian을 포함하는지 확인했다. Teacher floor가 활성화된 좌표를 포함해
sampling/density의 일치, state 간 분리, 생성 component 포함을 확인했다.
NLL의 전체 결합 계산과 가중 moment 계산을 비교하고, μ/logσ 미분을 독립 해석식과
유한차분으로 검증했다. Q/teacher/OT에 gradient가 흐르지 않는 테스트도 통과했다.

Sinkhorn은 raw 거리, epsilon=.25, 100회, column mass floor=1e-20,
행별 NLL 정규화 규칙으로 대조했다. Critic loss는 두 MSE의 합이며, 두 critic을
함께 gradient clipping하는 규칙과 terminal/timeout 처리를 확인했다.
Guard는 old/new에 같은 난수를 쓰고, generating+15 vs independent16 component로
entropy bracket을 만들며, 거절 시 optimizer 상태까지 복원하는지 검증했다.

원본 소스 비교는 당시 `algorithm.py`와 `soft_improvement.py`의 함수 본문을
`git show`로 불러와 동일한 상태·난수를 입력한 검사다. 작은 네트워크/고정 Q fixture와
실제 Adam·clipping 설정, N16/K64/M16/J8/BG32를 사용했다. 공통 수치 helper는 현재
구현을 사용하므로 독립 NumPy 대조를 별도로 수행했다. 추가 진단 키
`backup_policy_density_exact`를 제외한 기존 critic 지표도 array-equal이었다.
이는 지정 입력에 대한 코드 경로 확인이며, 1M 학습을 새로 재현한 검사는 아니다.
재검토 스크립트는 `review/check_historical_updates.py`다.

실제 Humanoid integration은 최종 256×3 네트워크와 OT/entropy/guard 크기를
유지하고 테스트 시간 절약을 위해 warmup=2, batch=4, replay=32, total=8로 실행했다.
6번 critic 업데이트/actor 제안, 수락 횟수에 따른 actor step, time-limit bootstrap,
저장 actor의 행동 일치 및 critic 체크포인트 존재를 검사했다. 이 축소 설정은
실험 기본값에 반영하지 않았다. 검증은 CPU에서 실행하여 기존 GPU 작업과 분리했다.

## 경로 이동과 기존 실행의 호환성

최종 알고리즘 설정은 `configs/v2/final.yaml`에 한 번 정의하고, 기본 두 이름은
그 파일을 읽는 alias로 만들었다. 원안과 비교 구성 정의는 `configs/archive/v2/`,
관련 문서는 `docs/archive/v2/`에 보관했다. 기존 문서 위치에는 이동 안내를 남겼다.
원안 전용 calibration/screen/테스트는 `archive/v2/original`을 명시하도록 고정했다.

`outputs/`, 저장 정책, 평가 배열, W&B 기록, 기존 실험 protocol은 이동·수정하지 않았다.
공통 코드 및 기존 서비스가 참조하는 script 경로도 유지했다. 기존 최종 평가 도구의
파일 해시 검사는 설정 이동을 인식하도록 보완했다. 실행 source의 해시는 여전히
정확히 일치해야 하고, 설정 파일 차이는 **기록된 이전 해시 + 전체 이동 후 설정 해시 +
resolved 값 동등성**을 모두 통과해야만 허용한다. 알 수 없는 이전 해시나 학습 source
변경은 거절하는 테스트를 포함한다. 변경 허용 내역은 평가 결과에 기록된다.
확인 명령은 기존 실험을 시작하거나 중단하지 않으며, 미완료 실행은 계속 미완료로 표시한다.

## 의사코드 검토에서 보완한 재현 세부 사항

- Critic scalar 출력의 마지막 축을 제거하여 [B,1]과 [B]의 잘못된 broadcast 방지.
- Critic MSE 합산, action 차원 NLL 합산, clipping과 Adam 순서 및 epsilon 위치 명시.
- Guard 밀도당 M16이지만 base latent는 **M+1=17개** 생성하는 점 명시.
- Teacher의 실제 샘플링과 밀도에서 동일한 σ floor를 사용하는 점 명시.
- Actor/critic initializer, GELU 근사, latent 차원, 기본 Humanoid constructor 명시.
- Step 5001 첫 업데이트, 평가 시점과 저장 시점의 차이, 평가의 policy RNG 소비 명시.
- 원본 actor/critic checkpoint만으로 exact training resume가 되지 않는 점 명시.
- 문서용 수치 snapshot에 interpolation 문자열이 남았던 점을 원본 run JSON에서
  다시 추출하여 수정하고 최종 설정 동등성 테스트를 재실행했다.

## 남는 해석 범위

선택된 v2의 실제 4시드 실험 결과는 그대로 보존했다. 더 강한 이전 10% 탐색 기준보다
평균 보상이 낮고, 학습 중 일시적 하락이 있었던 사실도 변경하지 않았다.
사용자의 최종 버전 선택은 다른 알고리즘보다 모든 지표에서 우세하다는 선언이 아니다.
유한 M/K, Gaussian projection 및 replay 표본 평균 guard는 실제 매 업데이트의
global policy improvement를 인증하지 않는다. 이론 조건은 [THEORY.md](THEORY.md)에 있다.

다른 framework에서 동일 분포와 수식으로 구현해도 PRNG·연산 순서·GPU 차이 때문에
동일 seed가 동일 궤적을 보장하지 않는다. 다른 환경은 설정 선택 가능성을 검증했으며,
새 1M 성능 실험은 이번 경로 정리/명세 작성 작업에서 실행하지 않았다.

## 현재 서버 통일 검증

최종 릴리스 `894e4aa030ef88da309f1a93912d5921d4db3e5e`를 `/root/OptiQ`의
`v2`에 반영했다. 이전 로컬 Gaussian-W2 코드는 별도 보존하고, 기본 설정·실행기·
관리형 worker·문서 경로는 성공한 checked K64를 가리키도록 정리했다.
릴리스 대비 수치 설정 변경은 없고 `output_root`만 저장소 밖으로 옮겼다.

- 전체 회귀검사 181개 통과. 이후 경로 테스트 모듈 5개도 통과했다. 두 검사는
  중복되므로 186개의 독립 테스트로 합산하지 않는다.
- 성공 당시 actor/critic/guard와 현재 구현에 동일한 입력·난수를 주어 CPU와 GPU
  각각에서 비교한 초기 상태, 행동, 업데이트 상태와 기존 지표가 array-equal이었다.
  실제 Humanoid 초기 관측과 256×3 네트워크, N16/K64/M16/J8/BG32를 사용했다.
  한 업데이트 경로의 대조이며 전체 1M 궤적이나 성능의 재현 증명은 아니다.
- 실제 Humanoid GPU 검증은 기본 5K warmup 뒤 100회 업데이트까지 실행했다.
  Guard는 100회 제안 중 37회를 수락했고 actor/critic 상태는 모두 유한했다.
  최종 체크포인트 저장과 TrainState 복원도 통과했다. W&B run은 만들지 않았다.
- 다른 작업 디렉터리에서 실제 실행기의 seed 0·1·2·3 설정을 확인했다.
  네 worker는 1M, 같은 기준 프로젝트, 외부 결과 경로를 사용하며 자동 시작하지 않는다.

상세 검증 자료는 이 서버의 `/root/anal/v2_unification_20260911/`에 있다.
Git에는 분석 출력·비밀 환경변수·체크포인트를 포함하지 않는다.
현재 경로는 [INSTANCE.md](INSTANCE.md)를 따른다. 위의 92개 테스트 기록은
원래 릴리스의 검증으로 그대로 보존하며, 이 절은 후속 통일 검증이다.
