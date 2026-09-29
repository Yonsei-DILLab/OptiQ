# 서버 설정 및 1M 큐 상태

최초 기록: 2026-09-26 13:25 KST. 아래 최초 배정은 13:32에 변경됐으며 현재 배정은 문서 끝의 갱신을 따릅니다.

추가 서버 `vast-heechan-6` (136.63.24.6:41636, RTX 5090×4)의 Codex 0.157.1 ChatGPT 로그인과 doctor 21개 항목을 통과했습니다. CUDA·PyTorch·JAX 연산, 실제 batch4096 사전학습·평가·저장을 확인했습니다.

|서버|GPU|배정|현재|
|---|---:|---|---|
|vast-heechan-199|4|12개|실행 4, 대기 8, 완료 0|
|vast-heechan-46|8|13개|실행 8, 대기 5, 완료 0|
|vast-heechan-6|4|6개|실행 4, 대기 2, 완료 0|

실행에는 사전검사 단계가 포함됩니다. 46 서버의 8개 작업도 사전검사를 모두 통과하여 본학습으로 전환됐습니다(13:26 확인). 슬롯별 독립 큐가 다음 작업을 자동 배정합니다.

학습 소스: `a55aaf13f7803b5cf8da7ba56f318675c7794171`. 추가 서버 배정·수집기 소스: `7e6c83edf8d7d02156ef46adf6820ecd89bfa6f5`. GitHub `maze-deadline-1m-20260926` 브랜치로 푸시했고 세 서버에 공유했습니다.

신규 31개 + 검증 완료 과거 2개(8/16-Way T1 재사용), 총33개 결과가 대상입니다. 새 런은 256env, batch4096,16updates/256transitions, 1,000,192 transitions이며 기존 T1 2개와 UTD가 다릅니다. 과거 결과를 순수한 temperature 대조군으로 단정하지 않습니다.

46/6의 새 런타임은 복사 대신 고정 버전 uv 설치로 준비했습니다. PyTorch2.7.1+cu128 / JAX0.4.33 / Python3.11.16이며 전체 패키지 목록과 준비 기록은 hosts/<host>/runtime-readiness 에 보관합니다.

완료 여부는 마지막1M 평가·checkpoint/replay·학습 update 수·원시 goal counts·SHA256 검증으로 판정합니다. 현재 이 문서는 서버 설정 완료 보고이며 전체 학습 완료 보고가 아닙니다.

## 13:57 KST 갱신

아직 시작되지 않은199의 8개를 원자적으로 이관했습니다. 현재 배정은199 4개(DIPO3+Hard MEOW),46 15개(Way OptiQ10+MFPO3+MEOW2),6 12개(PointMaze OptiQ3+SQL3+SAC3+TD3 3)입니다. 기존199 항목은 transferred로 보존하며 중복 실행하지 않습니다. 이관 계획/도구 커밋은 b3e5d468dd73a5c662d73a0bda4ee0d33b62d1b9입니다.

46의 Medium MEOW 사전검사에서 모델 생성 전 tyro 누락으로 실패했습니다. tyro0.8.14와 관련 CLI 의존성3개만199와 같은 버전으로 추가했으며 NumPy/JAX/PyTorch와 학습 설정은 바꾸지 않았습니다. 실패 디렉터리·로그·명령·job은 preserved-attempts/meow-missing-tyro에 보존하고 복구 근거는 hosts/vast-heechan-46/meow-dependency-repair.json 및 meow-runtime-readiness.json에 저장했습니다. 조사에 따른 일회성 복구 코드/프로토콜은8930bbac6de5f3984be650bbaee5655d3fa0e3bf이며 GitHub와 세 서버에 공유됐습니다. Medium/Simple MEOW 모두8448step/16update GPU 사전검사와 저장 검증을 통과한 후 원래 frozen a55 소스로 본학습 중입니다.

이전 취소 캠페인의 학습/컨트롤러 프로세스가180/199에 남아있지 않음을 /proc의 실제 argv로 재확인했습니다. cancellations/*-verified.json에 점검 시각과 결과를 보관합니다. 다른 사용자의 프로세스나4090 작업은 이번 대상이 아닙니다.

로컬16/33개 검증까지 완료된 상태이며 전체 완료를 의미하지 않습니다. 이후 최신 상태는 status.json, archive-manifest.json과 최종 completion-audit.json에서 확인합니다.
