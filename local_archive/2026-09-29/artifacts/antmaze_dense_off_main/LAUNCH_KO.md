# Dense AntMaze 16개 실행 기록

- 캠페인: `antmaze-dense-noveld-off-main-s0-20260923`
- 소스: `26336810f7ea4ca61210ea70c6aeeae9f7acaed0`, `direct-gmm-trg` 푸시 및 양 서버 공유 완료.
- v1/v2/v3/v4 × OptiQ/SAC/DIPO/MFPO, 각각 seed 0. 180 서버는 v1/v3, 199 서버는 v2/v4.
- 보상: 다음 위치에서 가장 가까운 목표까지의 거리의 음수. NovelD OFF. 추가 sparse 도달 보너스/정규화 없음.
- 256 환경, batch 4096, 256 transition 수집당 8 learner updates. 기존 모델·학습률·OptiQ T=.25 등 유지.
- 원본 예산: v1/v2 3,008,256, v3 4,008,448, v4 5,008,384 total transitions. 원본 3M/3M/4M/5M에 warmup 및 strict-stop 계산 포함.
- 평가 250k마다, 전체 체크포인트 최종만. 주 궤적은 매 episode 랜덤 시작 위치에서 direct policy로 평가. 고정 시작 및 OptiQ μ-only는 별도 보조 결과.
- 8 GPU에서 먼저 DIPO 4개·OptiQ 4개, SAC/MFPO는 개별 슬롯이 비면 2초 이내 순차 시작. 미로/방법 전체 완료 장벽 없음.
- 실패 시 해당 서버 대기를 보류하고 나머지 실행은 보존. 자동 재시작하지 않음.
- DIPO 수치 안정화 및 dense 호환 support [-6000,5] 적용. 51 atoms의 넓은 간격은 이전 검증 프로파일과 동일하며 이번 실행에서 추가 조정하지 않음.
- 10k-update 사전 실험은 별도 소스 c3366d1이며 8개 모두 검증됨. 목표 도달 성능 통과를 뜻하지 않음.
- W&B: https://wandb.ai/OptiQ/gmm-trg , group은 캠페인명과 동일.

상태/원자료 수집:

```sh
/tmp/optiq-antmaze-report-20260922/bin/python -m antmaze_experiments.collect_dense_off --root artifacts/antmaze_dense_off_main --stage main
```

모든 작업 완료 후에는 같은 명령에 `--complete`를 추가하면 config·원시 궤적·학습 방문 좌표를 보관하고 서버에 남은 최종 체크포인트 SHA256 및 저장 검증 결과를 확인한다. 기존 `dense_off_report`는 8개 probe 전용이므로 본실험에 그대로 사용하지 않는다.
