# TRG SingleQ MC64 MuJoCo 실행 기록

- 원본: `direct-gmm-trg@4ca69473515b5083d6e651845ded39da34e14538`의 truncated MLL 구현
- 실행 소스: `heejoon@52ae6c4dd2502cb3384a410bf4197ad589306492` (학습 전 commit/push)
- 서버: `heejoonorm@103.177.249.208:37047`, RTX 5090 4장
- 경로: `/home/heejoonorm/OptiQ/trg_single_mc/52ae6c4dd250`
- tmux: `optiq-trg-singlemc`
- 조건: single critic, K64, N64, M64/M256, temperature0.25, 1Msteps
- 환경: Humanoid / HalfCheetah / Ant 각각 seeds0–3 =24runs
- 우선순위: Humanoid → HalfCheetah → Ant, cross-environment dependency 없음
- W&B: `OptiQ/DirectGMM_heejoon`
- 그룹: `20260921_TRG_SingleQ_MC64_N64_M{64,256}_T025`
- 단위 검증과 24config 검증, GPU 6smoke 완료 후 자동 시작

진행 상황은 서버 STATUS.json, 개별 LAUNCH.json/STATUS.json, W&B URL과 로그로 확인한다.

## 초반 실행 검증 완료

- 수식·sampling·gradient 검증9개 통과, 24config 검증 통과.
- 3환경×2크기 GPU smoke6개 통과. W&B 온라인 완료 기록 확인.
- Humanoid8개 실행 중, HalfCheetah8개/Ant8개 대기; GPU당2개 배치.
- 초기 점검: Humanoid 모든 run이24,754~29,566steps, finite actor/critic losses.
- dildata 자동백업:3분마다, 읽기 전용 공개키. 첫 동기화exit0,120MB 보관 확인.
- 저장경로: `dildata:/data1/heejoonorm/OptiQ/studies/20260921_trg_single_mc/campaign/`
- 소스백업389개 SHA256 일치 확인. 실행 source commit52ae6c4는 변경하지 않음.
- 실행별 URL/초기steps: `receipts/EARLY_PROGRESS.json`.

| Run | W&B |
|---|---|
| humanoid_m256_s0 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/mkf9bdj9) |
| humanoid_m256_s1 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/y4yrqrma) |
| humanoid_m256_s2 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/vldh3nbo) |
| humanoid_m256_s3 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/h0840azi) |
| humanoid_m64_s0 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/0f3kdhal) |
| humanoid_m64_s1 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/wbk0zt5x) |
| humanoid_m64_s2 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/8m3c93a6) |
| humanoid_m64_s3 | [run](https://wandb.ai/OptiQ/DirectGMM_heejoon/runs/r9cx7zp6) |
