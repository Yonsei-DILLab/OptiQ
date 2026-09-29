# Ant-v4 Direct GMM fixed64: 4-seed campaign

요청대로 기존 v5 RL 학습 기본값을 유지하고 Ant-v4, T=.25, N=M64, fixed64 latent prior로 실행. actor와 critic은256x2 GELU. seeds0,1,2,3은 vast-heechan-199 GPU0,1,2,3에 독립 supervisor worker로 배치. 목표각1M 환경step, 성능기반 조기종료없음.

## 유지된 학습 기본값

- replay batch256, UTD1, policy_delay1
- actor/critic Adam learning rate3e-4; clipping없음
- gamma.99, target tau.005, replay capacity1M, warmup5000
- sigma 초기.5, mean head 초기화scale1e-4; toy의scale1로 변경하지 않음
- plain TD, entropy backup/guard/annealing/uniform replacement없음

## 요청된 변경

- 기존v5 Direct GMM NLL profile 상속, OT/Sinkhorn없음
- finite latent64: codebook seed20260911, 정규화한64x8 Gaussian codes. 네 training seed에서 동일codebook으로 초기화/환경난수만 달라짐
- 매state actor update에 전체64codes 사용; teacher는IID component mixture에서64후보
- collection/TD도 이64codebook에서균등선택. state가달라지면conditional mu/sigma는달라짐

## 평가 호환 변경

기존zero-z/stochastic-z dual evaluation은continuous prior만허용. 이번profile은dual_mu_eval=false로설정하고기존finite-policy-aware평가를사용: stochastic_z/epsilon0,10episodes,step1및5000마다. 학습률/UTD/batch 등training설정변경없음. Checkpoint50000마다.

## 실행 링크

|Seed|GPU|W&B|
|---:|---:|---|
|0|0|[run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/gxv5q9i8)|
|1|1|[run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/vso9d8ls)|
|2|2|[run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/670yefgm)|
|3|3|[run](https://wandb.ai/OptiQ/v5-heechan-gmm/runs/a72pwt97)|

Source commit: `25a692387159c3a35993446ffec6f847dd081104`
Frozen source: `/home/heechan/OptiQ-ops/sources/25a692387159c3a35993446ffec6f847dd081104`
Campaign: `/home/heechan/optiq-experiments/ant-direct-gmm-fixed64-20260920T150159Z`

실행전 full-size(batch256,N64,M64,256x2) 실제Ant 사전검증 통과:6actor/critic updates, fixed codebook보존, Sinkhorn금지, finite평가,serialization. 짧은검증은실제4runs의1M학습과별도이다. Source와profile을commit후frozen clone으로실행; push없음.

[Resolved config](resolved-seed0.json) · [Manifest](manifest.json) · [Latest checked status](status.json) · [Preflight](preflight_result.json)

## W&B 프로젝트 이동 (2026-09-21 KST)

사용자 요청에 따라 기존 4개 run을 `OptiQ/v5-direct-gmm`에서 `OptiQ/v5-heechan-gmm`으로 서버 측 이동. Run ID와 기존 기록을 유지하며 학습 프로세스를 재시작하지 않음. 위 실행 링크는 이동 후 주소. 원본 resolved config와 frozen source에는 실행 당시 프로젝트명이 보존되어 있음.

이동 후 원래 W&B 업로더는 이전 프로젝트 경로에 404를 받아 전송이 중단됨. 학습 프로세스는 정상 유지. W&B 0.29.0 공식 `wandb beta sync --live --project v5-heechan-gmm --entity OptiQ`를 4개 원본 `.wandb` 파일에 연결한 supervisor worker `ant-dgmm-f64-project-sync-20260920T150159Z`로 기록을 복구하고 이후 로그를 지속 전송. 학습/최적화 설정은 변경 없음.
