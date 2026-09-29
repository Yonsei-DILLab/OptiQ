# OptiQ AntMaze DACER entropy 양수 target 실험

- v1/v2/v3/v4 × 차원당 target +0.1/+0.5/+0.7/+0.9, seed0: 총16개.
- DACER 조절500 learner updates, T1, 500k post-warmup 예산.
- 실제 총508416 interactions,15632 learner updates, DACER32회.
- 보상100*(현재-다음 목표거리), bonus0, step penalty0, NovelD OFF.
- 256x3 actor/critic,256env,batch4096,8updates/256,나머지 RL 설정 유지.
- 100k마다40회 평가/정책저장, 마지막100회/전체checkpoint.
- v1 랜덤 시작, v2-v4 원래 고정 시작; 직접정책과 mu-only 구분, 평가 외부잡음 없음.
- 두5090서버에서8개 본학습,8개 대기. 개별완료 직후2초 간격 backfill.
- 첫8개 실제GPU 사전검사, 보상/전체체크포인트/target/interval 검증 통과.
- local20개 및 각server20개 tests 통과; 기존 frozen source/결과 보존.
- Commit: 2564b59faa0d319eece496b93eff0f19359efc37, direct-gmm-trg-antmaze.
- GitHub 푸시 및 두 서버 공유 후 등록.4090(vast1) 사용 안 함.
- W&B: https://wandb.ai/OptiQ/antmaze/groups/antmaze-optiq-dacer-hpos-i500-500k-s0-20260925

실시간 확인: `/tmp/optiq-antmaze-report-20260922/bin/python artifacts/antmaze_dacer_positive_500k/collect_status.py`
실행시점 증거: launch-verification.json; 이후 상태: latest-status.json.
