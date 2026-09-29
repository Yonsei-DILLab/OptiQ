# AntMaze dense + NovelD 16개 캠페인

- OptiQ / SAC / MEOW / MFPO × v1/v2/v3/v4 × seed0, 각각 1M 환경 interaction.
- 서버180: v1/v3 8개, 서버199: v2/v4 8개. GPU당 1개, 슬롯마다 2초 간격 backfill.
- 원격 경로 및 supervisor: `/home/heechan/optiq-experiments/antmaze-dense-noveld-1m-s0-20260922`.
- 학습 source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`, branch `direct-gmm-trg`.
- W&B: https://wandb.ai/OptiQ/gmm-trg (group `antmaze-dense-noveld-1m-s0-20260922`).
- 기존 취소 실험은 재개하지 않는다. 초기 개발 preflight9971bae 로그는 보존한다.

## 학습과 평가

환경 보상은 다음 위치에서 가장 가까운 목표까지의 음의 유클리드 거리다.
Replay는 이 환경 보상만 저장한다. 학습 minibatch에서 DDiffPG NovelD 보너스를
재계산하여 더한다. 1환경, batch256, UTD1이며 각 방법의 native 모델/LR는 유지한다.
OptiQ는 T=.25/beta1/DACER true/mean-init1/256x2/랜덤latent/N=M64/logstd[-5,-1]이다.
MaxEntDP D.2에 설명된 dense 보상을 따른 것이며, NovelD는 사용자 지정 추가 조건이다.
공개되지 않은 MaxEntDP AntMaze 전용 코드까지 정확하게 재현했다는 의미는 아니다.

성능 평가(native): SAC는 tanh(mu), MFPO는 Q-best-of10, MEOW는 prior-center,
OptiQ는 random-z mu-only. 경로 다양성 평가(policy)는 각 정책에서 직접 샘플링한다.
이때 OptiQ conditional sigma가 포함되므로 native mu-only 결과와 구분한다.
OptiQ zero-z mu-only도 별도 저장한다. 평가에는 NovelD 보상이나 외부 DACER 잡음이 없다.
매25k에 각10episode; 최종1M에는 평가 모드마다 natural/reset고정 각각100episode다.
각 학습정책 하나의 여러 rollout을 분석하며, 서로 다른 정책을 섞지 않는다.

## 저장 및 재개

100k마다와 최종1M에 `runs/<job>/resume/step_<10자리 step>/`에 저장한다.
`state.pt`는 model/target/optimizer/entropy/DACER/NovelD/RNG/시뮬레이터 상태와
카운터를, `replay.npz`는 replay의 실제 보관 transition을 담는다.
`manifest.json`에 파일 SHA256 및 전체 상태 digest를 기록한다.
이어 학습할 때 같은 source에서 `python -m antmaze.multimodal.dense_noveld_run
--task <vN> --method <method> --resume <snapshot> --steps <새 총예산>
--output <새 결과경로>`를 사용한다. 추가 예산 실행에는 별도 사용자 지시가 필요하다.
사전 검사는 272step/16update를 저장한 뒤 새 프로세스에서 전체 상태의 일치를 확인하고
280step/24update로 계속 학습한다. 이 검사를 16개 환경/방법 조합 모두에 수행한다.

## 상태 확인

`python3 artifacts/antmaze_dense_noveld_1m/collect_status.py`는 읽기만 수행하며,
최신 status/manifest/진행/config/W&B/checkpoint/재개검증과 실제 PID를 수집한다.
`latest.json`, `vast-heechan-180.json`, `vast-heechan-199.json`의 수집 시각을 확인한다.
실패 시 대기 큐는 보류하고 실행 중 작업과 로그를 보존한다. 자동 재시작은 없다.
