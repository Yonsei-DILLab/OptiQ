**Ant/Humanoid temperature → beta 캠페인 실행 기록**

W&B: [OptiQ/gmm-trg](https://wandb.ai/OptiQ/gmm-trg). Branch `direct-gmm-trg`, frozen commit `84f1e0a884349d6c4b0dae521839a8d4e5f46437`.

1단계: Humanoid T={0.5,0.1}, Ant T={0.1,0.05}, beta=1, seed0..4 → 20 runs.

2단계: 각 환경의 1단계 10개 run이 모두 1M step을 끝내면 선택한 T에서 beta={0.5,0.9}, seed0..4를 새로 학습 → 추가20 runs. 선택 T의 beta=1 대조군은 첫 단계 결과를 재사용한다.

선택 기준: stochastic_z reward의 마지막100k(900000<step<=1000000) 평균. 평가20회×10episode를 각 seed에서 평균내고 5seed를 같은 가중치로 평균. zero_z와 학습곡선도 비교한다.

모델256x2, log sigma[-5,-1], initial log sigma=-1, 랜덤 latent, N=M64, batch256, UTD1, actor/critic LR3e-4, clip없음, warm-up5k, 1M step, eval5k×모드별10episodes 유지. DACER/NM/density correction 제거는 이번 실행 범위에 포함하지 않는다.

60개 가능한 설정 조합에서 나머지 algorithm config 동일성 검사 통과. 마지막100k 경계/미완료seed 선택 차단 검사 통과. GPU에서 matched RNG beta .5/.9의 teacher ESS와 actor loss 차이 및 동일 critic loss를 확인했다.

현재 최초8개 run은 두 서버 GPU8개에서 실행 중이며 나머지 temperature12개는 큐에 있다. 아래는 실행 시점의 최초 run 링크다.

| 서버 | run | W&B |
|---|---|---|
| 180 | humanoid-trg-temperature-T0.1-b1-s0 | [d5dvv850](https://wandb.ai/OptiQ/gmm-trg/runs/d5dvv850) |
| 180 | humanoid-trg-temperature-T0.1-b1-s1 | [7izdnurw](https://wandb.ai/OptiQ/gmm-trg/runs/7izdnurw) |
| 180 | humanoid-trg-temperature-T0.5-b1-s0 | [8x3o6h2l](https://wandb.ai/OptiQ/gmm-trg/runs/8x3o6h2l) |
| 180 | humanoid-trg-temperature-T0.5-b1-s1 | [ak9u8pyx](https://wandb.ai/OptiQ/gmm-trg/runs/ak9u8pyx) |
| 199 | ant-trg-temperature-T0.05-b1-s0 | [f9yd6y0s](https://wandb.ai/OptiQ/gmm-trg/runs/f9yd6y0s) |
| 199 | ant-trg-temperature-T0.05-b1-s1 | [d1fb1e1q](https://wandb.ai/OptiQ/gmm-trg/runs/d1fb1e1q) |
| 199 | ant-trg-temperature-T0.1-b1-s0 | [92mxpw5c](https://wandb.ai/OptiQ/gmm-trg/runs/92mxpw5c) |
| 199 | ant-trg-temperature-T0.1-b1-s1 | [vz03vg12](https://wandb.ai/OptiQ/gmm-trg/runs/vz03vg12) |

서버별 Supervisor 서비스: `trg-temp-beta-20260921-humanoid` (180), `trg-temp-beta-20260921-ant` (199). 서버의 캠페인 경로는 `/home/heechan/optiq-experiments/trg-temp-beta-20260921`이다. controller는 환경별 단계 완료를 확인하고 자동 전환하며, 실패한 seed가 있으면 해당 환경의 beta 시작을 차단한다. 미완료 run을 자동 중복 실행하지 않는다.

30분 간격의 같은 대화 heartbeat `ant-humanoid-temperature-beta`도 생성했다. routine 진행에는 알리지 않고 실패·최적T 선택·beta 전환·완료에만 알린다.

`host180.json`, `host199.json`은 실제 실행 config와 W&B cloud 상태의 시점별 snapshot이다. `source.bundle`은 실행 전 커밋한 전체 소스의 로컬 백업이다. `validation.json`은 GPU 검증 결과다. `collect_snapshot.py`는 읽기 전용 상태 수집기다.
