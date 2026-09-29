# DDiffPG 원본 Sparse + NovelD · 256 병렬환경 실행 기록

검증 시각: 2026-09-23T03:39:48.919821+09:00

OptiQ/SAC/DIPO/MFPO × v1/v2/v3/v4 × seed0 = 16개를 새로 등록했다. 현재 8개 본학습과 8개 대기 작업이다. 기존 dense64 캠페인은 양 서버에서 중단했고 이전 로그·소스·부분 결과는 보존했다. 원본 환경 검증 및 현재 8개 작업의 실제 256환경·batch4096·8회 업데이트 사전검사가 통과했다. 대기 작업도 자기 GPU에서 사전검사를 통과한 직후 학습하며, 다른 미로/방법 완료를 기다리는 장벽은 없다.

## 원본 및 실행 소스

- 공식 DDiffPG HEAD: `7edd06c4799abbab0f8fa534c21deb56253b018e`. `antmaze/` 159개 파일의 원본 일치를 검증했다.
- 실행 소스: `0ebd8d26c711d79723f457343b36eb8788bd87b8`. `direct-gmm-trg`에 커밋·푸시했고 양 서버 canonical 및 frozen source가 일치한다.
- 공식 [default.yaml](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/default.yaml)은 **num_envs=256**, eval_num_envs=20이다.
- [SAC](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/algo/sac_algo.yaml) / [DIPO](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/algo/dipo_algo.yaml) baseline 설정은 batch4096, update_times8, warm_up32, replay1M이다. DDiffPG 알고리즘 자체의 cluster replay/warmup500을 네 방법에 이식하지 않았다.

## 공통 설정

|항목|적용값|
|---|---|
|환경|원본 MuJoCo2.1 / mujoco_py2.1.2.14 / Gym0.23.1의 DDiffPG AntMaze|
|병렬 수집|정책 1개당 CPU 환경256개, GPU learner1개|
|학습 비율|256 transition 수집 → learner8회 및 RND8회 업데이트|
|Batch / replay|4096 / 1,000,000|
|Warmup|32 vector steps = 8192 transitions|
|환경 보상|원본 sparse: 도달 전0, 목표 도달 시10 또는20|
|NovelD|0.01 × max(n(next) − 0.5 n(current), 0), normalization=False|
|RND|원본29D 관측 + xy Fourier10bands, L2error, AdamW1e-4, clip1|
|평가/저장|사용자 지정 평가250k마다, full checkpoint는 최종만|
|평가 횟수|중간20episode/mode, 최종100episode/mode/reset|
|시드|각각 seed0 1개|

## 환경별 원본 설정과 예산

|미로|목표 좌표|목표 보상|episode 제한|시작 상태|원본 max_step|warmup 포함 실제 interaction|
|---|---|---|---:|---|---:|---:|
|v1|(-8,0)|10|500|xy [-2,2] 무작위|3,000,000|3,008,256|
|v2|(-8,8), (8,0)|20,10|500|고정|3,000,000|3,008,256|
|v3|(-12,12), (12,-12)|10,10|700|고정|4,000,000|4,008,448|
|v4|(-16,4), (-16,-4)|10,10|700|고정|5,000,000|5,008,384|

원본 목표 도달 반경≤0.5, 도달 시 종료를 유지한다. **v2는 목표 보상이 비대칭이므로 양쪽 방문 비율이 같아야 한다고 해석하지 않는다.** 원본 timestep0.02, frame_skip5, RK4, actuator gear30, maze scale4, 관측29D/행동8D를 확인했다. 원본 등록의 eval=True 때문에 goal vector를 관측에 추가하지 않고 base Ant locomotion reward/fall termination도 사용하지 않는다. 원본 학습 counter는 warmup 제외 global_steps가 max_step을 넘으면 끝나므로 마지막256 묶음까지 포함한 예산을 위와 같이 계산했다.

## 알고리즘 설정과 남겨둔 차이

SAC/DIPO는 원본 learner와 optimizer를 사용한다: actor/critic AdamW LR3e-4/5e-4, gamma0.99, tau0.05, gradient clip1. DIPO는 원본5 diffusion steps,20 action-gradient updates, actionLR0.03, critic support[0,5]/51atoms이다.

OptiQ/MFPO는 원본 DDiffPG에 없는 방법이므로 각자의 기존 모델/목적함수/optimizer를 공통 수집 프로토콜에 연결했다. OptiQ는256×2GELU, T0.25, beta1, DACER=true, mean-init1, random latent, N=M64, log sigma[-5,-1]/init-1, actor/criticLR3e-4이다. MFPO는 native256×3 및2flow steps를 유지한다. 따라서 네 알고리즘이 모두 DDiffPG와 동일한 알고리즘이라는 의미는 아니다.

평가250k/최종저장과 seed0는 기존 사용자 지시를 유지한 부분이다. Native 평가는 SAC mean, MFPO Q-best-of10, OptiQ random-z mu-only, DIPO native diffusion이다. 직접 정책 샘플 평가는 OptiQ conditional sigma를 포함하며 외부 DACER/추가 Gaussian 탐색 잡음은 넣지 않는다. NovelD 보상도 평가에 넣지 않는다. 자연 초기 상태와 고정 full-state 결과를 분리하고 실패도 포함한다.

## 현재 실행

|서버|GPU|정책|실제 step|learner updates|W&B|
|---|---:|---|---:|---:|---|
|vast-heechan-180|0|v1-dipo-s0|180,224|5,376|online|
|vast-heechan-180|1|v3-dipo-s0|167,936|4,992|online|
|vast-heechan-180|2|v1-optiq-s0|446,464|13,696|online|
|vast-heechan-180|3|v1-mfpo-s0|532,480|16,384|online|
|vast-heechan-199|0|v2-dipo-s0|20,480|384|offline|
|vast-heechan-199|1|v4-dipo-s0|20,480|384|offline|
|vast-heechan-199|2|v2-optiq-s0|40,960|1,024|offline|
|vast-heechan-199|3|v2-mfpo-s0|36,864|896|offline|

각 GPU가 비면2초 간격으로 다음 대기 작업을 시작한다. 실패 시 해당 서버 대기 작업을 보류하고 실행 중 작업은 보존하며 자동 재시작하지 않는다.

199 서버는 외부 DNS/네트워크 장애로 W&B를 offline 기록 중이다. 학습 로그는 보존하며 별도 sync supervisor가 연결 회복 후 완료 run을 OptiQ/gmm-trg로 업로드한다. 현재199의 W&B 클라우드 반영 완료를 의미하지 않는다. 180은 online 기록 중이다. 코드도199에는 검증한 Git bundle로 전달했다.

256 CPU worker에 필요한 파일 한도를 프로세스 내부에서4096으로 올렸고 양 서버 실제256환경 검사에 통과했다. 서버 전역 설정/드라이버는 바꾸지 않았다. Python3.11/torchcu128은5090 호환을 위한 runtime이며 원본 MuJoCo 물리/맵은 유지한다. CPU 병렬화이므로256배 속도 향상을 뜻하지 않는다.

## 보존된 증거

- `launch-verification.json`:8개 본학습 진행/256 worker/업데이트 수/소스 일치 검증.
- `vast-heechan-180.json`, `vast-heechan-199.json`:manifest,환경 검사,현재 job/preflight/run config/progress,실제 learner PID,GPU 상태.
- `PROTOCOL.md`:전체 세부 설정 및 원본과의 차이.
- `collect_status.py`:양 서버 읽기 전용 상태 수집.

본 기록의 step 수는 실행 확인 시점의 스냅샷이며 학습 결과/성공 성능 보고가 아니다.
