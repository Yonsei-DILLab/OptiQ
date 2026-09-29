# 공식 DDiffPG AntMaze 64환경 실험

OptiQ/SAC/DIPO/MFPO × v1–v4 × seed0, 16개 fresh run.
각 run은 64개 환경의 합산 1M interaction이며 warm-up 8192개를 포함한다.
전 방법 batch4096, 64개 수집당 2회 업데이트: 최종 actor/critic/NovelD 각30994회.
공식 sparse 보상0/10/20과 원본 NovelD0.01. 기존 dense 결과와 별도 캠페인이다.
기존 v1 DIPO는 중지했고 로그·결과·frozen source를 보존했다.

코드 c85aa5f358f8853ef34982b8204c87f8fa227f9f, direct-gmm-trg에 commit/push, 양서버 공유.
공식 DDiffPG159파일은7edd06c 원본 그대로다. MFPO submodule은d8b3977.
양서버에서 실제 MuJoCo2.1/Gym0.23.1을 사용한다. Gym 난수직렬화 호환 및 실제 시뮬레이터 시딩은 antmaze_experiments 어댑터에서 처리한다.
Diffusers0.18.2는 PyTorch backend만 사용하며 OptiQ/MFPO의 JAX 학습은 정상 사용한다.

8GPU를 두 서버 각4개 독립 큐로 운영한다. GPU가 비면2초 이내 다음 작업을 시작하며, 미로/방법별 전체완료 장벽이 없다.
180은v1/v3,199는v2/v4. 각 run은 자기 preflight 통과 후 본학습한다.
실패하면 해당 서버 대기를 보류하고 이미 실행중인 작업은 보존한다.

검증: 네 방법 모두64환경·실제batch4096으로8320 interactions/4업데이트를 실행했다.
actor/critic/RND predictor의 유한한 변화, RND target 불변, 정확한 업데이트 횟수,
full checkpoint 재읽기와 SHA256, native/direct/OptiQ zero_z 평가 및 동일 full-state 시작 검사를 통과했다.
양서버 v1-v4 환경목표보상/500·700step timeout/terminal observation도 검사했다.

평가는250k threshold마다10episode, 최종100episode/mode/reset이다. 64개 배치경계 때문에
중간 실제 평가 step은250048/500032/750016, 최종은1000000이다.
Full checkpoint는최종1M만 저장한다. 학습탐색xy와 최종 정책rollout은 별도로 기록한다.
주 경로다양성 평가는direct-policy이며, OptiQ의conditional sigma 포함/no external DACER noise다.
SAC mean·MFPO Q-best-of10·OptiQ random-z mu-only는native 보조 평가다.
DDiffPG DIPO의native diffusion은초기 및역과정 noise가 있는원본샘플러다.

W&B: https://wandb.ai/OptiQ/gmm-trg
Group: antmaze-upstream-64env-1m-s0-20260923

상태수집: python3 artifacts/antmaze_upstream_64env_1m/collect_status.py
이 문서는실행등록보고이며, 사전검사수치를1M 실험결과로보고하지않는다.
