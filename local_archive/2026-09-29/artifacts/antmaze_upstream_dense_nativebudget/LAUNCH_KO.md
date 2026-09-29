# DDiffPG 공식 환경 기반 dense AntMaze 실행

- 코드: direct-gmm-trg a946a77226629688346d6f4c69952131fb528ed0, push 및 양 서버 공유 완료.
- 공식 antmaze/ 159개 파일은 unchanged. 외부 wrapper에서 reward를 최근접 목표까지의 음수 Euclidean 거리로 교체. sparse 성공 보너스는 더하지 않음.
- NovelD .01 유지. 이 값은 DDiffPG에서 가져왔으며 MaxEntDP dense 실험의 공개 기본값이 아님.
- v1/v2 각각3M, v3 4M, v4 5M total environment transitions. 각 OptiQ/SAC/DIPO/MFPO seed0, 총16개60M.
- 공통64env/batch4096,64transitions당2learner+RNDupdates, warmup8192 포함, replay1M, eval250k, 최종 full checkpoint.
- DIPO support만 dense 호환을 위해[0,5]→[-6000,5],51atoms 유지. 음수 목표값을0으로 잘라버리는 문제를 방지하기 위한 설정이며 최적값 탐색 결과는 아님.
- 실제 MuJoCo2.1/mujoco_py2.1.2.14/Gym.23.1. 원본 map/reset/종료 조건 유지.
- 두 서버 모두64env×4maze 거리보상/도달/시간제한/terminal-observation검증, replaywrap 및 음수 distributionalprojection 검사 통과.
- 각 작업은8192warmup+128transitions=8320steps,4회 실제 batch4096update, 저장/readback, 전체 평가모드 검사를 통과한 뒤 본학습.
- 서버180: v1/v3, 서버199: v2/v4. 각4GPU, 독립FIFO2초 backfill, 다른maze/method완료장벽없음. 실패시그서버대기보류/live보존.
- 이전 sparse1M 캠페인은 중단, 로그와 frozen source 보존. 이전 취소 캠페인 재개 없음.
- W&B OptiQ/gmm-trg group antmaze-upstream-dense-nativebudget-64env-s0-20260923.

MaxEntDP 확인 내용은 MAXENTDP_CHECK_KO.md 참조. 현재 실행은 DDiffPG 기반 dense+NovelD 비교이며 정확한 MaxEntDP AntMaze 보상/학습 재현으로 주장하지 않는다.
