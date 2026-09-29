# MaxEntDP AntMaze 설정 확인 (2026-09-23)

결론: dense 보상에서 NovelD를 사용했다는 근거를 찾지 못했다. 논문은 dense 거리 페널티를 명시하지만 NovelD/RND를 명시하지 않으며, 공개 코드에도 이 보너스 구현이 없다. AntMaze 전용 online 환경/실행 설정은 공개되어 있지 않아 미공개 실험 설정까지 부재를 확정할 수는 없다.

## 확인된 논문 내용

- Appendix D.2: DDiffPG의 AntMaze 지도를 이용하고 sparse 대신 최근접 목표까지의 거리 페널티를 환경 보상으로 사용.
- Figure8/9: 1M environment interactions 이후 궤적 및 state coverage 비교. 1M gradient updates와 구분.
- 비교 방법: MaxEntDP와 SAC.
- 탐색 목적: 정책 엔트로피가 들어가는 MaxEnt RL. NovelD/RND의 상태 novelty 보너스와 다른 항이다.
- AntMaze 전용 온도, reward scale/정확한 구현, parallel env 수, 추가 NovelD 계수는 확인 불가. 일반 Ant-v3 온도를 AntMaze로 대입하지 않는다.

논문: https://arxiv.org/html/2502.11612v3#A4.SS2

## 공식 코드에서 확인된 일반 기본값

2026-09-23 git ls-remote로 공식 main=8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077 확인. 기존 같은 commit의82개 Python/YAML/Markdown 파일 감사와 실행 경로를 재점검했다. NovelD/RND 보너스를 찾지 못했다. AntMaze 문자열이 있는 offline D4RL/IQL 코드는 이 online 실험의 설정이 아니다.

- 단일 gym env, batch256, UTD1, warmup10k, max_steps1M, eval10k마다10episodes.
- actor/critic256x2, LR3e-4, gamma.99, tau.005, replay는 max_steps 용량.
- config temp.1, backup_entropy=True, diffusion20steps, QNE samples500, log probability samples50, evaluation best-of10.
- 이 목록은 공개 일반 기본값이며 AntMaze 전용 실행값으로 검증된 목록이 아니다.
- Critic target: r + gamma*mask*(min Q_target - temp*log_pi). 별도의 RND 보너스 없음.

실행: https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/train_score_matching_online.py
설정: https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/configs/max_entropy_learner_config.py
학습: https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/jaxrl5/agents/score_matching/max_entropy_learner.py

## 현재 승인안과 구분

우리 dense+NovelD.01은 DDiffPG의 NovelD를 유지하고 환경 보상을 dense로 바꾸는 별도 실험이다. 64env/batch4096/64수집당2업데이트는 사용자가 선택한 비교 조건이며 MaxEntDP의 일반 기본값이 아니다. v1/v2 3M, v3 4M, v4 5M은 DDiffPG preprocess_cfg의 예산이다. 조사 질문만으로 NovelD나 알고리즘 목적을 임의로 변경하지 않는다.
