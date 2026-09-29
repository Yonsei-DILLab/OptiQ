# AntMaze NovelD 설정 조사 — 2026-09-22

v1은 기존 계수 0.01을 유지하고 DIPO seed0·1M만 추가한다. v2~v4의 새 계수
실험은 아직 등록하지 않았다. 이번 조사는 계수 선정 근거를 찾는 작업이며,
ERIR, xy-only novelty, 보상 정규화, 추가 shaping, soft TD로 변경하지 않았다.

## 공개 문헌과 코드에서 확인한 범위

| 출처 | NovelD에 관해 실제 확인한 것 | 한계 |
|---|---|---|
| DDiffPG | 공개 intrinsic.py에서 0.01을 직접 곱한다. novelty 차분은 next−0.5×current이며 normalize=False. v1~v4별 계수 분기가 없다. | sparse 보상, 탐색 전용 Q 및 mode별 학습을 사용하는 구조여서 우리의 dense+NovelD와 숫자만으로 탐색 강도를 대응시킬 수 없다. |
| MaxEntDP | Appendix D.2는 가장 가까운 목표까지의 거리 페널티와 1M interactions를 명시한다. | 해당 절에 NovelD 사용·계수는 명시되지 않았다. 공개 저장소에도 NovelD/RND 구현과 이 AntMaze 실험 전용 설정이 없다. 미사용으로 확정하거나 0.01로 추정할 수 없다. |
| MFPO | E.6과 Figure10은 SAC/BPTT/MFPO의 AntMaze 궤적을 100k interactions 후 비교한다. | NovelD 사용·계수와 AntMaze 보상의 상세 정의는 명시되지 않는다. 공개 코드에서도 이 실험 전용 환경/NovelD 설정을 확인하지 못했다. |

DDiffPG의 환경별 preprocess 설정은 v1 3M, v2 3M, v3 4M, v4 5M의 max_step을
지정한다. 이들은 공개 코드의 예산이며, NovelD 계수를 환경별로 바꾸는 설정은
아니다. 논문의 실험 전부가 공개 기본값 그대로 실행됐다는 증거로 확대하지 않는다.

DDiffPG의 explore Q는 intrinsic만, 나머지 mode Q는 환경보상+intrinsic을 받는다.
우리 실험은 하나의 보상으로 dense+intrinsic을 합쳐 학습한다. 따라서 원 코드의
0.01이 dense 보상하에서도 같은 탐색 효과를 낸다고 단정할 수 없다.

v2는 별도 주의가 필요하다. DDiffPG 환경은 먼 왼쪽 위 목표에 더 큰 도달 보상을
준다(공개 코드에서 20, 가까운 오른쪽 목표는10). 우리는 목표별 차등 보상 없이
매 step −nearest-goal distance를 사용한다. 원문에서 v2의 먼 목표를 찾은 결과를
NovelD 계수만의 효과로 해석할 수 없다. 보상은 이번에 변경하지 않았다.

MFPO config의 temp=0.01은 초기 entropy temperature다. MaxEntDP의 temp 역시
entropy temperature다. 이 숫자들은 NovelD 계수가 아니다.

## 우리 실험의 저장 설정

실제 양 서버 runs/*/config.json을 읽은 결과:

- 최초 dense+NovelD 1M 캠페인은 v2/v3/v4 모든 등록 방법이 0.01이었다.
- 후속 계수10 캠페인에서는 v2 OptiQ 144k, SAC 101k까지의 중단 기록이 있다.
  v3/v4는 해당 캠페인 본 학습을 시작하지 않았다.
- v3 OptiQ 별도 100k 실험은 0.1,1,5,10,50,100을 완료했다.
- v4 OptiQ 별도 100k 실험은 50,100을 완료했다. 0.01의 실제100k checkpoint를 대조군으로 보존했다.

원자료: existing-coefficient-audit.json, remote-coefficient-audit.json.
학습량이 다른 결과를 최종 성능 순위로 섞어 비교하지 않는다.

## 다음 설정에 대한 판단 — 아직 실행하지 않은 제안

공개 문헌에서 'v2는 X, v3는 Y, v4는 Z'라는 dense+NovelD 표준 계수를 찾지는
못했다. 문헌값 복제가 아니라 우리 환경에서의 계수 탐색으로 명시해야 한다.

- v2: 10에서는 공간 방문은 늘었으나 먼 G2 방향은 여전히 방문하지 않았다.
  우선 0.01 대조군과 1·5 같은 중간 강도를 비교할 근거가 있다. 총 방문칸만
  늘어나는지, 먼 목표 쪽 통로에 실제로 진입하는지를 분리한다.
- v3: 100k에서 5는598칸,10은901칸,50은1277칸으로 탐색을 넓혔다.
  50의 학습 최소 G1/G2 거리는1.88/2.08m였다. 다만 NovelD minibatch 평균은
  +35.91로 dense −13.45보다 컸다. 50을 성공 정책의 권장값으로 확정할 수는 없다.
  사용자 목표가 탐색 범위이므로 5·10은 주 후보,50은 이미 확보한 강한 탐색 참고군으로 유지한다.
- v4: 기존0.01 대비50·100만 시험했고, 1·5·10 중간 구간의 자료가 없다.
  100이50보다 방문칸이 적었다(431 대511). 중간 구간을 먼저 채우는 것이
  무조건 계수를 올리는 것보다 근거가 명확하다.

다음 비교는 dense/NovelD 구조, 모델, LR, batch, UTD를 고정하고 계수만 바꾸며,
공간 coverage·반대쪽 탐색 깊이·최근 방문 편중을 우선한다. 최종 정책의 여러
경로 유지 여부는 별도의 직접 샘플링 rollout으로 본다. 기존 v1의0.01은300k에
편향이 있었지만1M에서 두 성공 경로를 보였으므로100k로 장기 경로 다양성을
확정하지 않는다. 모든 현재 결과는 training seed0 하나다.

## 출처와 코드 버전

- [DDiffPG intrinsic 구현](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/utils/intrinsic.py#L47-L60)
- [DDiffPG 기본 설정](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/default.yaml#L31-L35)
- [DDiffPG 환경별 설정](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/utils/common.py#L36-L59)
- [DDiffPG Q별 보상](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/ddiffpg.py#L195-L218)
- [DDiffPG 목표별 보상](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/env/d4rl/locomotion/goal_reaching_env.py#L37-L46)
- [MaxEntDP D.2](https://arxiv.org/html/2502.11612v3#A4.SS2)
- [MaxEntDP 공개 online runner](https://github.com/diffusionyes/MaxEntDP/blob/8adfc7e5a3eb4e5dd09eea28eb8adfafa6c91077/examples/states/train_score_matching_online.py)
- [MFPO E.6](https://arxiv.org/html/2604.14698#A5.SS6)
- [MFPO 공개 online runner](https://github.com/dongxiaoyi-xyz/MFPO/blob/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/train_online.py)
- [MFPO temperature 설정](https://github.com/dongxiaoyi-xyz/MFPO/blob/d8b3977d29d4ef2d315e871337e5826f2eb79eb2/configs/mfpo_config.py#L17-L18)

공개 기본 브랜치 HEAD를 직접 clone하여 전체 추적 파일을 검색했다. 코드 검색
증거와 SHA는 literature-code-audit.json에 보관했다. MaxEntDP의 AntMaze 문자열
6건은 offline D4RL/IQL 관련이며, 위 online dense 실험의 전용 설정이 아니다.
