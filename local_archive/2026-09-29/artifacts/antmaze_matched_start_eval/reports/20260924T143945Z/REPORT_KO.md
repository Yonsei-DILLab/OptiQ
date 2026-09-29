# AntMaze 학습과 동일한 시작점 평가

최신 사용자 지시: v1은 학습과 동일하게 랜덤 시작 유지. v2/v3/v4는 원래 원점·자세·속도로 고정. 평가 소스061e9888e1f3d0e59b5037441fa146c1eff25e2a, 학습소스4d75715 또는0751e86은보존.

새 실행은 eval_starts=upstream을기본으로사용한다. 이미실행중인frozen학습의in-memory평가설정은바꾸지않고,보존체크포인트를CPU에서40회(최종100회)씩재평가한다. v2-v4 기존랜덤평가는보조자료이고새고정평가가주결과다. v1은재평가대상에서제외했다. 학습재시작/모델변경없음.

기존평가구현의선택적RNG필드를JAX배열로변환하던오류는독립평가seed로수정했다. 최초평가시도의실패자료를보존했고학습에는영향없었다. 수정소스로검증및재평가완료.

|미로|현재 실험 checkpoint|고정 성공|고정 통로 방문|같은 checkpoint 랜덤 방문|
|---|---:|---:|---|---|
|v2|750080|40/40|{'right': 40}|{'right': 38, 'uncommitted': 2}|
|v3|1000192|40/40|{'right': 40}|{'right': 40}|
|v4|500224|1/40|{'lower': 40}|{'lower': 20, 'upper': 16, 'uncommitted': 4}|

v4의500k정책은랜덤시작시위16/아래20/미진입4였지만원래고정시작에서는아래40/40이었다. 따라서랜덤시작에서보였던양갈래는동일한출발상태에서의경로다양성을입증하지않는다. v2/v3도고정시작에서는모두오른쪽이었다. 단일학습seed,40회샘플의해당checkpoint결과로한정한다.

직접정책샘플링은random latent+conditional sigma이고외부DACER잡음이나NovelD보상을추가하지않았다. Native mu-only는별도보관했다. 같은checkpoint쌍끼리만비교했으며미로별학습량은다르다.

검증:모델/optimizer복원일치,checkpoint SHA 전후일치,rollout후모델불변,고정평가의40개initial_full_state동일/XY[0,0],native와policy시작상태동일,progress 보상재계산,로컬수집SHA256. 자세한수치는analysis.json과각run verification/provenance.json에있다.

![고정과 랜덤 시작 비교](fixed_vs_random.png)
