# Step penalty 고정 시작점 확인

보상 100*(d_current-d_next)-1, geodesic, B=0. T=1, DACER/NovelD OFF.
각 체크포인트 seed0의 direct policy(random z+conditional sigma)를 원래 학습과 같은 고정 전체 상태에서 40회 평가.

- v2 500,224 steps: 통로 {'right': 40}, 성공 40/40.
- v3 3,000,064 steps: 통로 {'right': 40}, 성공 40/40.
- v4 2,500,096 steps: 통로 {'lower': 40}, 성공 40/40.

해당 체크포인트에서는 모두 단일 통로이며, step penalty를 줬다고 양쪽 경로가 유지되지는 않았다. penalty 없는 비교군과 학습량이 달라 penalty의 인과효과를 분리한 결과는 아니다.

원자료 SHA256·전체 초기 상태 일치·모델 불변 검증 통과. 일부 서버 재평가의 W&B 업로드는 API 인증 누락으로 실패했으나 검증된 로컬 궤적/지표는 정상 보관됐다. 원래 학습 로그와 구분한다.

![고정 시작 평가](step_penalty_fixed.png)
