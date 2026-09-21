# 동일 cloud에서 Sinkhorn row-argmax와 Monge 선택 비교

목적: 기존 12-run pilot의 저장된 assignment audit만으로, 동일 source row가
Sinkhorn row-argmax와 quantile Monge에서 비슷한 action을 받는지 확인한다.
새 학습, 새 candidate 표집, 추가 OT solve는 하지 않는다.

- 입력: 각 run의 8개 assignment audit, 총 96개. 데이터는 수정하지 않는다.
- N=256 source, M=1024 candidate, epsilon=0.05, Sinkhorn 30 iterations.
- 후보 index가 같은 비율, 실제 action 절대 거리(평균/중앙값/P90), 거리≤0.05
  비율, 정답 density valley로 나눈 동일 mode 영역 비율을 계산한다.
- Exact OT row-argmax vs Monge를 대조군으로 사용한다.
- Monge는 원래 weighted teacher의 midpoint quantile 대표점 256개를 사용한
  256×256 empirical bijection이다. 원래 arbitrary weights 보존을 주장하지 않는다.
- 최종 35K baseline cloud, seeds 0,1 평균을 주 표로 보고한다. 그림은 최종
  seed 0을 두 환경 모두 제시한다. 다른 방법이 만든 cloud와 모든 저장 시점도
  CSV에 포함한다. 반복 시점/row를 독립 seed로 보지 않는다.
- 각 row 전체를 uniform하게 섞은 soft mixture와 row-argmax hard target을
  비교하여 teacher CDF 왜곡을 구분한다. Raw plan의 row/column residual을 기록한다.
- Quantized-teacher Sinkhorn의 column residual/CDF는 quantized teacher 기준으로
  계산한다. 원래 audit에 저장된 original-teacher residual과 혼동하지 않는다.
- 선택 target histogram은 실제 선택점 N=256, 64 bins, smoothing 없음이다.
  Actor에서 32,768개 뽑은 평가 density와 명확히 구분한다.
- 입력 SHA256, 학습 commit, 후처리 commit을 `selection_provenance.json`에 기록한다.

주 결론은 현 구현의 epsilon과 유한 iteration에 한정한다. 이 후처리로 epsilon
효과와 미수렴 효과를 완전히 분리했다고 주장하지 않는다.
