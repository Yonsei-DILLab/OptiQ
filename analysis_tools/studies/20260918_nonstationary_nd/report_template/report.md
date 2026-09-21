# Non-stationary Q: 어떤 방법이 가장 빨리 적합하는가?

1. 정확한 환경·비교 설정, epsilon과 cost scale, fixed sigma와 legacy 차이
2. 완료 수 / 실패 / reference가 불확실한 run을 먼저 표시
3. 두 mode, 기존 mass, 기존 split: 변화 전 → 직후 →100/500/1000/5000updates
4. 1D 직접 histogram,2D histogram surface+heatmap,4D/8D coordinate별 histogram
5. Epsilon12개 small multiples: tracking TV,AUC,회복 시간,실제 학습 compute시간
6. 동일 learned-Q replay와 closed actor–critic 분리
7. Proposal →weighted teacher →actor; OT marginal residual과 sigma/component 진단
8. 원본/정렬 assignment heatmap; GMM effective assignment 명시
9. 절대 fitting 한계와 adaptation 속도를 구분하고 미회복은 censored로 표시
10. 전체4seed 결과와 hardware/source/commit provenance

결과가 아직 없는 부분을 우열에 대한 예상으로 채우지 않는다.
