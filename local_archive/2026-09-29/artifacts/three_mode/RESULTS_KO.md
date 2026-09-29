# heejoon Direct GMM 3-mode quick experiment

각 설정 4 seeds, 5000 updates. 목표: -0.6, 0, 0.6의 동일 질량 Gaussian (std .1), [-1,1] 정규화. 원본 actor/Direct GMM loss/Adam 유지.

- 매 step z 재추출, N=M=2048: histogram TV 평균 0.2227, specialist 비율 44.71%. 양쪽 모드부터 형성, 중앙 미적합.
- 학습 z64 고정, N=M=64: TV 평균 0.3661, specialist 비율 0%. 64개 conditional이 비슷한 넓은 분포로 남음. Fresh-z 평가 TV 0.3662.
- Specialist: conditional의 한 mode basin 확률 >=0.8. Basin 경계 -0.3,0.3. 영역에 샘플이 존재하는 것과 좁은 모드를 맞추는 것은 다름.
- TV는 256-bin histogram과 analytic CDF 질량 비교. 그림은 가독성을 위해 128 bins, KDE 없음. 표준편차는 seed 간 sample SD.
- N/M과 z 고정 여부가 동시에 달라 고정 z의 독립 효과로 해석할 수 없음. 5000-update의 빠른 fixed-Q 실험 결과이며 장기 수렴 결론 아님.

실행 전 서버 heejoon 커밋: baseline e1f8ced729b2cea14ae8918e882380cab54ff38d, fixed64 25850796ce0bd158a92ad9f13324e99615110f96. 원격 push 없음.
원격 코드 /home/heechan/OptiQ-heejoon/experiments/quick_three_mode, 결과 /home/heechan/optiq-experiments/three-mode-20260920.
검증: 8 runs 모두 5000 업데이트 완료, 모든 48 sample snapshots finite, conditional basin 확률 합 1, fixed64 latent bank가 모든 snapshot에서 동일.
