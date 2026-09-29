# Latent 고정 vs 매번 재추출: 업데이트 수 기준 비교

동일 조건: N=M64, batch32, Qlogf/T1, actor256x256 GELU, mu 출력 초기화 scale1, 초기 sigma.5, Adam3e-4, zero latent skip. 각2 seeds, 100K updates. 초기 parameter와 teacher RNG stream 일치. Fixed는 seed별 동일64 bank를 사용, Fresh는 매 step/각 batch마다 새로운64 Gaussian z를 추출한다.

각 정책의 실제 prior에서 평가한 평균 histogram TV (32768 samples,256bins; 낮을수록 좋음):
|updates|fixed64 prior|fresh Gaussian prior|
|---|---:|---:|
|10K|.1599|.2062|
|20K|.0740|.0749|
|100K|.0624|.0397|

TV<.1의 첫 관측 checkpoint는 두 방법/두 seed 모두20K. 평가 간격 때문에 정확한 최초 통과 step은 모름. Fixed가 초반 일부구간 앞섰으나, 뚜렷한 전체 수렴속도 우위는 확인되지 않았고 최종 fitting은 fresh가 더 좋았다. 이 결과는2 seeds의 해당 toy 설정이며 일반적 우위/통계적 유의성 주장 아님.

동일한 새 Gaussian-z로 평가하면 100K TV는 fixed-trained .1440, fresh-trained .0397. 고정64 prior를 사용하는 모델에 새 Gaussian-z를 주는 것은 배포 prior 변경이므로, 이것을 fixed64 정책 자체의 오차와 혼동하지 않는다. 고정하면 finite64 mixture를 학습하고, 재추출하면 continuous-z 정책을 학습한다. 따라서 완전히 같은 정책분포에서 gradient noise만 바꾼 실험은 아니다.

Fixed rerun의 모든 metrics(시간 제외)가 이전 init1 결과와 일치. 두 방법의 seed별 step0 모든 metrics(시간 제외)가 일치. 모든4runs가100K완료 및 finite TV 확인.

실행 전 서버 heejoon commit 0896c2c14a016ae937e552ffd6ccab5490aaa7ac. 경로 /home/heechan/OptiQ-heejoon/experiments/quick_three_mode/compare_latent.py; 원격결과 /home/heechan/optiq-experiments/three-mode-latent-20260920. 원격 push 없음. wall time 성능 비교를 주장하지 않는다.
