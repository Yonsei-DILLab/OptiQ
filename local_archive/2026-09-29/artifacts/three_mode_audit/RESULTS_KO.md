# 3-mode fixed64 재검증

사용자 요청 반영: Q=log f, T=1, batch32, N=M=64, 고정 z64, 100000 updates. 초기화 대조군 포함 seeds0/1. 공유 actor 256x256 GELU, Adam3e-4, 초기 sigma .5, teacher floor .05, latent skip0 유지.

|설정|seed0 TV|seed1 TV|관찰|
|---|---:|---:|---|
|기존 초기화, batch1,100K|.3664|.3675|두 seed 모두 넓은 분포 유지|
|기존 초기화, batch4,100K|.3665|.3668|두 seed 모두 넓은 분포 유지|
|기존 초기화, batch32,100K|.3667|.0471|seed1은 3-mode fitting 성공, seed0은 미분화|
|평균 초기화 scale1, batch32,100K|.0734|.0514|두 seed 모두 3-mode 분화|

상단 두 조건 Q=.25log f/T=.25, 하단 두 조건 Q=log f/T=1; Q/T가 같아서 같은 target. 이 온도 표기 변경을 tempering 효과로 해석하지 않는다. 후보 총량은 초기5K batch1에서32만, 첨부100K batch4에서2560만, 새100K batch32에서2억480만이다.

평균 출력층 variance initialization scale만 1e-4에서1로 변경. seed0의 초기64개 mu 표준편차는 .0004157에서 .04157로100배 증가. 아키텍처/latent skip/최적화식 변경 없음. 기존 설정은 거의 같은 conditional로 시작하며, 넓은 단일 분포에 머무는 최적화 현상에 민감하다는 실험적 근거다. 안정성 일반화를 위해 더 많은 seed가 필요하며 원인이 이것 하나뿐이라는 증명은 아니다.

Fixed64의 기본 평가 분포는 해당64개 code의 uniform mixture. 새 z 평가 TV는 초기화 대조군 .1504/.1377로 고정 mixture 평가와 다르다. Target equal mass 대비 init1 최종 mass는 seed0 .3821/.3514/.2665, seed1 .3585/.3418/.2997로 완벽한 질량 일치는 아니다. TV는256bin, 그림은128bin 원본 histogram이며 KDE 없음.

코드 검산: 독립 NumPy 수식과 NLL 오차0, mu gradient 오차3.73e-9, log-sigma gradient 오차1.40e-9, proposal tanh Jacobian 오차7.15e-7. batch1 rerun5K 시점의 두 seed mu/sigma/z/samples가 이전 결과와 bitwise 동일.

원격 브랜치 v5-gmm40@a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71 확인. 현재 commit의 gmm40/optiq.py는 매 업데이트 z를 재추출하는 conditional OT adapter이며, 첨부 이미지의 finite-uniform64 Direct marginal GMM 실행 소스는 현재 branch에서 확인되지 않음. 이미지로 확인되는 N/M,batch,T,updates,skip와 코드에서 확인한 actor defaults를 구분한다. 정확한 screenshot 재현을 했다고 주장하지 않는다.

최종 소스는 서버 heejoon e905ec1becbd0499f1c9fd5a2877d10c0fec1328에 실행 전 커밋. 원격 push 없음. 원격 결과 /home/heechan/optiq-experiments/three-mode-T1-b32-20260920. 실행시간은 actor학습/평가 구간 약18~21초 per run, import/프로세스 시작 및 전송 시간 제외.

판정: 앞선5K/batch1 결과를 fixed64 모델의 표현력 한계로 해석할 수 없다. 큰 batch와 초기화 대조에서64x64로도3mode를 학습했다. 첨부GMM40 대비 정확한 차이는 원본 실행 source/config가 있어야 완전히 확인할 수 있다.
