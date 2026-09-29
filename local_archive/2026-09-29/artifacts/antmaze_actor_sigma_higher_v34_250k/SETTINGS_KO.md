현재 log sigma 상한 비교의 설정 확인

공통 기본값 캠페인이 아니다. 직전 O/P 대조군을 환경별로 그대로 이어받아
각 환경 안에서 상한만 바꾼다. 이 전제를 사용자에게 명확히 공개했다.
보상/temperature/gamma/DACER/teacherfloor를 기본으로 되돌린 실험으로 부르면 안 된다.

v3: 100 * (d_t - d_next), d=min_i(w_i * max(geodesic_i - 0.5,0)),
w=[0.9419037740036652,1]. 고정 원점의 각 목표 잔여거리로 정규화.
v4: 100 * (d_t - d_next), d=min_i geodesic_to_goal_center_i.
둘 다 step penalty 0, 성공 bonus 0, NovelD/RND OFF, gamma=.999.
목표 Euclidean 반경 .5 진입 시 종료. 시간제한700, timeout은 bootstrap.
DDiffPG 원본 지도/물리/시작점 유지; v3/v4 full state 고정 평가.

v3 T3/teacher std floor1, v4 T1/teacher std floor.5.
공통 DACER ON,target_entropy_per_dim=+.7,total5.6,interval500learnerupdates,
noise_scale=.1,initial_alpha=.27,initial added noise std=.027,alphaLR=.03.
평가와TD타겟에 외부DACER잡음이나NovelD보상을 추가하지 않음.

OptiQ Direct GMM/TRG, direct marginal GMM NLL. Actor/twin scalar critics
256x3 GELU. Actor input=29obs+8randomnormalz,criticinput=29obs+8action.
N=M64,randomlatent,density correction beta1; teacher Q=twin online mean.
Critic is ordinary TD with min target doubleQ,entropy backup coefficient0.
Critic tau=.005,target actor policy_tau=1,actor/critic update delay1.
ActorAdamLR3e-4,criticAdamLR5e-4,gradient clippingnone.
256env,collect256/update8,batch4096,replay1M,warmup8192.
Initial mean-head variance scale1, initial log sigma-1,lower-5.
Only upper0/1/2/3/inf; archived-1control; boundedactions[-1,1]unchanged.
250kpostwarmup,258304total/7816updates;40episode/modeeach50k,100final.
Direct conditional sigma included; native mu-only supplementary.

All8finite-cappreflight and actualmainverified;2unboundedpending, immediatebackfill.
Source b111a993b1e1bf895966dac4469abdfdb2c37c05, direct-gmm-trg-antmaze,
committed/pushed/frozen onboth5090hosts, W&BOptiQ/antmaze. No4090.
