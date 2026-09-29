최신 완료 결과: [256×2 재현 보고서](RESULT_256x2.md). 500k에서 cap -4/-4.5 모두 μ-only near≥90%, coverage40/40, 각 5개 새 latent 키에서도 통과. 학습 seed0만 확인. 아래 내용은 실행 당시의 누적 기록이며 ACTIVE/pending은 과거 상태입니다. 등록된 작업은 모두 완료했습니다.

# GMM40 random-latent μ-only parameter search

Goal remains active: coverage40/40 AND μ-only near>=90%, unchanged OptiQ TRG algorithm.
256x2 screen1 completed all4 seed0 runs at100k. Results in screen1-summary.json.
No passing result: init16,N=M256 covers40 but near76.74%.

256x3 and512x3 matched controls were requested by user, registered on180.
Frozen source cb35b8d64bc6422d53c56c50378d2b44e2ea1bc7, pushed direct-gmm-trg.
Roots /home/heechan/optiq-experiments/gmm40-mu90-{256x3,512x3}-20260921.
Queue /home/heechan/optiq-experiments/gmm40-mu90-depth-queue-20260921.
User Supervisor services gmm40-mu90-depth-queue-20260921-gpu0..3.
Each slot runs its256x3 then512x3 profile;4matchingprofilesperarchitecture:
mean-head variance scale16/256 crossed withN=M64/256.
100k,seed0,batch256,T1,Adam3e-4,beta1,logσ[-5,-2],initial-2.5,teacherfloor.05.
No fixed latent. μ-only reporting; fullpolicy retained supplemental.
256x3 all4 GPU preflights passed and actualconfig width256/depth3verified.
512x3 pending; actual GPU preflight runs at slot acquisition, CPU architecture checks passed.

Check status-gpu*.json, failure-gpu*.json, logs, per-run update_count_audit.json.
Never duplicate launch or restart. Preserve frozen source. Failures block only thatslot.
Per-profile finalμ evaluation is results/<job>/evaluations/step_0100000/metrics_mu_only.json.
Check all12profiles and collect configs/metrics/μsamples/checkpoints; plot matched comparisons.
A passing seed0 needs independent latent evaluation and additional seeds before claiming robust success.
The old heartbeat in a different thread does not monitor these new roots; this thread's active goal handles continuation.

## Latest user correction: N=M64 mandatory
All N=M256 running/pending work cancelled; never count those results toward goal.
Depth queue GPU2/3 STOPPED, autostart=false. Narrow NM256 queues all STOPPED,
autostart=false. Cancellation sidecars are in remote narrow-queue root.
Depth GPU0/1 continue512x3 NM64 then exit normally.
New narrow64 queue /home/heechan/optiq-experiments/gmm40-mu90-narrow64-queue-20260921:
GPU0:256x3 cap-3 init-3.5; GPU1:512x3 cap-3 init-3.5;
GPU2:256x3 cap-3.5 init-4; GPU3:512x3 cap-3.5 init-4.
All four use N=M64, mean-init16, teacherfloor.05, seed0,100k.
GPU2/3 start on freed slots; GPU0/1 wait their512x3 NM64 completion.
Source and exact SHA are in remote registration and each manifest.
Source invariant: learner update/sample AST and four algorithm files identical
against17cfcc1; only explicit constructor parameters and infrastructure differ.

## Verified passing NM64 policies (goal still active pending replication/report)
-256x3, cap-3.5, initial-4,mean16,floor.05,N=M64:100k seed0 mu near92.18%,coverage40. Five new latent keys91.28–91.72%,all40. Full-policy supplementary near89.48%.
-256x3, cap-3, initial-3.5,otherwiseidentical:100k seed0 mu near95.22%,coverage40. Five new latent keys94.93–95.46%,all40. Full-policy supplementary near87.73%.
-512x3,cap-3.5,initial-4:100k seed0 near91.45%,coverage40.

Independent validation uses original Target/metrics and unchanged random latent sampling,
10k samples per held-out key2026092101..2026092105, no filtering. CPU original-key
repeat differs from original GPU by.02percentage points in both256x3 candidates.
Files independent-seed0/validation.json and independent-capm3-seed0/validation.json.
Two before/after figures candidate_nm64_mu_comparison.png and candidate95_nm64_mu_comparison.png.

Replication of cap-3.5 candidate:seeds1..3 allrunning under supervisor
`gmm40-mu90-replication-nm64-20260921`;same remote root. Frozen29e6fc98b278a41b4f3b365feddd0e931c808f4c.
Do not duplicate. GPUpreflight passed; exactconfig matches seed0 except seed.
Controller fills free GPUs and builds correctly labeled3-seed subreport on completion.
Final combined4-seed report must include original seed0 from e0f1ba0 source separately.
Remaining512x3 cap-3 seed0 was also running; finish/collect its result before finalreport.
Check actualprocesses/currentstate. Keep goal active until remaining work and report done.

## New user request: smaller sigma (pending)
Six matched runs registered under small-sigma queue, frozen137171ec724a2be77c9556c93858fc6c14b60234.
Root /home/heechan/optiq-experiments/gmm40-mu90-small-sigma-queue-20260921;
services suffix-gpu0..3. Read registration/status/failurefiles and actualSupervisor.
Each plan's initial logσ=-4.75 (interior to all bounds),cap[-3.5,-4,-4.5]
crossedwithwidth256/512,depth3. Otherparamsidentical:mean16,N=M64,batch256,
T1,beta1,Adam3e-4,teacherfloor.05,randomlatent,seed0,100k.
GPU0:256cap-3.5 then256cap-4;GPU1:512cap-3.5 then512cap-4;
GPU2:256cap-4.5;GPU3:512cap-4.5. Do not duplicate or restart.
Plan filesmu90_small_sigma_grid.json andmu90_small_sigma_0..5.jsonin frozen source.
collect.py accepts eachcampaignslug;report_small_sigma.py builds6-run finalplots
onlyafterall100k audits pass. Preserveold cap-3.5initial-4 result; new control
changesinitialto-4.75 to isolate cap effectswithinnewgrid.

Replication cap-3.5 seeds1..3 completed100k:seed1 near92.08%,coverage35;
seed2 near89.57%,coverage40;seed3 near92.78%,coverage40.
Do notclaimalltrainingseedsreproduce40/40+90. Seed0and3pass,1and2donot.
512x3cap-3 seed0 finishednear92.94%,coverage38 (failscoverage).
Allresultsdo not imply the near95cap-3 seed0 is stableacrosstrainingseeds;
only its fiveheld-outlatentsetswere independentlyverified94.93–95.46%,all40.

## Latest user request: reproduce with256x2 ifpossible
Four newruns, frozen6aee290c068b9efa58092f6a23c1b4c3cc5f5d8d, pushed.
Queue root /home/heechan/optiq-experiments/gmm40-mu90-256x2-repro-queue-20260921,
supervisor suffixgpu0..3;sourcefilesmu90_256x2_grid.json andmu90_256x2_0..3.json.
Primary:256x2 cap-4.5 initial-4.75 (GPU2), otherwise exactlysuccessful512x3smallσ.
Controls:cap-4 initial-4.75 GPU3;cap-3.5 initial-4.75 GPU0;
cap-3 initial-3.5 GPU1 (matches95.22%256x3seed0 exceptdepth).
Allmean16,N=M64,batch256,T1,Adam3e-4,teacherfloor.05,randomlatent,seed0,100k.
User preference is256x2; do notmarkoverallgoalcompletebeforetrying/reportingthese.
Independent validation for anypassing256x2checkpoint usingexisting script.

Smallσ512x3cap-4.5,initial-4.75:100kmu91.31%,40/40;fresh5keys91.01–91.32%,all40.
Fullpolicy supplemental91.13%,40/40.Conditionalσmaxphysical.444 vsGTσ1.313.
Descriptivewithin-modeμRMSstdmean1.069 (nearest-mode,within3σ diagnosticonly).
Smallσ256x3cap-4.5=87.45%,40/40;cap-4=89.50%,40/40;
newinit-4.75cap-3.5control256x3=91.74%,35/40;512x3=91.14%,40/40.
512x3cap-4 was stilltrainingatlastcheck;finishall6andplotreport_small_sigma.py.

##256x2 continuation to500k (ACTIVE)
100k256x2failednear criterion:cap-3initial-3.5=78.08%,38/40;
cap-3.5initial-4.75=74.60%,34/40;cap-4=75.70%,40/40;cap-4.5=75.19%,40/40.
Continued exact cap-4/-4.5 checkpoints tofixed500k (noearlystop), two services:
`gmm40-mu90-256x2-500k-20260921-2` and`...-3`, GPU2/3respectively.
Root/home/heechan/optiq-experiments/gmm40-mu90-256x2-500k-20260921;
results/<mu90_256x2_capm4p5_s0_500k ormu90_256x2_capm4p0_s0_500k>.
Originaltraining source6aee290; resume_runner_commita661b78f08e9f63b30f778ce82959ef250f44f58.
Parameters,Adamstate,andRNGpreserved. GPUroundtrip preflight100000→100002
bit-identical aftersave/restore. Resumed100kmuarrays exactequaloriginalsamples.
run.py nowallowsTRGresume onlyifalllearner hyperparametersmatchparentconfig;
changedN/M,seed,width/depth,batch,T,sigma,mean-init rejectedbytestedguard.
50k evaluation intervals beyond100k. Algorithmcode unchanged andverified.
At250k:cap-4near84.65%40/40;cap-4.5near85.90%40/40.
Mustlabel500kresultaslongertraining,not100kreproduction.

Metadata minorfixpending: frozena661b78 campaign_job stilluseslegacyW&B
job_typegmm40-fixed-q-100k thoughnames/configstepsarecorrect500k.
Mutablebranchcampaign_job hasbudget-neutraljob_typeeditnotyetcommitted.
Afterfinish, usepublicW&B Run.job_type setterand Run.update tocorrectonly
thesetwo500k runs; preservehistory/artifacts. Setter supportwas beinginspected.
Recordpostlaunchmetadatacorrection sidecar; docsclaimedfixbeforeitactuallylanded,
soamendprovenance. Do notmodifyrunningfrozenfilesorrestarttrainingforlabel.

validate_latent_keys.py nowaccepts--step500000 andretainsoriginaltraining
source plusresume_runner_commit. Uploaded toserver180opsvalidate_mu90_latent_keys.py.
Usefreshoutfolder percheck. Re-check5held-outkeysonpassing500kcheckpoint.
