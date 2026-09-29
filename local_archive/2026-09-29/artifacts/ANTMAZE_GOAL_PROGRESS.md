# AntMaze stop and canonical OptiQ reset audit — latest 2026-09-25

User explicitly stopped all prior AntMaze experiments and requested a return to basic OptiQ settings, followed by a complete settings/evaluation audit. Both 5090 hosts have zero RUNNING supervisor entries after stopping four W&B-sync watchers (180 sigma higher; 199 sigma higher, dense anneal baselines, dense T1). NVML had zero GPU compute jobs. All AntMaze supervisor configs have autostart=false and autorestart=false. Some historical status.json files still contain stale pending/running lists; controllers are stopped and reject restarting when status.json exists. Do not resume or launch any historical campaign. vast1/4090 remains GMM40-only. Frozen sources/checkpoints/results preserved.

Canonical Direct GMM/TRG defaults were composed on server180 from source b111a993: 256x2 actor/critic GELU, direct marginal GMM NLL, 64 policy samples and one proposal each, random normal latent per action, T=.25, beta1, mean-init1e-4, log sigma[-5,-1]/init-1, teacher floor exp(-5), DACER disabled, gamma.99, tau.005, Adam actor/critic3e-4, batch256/UTD1/warmup5000/replay1M, no grad clip. The user's selected AntMaze base differs explicitly: T=1, DACER off, NovelD off, 256env/batch4096/8updates per256transitions/warmup8192. Commit d3ae8fefccd9d68fe9a6326d3c2f21e452940638 selects this as the OptiQ `basic` default, retaining core 256x2/init1e-4/actor+critic LR3e-4; the old 256x3/mean-init1/critic5e-4 adapter is explicit `legacy` only. Pushed to GitHub direct-gmm-trg-antmaze and fast-forwarded both5090 canonical checkouts. All previous frozen experiments and job logs are untouched. No new experiment was launched; real GPU preflight is deferred until a separately authorized run. Source-controlled audit: antmaze_experiments/DEFAULT_RESET_AUDIT_20260925.md. Local selected-default tests: 16 pass, both server smoke tests: 8 pass each.

Evaluation audit: current AntMaze default250k/20 episodes/mode, final100/mode; recent sigma W was50k/40. Direct policy fresh N(0,I) z and conditional truncated sigma per action; native fresh z mu-only; zero_z separate. Upstream eval starts v1 random and v2-v4 original fixed full state. No DACER behavior noise or NovelD reward at eval. Judge route success separately from entrance. Official AntMaze reward baseline sparse0/10 or v2 target20/10, original horizons500/700; OptiQ has no built-in AntMaze reward. NovelD is not part of core OptiQ. All earlier runs/results remain historical, not a default baseline.

---

# W completed8 finite-cap runs;2uncapped failed config serialization — latest2026-09-25

User asked progress. Current BOTH hosts GPU0..3 confirmed free by NVML andlocks.
No live/pending Wjobs. Sourceb111a99 preserved, no automaticrestart performed.
Finitecaps0/1/2/3 v3/v4 all258304total/250112postwarmup/7816updates completed.
Final100direct,randomz+conditionalnoise,originalsamefullstate,failuresincluded:
v3cap-1(control)R44success;0R15;1R12;2R2;3R4. Leftsuccess0inall.
v3entriesL/R/none respectively11/87/2,12/86/2,2/96/2,0/97/3,3/84/13.
v4success0all;cap-1U71D29;0U95D5;1U61D39;2U34D64none2;3U99none1.
Highercapdidnotimprove successfulmultimodalityinthese250k,singleseedconditions.
Nativev3success80/33/24/16/1 forcap-1/0/1/2/3,allright;v4nativeallzero.
All8traininggoalvisitszero. Lastbatchrawactorstdmeanv3 .942/1.264/1.425/2.166;
v4 .714/.743/.798/.756. Do notcallrawsigma actualtruncated-actionstd.

Unbounded2failed BEFORE maintrain/collection inpreflight: run.write(config)
usesjson.dumps(allow_nan=False),butnativealg.actor.log_std_max=inf.
ValueError:Out of range float values are not JSON compliant:inf.
Initial scratchsampler/NLL/gradientchecks passed;this is metadata serialization,
notobserved numerical divergence. Failurelogs saved locally. No hiddenfinitecap.
No automaticretry. Fixneedsportableencoding/restoration ofexplicitunbounded
config inallconfig/proof/checkpointconsumers ifpursued;donotalteralgorithm.
Userwasquestioning teacherflooranddefaultscope;donotsilentlyresetsettings.

Reporting artifacts/antmaze_actor_sigma_higher_v34_250k/report/results.json,
REPORT_KO.md,v3/v4_final_policy.png,native supplements,learning_curves.
All120rawpolicy/native rows checked (10completedpolicyconditionsincluding2controls),
rawSHA/start/endpointreward/goals/fullcheckpointserverproofs/initialhashes verified.
Figuresvisuallyinspectedwithfixedmargins. W&B8finishedandfinalmetricsverified
inwandb-final-verification-host.json BOTH;noAPIwrites/historyrepair.
Fullcheckpointarchivecopybeingstartedunderunifiedsessions;consume nextturn.
U lowercap4finalresults/W&Bverified;3checkpoints localSHAverified. v3capm3
archive hadtransientSSHbrokenpipe;read-onlypartialcopyretryinprogress.
Do notconfuse downloadretry withtrainingretry. Frozenresultsallpreserved.

No commondefault claims:Wv3normalizedgeodesic100/T3/teacherfloor1;
v4geodesic100/T1/floor.5,bothgamma.999/H+.7every500/DACERon/NovelDoff.
Userlastaskedifmu-only/z0cause:confirmedpolicy=randomfresh8Dnormalz peraction
+conditionaltruncatedsigma;native=randomzmu-only;zero_zseparate. fixedmeans
initialfullsimulatorstate,notlatent. Earlieralgorithmquestionaboutteacherfloor
wasinterrupted/unanswered:optionexistedcommit58555ac,default exp(-5)identity;
AntMazeoverrideadded4607dd5. Raisingit changesproposaldistribution;do notclaim
it is unchangeddefaultOptiQ solelybecause pinnedcomputationalfilesunchanged.

Goal ACTIVE,notachieved;noagents/newautomation/4090/cancelledresumes.

---

# W upper-sigma grid registered; user requests default-settings audit — latest2026-09-25

Goal remains ACTIVE; v3/v4 successful multimodal retention is unresolved.
LATEST USER asks whether settings other than sigma are defaults and requests
reward/structure report. IMPORTANT: they are NOT common defaults. W inherited
O/P maze-specific controls: v3 normalized remaining geodesic100/T3/floor1;
v4 original geodesic100/T1/floor.5; bothgamma.999,H/d+.7/500,DACERon.
Explicitly disclosed this in commentary; do not claim baseline/default settings.
No additional reward/config change or cancellation was requested in that question.
Need clarify any future reset of what baseline means before changing existing jobs.

W explicit user grid: v3/v4 x log upper0/1/2/3/unbounded, initial-1/lower-5.
Ten fresh250k postwarmup seed0, controls archivedcap-1; source
b111a993b1e1bf895966dac4469abdfdb2c37c05, committed/pushed/synchronizedbothhosts.
All9 pinned computational files unchanged. Eight regression tests passed locally
andbothhosts. Frozen source and supervision use
antmaze-optiq-sigmacap-higher-v34-250k-s0-20260925 on180(v3)/199(v4).
Five perhost,4GPUs/host, existinglocks,2sec independentbackfill,no4090.
First8finitecaps assigned; eachuncappedjob pending. No automaticextension/retry.
Actual v3 fourpreflight/main validated, model/optimizer/replayreward/readback,
exactinitialactorcriticcontrols, finiteNLL/gradient/densitychecks andsource/cwd/PID.
Mainv3~45k-49k latest. Allfourv4actualpreflight/mainnowverified at20k-28k.
All8learnerPIDs/configs/frozenCWD/initialhashes/replayproofs verified;2uncappedpending.
Unbounded really passes+inf toexistingclip, not a finite substitute; queued
jobwillrun itsownstress/fullpreflightbeforemain. RuntimeconfigInfinity is
accompaniedbyexplicitupperremovedflag/nullmetadata. Existingfiniteguard remains.
artifacts/antmaze_actor_sigma_higher_v34_250k hasvalidatedplans/registration/
source-sharing/collector/verify_launch.py/settings-audit.json.

U lowercap+init-2/-3 COMPLETE4/4, frozen6e9090c preserved. Finaldirect100:
v3cap-1R44success;cap-2R15;cap-3R14, allleft0. v4allzero;
cap-1U71D29entries,cap-2D96U3none1,cap-3D91U9. Notpurecapeffectbecauseinitchanged.
Allraw72reportrows/finalproofs verified; W&B4finished/metricsverifiedno repairs.
V4bothfullcheckpoints archivedSHAverified. V3archivesession76578currentlyactive;
consumeitbeforelaunchinganotherarchivecopy. LocalartifactUreport/REPORT_KO.md.

V frozen teacherMCdiagnostic COMPLETE2/2, reporting637fdb7/trainO/Punchanged.
LocalrawNPZSHAandallstatisticrecomputationverified; report at
artifacts/antmaze_teacher_mc_diagnostic/report/REPORT_KO.md.
M64ESSmean32.63v3/26.24v4,notonecandidatecollapse. Outputgradientcos~.21 vs
independentfinite4096reference; notbatch4096network-gradientvariance.
NoM/Ntrainingchangeornewrollout; allmodel/optimizer/RNGpreserved.

Do notduplicate Aentropy16orcancelledcampaigns. No newagents/automation/4090.
S/T andpositivev1recordsbelowremainvalid. Currenttaskfinish: verifyv4actualmain,
collectsettings/sourceproofs,respondto user's defaultsquestion accurately.

---

# U sigma-cap actual main verified; S/T complete — latest2026-09-25

Goal ACTIVE. PROGRESS this continuation; v1 positive same-origin retention is
preserved, but v3/v4 successful multimodal retention remains unproved. Original
entropy16 grid A is fully complete; do not duplicate the old visible request.
No subagents/newautomation/4090/cancelled resumes. Current clean local/tracking,
GitHub and BOTH canonical/frozen source:
6e9090c8b9bb0ed7d1700ab6339a481bc97bf84d (direct-gmm-trg-antmaze).
Commit: Screen AntMaze conditional sigma caps with unchanged OptiQ updates.
All nine pinned computational files byte-identical to2564b59. Upstream antmaze/,
reward/physics/shared defaults untouched. Seven profile/default/manifest and
collection regression tests passed locally and BOTH servers before registration.

U NEW LIVE host199 root/supervisor:
antmaze-optiq-sigmacap-v34-250k-s0-20260925 (+wandb-sync).
Four fresh seed0: v3/v4 x actor cap/initial log sigma -2/-3; lower-5. These are
paired cap+initial changes, NOT isolated cap effects. Default[-5,-1]/init-1
remains unchanged. Original256env/eight updates, NOT T32 profile. v3 normalized
geodesic progress100/T3/teacherfloor1;v4 original geodesic/T1/floor.5. gamma.999,
H/d+.7/500,DACERon,randomnormalz,N=M64,256x3GELU,meaninit1,Adam3e-4/5e-4,
tau.005,batch4096,replay1M,NovelDoff/noB/nocost/native700/fullfixedorigin unchanged.
250k postwarmup=258304total/7816updates/16regupdates;40direct+40nativeeach50k,
100final/fullcheckpoint. No automaticextension/retry. Current aim: determine
whether reduced conditional action spread improves precise goal acquisition,
while teacher proposals remain broad; it could instead reduce exploration.
DACER adaptation to changed entropy is retained and is not held behavior fixed.

Real preflight and main verified ALLFOUR:8448transitions/eightactualbatch4096
updates,256simulators,fullreplay/rewardreadback,sampler/serialization/RNG checks.
Actual initial critic EXACTsame as O/P. Restoring ONLY sigma bias to-1 in a
scratch actor tree reproduces exact archived control actor hash. Every other
native cfg value matches control: only log_std_max/initial_log_std/output_root
differ. Actual GPU learner/cwd/cmdline/frozen source verified at12k..20k.
Learners at verification: v3capm2 585210, v4capm2 584814,v3capm3 585222,
v4capm3 584672. Main statuses clean, no failures/pending. W&B IDs respectively
98lhm34t,ke7x8zyn,id437ib9,kyd0l8d7, projectOptiQ/antmaze.

Local artifacts/antmaze_actor_sigma_v34_250k has precommit-validation,
validated/frozen-launch-plan,source-sharing BOTH,registration,actual-launch-
verification, collect_results.py,verify_launch.py,archive_checkpoints.py,
report_results.py. Reporter now validated26rows including first U50k raw evaluations. It compares -1/-2/-3 at exactlymatched transitions, raw starts/
reward/goals/SHA and finalreplay. Newly generated reports are posthoc/SHA-recorded.
Always collect THEN report sequentially. Never relabel mu-only as direct or
strip sigma post-hoc. See ACTOR_SIGMA_SCREEN_PROTOCOL.md in newsource.

S fixed64 COMPLETE2/2, source9b733bad frozenpreserved. Final direct100:
v3R92/L4/none4,successR15/L0 vs randomcontrolR87/L11/none2,successR44/L0;
v4D71/U27/none2,success0 vscontrolU71/D29,success0. Nativev3R56successful,
nativev4U1successful. No evidence fixed64 helps; do notextend. All48reportrows,
raw/config/result/fullcheckpointSHA verified LOCAL. v3ckpt89320267B SHA
bbc9a8489fb138693a977a949c3e7c50021a47e54019da2b4b338b12d1231b34;
v489328907B SHA0136085d31a0c391b7b409b17e274a779a88ffbf5a879350a952e0de3e80b131.
S finalfigure visuallychecked. BOTH W&Bfinished/finalmetricsverified.
v4 API summary wasstale despite uploadedwandb-summary.json correct. Narrow
metadata-only patch from uploaded file verified against rawresult; no resume,
history/config/training/frozenchanges. First POST-write verify hit SDK busy
service while creatingsecondApi; separate read-only process verified successful
existingwrite WITHOUT repeatingpatch. Sidecar saved remote/local
wandb-summary-repair-1727hgdu.json; final verification in
artifacts/antmaze_fixed64_v34_250k/wandb-final-verification.json. Do not rerun
repair script (refuses existing sidecar). v3JSONmeanrounding2.27e-13 checked
with1e-9absolute tolerance; episodes/success counts exact. No unresolvedissue.

T env32 COMPLETE2/2, source7d1b9f2 preserved. Final258304direct100:
v3R88/L7/none5,successR48/L0 (nativeR48); v4lower100,success0 (nativealso0).
Control256directv3R44, v4zero butU71/D29entries. T training28goals ALLgoal2
(right),355episodes;v4zero. Smallerconcurrency increased goalvisits but did
notretain successfulmultiplepaths. Noextension. All48reportrows and FOUR
control/treatmentfinalproofsverified. Tfinalfigure visuallychecked.
BOTH W&Bfinished/finalmetricsverified, no repairs needed.
Bothfullcheckpoints LOCALsize+SHAverified: v389058699B
35bd5bf787eb29bebd41a8bd6d1a5bd719e25c62d805846890d5bea23baab47b;
v489053387B e2db01d6d74f0f1497ac8fcdaf89c0349dd82a1b0a1682eb00fa9199dd06cf1f.
Treport/W&Bverification/artifacts in artifacts/antmaze_env32_v34_250k.
There are no pending archive/tool sessions at this update (all previous sessions
consumed). U training is supervised, not a local shell session.

Next: collect U actual50k/100k trajectories; compare oldO/P at samebudget;
inspect failures/branch successes before any futuredecision. Do not invent
success from entrances. Keep activegoal, no premature completion. All existing
M v1 positive1M sameorigin evidence and Q/R diagnostics remain as earliernotes.

Latest U collected progress:
[
  {
    "id": "v3-optiq-startnorm-geodesic-T3-teacherfloor1-capm2-250k-s0",
    "step": 49152,
    "updates": 1280,
    "successes": 0,
    "url": "https://wandb.ai/OptiQ/antmaze/runs/98lhm34t"
  },
  {
    "id": "v3-optiq-startnorm-geodesic-T3-teacherfloor1-capm3-250k-s0",
    "step": 49152,
    "updates": 1280,
    "successes": 0,
    "url": "https://wandb.ai/OptiQ/antmaze/runs/id437ib9"
  },
  {
    "id": "v4-optiq-geodesic-T1-teacherfloor0.5-capm2-250k-s0",
    "step": 49152,
    "updates": 1280,
    "successes": 0,
    "url": "https://wandb.ai/OptiQ/antmaze/runs/ke7x8zyn"
  },
  {
    "id": "v4-optiq-geodesic-T1-teacherfloor0.5-capm3-250k-s0",
    "step": 49152,
    "updates": 1280,
    "successes": 0,
    "url": "https://wandb.ai/OptiQ/antmaze/runs/kyd0l8d7"
  }
]
Updated 2026-09-25T01:59:08.649484+00:00

---

# T32-env actual main verified; S finishing; Q stopped — latest2026-09-25

Goal ACTIVE. This turn progresses two independent hypotheses without changing
any of nine pinned computational files. Latest local/GitHub/BOTH canonicals and
frozen source7d1b9f2ee63ba959e66f3453f2587cdbd6d993a3, clean/tracking synchronized.
No4090/subagents/newautomation/cancelled resumes. Original entropy16 grid A is
already fully complete, raw/checkpointproof/W&Bfinished verified; do notduplicate.

T NEW host180 root/supervisor antmaze-optiq-env32-v34-250k-s0-20260925,
source7d1b9f2, actually GPU preflight AND main verified. v3normalizedgeodesic/
T3/teacherfloor1 andv4originalgeodesic/T1/floor.5, normalrandomz, gamma.999,
H/d+.7/500, otherO/Pcontrol settings unchanged. Only collection cadence changes:
32env/one update replaces256env/eight; batch4096,global1/32ratio,warmup8192,
258304total/7816updates remain identical. Original700/fullorigin retained.
Intermediate40episodes/mode at EXACToIdcontrol steps50176/100096/150016/200192/
250112;final100/completecheckpoint. 32env gives8072physicalsteps/env byend vs
1009old;700ratherthan5600updates during700steptrajectory. Hypothesis,notcausalproof.
Collection order/replay correlation necessarily differ; cannotclaim identicaldata.
No automaticextension/retry. Existing absent option remains256/eight; testlegacy
counts/clocks at allblocks passed4unit tests locally andonbothservers; runtime
preflight8448/eightupdates/32simulators/fullreplayreadbackpassed. Actualinitial
actor/critic hashes in BOTH mainandpreflight exactlyequalO/Pcontrols. Native
learningcfg differs ONLY output_root. Actual180GPUmain PIDs565431(v3,at49152),
565247(v4,at45056) verifiedfromNVML/cwd/cmdline. W&Bv3zns9ja7y,v4uv0ptbcs.
Local artifacts/antmaze_env32_v34_250k: validated-launch-plan,source-sharingboth,
registration,actual-launch-verification,collect_results.py,verify_launch.py,
report_results.py,archive_checkpoints.py. Reporting is posthoc SHA-recorded.
Firstcollectedstatusmain8192 wasold; actual-launch-verification isnewerauthority.

S host199 root antmaze-optiq-fixed64-v34-250k-s0-20260925, source9b733bad...
remainsunchanged. Real main verified at98304each, GPUlearner572040(v3)/571773(v4),
fixedcodebook initial/main parameterhashes matchO/P, all sampler/serialization
checks passed, core9unchanged. v3W&Bkur0b215,v41727hgdu. Bothreached258304/7816
andfinalevalrunning. v4 final100direct entrieslower71/upper27/none2,success0;
mu-onlylower75/upper25,upper1success. v4resultcompleted andcheckpointproof
collected. v3final100/resultnotyetverified. At250112direct40v3 R34/L4/none2,
R6success,0left; randomcontrolR35/L5,R16success. No demonstratedfixedpriorbenefit.
Local artifacts/antmaze_fixed64_v34_250k reporters validate rawreward/fullstarts
and same-stepcontrols; finalproofs/archives must finish beforecompleteclaim.
Archiveexec31689 currently copiesv4 only(v3wasnotready); W&Bcheckexec66285pending.
Nootherpendingtool sessions atthisnote. Do not rerun archive until31689consumed.

Q host180 teacherfloor1retention (a5de32,6c82yv3s) wasSTOPPED EARLY atlastlogged
696320/21504updates after direct500224/550144/600064 allright40 andnoleftsuccess.
Exactoldcontroller/job/learnerPIDsmissing/PGIDempty, bothsupervisorsSTOPPED.
Stopcommand initially exitednonzeroafteractualstop; read-only verification
confirmedtermination andmetadata finalizedwithoutrepeatingstop/restarting.
screen-stop.json recordsobservationissue. Alllogs/intermediatepolicies preserved;
NO1M/fullfinalclaim. W&Bsummary stoppedearly verified; APIstate stillrunning
at01:10washeartbeatlag, not actualtraining. Qreport updated38rows+10exactprefixchecks.

R physicalcontactreport corrected: not literally zero goal progress. Last200
geodesic distance decreases meanupper.364m/lower.856m, whiletravelpath~9m and
net.815/1.208m,contacts6.7/6.0%,0goals/native700. Allpairedraw/actions/checkpoint
stillidentical; derivedmetrics/reportSHAupdated. No new rolloutortraining.
V1positive1M same-full-origin retention preserved; v3/v4 goalstillunproved.

Updated 2026-09-25T01:33:50.325968+00:00

---

# R contact diagnostic complete; Q reaches ~450k — latest2026-09-25

Goal remains ACTIVE. New evidence this turn: frozen v4 policy's failure is not
well described as continuous physical wall contact. R paired CPU evaluation
completed, exact actions/XY/returns/full starts/model+optimizer/RNG/checkpoint
invariance verified; all raw arrays archived with SHA and locally validated.
New canonical/GitHub/local/BOTH-server frozen source f21108190afa212a811b78ca222b2bf7ce086489.
Core7 unchanged from2564b59; no learning/TD/loss/optimizer/sampler change.
Prelaunch59bd6aa/dda9 were not executed; f211081 fixes diagnostic local variable
shadowing before any inference. Q still runs original a5de32, not newsource.

R host199 supervisor antmaze-v4-wall-contact-diagnostic-20260925 is EXITED;
observed actual CPU PID565219, CUDAempty/JAXcpu/source/cwdverified. Noauto-retry.
Original P floor.5 v4 checkpoint258304, source555bb7e, native700 horizon,
40directpolicy same-full-origin episodes with/without instrumentation.
Both conditions U26/D14, goals0. Last200 averages: uppercontact6.73%, path9.27m,
net.815m, torsoheight.582m, finaldistance6.820m; lowercontact5.96%, path8.98m,
net1.208m,height.573m,distance6.088m. Noepisode has >50%tailcontacts AND net<1m.
This is local movement with little net progress, not evidence for constantwall
contact. Readouts are pre-action last-physics-substep cache, not allsubsteps or
force measurements; do not claim wallcontact has no causal effect. Do not infer
body-inflation or any remedy from contact correlations. No newtrainingvariant
was launched from this diagnostic. Map drawn from rewardgeometry was checked
exactly against actual physical block boxes. Figure visually verified.
Local artifacts/antmaze_v4_wall_contact_diagnostic/{raw,report,execution-verification.json};
report/REPORT_KO.md, results.json, wall_contact_diagnostic.png. Reportcode is
post-hoc/sha-recorded, training/evaluation sources separate. W&Bc8ce7ab00951
APIverifiedfinished, parenttwlpwini, success0,40episodes; metadata archived.

Q onlyliveGPUtraining: host180/GPU0 controller560894/job560959/learner561664,
sourcea5de32fe885b79700ab4566bf36fe91de2ee3860, allactualhandles/cwds verified
at364544total/11136updates. Newer collectedprogress446464total/13696updates,
105training-successes. Direct400128: R39/L1, R35successful/40, noleftsuccess;
mu-only450048: R40, R31successful/40. No failure/pending/screenstop. Do not
confuse modes/evalsteps; direct450 stillnotcollected atthisnote. Reporter29rows,
10exactsharedprefix checks: bothpolicy/native50k..250k match earlier O rawarrays.
Apply committed earlyscreen only FROM500k if leftentry<=1/40 and0leftsuccess
for3consecutive directevals. No Qrestart/extension/parameterchange.

P BOTH fullcheckpoints now fully downloaded locally and SHA verified (90102
completed). .5 bytes89322187 SHA5d082deaddef3831dbabd52b9e3744280900215b07063b638fb8486a9512e437;
1 bytes89322315 SHAe4eaec06e3108158b117d7c5a25c658b72a4be265f9297e6f322495b9dd75747.
archive-verification.json perrun provescomplete; earlierpartial notesobsolete.
No pending localtool sessions atthisnote. 4090 untouched; no agents/automation.
V1positive1M same-origin retention evidence preserved; v3/v4 stillnot solved.

Updated 2026-09-25T00:53:41.657078+00:00

---

# Q actual main verified — latest2026-09-25

Goal remains ACTIVE. NEW Q1M v3teacherfloor1 actualGPUpreflight and mainboth
verified fromfrozen a5de32fe885b79700ab4566bf36fe91de2ee3860. Onlynewtrainingjob.
Host180/GPU0/learnerPID561664/processgroup560894,
controller560894/job560959. Atverification step49152,
updates1280. W&Bhttps://wandb.ai/OptiQ/antmaze/runs/6c82yv3s.
Actual8update/batch4096GPUpreflight/fullcheckpointrewardreadbackpassed;
initialactor/critic hashes exactlymatchcompletedOcontrol; nativealgandDACER
configsdonotdiffer atall. Mainconfig1008384total/31256updates,256env/8updates,
T3/gamma.999/teacherfloor1/normalizedgeodesic verified. Core7unchanged; actual
GPUmainPID/cwd/sourcechecked. Localactual-launch-verification.json recordsall.
Firstread-onlyverifierwasrunduringmainwarmupandassertedupdates>0; readiness
checknowwaitsforpositiveupdates,allstrictchecksretained. Posthocobservationsaved;
nottrainingfailure,no traincodechange/restart. Currentrealmainverifiedhealthy.
Qcollectorandreporterready. Checkshared50k..250kpolicyarraysagainstOcontrolas
newrawarrives. From500kscreenaccordingtocommittedprotocol. Noadditionalhypotheses
registered;allothercompleted/stoppedtrainingpreserved.

P v4both250kcomplete/W&Bfinished/rawverified asnextsection. Fullcheckpoint
resumablecopy execsession56867 stillongoing;partialfilesmustnotbecalledverified.
Usewrite_stdin toconsumecompletion; thencheckarchive-verification.json perrun.
Atlastfilecheck~62MB/58MB eachof~89MB. Raw/config/36reportrowsalreadylocal,
finalmatchedtrajectoryfigureQApassed, bothdirect100success0.

Updated 2026-09-25T00:33:28.424002+00:00

---

# Q registered; P complete — latest2026-09-25 continuation

Goal ACTIVE. v1 same-origin positive evidence persists; v3/v4 not solved.
Newest clean/pushed/shared/frozen source a5de32fe885b79700ab4566bf36fe91de2ee3860.
Q NEW registered180 only: antmaze-optiq-v3-teacherfloor1-retention-1m-s0-20260925,
supervisor same plus-wandb-sync. One fresh v3 seed0 normalized-geodesic/T3/
gamma.999/teacherfloor1/Hperdim+.7/500 confirmation. Only jobid/hypothesis/budget
change versus actual completed O d259301 floor1 manifest. Max1008384total,
31256updates/63regupdates.40each50k,100final; original fixed fullstart/native700.
Core7 byte-identical to2564b59; no learning/runtime/reward/physics edits.
Manual evidence-reviewed budget confirmation under activegoal, not oldqueue
resume or independentseed. From500k screen if leftentries<=1/40 and0leftsuccess
for3consecutive directevaluations. Noextension orautorestart. Compare short/long
sharedprefix before claiming continuation. Do not extend failedv4 screens.
Registered and controller started; REAL preflight/main proof PENDING atthisnote.
Local artifacts/antmaze_teacher_floor_retention_1m/{collect_results.py,
verify_launch.py,report_results.py,validated-launch-plan.json,source-sharing-199.json}.
Reportervalidated12shortcontrolrows; no newQevalyet. Actor/critic initialweights
andactual GPU/config need verifier after mainprogress appears.
Previous e99d86ce444a3d8047fb0ed5baeddbcdc0f135f8 was dry-run only; a5 fixes stale
manifestdescription(finalbudget258304->1008384/controlfloor1) BEFORE any Qpreflight.
Bothcommits/history/sources preserved; no maintraining frome99. No199Qjob.

P v4 teacherfloor .5/1 now COMPLETE2/2 at258304total/7816updates, all100final
policy/native/zero_z evals. Source555bb7e remainsfrozen. Rawstarts/reward/SHA/
checkpointreadback/proofsverified; report36rows/final_verifiedall3. W&Btwlpwini
andj34ro45h bothfinished; allfinalsuccess0 ineachmode verifiedinAPI, sidecar
wandb-final-verification.json locallycollected. Directfinal100:floor.5 U71/D29,
floor1 U75/D24/none1; zerogoalsboth. Mu-only .5U68/D32,1U77/D23; zero goals.
Widerteacher is NOT a demonstrated successful route solution forv4.
Finalmatched directfigure visuallychecked; primarybothconditions100episodes.
P localfullcheckpoint archive copying viaresumablecompressedrsync, execsession
56867 stillrunning atthisnote. DirectSSHcat timedout180s; sourcecheckpoints intact,
NOT a learningfailure. Partialfiles clearly .partial and not called verified.
archive_checkpoints.py preservespartial/resumes/checks exactSHA beforefinalrename.
Only after itcompletes may claim localfullarchivesverified. Raw/config/results
alreadyarchived in artifacts/antmaze_v4_teacher_floor_250k/results/vast-heechan-199.

O v3floor.5/1 andMv1 completefullcheckpoints were alreadyarchived andverified.
No new latent/sigma/envcount/actor-delay/body-geodesic changes made. Read-only
inspection considered these mechanisms, but no such ablation/diagnostic was
implemented or launched. No4090/agents/automations/cancelled resumes.

Updated 2026-09-25T00:30:56.063084+00:00

---

# O finished and fully archived; P continues — latest 2026-09-25

Goal ACTIVE. O v3 teacher-floor screens are COMPLETE2/2 at258304total/7816updates,
all100finalmode evaluations, fullcheckpoint/replay readback and W&Bfinished
verified. No extension/restart registered. Source remainsd25930197ee9ba060a3aa84c3b6fea419dfe856d.
Final direct100: controlfloor-exp(-5)R84/L16entries,successfulR4;floor.5R84/L15/
none1,successfulR8;floor1R87/L11/none2,successfulR44. Noleftsuccessinanycondition.
Finalmu-only:controlR38success;floor.5R29;floor1R80. Allsamefixedfullorigin,seed0.
Both original actor/critic/replay checkpoints now archivedlocally under
artifacts/antmaze_teacher_floor_250k_r2/checkpoints/<job>/checkpoint-final.pt,
~89MB each, exactSHA/bytes verified (archive-verification.json).
SHA .5=6580f63a924d184968daecc6d624ddd4f35b74d10e007617365924a76d4a6b57
SHA 1=613867a725feff0f6e57ff05b5d4147f0755b029e2e58f11f847c1363b53f1a8.
Reporter36rows/final_verifiedall3; finalmatched directfigure visuallychecked.
W&B4qpoemxl/hrhjsjsx finished, finalpolicy.08/.44 andnative.29/.80 verified,
zero_z0both. Sidecarwandb-final-verification collected. Nevermergeevalmodes.

NEW fullreplay chronology check (read-onlyCPU180): BOTH O datasets have0actual
goal-radius visits and0goal-terminaltransitions; training-successes.json0 agrees.
256environments means1009transitions perenvincluding32warmup; there has onlybeen
one timeout perenv by250k. Closesttrain distances[goal1,goal2]:floor.5[6.079,1.843]m,
floor1[5.694,3.925]m. SavedfinalpolicyevaluationR44success is distinct fromonline
trainingvisits underchangingpolicies. This counterexample means thiscase's
imbalance is not solely triggered by collectedfirstgoalterminal; itdoesNOTprove
a specificcause or solution. report/training-reach-verification.json records
fullreplay/checkpointSHA andsame localarchivehash verification. LocalTorchcould
notunpickleJAXtypes (localJAXabsent), so no packages installed; verified withthe
compatible frozenserverruntime, CPUonly, not training. LocalbytesSHAverified.

Teacher diagnostic: archived actualCSV8minibatches at160000total (updates4737..4744),
notentiretraining ororiginconditionalstats. ESS/64mean:old19.2678,.5=30.7748,1=33.8654;
sourceQstd2.765/2.839/2.966; stdatuppercapall. CSVrawSHA+scopedsummary stored
report/teacher-diagnostics/. Diagnostic5000 modulus and256step collection align
onlyevery160000; do not claim dense time-series or change frozenlogging.
Thisimproves effectiveweightedcandidates butnot minority successfulroutes.

M positivev1final fullcheckpoint ALSO now archivedlocally:
artifacts/antmaze_increasing_temperature_1m/checkpoints/<v1job>/checkpoint-final.pt
312565067bytes, SHAed28026f23ac6e382d46467de581d32914b864cea6076022df2f81047355286a,
archive-verification.json provesidenticalbytes. Mv4 remainsstopped946176.

P newv4 .5/1 teacherfloor on199 remainsrunning from555bb7e3c0101ee939545cc35d4d0729291781ec,
actualmain/preflight verified, nofailure/noqueue. Latestcollected~100k atthis
note; usefreshstatus/report. No new sigma/latent/envcount/actor-delay orother
hypotheses have beenimplemented or launched. Corealgorithmstructureunchanged.

Earliernotesbelowpreserved; this supersedes O pendingfinaleval status.

---

# Latest update — new v4 teacher-floor pair actually running

Current clean/pushed GitHub and BOTH server canonical/frozen HEAD:
555bb7e3c0101ee939545cc35d4d0729291781ec. Only new registrar/protocol/AGENTS changed
fromd259; all7computationalcore files remain byte-identical to2564b59.
Goal ACTIVE. No4090, no new automation/subagents, no old job resumes.

P NEW ACTUALLY VERIFIED on199:
root/supervisor antmaze-optiq-v4-teacherfloor-250k-s0-20260925 (+wandb-sync),
source555bb7e. Freshv4seed0 original-geodesic,T1,gamma.999,H/d+.7/500; teacher
floor.5/1only vsactual438907f3 archived250k control. No normalizedreward forv4.
Actor sigma[-5,-1] and every learning/evaluation/reset/physics default preserved.
258304total/7816updates,40each50k and100final/fullstate, noautoextension/retry.
Actual8448/8batch4096 preflights/fullreplay/proposal-density passed; initial
actor/critic hashes identical to archivedcontrol; actualmainconfig differs only
actor.teacher_std_floor andits2existing aliases. Source/core7, GPUmainPIDs,
cwd andruntimeverified at16384total. GPU0learner556120, GPU1learner555853,
job PIDs554650/554651. Readactual-launch-verification.json forcontrollerPGID.
W&B floor.5 twlpwini / floor1 j34ro45h, projectOptiQ/antmaze.
Local artifacts/antmaze_v4_teacher_floor_250k contains manifest/resultscollector,
control-validated report, immutable-source verification and sharingproof.
verify_launch.py isREAD-ONLY postlaunch withrecordedscriptSHA;reporter isseparate.
199 prior unrelated W&Bsync services284013/282542 leftuntouched; noothercompute
wasrunning atPregistration. Existing cancelledqueues preserved.

O v3 floor.5/1 on180 bothreached258304/7816 withnolearnererror; final100episodes
permode andzero_z stillunderway atlastread. Latest floor1.0 nativefinal100:
right97left3,successR80; direct25011240:right35left5,successR16. Floor.5direct
25011240:right34left6,successR2. No minority successverified. Do NOT claimwhole
campaigncompleteuntilresult/checkpoint/rawverified. NoOextension launched.
O source d259301 remainsfrozen despite newer555source. Oreportlast33rows;
final_verified onlyhistoricalcontrol beforelatestcollection. Collect/reportreread.

M v1same-origin finalpositive andMv4earlystop946176 remain aspreviousupdate.
Strong2x3same-origin retentionfigure visuallycheckedagain andqueuedinCodexpanel:
artifacts/antmaze_increasing_temperature_v1_origin/report/fixed_vs_increasing_temperature.png.
All500k/750k/1Mv1routecounts are verified one-seed evidence; v3/v4unresolved.

Earliernotesbelowpreserved. Thisentrysupersedes any olderHEAD/livecounts.

---

# Update — M v4 screened, O two screens finishing — 2026-09-25

M v4 is now STOPPED EARLY, last logged946176total/29312updates. Exact supervisor
and sync stopped; group549231 verifiedempty, completedv1preserved, O557499/557506
liveprotected. 700160/750080/800000direct40:lower39upper1,zero successes;
850176/900096:lower40,zero successes. Latestfixedfullorigin/rawSHA checks verified
before stopping. No final1M/fullstate/final100claim. screen-stop.json, backups,
updatedstatus/job and W&B screen/* metadata verified652g24xq. Local Mcollector
now includes screen-stop/actualverification/backups/W&Bsidecar; reporter exposes
both oldGv4 and newMv4stops explicitly. Frozen training untouched.
O r2 .5/1 reached245760total/7424updates and finalscreen evaluations are imminent.
Two preflight/main actual source/core/config/initialweights/PID checks were saved
locally in results/vast-heechan-180/actual-launch-verification.json.199source-sharing
proof and original failed4607 preflight-analysis are now locallycollected.
Oreport28rows. At150016direct40 floor.5 left17/right11/none12,successright1;
floor1left2/right16/none22,successright3. No leftsuccessfulroute yet; no efficacy
claim or automaticextension. v3floorcontrol at150kleft7right14none19successR3.
Matched100kdirectfigure was visually checked. All7computationalfilesunchanged.

Earlier notes below retained; this supersedes Mv4 LIVE.

---

# Current AntMaze goal handoff — latest 2026-09-25 continuation

Goal ACTIVE; do not narrow completion to v1. Original positive-entropy16 sweep
A is complete, including recovered199 raw files. Do not duplicate it.
Newest clean source d25930197ee9ba060a3aa84c3b6fea419dfe856d is committed/pushed,
shared with BOTH180/199 canonical repos and frozen snapshots. 199 SSH recovered;
its guide was read, A/B missing data recovered and verified; no jobs restarted.
No AntMaze4090, no new automation/subagents. Seven computational core files
remain byte-identical to2564b59. Teacher-only floor adapter forwarding changed,
not core samplers, losses, TD, optimizer or regulator.

O R2 LIVE AND ACTUALLY VERIFIED:180 root/supervisor
antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2 (+wandb-sync), source d259301.
Two fresh v3seed0 teacher floors .5/1, normalizedgeodesic,T3,gamma.999,H/d+.7/500;
only existing teacher floor vsHcontrol exp(-5), actor sigma remains[-5,-1].
Max258304total/7816updates;40eval each50k,100final/fullstate. Both actual8448/8
batch4096 preflights, proposal sampling/density, fullreplay checkpoint readback,
initial actor/critic identity withHcontrol, source/core/config/PID verified.
Learners557499(GPU0),557506(GPU2); controller556100/sharedPGID556100.
Latest actual verification step69632 each; W&B4qpoemxl/hrhjsjsx.
Remote actual-launch-verification.json proves checks; collect it locally.
Local artifacts/antmaze_teacher_floor_250k_r2. Prior un-suffixed4607dd5 campaign
FAILED PREFLIGHT ONLY: periodic proposal diagnostic absent at8448 caused new
validator KeyError after numerical preflight passed. No maintraining launched.
Old error/source/logs preserved; d259 validates resolvedfloor and alwayslogged
actor_std_mean; optional periodicmetric checked only when present. No logging
interval/core change. Failedold controller terminal; sync stopped; neverresume.

M v1 COMPLETE1M: sourceacb0f2a85f9947de6500d6b1d7cae1d7944e3685, original
geodesic/noCost/noB, gamma.999,H+.7/500, linear T1->3 over1Mpostwarmup.
1008384total/31256updates, finalfullcheckpointSHA
ed28026f23ac6e382d46467de581d32914b864cea6076022df2f81047355286a.
Same-full-origin supplement100direct episodes:
500224 successesU71/D11/fail18;750080U82/D17/fail1;finalU86/D10/fail4.
Finalmu-onlyU94/D4/fail2. Allsameinitialposition/pose/velocity, checkpoint/raw
SHA, reward/modeloptimizer preservation verified. Primaryv1 random unchanged.
Retained BOTH SUCCESS ROUTES through1M for ONEseed/ONEorigin; imbalanced,
not proof for latertraining/otherstates/v3/v4 or instantaneous actiondensity.
Compared toGfixedT1sameorigin500U49/D40,750U4/D95,finalD100.
Local artifacts/antmaze_increasing_temperature_v1_origin/report/
fixed_vs_increasing_temperature.png was visually verified, source/proofs saved.

M v4 STILL LIVE at handoff: GPU1 learner550730/controller549231/sharedPGID549231.
Latest direct750080 lower39/upper1/40,success0, meanfinalgeodesicdistance3.505m.
550successlower1;600/650/700/750success0, minorityentries2/2/1/1. Notyetstopped.
Refresh before deciding earlyscreen per committed M protocol. v1 is complete;
verify that before stopping entireMgroup. Protect OPGID556100. Nevercallv4final1M.
Mroot antmaze-optiq-geodesic-T1to3-retention-1m-s0-20260925, W&Bv4652g24xq/v17rw82x8w.
Local artifacts/antmaze_increasing_temperature_1m. Collector now includes
screen-stop/actual-launch-verification andstopbackups for prospectiveuse.

L both STOPPED EARLY598016 after repeated one-sided direct450/500/550;
noleftsuccess, notfinal1M. Exactgroup546235 empty; W&Bstopsidecars verified.
Local artifacts/antmaze_normalized_discount_retention_1m58rows/10prefixchecks.
N pairedhorizon complete: v3H T3,100samefullorigin pairedprefix rollouts:
700successR14/L0/fail86;1400R65/L1/fail34. Extra52mostlyright; supplementary
1400 is notnative700success. W&B authlogging-onlyrepair verifiedbdf1d90f0d88;
originalerrorpreserved, no training/evaluationrerun.
A all16complete/finalrawverified, B all8completeincluding199; B W&Bfinished.
See artifacts/antmaze_dacer_positive_500k and antmaze_horizon_temperature_250k.

Earlier entries below are historical; this section supersedes their live statuses.

---

# Current AntMaze goal handoff — 2026-09-25 08:01 KST

Goal ACTIVE, current continuation PROGRESS. Do not mark complete or narrow success
to historical v1. Original user entropy16 sweep is already completed; no duplicate
launch. Current clean GitHub/180 HEAD acb0f2a85f9947de6500d6b1d7cae1d7944e3685.
199 source-only sharing remains pending SSH recovery; no unknown jobs touched.
No AntMaze on4090, no new automations or agents.

End-of-turn refresh: actual progress Lboth348160, Mv1286720/T1.557056 and
Mv4147456/T1.278528; statuses4running/0failed/0pending. New direct300032 L:
.999 L13/R27,successfulR19; .99999L2/R38,successfulR19. No leftsuccess yet.
Lreport38rows/10exactprefixchecks; route-progress diagnostic regenerated.
Mv1direct250112 randomprimary U23/D15/none2,successfulU1; Mv4direct100096
U8/D14/none18,zero success. Mreport80rows. Do not claim early Mv1both-route
success or conditional retention. No tools/sessions left running locally.

M NEW AND ACTUALLY VERIFIED: v4/v1 fresh T1->3 linear over1M postwarmup, original
geodesic progress100/no cost/no bonus/NovelDoff, gamma.999,H/d+.7/500. Root180 and
supervisor antmaze-optiq-geodesic-T1to3-retention-1m-s0-20260925 (+wandb-sync),
training source acb0f2a85f9947de6500d6b1d7cae1d7944e3685. Same defaults/budget/eval
as original G control:1008384total/31256updates;40each50k and100final/fullstate.
v1 primary random, v4 original fixed full state. v1 same-origin supplement still
required at promising later checkpoints before any conditional retention claim.
Only temperature_schedule differs from G (plus job identity/hypothesis). Seven
pinned algorithm files remain byte-identical to2564b59; scheduled_temperature
interpolation AST unchanged. Parser/CLI now allow positive increasing endpoints;
controller accepts/verifies explicit schedule with explicit DACER target. Do not
claim every source file unchanged: validation-only changes are disclosed in
INCREASING_TEMPERATURE_PROTOCOL.md. Seven schedule tests passed locally/remotely,
including exact old decreasing values. Committed/pushed/frozen BEFORE preflights.
Both real8448/8batch4096 preflights, full replay/checkpoint readback, initial model
hash identity with G, source/core/config/current temperature verified. Local
artifacts/antmaze_increasing_temperature_1m/actual-launch-verification.json proves
GPU1 v4 learner550730(job549296), GPU3 v1 learner550454(job549297); controller549231.
Verified v4step49152/T1.08192, v1step98304/T1.180224. Both now beyond these points.
W&B v4 https://wandb.ai/OptiQ/antmaze/runs/652g24xq
W&B v1 https://wandb.ai/OptiQ/antmaze/runs/7rw82x8w
Collectors/report are ready and source-bound; latest report75rows, matched policy
figure visually checked. At v1 150016 direct U18/D13/none9,zero successes; v4last
complete direct50176 U6/D12/none22,zero. Early observations, not failure/success.
M/L controller families share process groups across child jobs: never kill an
entire group to stop one candidate while its other learner is live.

K COMPLETE2/2 AND FINAL VERIFIED: v4 original geodesic,T3/T10,source4d272c5.
Both258304total/7816updates; raw100direct/native and final fullstate/replay verified.
Direct T3 U38/D45/none17,0success,finaldistance10.589m; T10U2/none98,0success,
finaldistance15.697m. Matched originalT1 D73/U25/none2,0success,4.384m.
No automatic K extension. Local artifacts/antmaze_v4_geodesic_temperature_250k
report final_verified has original-T1,T3,T10; matched policy figure QA passed.

L STILL LIVE on180 GPU0/2,sourcef5b3fcb; no edits to frozen source. Latest report
has10 exact shared raw-prefix checks against H control,34rows. Direct250112/40:
gamma.999 R33/L7,successfulR8; gamma.99999 R37/L3,successfulR11. Neither has left
success. Native300032: .999R34/L6,successR28; .99999R36/L3/none1,successR17.
Do not describe this as retained successful multimodality or gamma causal failure
from one small checkpoint. Keep planned250/500/750 screening; no blindextension.
Read-only route progress script added under local artifact folder (not training
source), with rawSHA checks and all failure denominators. At250k left episodes all
reach original700step limit: .999 finaldistance3.921m, .99999 5.575m. This suggests
incomplete acquisition, but does NOT establish extra time would reach a goal.
No horizon override, new rollout, or training change was made for this diagnostic.
report/route-progress-diagnostic.json preserves group counts andlast100 gains.

Older notes below remain history; newest status above supersedes them.

---

# Current AntMaze goal handoff — 2026-09-25 07:45 KST

Goal ACTIVE. This turn is PROGRESS: completed/validated H, established I final
same-state collapse, preserved screened Gv4, committed/launched/actually verified
new L discount pair. Do not mark goal complete or narrow it to historical v1.
Current clean pushed GitHub/180 HEAD f5b3fcbfff3e61ecac4517facaa446393045c5a9.
199 SSH still times out (retried this turn), source-only sync pending; no unknown
jobs touched. No4090 AntMaze, no new automation, no subagents. Core7 files are
byte-identical to2564b59; no algorithm/runner/reward implementation changed this turn.

L NEW: antmaze-optiq-v3-startnorm-discount-retention-1m-s0-20260925 on180,
source f5b3fcbfff3e61ecac4517facaa446393045c5a9. Supervisor same name plus
-wandb-sync. v3 seed0 normalized geodesic/T3/Hperdim+.7/interval500; gamma.999
longer-budget control and gamma.99999 discount-only treatment. Maximum1M
postwarmup=1008384total/31256updates/63regulator updates. Keep50k40episode
intermediate evals/policies and100final/fullstate. Fixed original full start.
Both actual8448/8batch4096 preflights and replay/full checkpoint readback passed;
initial actor/critic hashes equal H's actualT3 control; configs/source/core7 checked.
Actual learner PIDs547458(gamma999 GPU0),547448(gamma99999 GPU2); last verified
49152total/1280updates each. Main job PIDs546300/546301, controller546235.
W&B https://wandb.ai/OptiQ/antmaze/runs/zs3gh977 (gamma999)
W&B https://wandb.ai/OptiQ/antmaze/runs/xd4sqbye (gamma99999)
Preserve K liveT10 and any GPU locks. No checkpoint resume or independentseed claim.
Protocol antmaze_experiments/NORMALIZED_DISCOUNT_RETENTION_PROTOCOL.md; compare
actual archivedH job fields: .999 differs onlyid/budget/hypothesis, .99999 alsodiscount.
Screen250k/500k/750k and stop if minority path remains absent across two later
checkpoints with no minority success. No blind extension beyond1M or restart.
Local artifacts/antmaze_normalized_discount_retention_1m contains collectors,
validated-launch-plan.json, actual-launch-verification.json, frozen-reward report.
Reporter is ready; control-only12rows passed before new evaluations existed.
Must collect and verify .999 shared raw prefix vs H as checkpoints arrive.

Motivation stored in artifacts/antmaze_normalized_discount_probe/rescore.py/results.json:
rescored74 historical successful left paths vs4 normalizedT3 successful right paths
under SAME normalized reward, identical full starts. Means gamma.9991447.70/1326.34;
.999991644.86/1643.23; undiscounted1647.0563both. Different policies/budgets,
selectedsuccesses only (original100episodes andfailures preserved). This is reward
functional analysis, NOT new rollouts, not trueQ accuracy or proof gamma repairsit.
Gamma1 is analytic only. Previous no-success gamma9999 corpus does not establish
this result; never conflate the corpora. Near-unit gamma may itself hurtlearning.
Teacher-floor escalation was investigated but NOT launched: loggedstd overwhelmingly
at uppercap .367879, not near lowerfloor. Sampled CSV windows are not all-training
statistics or matched-origin diagnostics; do not claim sigma-collapsecausality.

I COMPLETED all3 checkpoints: v1 geodesic same-full-origin supplement, original
G training8e9d7d3; evaluator2617549.100directepisodes each:500224 upper49/lower40
success,fail11;750080 upper4/lower95 success,fail1;1008384 upper0/lower100 success.
Mu-only atfinal lower98success,fail2. Exact start/checkpoint/rawreward/model-RNG
preservation checks passed. v1 primaryrandomfinal98%stillbothpaths, but same-state
FINAL COLLAPSED. Do not claim Gv1retains conditional route diversity. This differs
from historicalE denseT3, which DID retain bothsuccessroutesat2.75M/final3M with
independent evalRNG confirmation. Goal still notprovedv3/v4.
Local artifacts/antmaze_geodesic_v1_origin_supplement/report/results.json,
same_state_policy_retention.png and same_state_routes.png were visuallychecked.

G v1 COMPLETED1M, final checkpoint/source verified. Gv4 intentionally STOPPED
EARLY lastlogged626688 after550144direct lower40/40,successfullower26, no upper
success. G controller/wandb-sync stopped; exact group536201 verifiedempty;
Gv1 and CPUevaluators preserved. status/job/screen-stop/W&B sidecars preserved.
No Gv4 final1M or final100/fullstate claim. Local Greport supports stoppedearly.
W&B Gv1 1a6v784a; stoppedGv4 0hia2p4m. NeverresumeG/F.

H COMPLETE2/2, fullcheckpoint/replayproof and100finalrawverified:
source db4ca0a446f5fe8df1dda02e261fa9154c66d9c9,
antmaze-optiq-v3-startnorm-geodesic-250k-s0-20260925.
At258304total T1directright87/none13,success0;nativeR74/none26,successR1.
T3directright84/left16,successR4;nativeR92/L8,successR38. No leftsuccess.
Original438controlT1 directR100successR25. Normalization is not provedsolution.
Local Hreport regenerated, matchedpolicytrajectoryfigureQApassed; final_verified
includesoriginal-T1,normalized-T1,normalized-T3. Exact frozen reward validated.
L extends onlyT3candidate; H T1 is not re-run. Bothrewarddetails changed together:
fixed per-goalweights AND success-regionendpoint. Do not overclaimonefactorcausality.

K: antmaze-optiq-v4-geodesic-temperature-250k-s0-20260925,source4d272c5.
T3 completed258304total/7816updates;100directU38/D45/none17,success0;
T10 reached258304budget and final eval/save underway atlaststatus.
T10 final100/fullcheckpoint NOT yetverified atthishandoff. Collect/report before
completionclaim. Originalgeodesic, gamma.999,H+.7/500, onlyteacherT3/T10 differs
fromoldT1 control. T3 acquisitionmuchslower(finalgoaldistance~11m) thanT1control.
Local artifacts/antmaze_v4_geodesic_temperature_250k collectors/reportready.
Do not automaticallyextendK; successfulroutes notentries arepositivecriterion.

Original userentropy16grid A is already complete and reported; do notduplicate.
It is not the same condition as G/H/I/L. Retain finalW&B/raw caveats fromoldernotes.

Earlier notes below are historical, not current status.

---

# Current AntMaze goal handoff — 2026-09-25 07:21 KST

Newest HEAD2617549326b25f3dd00987e06d75de6bdb5e685b, clean and pushed GitHub/180.
199 SSH still times out; no unknown jobs restarted. No4090 AntMaze work.
Goal ACTIVE. This continuation is PROGRESS: two new reward-only screens launched,
real preflights/PIDs/source/replay checked; new same-state successful v1 evidence.

H: NEW v3 fixed-reference normalized geodesic250k T1/T3, source
 db4ca0a446f5fe8df1dda02e261fa9154c66d9c9,180 GPU0/3,root/supervisor
 antmaze-optiq-v3-startnorm-geodesic-250k-s0-20260925 (+wandb-sync).
Formula D(x)=min_i[max(d_geo(x,goal_i)-.5,0)*L/L_i], L_i=d_geo([0,0],goal_i)-.5,
L=min_iL_i; r=100*(Dcurrent-Dnext). No bonus/step cost/NovelD; gamma.999,H/d+.7/500.
Fixed per-goal weights AND success-region distance change together; not a claim
of equal discounted returns or full-body geometry. Physics/goals/terminal/reset
and seven algorithm files unchanged. Legacy reward values/specs unchanged.
v3 initial move[-.1,.1]: old -14.1421,new +12.9180; reverse direction+14.1421both.
16 meaningful tests passed; exact archived-control manifest differences verified.
Both actual8448/8batch4096 preflights/replay/checkpoints passed; initial actor/critic
hashes match original geodesic control. Actual GPU learners verified.
Lateststatus147456total each; current100k40episode results stillmostlyuncommitted,
zero goals; do not infer success. No automatic extension. Preserve G learners.
W&B T1 https://wandb.ai/OptiQ/antmaze/runs/xl27ae5r
T3 https://wandb.ai/OptiQ/antmaze/runs/81gulfd9
Local artifacts/antmaze_startnorm_geodesic_250k has collectors,actual-launch-verification,
validated-launch-plan, frozen-reward reporter and initial results. New reporter
uses exactdb4 reward for new profile, with untouched legacy control values.

I: NEW inference-only v1 geodesic SAME FULL ORIGIN supplement. Training source8e9d7d3,
report source2617549,root antmaze-v1-geodesic-origin-supplement-20260925 on180.
500224checkpoint (SHA ace4fb03dc498613ca89b34f58daec7f88ddf2ac0cec5c428c2ac9b6d665ed87)
100 direct: upper52/lower48entries, successfulupper49/lower40,11failures.
100 mu-only: upper52/lower48entries,successfulupper49/lower40,11failures.
Full-state/reward/rawhash/restoredparams/modeloptimizer/checkpoint checks passed.
Primary v1 random starts unchanged; same-state is explicit supplement only.
After this positive case,750080checkpoint100each CPU nice19 follow-up registered
under same committed evaluator, service rootname-step750k; still running.
500k service completed. No GPU use or training updates. No same-state retention
claim yet until latercheckpoint done. Local artifacts/antmaze_geodesic_v1_origin_supplement
contains collection script/results/report with scientific figure visually checked.

G: existing v4/v1 geodesic gamma9991M,source8e9d7d3 continues180GPU1/2.
v1 lateststatus1003520total near final;900096direct upper21/lower19 all40success.
500224randomprimarydirect upper19/lower16success/40;800000upper19/lower20success/40.
v4 lateststatus475136,failed to retain upper route:350208and400128directlower40,
zero goals;450048native lower38/none2,zero goals. Collect policy450/500kbefore
screen stop. Prefer wait untilv1finalcomplete then stop G service with onlyv4left;
do NOT kill shared controller processgroup whilev1live. Old F remainsstopped598016.
ReportG import fixed438 reward sourcebytes (no dependence on futureactiveworktree).

Earlier notes below are preserved history; newest section overrides statuses.

---

# Current AntMaze goal handoff — 2026-09-25 07:01 KST

IMPORTANT NEWEST: F is now STOPPED EARLY, not running or completed1M.
At450048 and500224 bothpolicy/native lost allleft entries/successes; at550144
directR39/none1,successfulR26;nativeR40,successfulR34. No recovery observed.
Active goal explicitly authorizes early screening. Stopped onlyF supervisor and
its wandb-sync; after supervisor stop, rechecked its orphan process group534370
and finished terminating that exact verified group. Final ps proves0live F
processes. Gcontroller job PIDs536266/536267 and learners537682/537411 preserved.
Last logged trainingstep598016; evidencecheckpoint500224 preserved andSHAverified.
There is NO completed1M/fullfinalreplay checkpoint or100-episode final evaluation.
Do not silently call this a complete run or automatically resume it.

F remote root now has screen-stop.json, preserved before-stop status/job/progress
under screen-stop-preserved-<timestamp>, status/jobstopped_early,originalmanifest
andfrozen source unchanged. W&B i97mi3u6 summary is annotated stopped_early=true,
evidence500224,lastlogged598016,planned_budget_completed=false. API state initially
still saidrunning due heartbeat lag; actualprocesses areauthoritativelyterminated.
wandb-screen-annotation.json records postlaunchmetadataonly,annotatedtrue.
Local Fcollect_status/results now collect these sidecars. Reporter explicitly
marks stoppedearly/planned1Mincomplete and displays latestmatchedmodecheckpoint.
LatestFraw550k wascollected/reward/state/hashvalidated. Noleftsuccess atany checkpoint.

Currently only G(v4/v1 geodesic1M) runs on180 GPU1/2.199 remainsunreachable;
unknown199jobs werenot touched. CurrentHEAD8e9d7d3 unchanged. GoalACTIVE.
Continuation progress includes two new verifiedG runs AND failedcandidateF
screening/termination, preservedartifacts and newevidence, not juststatuspolling.

Earlier06:56notesbelow superseded where theycallF live; the rest remainsvalid.

Latest continuation classification: PROGRESS (not an idle wait). Revalidated
live F at241k then397k using actual learner/GPU PIDs; collected new300k/400k
raw trajectories and exact prefix matches. Committed/shared and launched G's
two bounded candidate tests below, both actual real preflight gates passed.
Goal still ACTIVE: v3/v4 sustained multi-route goal success remains unproved.

Newest worktree/GitHub/180 canonical HEAD is now
8e9d7d3c2c79f797654ccfb21913d2c000b89f71, clean tracked tree, frozen source180.
Only registrar/protocol/AGENTS changed. Seven algorithm files remain byte-identical
to2564b59.199 SSH transport still times out; source sharing remains pending there.

G: NEW180-only root/supervisor
antmaze-optiq-geodesic-gamma999-retention-1m-s0-20260925 (+wandb-sync),source8e9d7d3.
Fresh v4/v1 geodesic gamma.999/T1/Hperdim+.7/interval500 seed0,1Mpostwarmup.
Compared actual archived438907f jobs: onlyid/steps/hypothesis differ. No algorithm,
reward implementation, learning setting, native physics or evaluation change.
1008384total/31256updates/63regupdates;50keval40each/final100;v1random,v4fixed.
Both real8448/8batch4096 preflights passed with reward/fullcheckpointreadback.
GPU1v4 learner537682, GPU2v1 learner537411;F remainsGPU0learner535043.
LatestG status v4total49152/1280updates andv1total98304/2816updates, no failures.
W&B v4 https://wandb.ai/OptiQ/antmaze/runs/0hia2p4m
W&B v1 https://wandb.ai/OptiQ/antmaze/runs/1a6v784a
artifacts/antmaze_geodesic_gamma999_retention_1m/{validated-launch-plan.json,
actual-launch-verification.json,collect_status.py,collect_results.py,report_results.py}.
Reporter checks raw rewardtelescope/fullstates/hash and finalproofs; short/long
sharedprefix assertions; never counts longerfreshsame-seed as an independentseed.
First collection had no evaluatedv4 yet; refresh rawcollection before reporting.
Only these2new jobs launched this continuation; v3geodesic collapsed and was not extended.
No199 job or cancelled campaign restarted. No4090 work.

F most recent verified trajectories400128total/40episodes:
directR37/L3,successfulR23;nativeR36/L4,successfulR21. At300kdirectL10/R29/none1,
successfulR11. Minority successfulroute remains0; T3 has not established retention.
Ten short/long sharedmode-checkpoint raw comparisons match exactly. F remains live.
LatestF trajectoryfigure was visually checked;400k data are not final1M.

Additional evidence: discount9999_rescore.json under horizon250k artifacts
re-scores the old250k first-action bank only. It contains NO successfulepisodes.
v3 left/right mean finite returns: gamma.99=546/551,.999=1097/1146,.9999=1226/1301;
v4gapalsoincreases. This differs from the earlier400k trajectory corpus, so
"raisinggamma always equalizes both routes" is unsupported. Do not launchgamma9999
merely extrapolating the earlier106→34gap. No newgamma setting was launched.
This is finite historical-return re-scoring, not retraining or trueQerror evidence.

The following earlier06:48handoff remains valid except for the newerHEAD/G and
Fprogress above. Previous sources/checkpoints remain frozen.

Goal remains ACTIVE. Real positive v1 same-state successful-route evidence is
now verified; the goal is not yet established for v3/v4. Do not repeat the older
"no success yet" notes below as current findings. This section supersedes those
historical updates, which are retained for provenance.

Current worktree tmp/reward-progress-worktree, branch direct-gmm-trg-antmaze,
HEAD 8eb830d553f93e1360bcee5f6c9649e6392e5d25. Tracked clean; GitHub and180
canonical/frozen source share HEAD.199 SSH is unavailable since about21:00UTC;
source-only sharing of438907f/5c3c966/ae55684/8eb830d to199 remains pending.
No unknown199 job was restarted or cancelled. No AntMaze work on4090.

A: requested positive DACER sweep16/16 FINISHED,500k postwarmup, H/d+.1/.5/.7/.9,
regulator interval500,seed0,T1,Euclidean progress100/B0/cost0/NovelDoff.
W&B final direct success%(100episodes each): v1[0,0,0,0],v2[100,100,100,100],
v3[1,48,78,19],v4[0,0,0,0]. Raw final100/checkpointproofs14/16 locally;
199 v3H.5 andv4H.7 finalraw are still missing, successrates use W&B summaries.
Do not relabel their500224step40episode data as final100. Report and links:
artifacts/antmaze_dacer_positive_500k/report/FINAL_WANDB_KO.md,
final_success_wandb.png. Plot visually verified. Existing collector/source preserved.

B:250k gamma/T screen180four completed;199four W&B records show crashed but
transport outage prevents process/failure-cause verification. Do not equate
heartbeat loss with confirmed physical learner error and do not resume blindly.
Completed180 v3gamma999_temp3 finaldirectL20/R80,successfulR33/100;
nativeL17/R79/none4,successfulR49. Both-entry candidate, not two successful routes.

C: v3gamma.999/T1 1M source d266263 COMPLETED and collected/verified.
1008384total/31256updates,checkpointSHA ea9d75b95f54829c252580e1cf7246c2f410de0d5821a1aa661c7bf28ae58c1c.
FinaldirectL99/none1,successfulL74/100; nativeL98/none2,successfulL71/100.
Right route lost around350-450k: gamma increase alone failed retention.
artifacts/antmaze_gamma999_retention_1m/report has current raw-validated curves
and final trajectories; plot QA passed. Short/long same-seed prefix matches exactly.

D: three reward-only geodesic250k runs source438907f ALL COMPLETED,180only.
root antmaze-optiq-geodesic-gamma999-250k-s0-20260925; same gamma.999,T1,H/d+.7,
interval500,only reward profile changed to existing progress100_geodesic_no_step_no_bonus.
v1finaldirectsuccessfulU17/D9/failed74,nativeU13/D6/failed81; native random starts.
v3finaldirectR100,successfulR25; nativeR90/none10,successfulR12; fixed originalstart.
v4finaldirectD73/U25/none2,success0; nativeD61/U39,successfulD1/U1; fixed originalstart.
These do not prove maintained successful multimodality. v1 randomstarts not same-state proof.
All3 real8448step/8updatepreflights and final258304step/7816update checkpointrewardproofs pass.
artifacts/antmaze_geodesic_gamma999_250k/report_results.py validates source,
geodesic reward telescope, fullstarts,rawhashes,finalproofs; reports preserve missing199 controls.
No D1M follow-up has been launched. Point-agent geodesic has no full-body inflation;
v3 nearest-goal geometry remains asymmetric.

E: historical official v1,T3,dense=-nearest-distance,DACER OFF,NovelD OFF,
source5baa5b3463416cdfdcad465e8f862f3729802568. Inference-only same complete origin
state (XY=0,pose/velocity identical),100episodes/mode, CPU-only,nice19, no training.
Supplementary to the preserved native random v1 primary evaluation. Verified:
checkpoint2M directU0/D0/failed100;2.5M U1/D1/failed98;2.75M U14/D32/failed54;
final3.008M U20/D71/failed9; same final independent evalRNG U26/D66/failed8.
Mu-only2.75M U24/D53/failed23; finalU22/D72/failed6; newevalRNG U16/D77/failed7.
Thus same-state two successful routes retained2.75M→final with repeated evaluation.
ONE training seed, not action-density multimodality or all-maze/all-state proof.
Original primary v1randomstartfinaldirect U51/D44success95/100 also preserved.
All checkpoints/parameters unchanged; rawSHA/reward/distance/initialstates verified.
Core actor/critic/TD/NLL/sampler6files identical; regulator delta diagnostics only and disabled here.
artifacts/antmaze_v1_T3_origin_supplement/report/{results.json,REPORT_KO.md,
late_same_state_routes.png,same_state_retention.png}; both scientific figures QA passed.
Inference sources5c3c966 andae55684. Original2.75M validation failed because old
float32 tolerance treated physicaldistance.50000356 as inside true.5 radius.
Only reporting validator corrected to actual float64physics distance, initialpointincluded;
repeatedpolicyarrays bit-for-bit unchanged. Original failedattempt/logpreserved.
radius-audit-correction.json records exact case. This was not a training/goal implementation bug.

F: CURRENT running single follow-up180, source8eb830d, root/supervisor
antmaze-optiq-v3-gamma999-T3-retention-1m-s0-20260925 (+wandb-sync).
Fresh v3seed0,gamma.999,T3,H/d+.7/interval500,Euclideanprogress100/B0/cost0,
1Mpostwarmup=1008384total/31256updates/63regupdates.50keval40each/100final.
Only budget/id differ from B'sgamma999_temp3; not a resume or independentseed.
Real8448/8 preflight/replay/checkpointreadback passed;7corehashes match2564b59.
Controller534370/job534435/mainlearner535043/GPU0 verified. Lateststatus176128total,
5248updates,11regupdates,no failure. W&B https://wandb.ai/OptiQ/antmaze/runs/i97mi3u6.
artifacts/antmaze_gamma999_T3_retention_1m/{collect_status.py,collect_results.py,
actual-launch-verification.json,report_results.py}. Collector180only. Report compares
completedC T1, BshortT3 andF, asserts exact same-seed sharedprefix, verifies finalproofs.
Latest report was50k; refresh collected raw results before reporting progress.
Inspect500k/750k/final1M successful routes, not entry counts. No additional job registered.

Runtime /tmp/optiq-antmaze-report-20260922/bin/python. Do not usePython3.14/NumPy1.26.
Use supervisor/GPU locks and /etc/vast-agents-guide.md; do not rerun passedpreflights.
No algorithm, optimizer, actor, critic, NLL, TD, sampler or nativephysics changes.
All earlier cancelled queues remain cancelled. No new automations. No subagents.

---

# Historical handoff notes (superseded by current section above)

Goal remains ACTIVE: without changing algorithm structure, demonstrate sustained
multiple valid goal-reaching routes via reward/hyperparameter changes. Mere
corridor entry, mixed seeds/checkpoints, or extra behavior-noise rollouts are
not completion. v1 evaluation remains native random starts; v2-v4 original
fixed full state. A candidate needs later-checkpoint retention and additional
independent rollouts/seeds before a broad claim. No success has yet been proved.

Previous goal turn: PROGRESS. Committed and launched16 positive-entropy jobs.
This continuation: PROGRESS. Collected/verified raw trajectories, found renewed
route loss, quantified discount timing preference using saved trajectories,
implemented and registered the next bounded eight-job hypothesis screen.

Active source worktree: /Users/yunheechan/Documents/ChatGPT/OptiQ/tmp/reward-progress-worktree
branch direct-gmm-trg-antmaze, HEAD eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45.
Both canonical servers and GitHub share this commit. Tracked tree clean.

Only servers vast-heechan-180 and vast-heechan-199; read each
/etc/vast-agents-guide.md. Never use vast1/4090 for AntMaze.

1. Current16-job target grid: /home/heechan/optiq-experiments/
antmaze-optiq-dacer-hpos-i500-500k-s0-20260925 on both hosts.
Frozen2564b59faa0d319eece496b93eff0f19359efc37; H/d+.1,+.5,+.7,+.9
xv1-v4,seed0,500k postwarmup (508416total); T1,gamma.99,DACER interval500.
Reward100*Euclidean delta-distance,B0,step cost0,NovelD OFF; all other latest
OptiQ settings retained.40eval each100k,100final. Independent2secondbackfill.
Collector artifacts/antmaze_dacer_positive_500k/collect_status.py; read-only
raw archive collector collect_results.py; figures/report report_results.py.
Use /tmp/optiq-antmaze-report-20260922/bin/python. Source is frozen, preserve it.
At400k v3 H+.7: left39/right0/none1,successfulleft20/40. v4 H+.9: upper35/lower5,
success0. Historical v3 control400k route MC gap106(gamma.99) versus34(.999)
when re-scoring the SAME paths; this is not a trained counterfactual result.

2. New8-job follow-up: /home/heechan/optiq-experiments/
antmaze-optiq-horizon-temperature-250k-s0-20260925 on both hosts.
Frozen eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45.
v3/v4 each gamma999,T3,both; v1 gamma999 and gamma999+T3. H/d+.7,interval500,
other settings same.250k postwarmup=258304total,7816updates;40eval each50k,
100final. Algorithm core7files byte-identical to2564b59, verified at registration.
Both supervisor controllers registered and confirmed live. They wait only
until predecessor PENDING jobs are assigned, then take freeGPU slots; no
all-completed barrier. Do not launch duplicates or assume queued preflights
already passed. Unit22tests passed locally and on eachserver. Actual per-job
GPU preflight will gate main training and verify live gamma/config/reward/checkpoint.
Local artifacts/antmaze_horizon_temperature_250k/{collect_status.py,collect_results.py,
registration-verification.json,algorithm-invariance.json}. A condition-aware
report for the NEW study still needs creation: do not reuse old report's
(task,entropy_target) dictionary because all newtargets=.7; key by condition.

Do not resume any previously cancelled campaigns. Inspect live controller AND
learner/GPU PIDs plus status/failure/config before acting. Pending jobs hold
on failure; preserve live jobs and audit causes. No algorithm/TD/loss/latent
architecture changes. Hyperparameter hypotheses are authorized by the active
goal; commit/push/share before new or changed training. Future conclusions
must distinguish entry diversity, successful paths, mu-only/native versus
conditional-sigma direct policy, and single-seed exploratory results.

Latest collected final500k: v1 H+.1,100episodes,upper12/lower1/none87,success0; v3 H+.7,left100/100,success78/100; v4 H+.9,upper97/none3,success0. These conditions fail sustained multi-route goal-reaching. Remaining grid jobs continue; new250k study is queued.

Controller scheduling correction: HEAD/canonical source is now7278a0ed0f37e536ae81663fbd1e08a49ae561b8. Follow-up TRAINING source remains eee04de. The follow-up queue had zero started jobs when controllers were stopped manually, then their service directory was switched to7278. This fixes priority dispatch/OS-lock gap by respecting predecessor running GPU assignments. Original16 entropy learners were never stopped. Audit sidecars controller-provenance.json and priority-dispatch-audit.json plus preserved old status/config are on each new-campaign root. Manifest/frozen training sources unchanged.23tests passed local+bothservers. Collect controller sidecars and confirm fresh PID/status; do not duplicate the eight jobs.


Continuation update 2026-09-24T20:53:42.251910+00:00 — PROGRESS, goal still ACTIVE.
Original entropy sweep latest verified status:13/16 completed,3 running/finalizing; no failures. Follow-up8:2 completed,4 running,2 pending; no failures. Controllers and job launcher PIDs verified against actual ps; GPU compute PIDs saved in artifacts/antmaze_horizon_temperature_250k/goal-watch-verification.json. Sources and code unchanged during this continuation.
Built and visually checked condition-aware report_results.py in horizon artifacts. It keys task+condition+evaluation mode, validates raw starts/goals/reward telescope/SHA256, separates direct policy from random-z mu-only, and adds exact-step-matched comparison figures. It never replaces frozen training or raw evaluations. Output report/results.json, REPORT_KO.md, latest_trajectories_{policy,native}.png, matched_trajectories_v3/v4.png. Current final250k results: v3 gamma.999/T1 direct n100 left54/right33/none13, successes0, closest-goal mean4.716m; mu-only left54/right20/none26,success0. v4 gamma.99/T3 direct lower34/upper27/none39,success0, closest-goal11.18m. These are entry-diversity candidates, NOT multimodal goal success. Inspect other6 follow-ups before selecting longer confirmation.
Entropy regulator actually changes extra behavior std: v3 H+.1 final.0104 versus H+.7 .0660 and H+.9 .0689, but +.7/+ .9 choose left100/100 at final (success78/19 respectively). No sustained successful two-route policy yet. Full target grid still needs final collection/report.
Historical v1 dense=-distance,T3 source5baa5b3 final3.008M audited from raw archives: natural direct routesU54/D46, successfulU51/D44,95%; fixed saved start (1.987,-1.452) direct lower100,successful94. Natural action route agrees with initial-y half96%; mu-only92%. This specific fixed-start result does not rule out other states, but random-start route counts alone are NOT same-state multimodality. See historical_v1_dense_T3_audit.json. No old experiments resumed.
Operational: full-host199 download can time out; per-job chunked collector succeeded with SHA/CRC validation. Use entropy artifacts/collect_chunked.py --host vast-heechan-199 (optional --job repeated). Collector temporary files now unique, supports decimal/underscore job IDs; reporting-only changes. Preserve delta collection metadata.
Important before any new seed confirmation: current native run.py/learners.py hardcode seed0, controller does not forward seed. All existing requested jobs are correctly seed0. Merely naming a new manifest seed1 would NOT create an independent seed. Implement/validate a real seed override in a separately committed runtime profile before any seed1 launch; do not alter current frozen sources. No new confirmation jobs have been registered this turn.


LATEST 2026-09-24T21:05:10.579818+00:00 — PROGRESS; goal remains ACTIVE.
Current branch and both canonical servers/GitHub are d2662636c4b8253efed8d9d0fc2c1af6a5b1da15; immutable worktrees exist onboth. Tracked worktree clean. Added only register_gamma_retention.py, GAMMA_RETENTION_PROTOCOL.md and AGENTS entry; all7core files still byte-identical to2564b59. Launch-plan diff from completed v3 gamma999 short job changes only id/steps/hypothesis.
NEW campaign C: server180 ONLY /home/heechan/optiq-experiments/antmaze-optiq-v3-gamma999-retention-1m-s0-20260925, supervisor same name + -wandb-sync. One fresh seed0 v3 gamma.999/T1/Hperdim+.7/interval500, same progress100 Euclidean B0/stepcost0/NovelDoff. Budget1008384 total=1000192 postwarmup,31256updates/63regulatorupdates,50k eval40each/100final. Priority predecessor is existing horizon screen local pending/reservations; uses freeGPU1, does not restart or displace other jobs. This is fresh same-seed longer-budget confirmation, NOT resumed state or independent seed. Source d266263, original sources preserved.
C actual preflight passed8448steps/8batch4096updates, reward/fullcheckpoint verified. Main live at81920total/2304updates in most recent read, controllerPID526662/jobPID526727/GPU1. W&B https://wandb.ai/OptiQ/antmaze/runs/hya3yj35. Actual source/gamma/config/process verified. Local artifacts/antmaze_gamma999_retention_1m/{collect_status.py,collect_results.py,validated-launch-plan.json,launch-verification.json}. These collectors query180only. Still need a C-specific long-horizon report (do not cap at short report258304 or call it a new seed).
A entropy16: BOTH controllers last successfully verified completed8/8, pending0/running0/failure0. Training of all16 is completed. Local final raw100episode data+checkpoint proofs exist14/16; only199 v3H.5 andv4H.7 final rawremain, their500224/40episode data are present. Other final500k conclusions unchanged: v1none succeeds, v2allright100success100, v3H.1success1right,H.7success78left,H.9success19left; v4H.1lower100success0,H.5upper99lower1success0,H.9upper97success0. Do NOT claim full-grid final raw verification until missing2collected. Reporter now validates per-run checkpoint/result proofs and flags final_grid_complete only16/16; currently14proofs/14finalrows.
NETWORK: around21:00UTC199 SSH became unreachable (TCP timeout direct; banner timeout via180ProxyJump). Earlier parentAallcompleted observation is valid; subsequent199 horizonB progress cannot be asserted. No restart/cancellation performed. Read-only network-observation.json stores scope. Do not infer training failure solely from SSH timeout. Retry with reasonable backoff next goal turn;180remains reachable and C is making real progress. This is NOT a goal blocker since180work continues.
B screen180final: v3 gamma999 policyL54/R33/none13 success0, nativeL54/R20/none26 success0; v4temp3lower34/upper27/none39success0; v1gamma999uncommitted100success0. v3gamma999_temp3 latest250112/40: right24,left13,none3, successfulright4. Its final100 should be collected from180; other4Bjobs are on199 and latest status needs network recovery. Consider longer combined-condition confirmation only after checking complete saved data; no such extra job is yet registered.
Goal still unachieved: no sustained both-route successful policy. C was selected because both entries persist even without conditional sigma and nearest-goal distance improved, not because it already succeeds. Assess500k/750k/final1M retention and goals, then actual independent seed/repeated-rollout validation if promising. Current runnerhardcodesseed0; namingseed1isnotvalid. No new seeds/hyperparameters/algorithm changes made.
