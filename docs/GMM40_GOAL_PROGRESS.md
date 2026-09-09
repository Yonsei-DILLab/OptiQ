# GMM40 goal research log (updated 2026-09-09)

## Latest user revision — all-metric wins are no longer required

At2026-09-09 the user asked about sufficiently increasing sample counts atT1 and explicitly said “모든 지표를 이길 필요는 없어”. This supersedes the original all-paper-metrics completion condition everywhere in historical notes and the older goal-registry wording. Preserve the same algorithm/native coordinates and prioritize original-GMM fidelity, sample-count adequacy and compute cost. Do not keep escalating tuning solely to cross every paper threshold.

Supporting actual evidence atT1,depth5: K2048/75k KDE h.35=6.985195,W2 4.769624; K4096/80k KDE6.951031,W2 4.492400; K4096/87500 KDE6.941509,W2 4.422149. All40modes. KDEis diagnostic, not paperCFMNLL; extra training is a confound. The currentK4096 T1 checkpoints have the best completed KDEdiagnostics; the newly evaluated85k control has h.35=6.936762. A matched-parent N8192 T1 trial is now running (1od7e84v). Independent seed0/1/2 common75k preparation and auxiliary CFM evaluations are active.


The original goal was to beat BOTH DiKL and iDEM paper GMM40 metrics with the existing OptiQ algorithm; the latest user revision above removes the all-metric requirement. Only hyperparameters and minor sampling changes are authorized; no SMC, gradient-objective substitutions, target-sample training or hidden output postprocessing. No success claim is currently justified.

## Latest progress: T1 sample-count comparison and three-seed prefixes

2026-09-09T01:15:18.876577+00:00. All three independent seeds0/1/2 completed the common75k prefix. Their continuation URLs are txznuwp8,wm31yxcb,9rf5hlty; no cross-seed state or seed selection. Final sample-count continuation is pending the single seed42 N8192 comparison, which is still live and approaching80000. No new sweep is planned merely to beat every historical threshold.

**Same20.48M additional queries:** K4096/80000 versusK8192/77500. Both40modes. W2 **4.492400→4.413451**, TV **.8250→.8214**, rawmeanlogp **-6.890193→-6.915973** (reference-6.873834). KDEh.2/.35/.5 **7.166435/6.951031/6.906294→7.107610/6.934381/6.899873**. Measured additionaltraining **558.239s→1198.637s**,2.147× at equalqueries. W2 difference is smaller than its resampling variation; KDE is not paperCFMNLL. N8192 intermediate export0xesphyc, KDEc0wvdx54,timingyzit5ifz. Read the actual paper summary for its runURL. Final80000 comparison remains pending.

[Detailed Korean comparison](GMM40_T1_SAMPLE_COUNT.md) and mode figure **ad5zpy6v** were created; the figure was visually inspected. The same prior-selectedcomponent13 is shown, with all40mode statistics retained. Mean normalized centererror .0780→.0587 andtailfraction.0432→.0463 (GT.0501) improve; variance ratios .9605/1.0945→.9892/1.1053 do not improve uniformly. Internal holes remain. No training gradient/precision or objective change.

HP19depth5final50k CFM **z7jwy3ni** completed100k: finaltestNLL **7.025284**,ESS1000 **.702401**,logZLB **-.174412**,bestvalidation7.005178 at70k,elapsed5356.767s. Its primary1e-3post ESS16 is **.814764±.053619** over8 independent small batches; strictpost is still running at the latest poll. Do not attach these metrics to the differentHP24/25 samplers. Other HP23/24 fullCFM fits remain active.

Added only `scripts/gmm40_checkpoint_evaluation_worker.py` and itsmanifest/tests: serialized checkpoint timing and exact export/sample/step hashes are checked; incomplete output directories are retained. Four tests and a real isolated200-step export/paper/KDE/timing smoke passed. Smoke samples are outside candidate exports and are not quality evidence. Supervisor `optiq-gmm40-checkpoint-eval-hp25-3` completed the real77500 pipeline and now waits for the actual final80000 checkpoint. It will execute100k rawg export, unchanged sample/KDE/timing metrics, full100kCFM and primary/strict ESS post. Do not duplicate these jobs. The previous OptiQ core hashes are unchanged.

Latest joined audit 61uk4sbj,`audit_20260909T0113`, covers65 exact candidates and explicitly respects the user revision. Every short exec handle collected; long work supervised. The goal remains active, with remaining work listed in`goal_progress.json`.

## Authoritative state

- Repo: `/workspace/OptiQ-heechan-no-anchor`, branch `heechan-no-anchor`.
- `benchmarks/gmm40/goal_protocol.json`: frozen core source hashes, paper metrics/thresholds and protocol caveats.
- `outputs/gmm40_tuning/goal_progress.json`: last snapshot; ALWAYS poll live supervisor/processes and read fresh run files.
- Original core hashes remain recorded in the protocol. As of HP8, `algorithm.py` adds only an optional `policy_latents` argument/input branch for a user-authorized minor sampling trial. The original IID bounded and unbounded updates are bitwise equal to the frozen goal source (two explicit regression tests). `policy.py` and `transport.py` still match goal-start hashes. No OT, argmax, MSE, critic/oracle, or model component changes.
- Training: `/workspace/.venv-optiq-no-anchor/bin/python` (JAX GPU, torch CPU).
- Auxiliary evaluation: `/venv/main/bin/python` (torch 2.11.0+cu128, torchdiffeq 0.2.5 installed). It does NOT have JAX. Keep auxiliary modules independent of `optiq_dime.runtime`/GMM sampler imports.
- W&B entity/project: `sae_project/optiq_dime_no_anchor`. Load repo/workspace `.env`; never print credentials.
- Everything is in the working tree; no new commit/push in this goal.

## Earlier snapshot after the user relaxed the all-metric requirement

Snapshot updated at 2026-09-09T00:54:18.225638+00:00. The current scope is original-GMM quality at T1, adequate sample count, compute cost and independent-seed confirmation. Missing a historical paper threshold is no longer a reason to continue tuning. The previous detailed00:30 snapshot is archived at `outputs/gmm40_tuning/goal_doc_snapshot_before_T1_N8192_20260909.md`; all source evidence remains intact.

### Completed HP24 and the T1 sample-count comparison

All four HP24 tuning runs finished. All40modes are covered. Same75k HP23depth5T1 actor/Adam/RNG parent, sigma1,beta1,lr1e-4,R1,epsilon1e-4/300,grid,native/unbounded; no OptiQ objective or model component changes. Every export is an independent100k IID rawg draw, training seed42. KDE values below are diagnostics, NOT CFM paperNLL. Full sample/query/timing records are preserved.

| Candidate | Update | Direct mean logp | W2 | TV | KDE h.2 / .35 / .5 |
|---|---:|---:|---:|---:|---|
| Parent K2048 T1 |75000|-6.876887|4.769624|.8274|7.286416 /6.985195 /6.917762|
| K2048 T.95 |100000|-6.833814|4.457590|.8212|7.361625 /7.006130 /6.922963|
| K4096 T.9 |87500|-6.808594|4.416435|.8125|7.281958 /6.980497 /6.913098|
| K4096 T.95 |87500|-6.839434|4.613220|.8237|7.186830 /6.954050 /6.903277|
| K4096 T1 |80000|-6.890193|4.492400|.8250|7.166435 /6.951031 /6.906294|
| K4096 T1 |85000|-6.890380|4.257551|.8280|7.123631 /6.936762 /6.899249|
| K4096 T1 |87500|-6.889038|4.422149|.8246|7.142812 /6.941509 /6.899921|

The85k checkpoint was exported specifically as the equal-query control for the new8192 trial, not as an independently replicated best model. W2/TV variability describes resampling, not training seeds. Direct GT reference meanlogp is-6.873834; GT-vs-GT empiricalW2=4.7402 andTV=.8223 illustrate finite-sample floors. Do not interpret beating a GT empirical point estimate as a better distribution than GT.

`benchmarks/gmm40/tuning_hp25_T1_N8192.json` starts one bounded extra trial onGPU3, **1od7e84v**, same75k parent and parameters with N8192. Final80000 adds5000updates and40.96M queries; checkpoint77500 adds20.48M. Compare final80000 against K4096/80000 for equal5000updates and K4096/85000 for equal40.96Mqueries. Compare intermediate77500 against K4096/80000 for equal20.48Mqueries. Do not compare different query budgets without labeling them. N8192 initial speed is about.47s/update, versusK4096 .11s/update; approximately40min for the new5000-update stage. Actual finish/quality remains pending; use live logs.

Read-only fixed-parent OT checks completed on bothN, three diagnostic RNG repeats: N4096 **777plmiq**, N8192 **li48fiou**. Source ESS/N is about.873 versus.876, so severe importance-weight collapse is not present in this snapshot. Selected unique fractions are about.698 versus.655: absolute distinct selected candidates still rise from~2860 to~5369. Mean selected movement decreases~.428→.342. Row-marginal TV~.0237→.0208; column errors~3e-7. These are instantaneous diagnostics, not learned quality gains, and no model was updated.

### Auxiliary CFM and independent training seeds

Current full100k auxiliary fits: HP19depth5/50k **z7jwy3ni** GPU3; HP23depth5T.9/75k **1ben65sb** GPU1; HP23depth5T1/75k **m9cy12rp** GPU0; HP24K4096T.9/87500 **l24lefhj** GPU2; HP24K4096T1/87500 **37pkugzq** GPU0. Each consumes its own fixed100k rawg export. Existing primary1e-3/strict1e-4 ESS post workers automatically wait for the full fit to finish. The newer T1 fit and post manifest are `cfm_post_queue_hp24_T1.json`. No fit exists for the85k control or for N8192 yet.

Best completed primary finaltestNLL remains HP14K8192/205k **7.000954** (**8nq9mblk**), a DIFFERENT model/long ancestry. Its completedpost **g5rjgpen** has primaryESS16 **.801855**, rawESS1000 **.675550**, logZLB **-.198524**. Strict1e-4 givesNLL7.005172,ESS1000.806220,ESS16.818546,logZ-.249931. Do not combine these CFM/ESS metrics with HP24 sample metrics. HP14K4096/210k finalNLL7.028359, primaryESS16.729335; HP13width512/75k finalNLL7.029454,primaryESS16.712770. Prior audits preserve all other candidates and tolerances.

Fresh independent bases: seed0 **v9vxwat8** completed50k; seed1 **s4w9g70p**, seed2 **seyivzv2** are training50k. All use the same HP19 depth5width512 first50k recipe, independent actor/Adam/RNG, measured timing,102.4Mqueries. `replication_polish75k.json` queues each seed's own50k→75k commonT1 continuation (K2048,lr1e-4,sigma1,epsilon1e-4/300). Seed0 continuation **txznuwp8** is running; others wait for their own base completion. No cross-seed state, no seed selection. This prepares the same75k prefix as the tuning comparison; final N continuation is not chosen yet. No completed full three-seed confirmation is claimed.

### Timing, source fidelity and reporting

Timing audit **vm4zu3nv** includes all HP24K4096 checkpoints. For T1/80k,85k,87500: cumulativequeries174.08M/194.56M/204.8M; estimatedtraininghours.839906/.991429/1.067206. Full measured time is absent because older parent stages predate instrumentation. Final stages have actual evaluation-excluded checkpoint timing. These are shared3090 data, not a matchedA100 speed comparison.

The previous reporting-only training-clock change passed26 tests including original bounded/unbounded update parity, and a real200-update GPU smoke **gvoh13xv**. No timing or algorithm code changed during this turn. Only manifests, supervised launch configs and audit wording changed. Audit now explicitly reads the latest user revision and says all-paper-metrics wins are not required; its legacy overall_goal_achieved=false is not a completion decision for the revised scope. Metric definitions are unchanged.

Latest joined audit **wn5bgmpl**, `audit_20260909T0053`, verifies64 exact checkpoint/sample identities. New85k export **7770oie0**, sample evaluation **m2yhrwv9**, KDE **mo996pzp** are complete. All short tool handles collected; main work is supervised. No commits/pushes during this goal. Prospective fresh finalreference seeds0/20261101/20261102/20261103 remain fixed and have not been evaluated; preserve all references if used rather than choosing one.

## What changed during tuning

The implicit actor class, Sinkhorn implementation, weights, row argmax and MSE remain unchanged. The common actor update now optionally accepts precomputed Gaussian latent draws; its default IID path is unchanged. The GMM adapter/runner exposes widths, exponential T/sigma/epsilon schedules, checkpoint continuation, target-free output initialization, and the optional randomized Gaussian latent grid. `coordinate_scale=1` throughout new tuning; no coordinate multiplier or clipping.

`initial_output_std=25` is a weight-initialization hyperparameter only: evaluate the random actor on 4096 independent Gaussian latents, then adjust only the final affine layer to zero mean and covariance 25^2 I. It uses no target samples/locations/logp, does not consume the training RNG, and does not add a runtime transform. All layers remain trainable. This was motivated by within-mode support gaps despite good global metrics.

Checkpoint continuation uses serialized actor+Adam state; recover the training key by applying `split(key,4)[0]` once per completed update, in a JAX loop. A test verifies resumed output/state/metrics bitwise against uninterrupted training. `--resume-checkpoint` requires matching seed, widths, native/bounded settings and initialization setting. The final `--updates` is the cumulative count.

`gmm40_tuning_worker.py` handles INT/TERM and forwards INT only to the training child, then waits. New supervisor configs use `stopasgroup=false`, `killasgroup=true`. For old workers launched before this fix, prefer signalling the exact training child and waiting for checkpoint/W&B completion if another stop is needed. The stopped HP2 N1024-sharp job retained its 10k checkpoint and 12,866-update evaluation; do not delete it.

## Findings

HP1: 16 configs, seed 42, 5k updates. T in [.5,1,2,4], sigma [4,8,16,32], beta=1, no anchor, N512/R4, epsilon=.01, 100 Sinkhorn iterations. T=1/sigma=8 reached 40 modes, but mean logp was -13.12. Native old T=.25/beta=.1/anchor runs covered only 6-8 modes.

HP2: 50k updates with longer training, width and bandwidth/epsilon schedules. Best practical route: N512/R4 (512x2048 OT), B1, width512x3 GELU, beta1, no anchor, T1, sigma8->2 and epsilon .01->.0001 exponentially over first15k, 300 Sinkhorn iterations, Adam .0003. Parent run is `outputs/gmm40_tuning/hp2/003-width512-eps0001/...` (seed42); exact path is in HP3 manifest.

HP3: continued that 50k parent to150k. Base T1/sigma2/lr.0003 ended at mean logp -6.9713 (GT -6.8679; abs error .1033), 40 modes. Alternatives use sigma1/lr.0001 or T.8/lr.0001. T.8 improves mean logp but did not solve the support problem below. No final held-out training seeds have been run yet.

## Independent sample evidence and remaining problem

100k independent DIRECT generator samples from the base 100k checkpoint: `outputs/gmm40_tuning/exports/hp3-base-100k/samples.npy` (W&B `nzjb8a80`). T.8 counterpart: `outputs/gmm40_tuning/exports/hp3-T08-100k/samples.npy` (W&B `pp62ktdg`). These contain no KDE noise, correction, resampling or target data.

Common paper-sample report: `outputs/gmm40_tuning/paper_eval_hp3_100k_v2/analysis.md`, W&B `95u7jnhb`.

- OptiQ mean logp -7.0518 vs GT -6.8738, abs error .1779, all40 modes, occupancy TVD .0371.
- 1000-point exact W2, 10 repeats: OptiQ4.9523, releasedDiKL6.7078, releasediDEM12.0069, independentGT4.7402.
- 2D200x200 TV, reference-derived bounds: OptiQ .8186, DiKL .8452, iDEM .8658, GT .8223. Fixed[-56,56]^2: .7299/.7615/.7816/.7210. Do not overclaim significance near the finite-sample TV floor; published histogram bounds are underspecified.
- First evaluator attempt (`paper_eval_hp3_100k`, W&B `9juq3gfe`) failed on a requires_grad tensor in released data. Fixed with detach/no_grad; v2 is complete. Never cite the failed report as the completed comparison.

**Important diagnostic**: fitting a narrow Gaussian KDE to100k g outputs and evaluating fresh target points gave NLL8.862/7.588/7.242 at bandwidth .2/.35/.5, vs true targetNLL6.867. T.8 was similarly poor. These are explicitly DIAGNOSTICS, NOT paper NLL. Figure `outputs/gmm40_tuning/density_visual_100k/density.png` (W&B `kil2quz4`) shows irregular within-mode support boundaries and bridges. Good mean logp/coverage/W2 alone do not prove density quality. The current HP5 initial-output-spread experiments address this using a hyperparameter only.

## Evaluation implementation

- `paper_sample_evaluate.py`: direct samples + released DiKL/iDEM; meanlogp/GT/error, coverage,1000pointW2 and200x200spatialTV. Reports both reference-derived and fixed histogram bounds, preserves overflow. Fresh final references/seeds are still needed after freezing final configuration.
- `idem_evaluate.py`: auxiliary OT-CFM fit on EXACTLY100k generated samples; unmodified public DEM MyMLP and CNF sources under `idem_reference/`, MIT license/provenance. Batch512, Adam .0005, gradclip.5, exact minibatch OT, sigma0, Gaussian prior std50. Exact-divergence dopri5 atol=rtol1e-3. Best validation checkpoint then held-out NLL; generated CFM samples only for the paper's normalized importanceESS/logZ. These auxiliary outputs NEVER replace direct OptiQ samples in sample metrics.
- `export_samples.py`: restore any fixed checkpoint and export independent direct samples. Run CPU JAX while GPUs train. Example use a new output dir each time; existing dirs deliberately cannot be overwritten.
- `kde_validation.py`: only a diagnostic for support holes, NOT an acceptable substitute for paper NLL. 5000 held-out target points x100k KDE centers at3 bandwidths takes several minutes on CPU.
- Latest tests:14 `tests/test_gmm40.py` tests passed, including source parity, unchanged bounded path, resume fidelity,2D TV vs marginals/overflow, initial covariance/RNG isolation. The auxiliary CNF analytic identity/linear-flow Gaussian check max error4.77e-7 (W&B `4vtlxr6b`). CPU CFM smoke completed (W&B `gjhj2ne3`).

## Paper requirements / unfinished work

DiKL Table1: target meanlogp -6.85; DiKL -7.21, iDEM -8.33, all modes. Report reference error so mode collapse cannot fake a win.

iDEM Table2 GMM: NLL6.96, normalizedESS.734, W2 7.42; Table5 TV.82, logZlowerbound-.340; training time .87h. NLL is held-out target likelihood under the auxiliary OT-CFM, NOT -meanlogp of g outputs. ESS is importanceESS under that model, NOT OptiQ candidateESS.

Primary refs: https://arxiv.org/html/2410.12456v2 ; https://arxiv.org/html/2402.06121v2 . Public code cloned read-only at `/workspace/DEM-reference` (8e9531987b0bfe4e770d5009c10a56b8434c0f0a) and `/workspace/fab-torch-reference`. The paper's covariance text conflicts with released code; released code uses softplus(1) sigma and matches our canonical target. Current public DEM TV code does not reproduce the paper's stated GMM2D histogram verbatim; both explicit range variants are therefore reported.

Need to finish tuning actual within-mode density; freeze a configuration; run independent training seeds0/1/2; obtain raw100k samples per seed; run the paper sample and auxiliary likelihood metrics; inspect every gate and source-fidelity condition. No goal completion claim until all are proved. Do not replace this objective with only logp/coverage/sample metrics.

## Last live poll before this goal-turn update

All four programs above were RUNNING. HP4 width1024 had reached60k,40modes,meanlogp-7.1892. Broad-init HP5 width512 at20k had32modes,meanlogp-11.4158; width1024 at10k had31modes,meanlogp-21.3637. Broad initialization has NOT shown a win yet. It may need a smaller actor learning rate and slower bandwidth/epsilon cooling because final-layer weight calibration amplifies sensitivity; this is a hypothesis, not a demonstrated cause. Do not conclude it works from its initial covariance test.

The GT-only CFM control reached40k with validationNLL6.970206 (still not an OptiQ result). Core hashes remained unchanged; all14GMM tests passed. Final confirmation seeds and actual OptiQ CFM metrics are still outstanding.


## Continuation update: 2026-09-08 19:48 UTC

The previous short turn was a verified wait: live supervisor processes and their native coordinate arguments were checked; nothing was restarted solely for the user's coordinate question. This continuation made progress (new completed evaluations/diagnostics, stopped failed hyperparameters with saved checkpoints, and launched independent likelihood evaluations plus hyperparameter trials).

HP5 broad-init lr.0003 lost modes: width512 stopped at65,861 updates (22modes), width1024 stopped at65,525. Both have `stopped.json`, final actor/Adam checkpoints and W&B results. HP6 now tests width512 with initial std25 vs10, lr.00003, slower50k sigma/epsilon annealing,150k total. At30k these retained39/40modes but still had poor meanlogp -17.42/-16.30; no performance win yet. W&B r4c7mwct/s6s4t2rl, GPUs1/3.

HP4 width1024 original initialization completed150k:40modes,meanlogp-6.9961,energyW1.1949. Its100k independent-export KDE NLL diagnostic was8.7739/7.5295/7.1957 for h.2/.35/.5 (GT6.8672). Slightly better than width512100k, still support mismatch. W&B egut2x51. This remains NOT paper NLL.

New direct100k exports: `exports/hp3-base-150k` (yxjuavhy), `exports/hp4-width1024-100k` (j2u1zre9). New same-protocol sample reports:
- `paper_eval_hp3_150k` (ge2fytor):meanlogp-6.95609,GTerror.08226,40modes,W2 mean5.00078,2DTV(reference range).8205,fixed-rangeTV.7260.
- `paper_eval_hp4_100k` (i472z5c4):meanlogp-7.01155,error.13772,40modes,W2 mean5.16111,2DTV.8267,fixedTV.7263.
These are tuning seed42 only; neither establishes all paper metrics or TV significance.

Actual OptiQ auxiliary CFM evaluations are now RUNNING independently from random initialization on GPU0 alongside the GT control:
- `optiq-gmm40-cfm-hp3-150k`, output `cfm_hp3_base_150k`, W&B m4t0e9uw.
- `optiq-gmm40-cfm-hp4-100k`, output `cfm_hp4_width1024_100k`, W&B hzdhobqk.
Each fits100k directg samples for100k CFMupdates. GTcontrol at70k valNLL6.98175 (best6.97021 at40k). No actual OptiQ NLL/ESS/logZ final yet. CPU Hungarian dominates CFM time, so the three CFM jobs shareGPU0 while main OptiQ training also uses that GPU; all fit comfortably in memory.

New read-only `benchmarks/gmm40/ot_diagnostic.py` calls the SAME GaussianKDE and Sinkhorn on a fixed checkpoint, without any actor updates. `ot_diagnostic_hp3_150k` W&B wchny3xg:atN512,R4,epsilon1e-4,300iterations, row marginalTV .0696-.1027 across3fresh clouds. Columns match near1e-6. Increasing iterations to3000 only reduces rowTV to.0446-.0652; epsilon1e-3/3000 reduces it below.008 but selects much more central rowargmax points and causes largerdisplacements. At defaultT1, weightedcandidate meanlogp~-6.85 while rowargmax targets~-6.61; R16 did NOT remove this concentration. These are diagnostic observations, not proof of a sole performance cause.

A temperature-only diagnostic atT1.5 (d9xnk4am) withthe same clouds changes selectedmeanlogp to~-6.98, though row marginalerrorremains. Motivated by this, HP7 continues thefixedHP3base150k to250k with lr.0001 and T1.25 or1.5, leavingN512/R4/sigma2/epsilon1e-4/300iterations unchanged. No core modification. GPU0/T1.5 is W&B lkb10hf7; GPU2/T1.25 is started only after HP4terminal status. Manifest`tuning_hp7.json`.

Paper timing clarification from AppendixF.3: published traininghours exclude evaluation, using periterationtrainingtime timesconvergenceiterations. Record both samplertraining and auxiliaryevaluation/search separately; hardwaredifferences stillprevent a strict matched-hardwareclaim. PaperAppendixF.4.1 says1000testpoints with seed0; publiccode regeneratestestpoints via mutableRNG. Finalreport should additionally state this protocol ambiguity and include an explicitly seeded0 reference as well asfreshindependentreferences, without claiming access to original unpublishedtestfiles.

Core sourceSHA256s remain equal togoal-start. No commit/push. No final trainingseeds0/1/2 yet; finalallmetricsgoalremainsactive.


Further completed evidence in this continuation:
- ActualOptiQ CFM validation at10k:HP3base150k7.457566 (m4t0e9uw),HP4width1024-100k7.370727 (hzdhobqk), versusGTcontrol10k7.209683. These are earlyvalidationvalues, notfinaltestNLL.
- HP3std1-150k export `exports/hp3-std1-150k` (56utky4t). KDEdiagnostic `kde_validation_hp3_std1_150k` (j5m5vyn5):NLL8.107495/7.302212/7.093227 at h.2/.35/.5, betterthanpreviousbase100k butnotpaperNLL.
- Its common sample report `paper_eval_hp3_std1_150k` (vatqw3w7):meanlogp-7.073295,GTerror.199461,40modes,W2mean4.705615,spatialTV.8321,fixedTV.7326. BetterW2, worseTV:do not cherry-pickmetricsacrossdifferentconfigurations.
- Consequently anindependent CFMfit onstd1directsamples is nowstarted onGPU1: `optiq-gmm40-cfm-hp3-std1-150k`, output`cfm_hp3_std1_150k`. ReaditsconfigforfreshW&BURL. Uses same100kupdatesandCFMprotocol.
- HP7T1.25 is W&Bsbxb85ls; at170k actorupdatesmeanlogp-7.00037,energyW1.20053,KS.0467,40modes. HP7T1.5 at180k:meanlogp-7.11995,energyW1.25664,40modes. Needfinaldensityevaluation;nosuccessyet.
- HP6initialstd25/10 reached70k with38/39modes,meanlogp-13.0156/-12.0649. LowerLRinitiallyretained40modes,butmodecountslaterdegradedagain. Stillrunning;donotdescribeitassolved.
- `goal_progress.json` refreshed fromliveartifact/supervisorstate. Allcorehashesunchanged,pythoncompileanddiffcheckpassed. Stillnofinal0/1/2confirmation.


## Continuation update: Gaussian latent strata (HP8)

Previous goal turn: progress. This continuation also made progress (completed HP6/HP7, new exports and density/sample evidence, an optional minor sampling implementation with regression verification, HP8 and two new independent CFM evaluations).

Read-only paired diagnostics on the frozen HP3base150k checkpoint used20freshclouds atN512/R4,T1,sigma2,epsilon1e-4,300iterations. IID runhp5tscaj versusgridrun0gi45ilp:
- Mean OTrowmarginalTV .0741977 -> .0396431.
- Mean policy mode-occupancyTV .1137891 -> .0689258.
- ESS1066.93 ->1136.13; selecteddelta1.1460 ->1.0155.
- Selectedmeanlogp-6.6194 ->-6.6214: strata reduce batch variation but do NOT by themselves remove argmax concentration.
Artifacts: `ot_latents_iid_hp3_150k` and `ot_latents_grid_hp3_150k`.

Implementation `benchmarks/gmm40/latent_sampling.py`: split the 2D standard-normal probability space into equal-probability16x32 cells forN512, independently jitter once percell, invert the GaussianCDF, randomly swapaxes and permuterows. No target information. Numerical endpointguards prevent infiniteinverseCDF values. Stratification correlates trainingrows while retaining the Gaussianprior; final draw/export/evaluation always remains IIDnormal. `--latent-sampling grid` is GMMopt-in; defaultiid is original. The commonupdate accepts `policy_latents=None` and keeps the exactoldrandomdraw in thatcase. This is a minor sampling change within the user's authorization, not a new OT/gradient/objectivecomponent. The original and new core hashes are BOTH recorded in `goal_protocol.json`; do NOT claim the algorithm source file is byte-unchanged afterHP8.

Tests:16GMMtests passed, then2additionalfrozen-sourcecomparisons passed forbounded/unboundeddefaultpaths. Both entireactor/optimizer/key/loss/metric outputs are bitwiseidentical to verifiedoriginalgoal sourcehash5ac2994e...; explicitly supplying originalIIDlatents also reproducesexactly. The Gaussian grid covers everyequalprobabilitycell and has the correctmean/covariance. `policy.py`/`transport.py` unchanged. No target samples usedforOptiQtraining, no new gradientobjective, noSMC, no postprocessing.

HP6final150k:initialstd25 had38modes,meanlogp-11.00275;initialstd10 had38modes,-10.53817. Broad initialization failed these trials evenwithlowerLR; no furtherbroad-inittrials launched.

HP7final250k (IID) retained40modes. T1.25 meanlogp-6.94616,energyW1.17763;T1.5 meanlogp-7.08890,energyW1.23499. New100k exports `hp7-T125-250k` (42oll0eh) and`hp7-T15-250k` (i4dhsgrj); both nowhaveindependent100k-updateCFMfits running. These are tuningcheckpoints, NOT finalconfirmation.

Earlier200kexports: `hp7-T125-200k` (dqvqpp8z),`hp7-T15-200k` (rh5zmpwp). Same sampleprotocolreports:
- T1.25 (`paper_eval_hp7_T125_200k`,oxy6vds8):meanlogp-6.99785,W2 4.89993,TV.8260,40modes.
- T1.5 (`paper_eval_hp7_T15_200k`,ll2igoto):meanlogp-7.12632,W2 5.05608,TV.8429,40modes.
KDElikelihooddiagnostic (NOTpaperNLL) h.2/.35/.5:
- T1.25:8.093203/7.281611/7.070725 (e8wbisza).
- T1.5:7.824439/7.198075/7.042240 (hg7rvr80).
HigherT improves local-supportdiagnostic butworsensTV; stillnoall-metricwin. 250k sample/KDEevaluations are alsoinflight; readtheirsummaryfileswhencomplete. Their sample scriptsuse--no-baselines toavoid recomputingunchangedreleasedsamplemetrics; samefixedreferenceprotocol makesearlierbaseline numbersreusable.

CFMlatestatthisnote:GT90kvalNLL6.97211;HP3base20k7.25873;HP4width1024-100k atCFM20k7.22139;HP3sigma1 atCFM10k7.27586. Still no finalOptiQNLL/ESS/logZ orfinaltrainingseeds0/1/2. Goalactive.


Completed HP7 final250k independent evaluations (all tuningseed42, notconfirmation):
- `paper_eval_hp7_T125_250k` (2virhuds):meanlogp-6.977886,error.104052,40modes,W2mean4.869016,TV.8262,fixedTV.7315,energyW1.175674.
- `paper_eval_hp7_T15_250k` (aq8etf9u):meanlogp-7.088980,error.215146,40modes,W2mean4.977827,TV.8342,fixedTV.7432,energyW1.222450.
- T1.25 finalKDEdiagnostic(plbafzoe):h.2/.35/.5 NLL8.044469/7.246524/7.044401.
- T1.5 finalKDEdiagnostic(yp7fp5nv):7.751163/7.166088/7.021364.
These support trying improvedbatchsampling but do NOT establish a paperNLL win. CFMfits forboth finalcheckpoints are runningonGPUs2/3. `goal_progress.json` now records both original and currentminorvariantcorehashes ratherthan anincorrectall-core-unchangedclaim. Currenthashes matchthe documentedminorvariant.


## Continuation update: native-coordinate confirmation and CFM numerical audit

The user's coordinate question was checked against current code and live runs: native experiments already use `coordinate_scale=1`, `unbounded_actions=True`, direct `x=g(z)`, no output clipping or proposal truncation. `scripts/run_gmm40_native.sh` records the requested original sigma8/anchor/T.25/beta.1 comparison; later goal tuning also remains native. No experiment was restarted just for this question. The auxiliary CFM Gaussian prior std50 is an evaluator prior, not a coordinate multiplier on OptiQ outputs.

GT-only CFM control finished100k updates (3attuvst), testNLL6.963286, normalizedESS.624616, logZlowerbound-.094471. This is evaluation calibration only, never an OptiQ result. Read-only numerical audit8s5jfpwq on the SAME fixed checkpoint:
- Old full1000 forward batch, tol1e-3: NLL6.963286, ESS.624616.
- Tol1e-4 on the same original samples: NLL6.963490, ESS.787588; regenerating at1e-4 gives ESS.814142.
- Tol1e-5: NLL6.960212, ESS.807057 on original samples, .819835 on regenerated samples.

Public DEM generates auxiliary samples in256-sized batches; its mutable-RNG/test-code details do not establish exact original paper reproduction. New `post_evaluate_cfm.py` evaluates a fixed checkpoint with both forward/reverse batches256, preserving the raw results. GT-only batched run5aey7oi8: tol1e-3 gives NLL6.963286, ESS.547334, logZLB-.048044; tol1e-4 gives NLL6.963490, ESS.805434, logZLB-.118145. Both results are reported, not the more favorable one. Adaptive solver errors materially affect ESS, even with exact divergence; strict-tolerance checks cannot silently replace the paper-tolerance metric.

`idem_evaluate.sample_cfm` now accepts optional batch_size; None retains historical full-batch behavior. Future auxiliary fits default to generation batch256; five currently running fits loaded the old code and remain unchanged. Reevaluate their fixed final checkpoints with the same disclosed batched/strict procedure. No OptiQ training or sample output changes. The first ad-hoc analytic test (x5cewr4p) failed an overly strict absolute-only float32 check at coordinate scale50: even the identity ODE has dense-interpolation rounding up to4.96e-5. Explicit relative+absolute error checks at3e-6 pass for identity and linear flows; permanent regression tests also verify RNG isolation. Preserve the failed run rather than deleting it.

HP8 final250k: all40modes. Grid sigma2/T1 meanlogp-6.791488; sigma1/T1 -6.942724; sigma2/T1.25 -6.900448; sigma2/T1.5 -7.054884. Independent100k export HP8T1.25 (l4l2t1a1) has common sample metrics5zk9097c: meanlogp-6.904849, GTerror.031015, W2mean4.817936, reference-rangeTV.8295, fixedTV.7300. KDE diagnostic tdg5c24z gives NLL8.149632/7.255657/7.031461 at h.2/.35/.5; NOT paper NLL. Grid did not establish a density win relative to matched HP7 IID. HP8sigma1 export2e6dqfnp is also complete; its independent evaluations are running.

Read-only OT resolution diagnostics used the same fixed HP3base150k checkpoint,2048 candidates,T1,epsilon1e-4,300iterations. K512/R4/sigma1 (svvhfdsa):rowTV.032719, modeTV.069414, ESS1463.95, weightedmeanlogp-6.865261, selectedmeanlogp-6.851824. K2048/R1/sigma1 (cqs1mcae):rowTV.021846, modeTV.037598, ESS1623.77, weighted-6.862178, selected-6.870562. Epsilon1e-5 worsened rowTV; sigma.5 reduced ESS to777.58. These motivate HP9 count-only/temperature trials, not a claim of final sample quality.

HP9 final175k available: K1024/T1 meanlogp-7.022587, K1024/T1.1 -7.125699, K2048/T1 -7.073139; all40modes. K2048/T1.1 was at170k (-7.129021) at last poll. Independent export/evaluation of K2048/T1 is underway. All are continued tuning seed42, not confirmation seeds. Latest actual-CFM validation: HP3base50k7.272085; width1024-100k atCFM50k7.363861; HP3sigma1 atCFM40k7.222756; HP7T1.25 atCFM20k7.198132; HP7T1.5 atCFM20k7.141124. These are current validation, not necessarily best, and not final test NLL.

No final trainingseed0/1/2 runs yet. All-paper-metrics objective remains active and unproved; this is another progress turn, not a blocked audit.


## Latest completed evidence and HP10 launch

The auxiliary analytic regression now passes all4 identity/linear/batch/RNG-isolation cases (vtn3j4ea). Run this independent Torch test with `--noconftest` in `/venv/main`: the shared repository conftest imports JAX/Flax, which this intentionally separate evaluator environment does not install. The verified run explicitly logged its junit/source artifact to W&B; no dependency installation or shared-conftest change was needed.

HP8sigma1 final independent samples (2e6dqfnp) evaluated as ojb8t997:meanlogp-6.943564,40modes,W2mean4.688435,TV.8278,fixedTV.7266. KDEdiagnostic ytxwhg14:NLL8.063173/7.244510/7.036390 at h.2/.35/.5.

**More OT rows substantially improve the local-support diagnostic**. HP9K2048/T1 final175k independent exportuxy9s550:paper sample evaluation1vh25t1h gives meanlogp-7.057109,40modes,W2mean4.661035,TV.8264,fixedTV.7374. KDEdiagnostic myu8db6y:NLL7.318141/7.027972/6.957976. K1024/T1 independent exportlftkl96h and KDEdiagnostic hnrr02lj:NLL7.541448/7.096692/6.987757. The narrower-bandwidth NLL improves markedly withmore rows, although no paperNLL win is yet established. Every density diagnostic uses100k directIID g samples,5000 heldouttarget points and the samefixedreference. No noise/postprocessing is added to those samples.

HP9K2048/T1.1 also completed175k:meanlogp-7.139350,40modes, W&Bwr93hud9. No trainingseed0/1/2 confirmation yet.

An independently initialized100k-update auxiliaryCFM fit forHP9K2048/T1 is nowRUNNING onGPU2: `optiq-gmm40-cfm-hp9-K2048-175k`, output`cfm_hp9_K2048_175k`, W&Bmewkbm43. It records generationbatch256 explicitly and uses the sameunmodifiedpublicCFMmodel/training. The fiveolderCFMfits continue; preserve theirrawfull-batchresults and applyfixedcheckpointpost-evaluation afterward.

HP10 startsfromthe improvedHP9K2048/T1 checkpoint175k and continues25k to200k. Manifest`tuning_hp10.json` is a matched2x2 hyperparametertrial: KDEsigma1 versus.75, actorlr1e-4 versus3e-5. K2048/R1, Gaussianlatentgrid, T1, beta1, epsilon1e-4,300Sinkhorniterations, seed42, native/unbounded allunchanged. Targetdensity evaluationsperupdate remain2048, so continuationbookkeepingis stillvalid. The hypothesis is finer local proposals and/or smaller updatejitter can improvewithinmode support; no mechanism/objective substitution.
- GPU0 sigma1/lr1e-4:optiq-gmm40-tuning-hp10-0, xrvzgx52.
- GPU1 sigma.75/lr1e-4:optiq-gmm40-tuning-hp10-1, ytvl0441.
- GPU2 sigma1/lr3e-5:optiq-gmm40-tuning-hp10-2, 9o1ytvui.
- GPU3 sigma.75/lr3e-5:optiq-gmm40-tuning-hp10-3, 6dapcsl7.
Allfour were verifiedstarted; currentconfigs showcoordinate_scale1 andunbounded_actionsTrue. Corehashes stillmatchthedocumentedminorvariant. `goal_progress.json` refreshedfromactualartifactsandliveSupervisor. Performancegatesunchanged,goalactive,notblocked,nofinalsuccessclaim.


## Continuation update: mode geometry, paper ESS correction and HP11/HP12

This turn made progress: completedHP10, new directexports andsample/KDEmeasurements, actualfinalCFMNLLs, a primary-source ESSprotocol correction with completednumericalcontrol, and freshtraining/initializationtrials. Notblocked and notcomplete.

`mode_diagnostic.py` is read-only and computes posterior-responsibility moments for all40targetcomponents. GMMreference calibration gives meanrelative covariance eigenvalues.979/1.026 and tailfraction.05006. HP3sigma1K512 gives.890/1.558, normalizedmeanerror.2209 andtailfraction.09892;HP9K2048 gives1.014/1.418,meanerror.1271 andtailfraction.08227. The figure stillshowswithinmode supportholes andbridges. W&B8s2cfxsa; figure`outputs/gmm40_tuning/mode_diagnostic_hp9_175k/mode_support.png`. This is diagnostic, not a paper metric.

HP10final200k, all40modes: sigma1/lr1e-4 meanlogp-6.994331; sigma.75/lr1e-4 -7.028455; sigma1/lr3e-5 -7.024956; sigma.75/lr3e-5 -7.052543. Direct100k exports`hp10-{0,1,2,3}-200k` arel2sfpss5/q9jozpsx/loteogar/o5dmakxe. KDEdiagnostic h.2/.35/.5:
- 0 (tcdp3jvg):7.288405/7.009601/6.943875.
- 1 (0jis6wad):7.302156/7.020251/6.953108.
- 2 (6ji8m134):7.307742/7.020954/6.951124.
- 3 (dhfajhsv):7.269678/7.011553/6.948152.
HP10-0 paper-sample report0dcsto50:meanlogp-6.999514,40modes,W2mean4.734609,TV.8312,fixedTV.7357. Small density improvements do not establish all-metric success.

HP9K2048/T1.1 independentexportcb4methu, KDE2q4fhkyc:7.239953/7.007839/6.955037. This is a diagnostic improvement at narrow bandwidth; no finalCFM fit onthisT1.1 checkpoint yet.

HP11manifest startsfromscratch withGaussian latentstrata andK1024/R2 orK2048/R1, widths512 or1024 (3layers), exactly2048candidates/update. Seed42,T1,beta1,lr.0003,50kupdates,sigma8->1 andepsilon.01->.0001 overfirst15k,300Sinkhorniterations, noanchor,native/unbounded. Allfourreached40modesby5-10k. W&B:8y6h9o1l (K1024width512),kt43erpn(K2048width512),yuka7897(K1024width1024),34qtcl4i(K2048width1024).
The K1024width512 final50k exportcl0ceo6b hasKDEqrlny8b0:7.608074/7.148890/7.029220. Width1024 exporth8xv85qu hasKDE9lfs158u:7.744195/7.169484/7.024411. Morewidth did not uniformly improve support; don'tchoose itfrommeanlogp alone. K2048finalexports/evaluations are stillneeded.

Primary-source recheck: https://github.com/jarridrb/DEM#ess-computation-considerations explicitlydiscloses originalESSsamplecount16. PaperF.2 specifies100kCFMtraining samples; public1000epochs×100dummybatches/epoch supports100kupdates, andauthorREADME recommendsbestvalidationcheckpoint. The sampler, generatedsampledataset andnumericperformance thresholds were not changed. ProtocolnowrecordspaperESS16 andupdatedrecommended1000, reportingbothalongwithnumericalconvergencechecks.
GT-only fixedcheckpoint controlbz6r2lig,8independent16-sample repeats:tol1e-3 ESS16mean.757534 (resamplingSD.140035) versusESS1000.547334;tol1e-4 ESS16mean.876856 (SD.047209) versusESS1000.805434. NLL6.963286/6.963490. These are notOptiQresults. `post_evaluate_cfm.py --ess-batch16-repeats 8` nowrecordsraw16samples andlogdensities, plus1000samples/logdensities, andboth tolerance results. The published.734 andourpreviousraw1000ESS must notbe describedas identicalevaluationconditions.

New initializationoption is solely a hyperparameter for existing weights: `initialization=identity, initial_output_std=1 or8, initialization_noise=.01`. It usesGELU(x)-GELU(-x)=x acrossfourchannels ofeachEXISTING Dense layer, retaining smallseededrandomkernels elsewhere. Everykernelandbias remainstrainable. No architecture, OT/argmax/MSE objective, targetgradient, samplingcomponentorpostprocessingchange. No targetlocs/samples/density queries areusedtosettheseweights. `coordinate_scale=1` always; thereisnofixedcoordinateconversioninforward. Defaultinitialization=random anditsRNG/output remainunchanged. All19GMMtestspass, includinganalyticlinearmap,parameter-shapeidentity,RNG/oracleisolation, actualsameOptiQupdate,andfrozendefaultbounded/unboundedfidelity (idxwx2iz).

Initial finite-difference diagnostic tmi6x01d on4096Gaussianpoints,seed42,width512x3: originalrandommap outputstd.00145/.00242,detJ positiveatallpoints (.23e-6 to4.81e-6). Identitystd1 hasdetJ .99978..1.00015;std8 has63.986..64.010. **Noinitiallocalfolding wasobserved intherandommap.** Do notclaimthatinitialfolding wasprovedtobethecause, orthatfinitepositiveJacobians proveglobalinjectivity. HP12testsinitialscale/conditioning andsubsequenttraininggeometry, anunprovenhypothesis.

HP12is matchedK1024/2048×initialstd1/8, width512x3, otherwiseHP11settings. Workers0/2runningonGPUs0/2since21:17 (mfvtnuex/gmuxzhut); workers1/3startedafterHP11terminalstatus. All4slotsused. At35k theK1024identitystd1/std8 variantsstillcover40modeswithmeanlogp-7.31477/-7.43594. Finaldirectsampledensityevaluations pending. Nofinaltrainingseed0/1/2 confirmationyet.

The firstactualCFMfinalresults are now authoritative negativeevidence:HP3baseNLL7.185580,ESS1000.381181,logZLB-.357201;HP4width1024 NLL7.200015,ESS1000.364497,logZLB-.485770. BothbestCFMcheckpointsare40kafter100k-fitbudget; noNLLwin. Theyremainpreservedwhilefixedcheckpoint ESS16/precisionpost-evaluations run. HP9K2048CFM20kvalNLL7.129310 ispromisingonlyrelative tooldcandidates,notgoalcompletion.


## Continuation update: larger OT resolution and honest resumed query counts

This is progress, not blocked. Four new independent100k exports were evaluated with fixedheldoutKDEdiagnostics (notpaperNLL), at h.2/.35/.5:
- HP11 K2048 width51250k:7.397567/7.067661/6.981603 (nxooutz1).
- HP11 K2048 width1024:7.373686/7.050187/6.970274 (gf0o9j1x).
- HP12 K1024 identitystd1:7.557278/7.142520/7.033351 (b1s065fm).
- HP12 K1024 identitystd8:7.729146/7.201645/7.065448 (13d179os).
The std1init isbetter at h.2/.35 butslightlyworse at h.5 thanmatchedrandomHP11K1024width512. No consistentinitializationwin. LargerK fromscratchisstrongerthanincreasingwidthalone. Still onlyseed42tuning.

Read-only OT diagnosticsonsameHP10best200k,grid,T1,sigma1,epsilon1e-4,300iterations,3clouds:
- K2048 (yonf897u):meanrowTV.01954,selectedmeanlogp about-.02 lessnegative thanweightedcandidatevalue.
- K4096 (tg9v4umf):meanrowTV.01252;meanselecteddisplacement.4911.
- K8192 (ir1vt0po):meanrowTV.009028;ESSabout7016of8192;selectedlogpwithin.004ofweightedlogpforeachcloud;displacement.4126.
These are finiteOTdiagnostics, notproof ofbetterfinaldensity. They motivateHP14 count-only/learning-rate tuning.

New`accounting.py` fixes only logging when resuming with changedK/R/B. It reads actualparentcheckpointsteps andsumseachsegmentquerycount, cachingtheinitialcountinnewconfigs. Previouscontinuationsallused2048queries/updateandremainvalid. Threeaccountingtests passed. Actualtwo-updateGPUresumesmokeiutljeel from200k atK4096 recorded409608192queries =200000*2048+2*4096;Adam/keyrestoreandoriginalOptiQupdateunchanged. SmokecheckpointisNOT anytrainingparent. Sourcehashesrecordedinprotocol;compileanddiffcheckpassed. Maincorevariantunchanged;nofinalconfirmationseedsyet.


Further completed at~21:52UTC:
- HP13width1024/512 bothcompleted75k. Trainingfinalmeanlogp-7.007186/-7.054362,40modes. Width1024independentexportbzb39wtd;sampleevaluation77olvnk3:meanlogp-7.012355,GTerror.138521,W2mean3.994822,referenceTV.8285,fixedTV.7322,40modes. W2singleseedresamplingcanbebetterthanindependentGT;donotclaimstatisticalwinfromthisalone.
- HP13width1024KDEv2nd0e70:7.332076/7.032316/6.959149 at h.2/.35/.5. This6.959isNOT paperNLL. NewactualCFMfitm0v4724s onGPU3 evaluatesitsfixed100kdirectsamples. Width512export/KDEinflight;checksummary.
- HP7T1.25finalCFM(z3xzrnur):testNLL7.140536,rawESS1000.303621,logZLB-.268499,best60k of100kCFMupdates. FailsNLL6.96 despitebetterlogZ. HP7T1.5 finalpending.
- HP9K2048 CFM(mewkbm43) at60k improvedvalidationNLLto7.022217, notfinaltest.
- HP14GPU0K4096lr1e-4 started12yofdml;GPU2K8192lr1e-4 startedp08yanvy. GPU1/3 areguardedqueuedbehindHP12finalcompletion. Launchwatcherexecsession31699 isstillrunning;do NOT double-start. Itwrites`hp14_launch_status.json`. Candidatequerycountatstart409600000verifiedforK4096;K8192willrecordafteritscheckpointload. LearningcurvesatnewKmustbeassessedafterupdates, notjustinitialsnapshot.
- Main4GPUworkandindependentCFMevaluationscontinue. Allgoalmetricsunproved;noheldout0/1/2confirmationyet.


## Continuation update: fixed-candidate evidence audit and HP15

Previousgoalturnwasprogress;thisturnalsochangedauthoritativestateandcompletednewmeasurements. No blocker.

HP12K2048init1/init8final50kmeanlogp-7.184245/-7.324268,all40modes. Independent100kexportKDEdiagnostics (h.2/.35/.5):
- init1:7.495737/7.126209/7.031234(uq7fqngb).
- init8:7.292468/7.061254/7.000574(hc46fynp).
MatchedrandomHP11K2048width51250k:7.397567/7.067661/6.981603. Thusinit8hasbetternarrow/middlesupportbutworsebroaderdiagnosticandenergy;neitherprovedwinner. ThismotivatesHP15continuedtraining,notagoalsuccessclaim. SameexistingtrainableweightsandoriginalOptiQupdate;noaddedmodelcomponent.

HP13width51275kKDE(kf7ln4p1):7.301926/7.018250/6.946465;NOTpaperNLL. Samecandidatepapersampleevaluationt4yuugl7:meanlogp-7.025193,40modes,W2mean4.206599,TV.8288,fixedTV.7339. ActualCFMnowrunning0kihldzp.
HP9K2048T1.1-175ksampleevaluationz89mv403:meanlogp-7.137830,40modes,W2mean4.636870,TV.8368,fixedTV.7448. WorseTVdespitebetternarrowKDEsupport;donotmixcrosscandidatemetrics.
HP14K4096lr1e-4 at205k (20.48millionextraqueries) independentKDE0x3owaa6:7.232306/6.994468/6.938444. ImprovesallthreebandwidthsversusparentHP10 (7.288405/7.009601/6.943875),butisNOTfinalNLLorproofallmetricswin. Trainingcontinuesto210k;8192comparisonscontinue205k.
HP7T1.5finalCFM(w3igxd31):testNLL7.110312,rawESS1000.412305,logZLB-.323458,best40kof100kupdates. BetterthanT1.25NLL7.140536,stillfails6.96.

New`audit_results.py` joins by exact100k sampleSHA and verifiescheckpointSHA. Itpreservesallreference/CFMpostvariantsandmissingvalues;alwayslabelsresulttuning,neverusesKDENLLorcurrentvalidationasfinaltestNLL. Sourceonlyreporting,noOptiQchanges. Live28candidateauditjga4bb9mcompleted;firstfailedW&Bsummaryupdatex1p559amwasretainedandfixedbydictargument.
Read-onlybaselineinventory:releasedDiKL/iDEMsamplefilescontainonly10000points,not100k. Theycannotbeusedas100kCFMtrainingdatasetswithoutchangingtheevaluationconditions. No10kduplication/resamplingorbaselineCFMsubstitutionwasperformed.

Allsamplersnative/unbounded. CorehashesstilldocumentedoptionalGaussianlatentvariant. No confirmationtrainingseeds0/1/2yet. Allpaper-metricgoalisactiveandunproved.


## Continuation update: matched query counts, completed K4096 and generator geometry

Previousgoalturnprogress;thisturnalsoprogress. Same-queryK4096/8192diagnosticsaboveusefixedcheckpointsandfresh100kIIDgexports,nottrainingevalbatches. K4096bothfinal210kmeanlogp-7.017711(lr1e-4)/-7.023358(lr3e-4);40modes. Eachrecorded450560000totaloraclequeries,includingthe2048candidateparenthistory. HP15bothstartedafteractualcompletion,watcher67379terminalandcollected. NewCFM7iw2e85xstartedonlyafteritsdirectexportcompleted.

Newread-only`map_geometry_diagnostic.py` evaluatesJAXJacobianandlatentnearest-modepartitionsontheSAMEsavedactors;nottraining,nocriticgradient,newloss,architectureoroutputpostprocessing. InitialdefaultGPUarithmeticvalidationfailed:vkq8m4jyretainedfirsttwoarrays;v2run3wn375qialsofailed;explicitnumericalprobeysc6hry9savedfullnumerics. HP10finite-differencerelativeRMSerrorswithdefaultprecision:.01541/.01602/.06217/.17899 atsteps.003/.001/.0003/.0001;forwardvsreverseautodiffrelativeRMS.001009. Thesefailureswerepreservedandnotreportedastrainingfailures.

JAXlocalinstalled0.4.33sourceand[officialprecisiondocs](https://docs.jax.dev/en/latest/201/precision.html)describeGPUdefaultreducedprecision(TF32whereavailable)andhighesttruefloat32. Diagnostic-onlyhighestprecisionresolvedagreementforallsamecheckpoints(5uv7518x):
- HP10FDstep1e-4relativeRMS.0002003;forward/reverseautodiff7.63e-7.
- HP14K8192:.0002541 /9.70e-7.
- HP12init8:.0021043 /1.72e-6.
Allstep-sizestudiesandrawarraysretained. Outputgriddefault-vs-highestRMS.00837/.00982 forHP10/HP14;modeassignmentdisagreement.0515%/.0624%;Jacobiansigndisagreement.0488%/.1465%. ThusnumericalprecisionexplainsthediagnosticfailurebutdoesNOTbyitselfexplainremainingdensitygap. `precision_comparison.json`storesgridconditions;thisfilewasaddedaftertheoriginaldiagnosticartifact,rawinputsarealreadyarchived. Existingtrainingprocesses,exportsandCFMmetricswereuntouched. No maintrainingprecisiontriallaunched.

Themapplotsretainthinmodepartitionsandorientationchangesacrossallthreecheckpoints;largerOTdoesnotremoveallgeometricdistortion. HP16thereforetestsbroaderidentityweightinitialization(std25)andK/lrusingtheexistinghyperparameteroption. NoJacobianpenalty,SMC,newnetworkcomponent,gradientobjective,trainabletargetinformationoroutputtransform. Thisisahypothesis,notaprovenfix. Latentstrataandnativecoordinatepolicyunchanged.

Corehashesremain thedocumentedminorlatent-inputvariant. Noconfirmation0/1/2,allgoalmetricsunproved. No blocker.


Latest beforegoalturnhandoff (~22:23UTC):HP9K2048CFMreached90kwithnewbestvalidationNLL7.008825(mewkbm43);finalteststillpending. HP14K4096lr3e-4finalKDE(nwneb93f)7.201117/6.992467/6.943091,isworsethanlr1e-4atallthreefixedbandwidths. NoadditionalCFMfitforlr3e-4launched. Its100kexportiscomplete. HP16GPU0freshidentitystd25/K2048/lr3e-4startedvh44ygjdafterHP15GPU0completed. OtherHP16workersremainunderliveguardedwatcher72740. Collectthishandlelater;doNOTstartduplicates. HP15GPU0finalresultsnotyetindependentlyexported/evaluated. GeometryandKDEexecevaluationhandlesareterminal/collected;only72740remainsrunningfromthisturn.

## Continuation update: 2026-09-08 23:06 UTC

Progress, not blocked or complete. HP9 actual final CFM/test and ESS post-evaluation completed; HP14 K8192 final exports/sample/KDE metrics and CFM launch completed; timing audit and two meaningful tests completed; HP18 two direct100k exports and all short sample/KDE evaluations completed; fixed HP13 OT diagnostic completed. HP16 stopped trials preserve exact actor/Adam checkpoints, final samples and logs. HP19 changes only existing hidden_dims, while HP20 continues a faster high-LR start using the same actor update and recovered Adam/RNG state. No core changes, gradient variants, coordinate transforms, hidden postprocessing, commits or pushes were made. The updated live section above records the specific values, W&B IDs and outstanding work.

HP15 final75k export diagnostics are also complete: init8/K2048 lr3e-4 gives7.313106/7.063368/6.997075 (zw53u2nz); lr1e-4 gives7.241012/7.020121/6.964334 (n6fmk2ef), for h=.2/.35/.5. Both retain40modes. This mixed diagnostic evidence has not established a CFM NLL improvement, so no additional long CFM fit was started for them.

## Additional completed work: 2026-09-08 23:18 UTC

HP17 both20k checkpoints exported100k IID direct samples: random5awhe45y and init8cexbailo. KDE diagnostics h=.2/.35/.5: random7.480725/7.117844/7.022616 (w6d3hpz4), init8 7.548564/7.143668/7.033853 (f224gkid). Initial high-lr training converged faster in meanlogp, but local-density diagnostics still lag the mature75k candidate. Both continue under HP20 at lowerlr3e-4 to50k; no final CFM fits for20k checkpoints were started.

Fixed HP13 candidate-count diagnostic n4w3u5uc and mode-support diagnostic mytjpldi completed and the figure was visually inspected. HP21 increases only random candidates per center at matched added target-density query counts. The automated CFM post queue is now supervised on four workers, with all inputs, source hashes, readiness checks and output paths recorded in the goal protocol. No pending tool sessions remain from the training launches or completed diagnostic/export commands. Source policy/transport hashes remain unchanged from the goal baseline; algorithm.py still contains only the previously verified optional latent input branch. All paper gates and fresh final seeds0/1/2 remain outstanding.

## Additional completed work: 2026-09-08 23:24 UTC

Both HP19 fixed25k intermediate checkpoints now have100k IID direct exports: depth5/width5129570xcsj and depth6/width256goa9jijt. Original coordinates, no output transforms or candidate noise. KDE diagnostics h=.2/.35/.5 are7.433927/7.077798/6.987444 (p6bhman7) and7.368353/7.076407/6.994266 (i0ar9c7l), respectively. Faster early meanlogp convergence has not yet beaten the mature HP13 fresh75k KDE support. No premature CFM fits were launched on these25k checkpoints. Both original training programs continue toward their configured50k budgets.

Read-only common-mode figure/statistics x6uwubd4 completed in `mode_diagnostic_depth_20260908T2321` and its figure was viewed. All40 component statistics are retained; the illustrated component13 was selected by the original HP13 reference, not chosen separately for each model. Mean tail fractions are.077901 for depth5 and.090694 for depth6, compared with.075625 for HP13 and.050060 for GT. Mean maximum variance ratios1.3623/1.4860 also indicate remaining local-shape error. These are diagnostics, not paper metrics.

Latest joined audit4yrcwp83 covers41 fixed candidates as of23:18. Two later HP19 exports postdate it. Every temporary exec export/diagnostic/start session from this turn has completed and been collected; no pending launch watchers remain. Long work consists only of supervised HP19/20/21 training, six CFM fits and four CFM post-evaluation queue workers. All four3090s were active at the last check. Core source hashes were reverified; no new algorithm changes, commits, pushes or confirmation seeds0/1/2. The goal remains active and unachieved, with genuine progress and no external blocker.

## Continuation update: 2026-09-08 23:52 UTC

Progress, not blocked or complete. Final HP19/20/21/22 training outputs are preserved; the existing random default and core update are unchanged. The wide initializer is only an optional initial-weight setting. All22 GMM regression tests passed, including actual immutable-source bounded/unbounded update equality. New CFM test/post results and current supervised work are recorded in the live section above.

Independent100k direct-sample KDE diagnostics, at h=.2/.35/.5 (not paper NLL):

| Candidate | h=.2 | h=.35 | h=.5 | W&B |
|---|---:|---:|---:|---|
|hp19_depth5_width512_50k|7.345827|7.019001|6.942784|dpsn4c58|
|hp19_depth6_width256_50k|7.382554|7.048949|6.961351|bgmcf5uz|
|hp20_fast_random_50k|7.272837|7.003381|6.937912|dz155cbu|
|hp20_fast_init8_50k|7.343053|7.038538|6.964229|4czqpiow|
|hp21_K2048_R2_85k|7.277253|7.004449|6.940149|6uw9e028|
|hp21_K2048_R4_80k|7.270705|6.998293|6.933696|ne4dcckc|
|hp22_wide_identity_0_20k|7.444797|7.131453|7.045522|ty5cqdgp|
|hp22_wide_identity_1_20k|7.441406|7.106708|7.013990|2x66rvmz|
|hp22_wide_identity_2_20k|7.432747|7.112741|7.024821|krmketch|
|hp22_wide_identity_3_20k|7.469099|7.117454|7.021410|endimjff|

Final training meanlogp and launch IDs for HP22 (all20k,40modes):

- 000-wide-identity-init8-noise0.0-lr0.0003-20k: -7.398458, ym2kgz2s.
- 001-wide-identity-init8-noise0.0-lr0.001-20k: -7.095724, rrmom9kh.
- 002-wide-identity-init8-noise0.01-lr0.0003-20k: -7.245025, jwge96y3.
- 003-wide-identity-init8-noise0.01-lr0.001-20k: -7.130163, pogfhci3.

HP23 is a matched2x2 continuation of depth5/fast-random parents and T1/T.9, all with lowerlr1e-4. Guarded starts verified actual successful HP22 completion and have all finished; no duplicates should be launched. New ordinary learning-rate and temperature settings are the only training changes in this round. Original GMM reference and all paper gates remain fixed. No final confirmation seeds0/1/2 have run.

Final23:54 check: HP9 T1.1 post-evaluation z9dp1tcm is complete; both tolerances are preserved above. The HP13 width512 post evaluator is live, has finished the primary tolerance and is writing secondary ESS16 batches; no failed.json exists in the post-evaluation directories. Worker states were inspected individually. All other temporary tool handles from this turn are terminal and collected.
