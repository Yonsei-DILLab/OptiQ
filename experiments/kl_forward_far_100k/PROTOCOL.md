# Forward far-mode experiment: continue 128x128 and256x256 to100K

User requested extending the existing128x128 and256x256 runs to100,000 total actor updates. Continue seeds0–3 from their completed20K checkpoints (8 runs), for80K additional updates. Preserve actor parameters, both Adam moments and optimizer count, TrainState step, and training PRNG key. This is continuation, not fresh initialization. Keep original20K campaign immutable.

Numerical implementation remains kl_forward_far_1d from1153ff9a797173681a980fffea6f2e75414fe40d: mode centers(-10,0,10),width0.1,equal target weights,action and mean box[-20,20],mean20*tanh(head),original mean initializer scale1,log sigma bounds[-5,-1],initial-1,temperature0.25,batch32,Adam3e-4,256x256 GELU,1D normal latent,forward marginal NLL with finite-mixture proposal. No other algorithm changes or interventions.

Only configuration differences: study name, total updates20K->100K, evaluation schedule. Evaluate at20K then every5K through100K, with the identical evaluation seed and32768 actual action samples;4096-bin unsmoothed histograms,per-mode zooms,TV,basinTV,W1,backup error,core masses and the original prespecified three-peak separation criterion. Independent evaluation never advances training RNG. Save full checkpoint every500 updates; snapshot starting20K checkpoint in each new run for provenance.

Before continuation, verify parent source/seed/config matches and checkpoint checksum equals the registered PARENT_CHECKPOINTS.json. After restoring, compare every serialized actor/optimizer/RNG leaf exactly. Re-evaluate20K and record numerical difference from original20K samples to expose any device-roundoff discrepancy. Changed GPU/compiler execution can produce later numerical trajectory differences even with exactly restored state; record hardware/software in RUN.json.

Commit extension config/runner/job/protocol to heejoon before launch. Record original numerical commit as well as extension commit; compare source hashes for actor/core/box_gaussian/distillation/evaluate. Eight independent Slurm jobs,1GPU2CPU16GB each,maximum8 concurrent,one-hour limit. Do not modify64/512 runs or unrelated administrative holds. Keep full data/checkpoints on dildata.

Primary question: does longer fitting restore the missing center peak and eventually all three peaks? Compare paired20K,50K,100K histograms for every seed; report per-seed recovery times among observed5K snapshots, not only averaged curves or basin masses. Do not redefine success thresholds after seeing results.

Parent: login4:/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/forward_far_20260923.
Extension: same extensions directory/forward_far_100k_20260923.
Central: dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_far_100k/campaign.
Report destination: reports/20260923_forward_far_100k.
