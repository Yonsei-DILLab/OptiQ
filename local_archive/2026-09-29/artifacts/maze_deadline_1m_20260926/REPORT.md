# 1M experiment results

Verified complete: 30/33.

All runs use seed0. Success is not mode coverage. Way T1 reused8/16 results use UTD1; new results use UTD0.0625 and cannot isolate temperature effects.

Removal SR5 curves are recomputed from raw five-episode groups to correct a frozen scorer boundary error. See reporting-corrections.json; training, success/goal counts, trajectories and raw archives are unchanged.

OptiQ primary evaluation: fresh random-z mu-only, without conditional sigma. Baselines: native samples. Historical sigma-included figures are supplementary; old Q probes are not mu-only. Missing obstacle-mu results are omitted.

|Run|Success|Goals visited|Goal counts|
|---|---:|---:|---|
|16way-optiq-t1-s0 (reused)|100.0%|4/16|[201, 0, 0, 0, 0, 0, 0, 0, 0, 298, 0, 0, 0, 0, 300, 225]|
|16way-optiq-t10-s0|100.0%|16/16|[32, 52, 106, 61, 16, 74, 141, 24, 31, 77, 79, 71, 29, 45, 131, 55]|
|16way-optiq-t3-s0|100.0%|13/16|[70, 84, 7, 145, 1, 176, 0, 0, 111, 77, 0, 96, 104, 7, 109, 37]|
|16way-optiq-t5-s0|100.0%|13/16|[40, 104, 8, 107, 0, 125, 41, 103, 20, 112, 0, 128, 0, 93, 91, 52]|
|4way-optiq-t1-s0|100.0%|3/4|[367, 324, 333, 0]|
|4way-optiq-t10-s0|100.0%|4/4|[303, 300, 210, 211]|
|4way-optiq-t3-s0|100.0%|4/4|[253, 273, 243, 255]|
|4way-optiq-t5-s0|100.0%|4/4|[246, 264, 263, 251]|
|8way-optiq-t1-s0 (reused)|100.0%|5/8|[142, 333, 139, 0, 215, 195, 0, 0]|
|8way-optiq-t10-s0|100.0%|8/8|[43, 194, 32, 243, 31, 238, 42, 201]|
|8way-optiq-t3-s0|100.0%|7/8|[126, 203, 72, 196, 123, 0, 128, 176]|
|8way-optiq-t5-s0|100.0%|8/8|[96, 174, 70, 177, 84, 179, 79, 165]|
|pm_hard-meow-s0|13.2%|1/8|[0, 0, 0, 0, 0, 0, 0, 66]|
|pm_hard-mfpo-s0|100.0%|1/8|[0, 0, 0, 0, 500, 0, 0, 0]|
|pm_hard-optiq-s0|100.0%|8/8|[52, 1, 6, 27, 126, 5, 265, 18]|
|pm_hard-sac-s0|100.0%|1/8|[0, 0, 0, 0, 0, 0, 0, 500]|
|pm_hard-sql-s0|42.4%|6/8|[65, 101, 14, 28, 0, 0, 1, 3]|
|pm_hard-td3-s0|0.0%|0/8|[0, 0, 0, 0, 0, 0, 0, 0]|
|pm_medium-meow-s0|100.0%|1/4|[0, 0, 0, 500]|
|pm_medium-mfpo-s0|100.0%|1/4|[0, 0, 0, 500]|
|pm_medium-optiq-s0|100.0%|4/4|[152, 6, 281, 61]|
|pm_medium-sac-s0|100.0%|1/4|[0, 0, 500, 0]|
|pm_medium-sql-s0|94.0%|3/4|[10, 437, 0, 23]|
|pm_medium-td3-s0|100.0%|1/4|[0, 0, 500, 0]|
|pm_simple-meow-s0|100.0%|1/4|[500, 0, 0, 0]|
|pm_simple-mfpo-s0|100.0%|1/4|[0, 0, 500, 0]|
|pm_simple-optiq-s0|100.0%|4/4|[121, 81, 108, 190]|
|pm_simple-sac-s0|100.0%|1/4|[0, 0, 500, 0]|
|pm_simple-sql-s0|100.0%|4/4|[2, 484, 2, 12]|
|pm_simple-td3-s0|100.0%|1/4|[0, 500, 0, 0]|
