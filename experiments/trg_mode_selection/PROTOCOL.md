# TRG mode selection preparation — 2026-09-21

User requested cancel all old experiments on31.148.50.247:11717 (RTX3090×4), then run TRG+mode selection with W&B OptiQ/DirectGMM_orthoGrad, Ant → Humanoid → HalfCheetah priority. Original jobs canceled; datasets/checkpoints preserved. Other hosts untouched.

## Important unresolved transfer from toy to MuJoCo

The frozen-Q mode-selection experiments used **known Q-density valley labels** before aggregating responsibilities by mode. Their H heatmap is not an unsupervised mode detector. The raw N×M posterior gamma may contain blocks, but equal conditional components give gamma=1/N for every candidate, regardless of the teacher's actual modes. Clustering gamma cannot reveal modes that leave no posterior pattern.

User suggested infer mode groups from the raw64×64 responsibility heatmap. First inspect original and reordered heatmaps from existing TRG checkpoints, at fixed probe state0 plus aggregate statistics over all saved states. Reordering does not create or verify Q modes. Exploratory graph-threshold group counts are not a training choice. The user authorized the new MuJoCo experiments; their mode-partition rule is not yet specified, so training has not been launched with an invented rule.

`inspect_responsibility.py` is read-only checkpoint analysis. Fixed RNG919, N=M64,128saved probe states if available, existing truncated-Gaussian actor/proposal unchanged, zero parameter updates. Store full gamma, outputs, samples and candidate posterior Bhattacharyya affinities. Plot original and reordered axes separately. Checkpoint hashes are recorded. ExistingTRG runtime source52ae6c4…; latest branch2bdeb9c… has identical core TRG numerical files (new additions are separate GMM40/other experiments).

Remaining launch settings tentatively inherit recentTRG comparison: singleQ, K64, N64,M64/M256, T.25, batch256,1Msteps, seeds0–3. No confidence weight. Preserve actual differentiable truncated normalizer. Queue ordering is a preference, no cross-environment gate. Source/config/protocol must be committed and pushed before new training. Authentication already exists on3090host. All new outputs remain under its dildata-backed legacy_monge subtree.
