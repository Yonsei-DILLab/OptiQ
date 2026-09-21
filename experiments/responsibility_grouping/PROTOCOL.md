# Responsibility-only grouping: same rule in 2D and 17D

Read-only checkpoint analysis, registered before computing clusters. No actor/critic update and no MuJoCo job launch. Use heejoon commit and input SHA256 manifest.

Inputs: TRG Humanoid seed0 at50K/650K, previously saved128 identical probe states, N=M64; and existing 2D Direct GMM double/tri mass trials at35K, N256/M1024, seeds0–3. Two-dimensional target modes vary along action1; action2 is an independent truncated N(0,.35²). This is the existing controlled easy 2D problem, not a new fully correlated target. TRG uses action-box truncated Gaussian; toy uses squashed Gaussian. Reconstruct each model's correct posterior in float64; same grouping code after that.

Candidate feature v_j = gamma[:,j]. Use Hellinger distance = Euclidean distance of sqrt(v)/sqrt(2), average-linkage hierarchical clustering. No action, Q, teacher weight, target boundary or mode count enters this clustering. Candidate count differs, so comparisons demonstrate the method, not a controlled dimension-only effect.

Continuous visualization: optimal leaf ordering of the dendrogram, no K required. For colored groups evaluate K=2..8 using ordinary unweighted silhouette in the same Hellinger distance. Exclude partitions with any group smaller than max(2,ceil(.01 M)). Select maximum silhouette; ties choose smaller K; require silhouette>=.25 and nonzero distances, otherwise K=1. Show ALL K scores and accepted counts. This is an exploratory predetermined rule, not a consistent mode-count estimator. Do not retune on the results.

Rows receive H_i,m=sum_{j in group m} R_ij, with R=J/alpha,J=w*gamma. Assign each row argmax H for display, with no confidence weighting or training. R original, action-PC1 order, responsibility grouping use identical values and common LogNorm[1e-5,1]. Display w and alpha statistics so row normalization does not imply uniform usage. Group color ID may be permuted by median first action coordinate for readable 2D plots ONLY, after clustering.

2D figures: saved32768-action64x64 histogram and exact-reference contours; actual candidate locations colored by discovered groups; student tanh(mu) centers colored by H argmax; candidate-level R heatmap. Show seed0 as fixed representative and allfour seeds in appendix. True basin labels are computed AFTER grouping only to report ARI and group/basin contingency. No oracle input to fit.

17D: state0 figures at both checkpoints, original/PCA/new order, plus firsteight states without outcome selection; counts across all128 probes. PCA is for the old visual comparison only, never a clustering input.

Stability: on all8 toy snapshots and first8 RL probes percheckpoint, retain80% of the existing student rows without replacement, recompute gamma by log normalization on SAME candidates, refit groups using SAME rule; 5 fixed seeds919..923. Report ARI with full-bank grouping and K variability. This measures sensitivity to the sampled student bank, NOT independent fresh actor-latent or candidate draws, nor training-seed robustness. The 2D four trained seeds are separate from this diagnostic.

Validation: posterior column sums, conditional row sums, J teacher marginal; repeatability; student-row permutation invariance of candidate distances; identical-student null yields K1; two artificial disjoint responsibility blocks yield K2; ARI identity/permutation tests. Verify all original input hashes and saved sample histograms. Report actual results including over/underpartitioning. Save partitions, distances, scores and source provenance outside Git; scripts/protocol in Git.
