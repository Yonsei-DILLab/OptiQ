# HalfCheetah SMEM+TR actor transfer repair

The original experiment was stopped by user request. Its committed implementation
and output data remain at `20260921_halfcheetah_smem_tr`. This directory is a new
version; no running source is modified.

## Diagnosis

Hypothesis: projecting independent per-state EM teachers into a shared actor
with cached Adam moments and an unweighted parameter regression yields an
unreliable descent direction for the actual weighted mixture likelihood.
Use the three original seed policies/critics at 50K and their saved replay probe
observations. Hold weights, candidate actions, state batches, random keys and
the teacher fixed. Compare the original ten regression steps with reset Adam
moments. Measure candidate NLL gain, exact joint KL and feasibility for each
step. This is a fixed-checkpoint diagnostic, not an RL performance comparison.

## Follow-up

Choose a repair only after diagnosing the fixed checkpoints. Document each
method change and retain KL radius 0.05, Gaussian bounds [-5,-1], environment
HalfCheetah-v4, temperature 0.25, 64 components/candidates, batch size 256,
critic, warmup and evaluation protocol. Label tuning pilots separately from
confirmation seeds. Compare return at equal environment steps and report
wall time as well. Any 1M run has its own frozen commit and W&B run ID.

## Selected repair (v2)

The six fixed-checkpoint checks reproduced rejection of all ten original
Adam candidates due to worsening NLL (usually KL remained below 0.05).
Backtracking the old regression alone yielded zero or ~1e-4 NLL improvement.
The EM auxiliary objective yielded 0.239--0.430 improvement in all six checks
within KL 0.018--0.050. These are empirical fixed-checkpoint results only.

Retain the independent generalized EM, split/merge, teacher projection and
importance weights. Replace the equally weighted parameter regression with
the generalized EM auxiliary objective for the shared actor:

    R_kj = responsibility of component k under the projected SMEM teacher
    L(theta) = -mean_s sum_j w_sj sum_k R_skj log p_theta(a_sj | s,z_sk)

R and w are stopped. Take one Adam candidate per outer actor update, with up
to ten halvings to find a strict mixture-NLL decrease and joint KL <= 0.05.
If rejected, retry from reset Adam moments; if still rejected, restore the
previous parameters and optimizer. The original actor counter advances once
per outer update. Accepted optimizer moments come from the accepted candidate.
This is an explicit shared-actor generalized M-step adaptation, not a claim
of the original variable-weight SMEM equations or a policy-return guarantee.
It is not a fallback to the Direct GMM marginal-NLL gradient.

Log per-interval means of acceptance, actual NLL gain, KL, momentum resets,
SMEM activity and ESS to eliminate phase aliasing from sparse diagnostics.
All comparison runs use the same evaluation protocol, and record elapsed
time and GPU type. Inner actor work differs by method and must be disclosed.

Seed 0 is the tuning gate: run from initialization toward 1M steps, require
mean of the last three evaluations at 50K to reach 1,000, otherwise stop at
50,001. This tests recovery from the near-zero plateau, not final superiority.
If it passes, continue the same run to 1M, launch unchanged seeds 1 and 2 and
three matching Direct baselines. Seed 0 selection is disclosed; seeds 1/2 are
the subsequent fixed-method checks. All runs are logged to the existing W&B
project with algorithm and experiment_revision fields, and no live renaming.
