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
