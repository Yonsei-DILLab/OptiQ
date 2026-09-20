**Ant/Humanoid temperature selection, then beta ablation (2026-09-21)**

Branch: direct-gmm-trg. W&B: OptiQ/gmm-trg. Commit all source and freeze it before launching.

Stage 1: Ant T={0.1,0.05}; Humanoid T={0.5,0.1}; beta=1; seeds 0,1,2,3,4.
Stage 2: per environment, wait for ALL ten full temperature runs to complete, select the best tested T, then launch beta={0.5,0.9} at that T with seeds 0..4. Reuse the five selected-T beta=1 runs as the control. Each beta run starts fresh; no checkpoint continuation.

Selection is the user's requested stochastic_z return averaged over the LAST 100,000 environment steps. With unchanged 5,000-step evaluation, use 900,000 < step <= 1,000,000 (20 evaluations, 10 episodes each). Average within each seed, then equally across five seeds. Report between-seed sample standard deviation. Exact ties use zero_z last-100k mean, then smaller temperature. Also report zero_z and learning-curve means. This picks the best tested temperature, not a global optimum; these same seeds are tuning seeds, not independent held-out confirmation.

Unchanged defaults: 1M env steps; 5k warm-up; batch256; UTD1; actor delay1; Adam actor/critic LR3e-4, betas .9/.999; no clipping; gamma .99; critic tau .005; actor/critic256x2 GELU; N=M64; normal fresh z; learned log sigma hard bounds[-5,-1], initial log sigma=-1; plain TD; density correction enabled; fixed beta; no extra exploration. Evaluation zero_z and stochastic_z are conditional-center actions without conditional noise, ten episodes per mode every5k. Diagnostic5k, checkpoint50k.

No performance early stopping. Failure/incomplete seed prevents selection and beta progression for that environment; other independent temperature runs may finish. No automatic retry or reuse of partial runs. Supervisor owns controller and child process groups. One run/GPU, four GPUs per environment, host180 Humanoid and host199 Ant. Environments have independent phase gates.

The older shared validator hardcodes beta=1. The dedicated experiment validates all its other conditions on a beta=1 copy, then passes the actual requested beta unchanged to the existing beta-parametric loss. Config comparisons enforce that no unrelated algorithm values changed. Both density_beta aliases match. No optimizer or loss formula is modified. A logging-only callback corrects stale pre-tanh metadata to describe the actual box-truncated policy.

Deferred: HalfCheetah, Walker, Hopper, DACER target entropy exploration, NM ablation, density correction removal. They are not part of this launch.
