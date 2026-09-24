# Six targets: finite GMM Forward-KL fitting

Status: pending; populate from completed trajectories, retain missing/failed runs explicitly.

1. Experiment setup: six targets, K, fixed/learned sigma, equal/learned weights; population quadrature versus previous SNIS neural study.
2. Main figure: six targets × five K, paired fixed/learned variance, sampled histograms and true density; per-seed figures plus average.
3. Table: 4-seed TV/W1, mode recovery counts, not only best seed.
4. Component means/sigmas/weights and responsibility-to-basin matrices: one-fits-many / many-fit-one where observed.
5. Trainable-weight control: capacity versus optimization.
6. Paper parameterization: structured and random initializations separately, objective/gradient/Hessian and perturbation diagnostics.
7. What is and is not supported about semi-implicit structure, latent resampling, and local minima.

Use 2^20 actual sampled actions and 512 histogram bins at the final checkpoint. Plot with standard Matplotlib, lightly filled learned histograms, unfilled dashed true density. Do not treat finite numerical tests as a proof.
