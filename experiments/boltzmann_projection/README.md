# Boltzmann Projection

Controlled experiments for critic-induced Boltzmann policy extraction.

The reproduction suite contains:

- `particle_refinement.py`: exact equal-mass refinement as policy capacity grows;
- `neural_distillation.py`: neural categorical OT amortization; and
- `density_correction.py`: correction under uniform and truncated-Gaussian
  proposals.

Scripts write generated results to `outputs/boltzmann_projection/`. Selected
paper-facing figures are versioned in `assets/figures/boltzmann_projection/`.
