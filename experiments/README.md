# Experiments

This directory contains runnable experiments that reproduce the controlled
results reported for OptiQ. End-to-end experiments import the implementation
from `optiq/`; analytic reference experiments remain self-contained so they can
isolate projection properties from implementation and optimization error.

- `boltzmann_projection/`: finite-particle projection and proposal-density
  correction experiments.
- `four_way_multigoal/`: multimodal policy extraction and rollout experiments.

Raw runs are written to `outputs/`. Only final, paper-facing figures belong in
`assets/figures/`.
