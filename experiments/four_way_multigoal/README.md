# Four-Way Multi-Goal: Skewed Proposal

This branch tests OptiQ with a pairwise-skewed, radially truncated Gaussian
proposal.  The proposal uses only the existing `proposal_std` and
`proposal_clip`; no separate gradient-step, parallel-scale, temperature, or
skew hyperparameter is introduced into the proposal geometry.

```bash
python -m experiments.four_way_multigoal.train_skewed \
  --seed 1 \
  --total-steps 100000 \
  --no-include-anchor
```

Outputs are written under
`outputs/four_way_multigoal/skewed_proposal/seed_<seed>/`.
The run also records 4,096 policy samples at the exact initial state in
`s0_actions.npz` and `s0_action_modes.png`. Deterministic proposal anchors are
disabled by default so all 80 candidates are IID draws from the density used
by the importance correction; pass `--include-anchor` only for an ablation.
