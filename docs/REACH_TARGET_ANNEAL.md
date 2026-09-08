# Reach Q/OT target temperature annealing

The target comparison changes the temperature in `Q / T` directly. The user
confirmed **10 to 0.25 over 400,000 environment steps**, exponentially, followed
by fixed temperature 0.25 through the end of a fresh 1,000,000-step run.

```text
T(t) = 10 * (0.25 / 10) ** min(t / 400000, 1)
candidate weights = softmax(Q / T(t) - 0.1 * log(g))
```

`g` is the truncated Gaussian KDE defined from 16 actor centers before drawing
four random candidates per component. Its std remains 0.2 and clip radius 0.5;
there is no `proposal_schedule` in this configuration. The 16 x 64 OT, no-anchor
setting, density beta 0.1, Sinkhorn epsilon 0.05 and 30 iterations, argmax target,
pointwise MSE, TD noise 0.2 / 0.5 and DIME critic all match the baseline.
The existing actor gradient calculation and critic loss are unchanged.

| Environment step | Q/OT temperature |
| ---: | ---: |
| 0 | 10.0000 |
| 80,000 | 4.7818 |
| 160,000 | 2.2865 |
| 240,000 | 1.0934 |
| 320,000 | 0.5228 |
| 400,000 through 1,000,000 | 0.2500 |

The schedule clock includes the 5,000-step random-action warmup. Thus the first
actor/critic training call at environment step 5,001 receives approximately
9.549 as its target temperature. UTD remains 2; the schedule is evaluated by
environment steps, not gradient updates. There is no stability detector or
initial hold period. `train/temperature` records the actual Q temperature;
`train/proposal_temperature=1`, `train/proposal_std=0.2` and
`train/proposal_clip=0.5` verify the fixed candidate generator.

## Basis for the starting value

The user supplied a DQS excerpt describing exponential annealing from 10 to 1
over 250k training steps, then explicitly selected the different endpoint 0.25
and a 400k **environment-step** decay period for this comparison. Using the same
step unit, matching the excerpt's logarithmic decay rate would take approximately
400,515 steps to reach 0.25; 400k is the selected rounded duration. This is a
chosen adaptation, not a claim that DQS uses these exact settings or step units.

Baseline Reach logs over environment steps 10k-50k have average ESS of 3.01
(seed 0) and 1.82 (seed 2), out of 64 candidates, at temperature 0.25. Average
maximum candidate weights are 0.725 and 0.810. The existing logged
counterfactual at temperature 1 raises ESS to 15.16 and 8.66 respectively.

An additional read-only audit froze each baseline's actor and critic at 50k,
used its 128 saved replay states, and generated three candidate replicates per
state. The candidates and KDE density are identical across temperatures. The
following statistics are over the resulting 384 state/candidate sets per seed;
the three replicates are not independent replay states.

| T | Seed 0 mean ESS | Seed 2 mean ESS | Seed 2 ESS 10th percentile |
| ---: | ---: | ---: | ---: |
| 0.25 | 4.35 | 1.88 | 1.00 |
| 2 | 36.83 | 28.99 | 3.06 |
| 4 | 47.65 | 43.00 | 8.32 |
| 8 | 52.00 | 49.64 | 25.24 |
| 16 | 53.54 | 52.48 | 44.14 |

These results motivate a high starting temperature when the aim is to weaken
early Q concentration. The user selected 10 after reviewing this evidence.
Counterfactual ESS does not predict the return of a newly trained policy, and
maximizing ESS is not the objective. The audit also compared 5,001/100k/250k
checkpoints on the same saved 50k states; the 5,001 result is therefore an
off-distribution probe, not the actual early training state distribution.

With density beta 0.1, the idealized weighted candidate density remains
proportional to `exp(Q / T) * g^0.9` on proposal support. This is partial density
correction; high temperature does not produce a uniform action distribution.
The candidate generator can still limit coverage. The finite OT and argmax
distillation do not guarantee exact sampling of a Boltzmann distribution.

## Matched runs and preserved comparison

Reach seed 0 runs on GPU 2 and seed 2 on GPU 3. These seeds were selected by the
lowest mean returns over the final 100 evaluation episodes of the three
completed baseline seeds; see the [selection table](REACH_PROPOSAL_ANNEAL.md).
All baseline learning/evaluation settings are inherited, including 1M total
steps, eval every 5k with 10 stochastic episodes, diagnostics every 5k and
checkpoints every 50k. The baseline `critic-dime-no-anchor` worktree and the
object-hold workers on GPUs 0/1 are unaffected.

The earlier candidate-width runs were deliberately stopped to switch to this
comparison. Seed 0 logged through step 16,971 and seed 2 through 16,909;
both retain evaluations through 15k and actor/critic checkpoints at 5,001.
Those logs/configs/checkpoints remain in their original output directories and
W&B runs `hg8mizrf` / `k7rlegyz`. They are partial runs, not completed baselines
and not initialization checkpoints for the new target-temperature runs.

## Launch and validation

```bash
bash scripts/run_reach_target_anneal.sh --list
CUDA_VISIBLE_DEVICES=2 bash scripts/run_reach_target_anneal.sh --task 0
CUDA_VISIBLE_DEVICES=3 bash scripts/run_reach_target_anneal.sh --task 1
```

Use `/workspace/.venv-optiq-no-anchor` and the ignored `.env`. The managed setup
is in `scripts/supervisor_reach_target.conf.example`; it does not automatically
restart completed jobs. Runs save to `outputs/optiq_dime_reach_target_anneal`
and a distinct target-temperature group in W&B project `optiq_dime_no_anchor`.
The run configuration, Git commit, packages and GPU assignment are recorded.

On 2026-09-08, all **64 tests** passed, including baseline configuration equality,
schedule endpoints/geometric decay/warmup clock, invalid settings, and actual
training calls with changing Q temperature and fixed KDE, TD noise, beta and
Sinkhorn epsilon. A full-network Reach GPU/JIT smoke run performed 16 updates
over 12 environment steps with a shortened schedule, reached and held 0.25,
and saved W&B artifacts:
[validation run](https://wandb.ai/sae_project/optiq_dime_no_anchor/runs/vmr97xpw).
After the user selected the final 400k duration, all 15 target-temperature tests
were rerun and passed with that resolved configuration.
