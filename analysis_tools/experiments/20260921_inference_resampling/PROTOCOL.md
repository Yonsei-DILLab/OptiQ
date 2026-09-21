Evaluate completed T=.25,beta1,DACER=true runs: HalfCheetah seeds0..4 and Ant seeds0..3.
Use the final1M actor AND current critic checkpoint, each run's frozen source/config.
No learning, no checkpoint mutation. Evaluation-only code committed before running.

Three modes: original one-mu, 64 fresh mu candidates weighted by exp(Q/.25),
and the same64 candidate mechanism with weights exp(Q/.25)/estimated_mu_density.
Use categorical sampling, never argmax. Use the same mean-of-two Q reduction as
training. Center log weights before softmax. Conditional sigma and DACER noise
are absent from ALL three modes.

The mu distribution is implicit. Estimate its density with256 separate Gaussian
latent mu outputs, diagonal Scott bandwidth per coordinate, floor.001 normalized
action units, and normalized box-truncated Gaussian kernels. The reference draw
is independent of candidate generation and selection. This is approximate IS,
not exact full-policy GMM correction. No claim of closer Boltzmann distribution
from return improvement alone. Q-only weighting targets q_mu(a)*exp(Q/T), not
the bare exp(Q/T) target. KDE bias, high-dimensional error and critic exploitation
can affect outcomes. Fixed hyperparameters; no performance-driven changes.

20 fresh reset seeds and20 policy seeds shared across all modes/trainingseeds.
Vectorize episodes; accumulate only each environment's first episode. Use
Gymnasium-v4 unchanged environment/time limit1000; no observation/reward normalization.
Report within-training-seed paired episode differences, cross-training-seed
mean/SD, per-seed results, ESS and maximum weight. Do not count episodes as
independent training seeds. This is final-checkpoint reevaluation, not the old
last100k average. CPU/GPU/library differences from original training are recorded.

Preflight: baseline parity with existing sample_action(no conditional noise),
sigma-head invariance in all three modes, finite normalized actions/weights,
uniform and additive-Q-shift weight tests, frozen input checksums before/after.
Keep original logs and artifacts unchanged. Hold GPU locks; use free0,1,3 on199,
leave unrelated GPU2 work untouched. No automatic retry or hidden fallback.

User follow-up after the three-way comparison: add inference-only best-of64.
Use the identical9 final actor/critic pairs and same20 reset/policy seeds. Draw
64 fresh mu actions exactly as mu_q64 does, then select argmax of the same
current twin-Q mean. No sigma or DACER noise; no learned parameters changed.
Training temperature remains.25; argmax ranking itself has no temperature.
Use separate campaign/manifest/output for this follow-up and preserve the
original three-way result. The shared evaluator now accepts an explicit mode
list; original frozen evaluation3524ea5 remains unchanged. Extra preflight
checks best-of64 has selection ESS=1 and nonnegative selected-Q minus mean-Q.
