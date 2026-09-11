# DMC Dog: OptiQ actor with DIME critic

This is the historical DMC experiment design, not the current scalar-critic
checked-K64 v2 protocol. See [README.md](README.md) and the
[v2 reproduction guide](docs/v2/REPRODUCIBILITY.md) for the active default.

## Question

Does OptiQ's one-step implicit actor match or improve DIME on high-dimensional
DMC Dog tasks without diffusion training or iterative inference?

## Suite

- Tasks: `dog-run`, `dog-trot`, `dog-walk`, `dog-stand`
- Seeds: 1 and 2
- Training steps: 1,000,000 per run
- Total runs: 8
- Evaluation: stochastic policy, 10 episodes at step 1 and every 5,000 steps

## Held fixed from DIME

- Replay capacity 1,000,000; batch size 256; learning starts at 5,000
- UTD ratio 2 and one environment step per collection cycle
- Two 2048x2048 categorical critics with 101 atoms on [-200, 200]
- CrossQ concatenated current/next batch and batch renormalization
- Critic distribution-entropy coefficient 0.005
- Critic Adam learning rate 3e-4, beta1 0.5, beta2 0.999
- Discount 0.99 and DIME's mean categorical target projection
- Hard critic and target-actor updates (`tau=policy_tau=1.0`)

## Replaced by OptiQ

- One-step actor `a = mu(s, z)`, with a 256x256x256 GELU MLP
- 16 policy samples and 5 local proposals per sample, including the anchor
- Exact box-truncated Gaussian proposals, std 0.2 and local clip 0.5
- Proposal-density correction with a uniform-action reference
- Q temperature 0.25; Sinkhorn epsilon 0.05 for 30 iterations
- Argmax row assignment and squared-error actor distillation
- Actor Adam learning rate 3e-4, beta1 0.9, beta2 0.999
- Actor update on every gradient step
- One target-policy sample with truncated-Gaussian TD noise (std 0.2, clip 0.5)

## Deliberate entropy difference

DIME's maximum-entropy term uses diffusion path costs that do not exist for a
one-step implicit actor. The supplied OptiQ configuration sets maximum-entropy
alpha and actor entropy loss to zero, so the hybrid also uses zero. DIME's
critic distribution-entropy regularizer (0.005) remains unchanged. This must be
reported as a deliberate algorithmic difference, not described as a perfectly
isolated actor-architecture ablation.
