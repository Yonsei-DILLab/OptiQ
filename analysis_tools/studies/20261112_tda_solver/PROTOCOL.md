# TDA solver repair and convergence validation

Authorized 2026-11-12: diagnose and fix negative persistence and far assignments,
validate at unchanged N/M, then increase N/M. Use dilg-01 GPUs 0 and 1 only.
Do not stop or modify existing jobs. No policy training or checkpoint edits.

Commit every numerical source/configuration/launcher revision to heejoon before
execution. Keep immutable per-revision source and provenance with full SHA,
input hashes, runtime versions, job IDs and wall time. Canonical archive:
`dildata:/data1/heejoonorm/OptiQ/studies/20261112_tda_solver`.
Remote compute directory: `/home/heejoonorm/tda-solver-20261112`.

## Phase A: reconstruct and diagnose

Reconstruct inference from msgpack params and saved BatchRenorm statistics, using
JAX 0.4.33 CPU and the inference equations in the available CrossQ actor source.
Do not restore optimizer updates or roll out new states. Validate all five
Hopper checkpoints against the saved N=1024 fixtures, including fixed latent
arrays, exact states, and different inference batch sizes. Acceptance: maximum
absolute mu/log_std error <=1e-5. Use fixed statistics, epsilon=.001, approximate
GELU, tanh centers, and saved log_std bounds [-5,-1].

After that gate, reconstruct N=32768 with PRNGKey(12345), original nonpartitioned
JAX Threefry behavior and original state banks. Record that this is reconstructed
input, not a byte-identical supplied N=32768 mixture. Diagnostic cases selected
from the prior audit before reexecution:
- Hopper seed3 states6,28 (large far and most negative persistence)
- Hopper seed0 state5 (negative persistence outside seed3)
- Humanoid seed3 state21 (far=18210)
- controls Hopper seed1 state0, Humanoid seed0 state0.

Use upstream equations unmodified for baseline diagnosis, loading functions
without executing the upstream CLI. Replay the original sampling generator
advances for preceding state indices. Save sampled actions and ascent endpoints
so revised solvers use identical actions. Diagnose fp32/fp64 score discrepancies,
stationarity residuals, fallback acceptance, assignment distances, endpoint
density exceeding assigned mode height, basin-edge height exceeding either
incident mode, and negative persistence pairs. Preserve all failures.

## Phase B: repair and test

Write and commit changes after the Phase A evidence. Required invariants:
- correct mixture normalization; stable density/responsibility evaluation;
- explicit convergence and free-coordinate projected-gradient verification;
- negative curvature on the constrained tangent/free subspace;
- no forced acceptance when all candidates fail;
- no assignment of unresolved samples to a distant mode by default;
- assigned verified mode height >= endpoint density within numerical tolerance;
- graph edge filtration compatible with incident births, with errors diagnosed
  rather than silently clipping invalid persistence or dropping states;
- zero materially negative persistence and zero unresolved/far assignments on
  validation cases. Numerical tolerances must be explicit and scale-aware.

Validate analytic gradients/Hessians against independent derivatives, known
unimodal/multimodal mixtures and separable SAC reference, then rerun the selected
original cases at N=M=32768 using exactly the saved action samples. Report which
errors were reproduced and corrected; do not infer cause from a single symptom.
Finite graph saddle approximations still require sensitivity checks and are not
certified continuous-path minima solely because midpoints are checked.

## Phase C: convergence, only after validation

Separate density components N from action/search samples M. Start matched
controls/problem states at (N,M)=(32768,32768),(65536,32768),(32768,65536),
(65536,65536). Hold states/checkpoints fixed and retain all outcomes. For new
N comparisons generate a maximum-size latent bank and use prefixes; separately
retain the original shape-generated N=32768 bank for baseline reproduction.
Expand to N=131072 and M=131072/524288 only after measured cost and resolution
diagnostics justify it. This numerical validation dependency is not an
environment-completion dependency: independent Hopper/Humanoid tasks may run
concurrently once eligible. Do not gate one environment on the other.

Cutoffs 1%,0.1%,0.01%,none are postprocessing only. Evaluate exactly tau=.5,1,2
and a full curve; report mean/std across training seeds only when an entire
seed/state protocol is completed. Pilot cases are not a replacement final table.

## Phase A diagnostic amendment

The first Apple-arm64 CPU restore attempt (commit 76311eb) failed the 1e-5
fixture gate: seed0 max mu error 2.32e-5, log_std 3.15e-5; latent arrays matched
exactly. No N=32768 inputs were generated. Preserve that failed attempt.
Repeat the same verifier on Linux x86 CPU with original JAX 0.4.33, NumPy
1.26.4 and SciPy 1.11.4 before attributing the difference to architecture.
Independently diagnose the supplied Hopper N=1024 fixtures (seed3 state6 and
state28), at M=32768, on GPU1. These are exact supplied mixture parameters,
not substitutes for the original N=32768 result; label them as small-N
diagnostic cases. Use their original 64-state-bank indices to replay RNG.

The Linux x86 repeat had the same ~3e-5 fixture discrepancy, so it is not
explained by ARM alone. Test the original constant-capture JIT semantics:
upstream extract.py closes over actor variables before jitting, whereas the
first reconstruction supplied them as dynamic arguments. Keep the 1e-5 gate
unchanged; preserve failed outputs and do not generate new mixtures until it
passes.
