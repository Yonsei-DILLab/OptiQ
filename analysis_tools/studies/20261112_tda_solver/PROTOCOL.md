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

## First repair candidate and CPU work while GPUs are occupied

The constant-capture JIT revision also failed with the same error. Run
fixture_probe.py to compare 1-, 8- and original 64-state extraction batches,
without weakening the restoration gate. Preserve all probe results.

The supplied N=1024 Hopper seed3/state6 run reproduced 5 far assignments but
no negative persistence. The repaired candidate uses fp64 ascent for every
sample, explicit dimensionless projected-gradient convergence (1e-8), verified
stationarity (1e-7) and negative precision-scaled curvature (<-1e-7). Candidate
deduplication is performed only after convergence (L_inf 1e-7), and assignment
must agree within 1e-6 and within 1e-7 nats in density. These are numerical
tolerances, not a definition of a persistent mode. No failed candidate or
distant assignment is silently accepted. A state with unresolved points fails.
The mixture normalization is corrected; the legacy -(D-1)log(N) constant
does not explain its negative persistence at fixed N.

Run independent density/autodiff/normalization, unimodal, bimodal 1D-root and
separable-SAC filtration tests before real cases. First validate the first 512
saved sample rows of the supplied N=1024 problem case on CPU, then its full
M=32768 actions. The 512-row run is a numerical smoke test only. GPU 0/1 may
be used once available; do not interrupt other users' processes. Keep the
legacy midpoint saddle approximation during this controlled solver comparison;
do not claim continuous-edge certification from midpoint checks.

Batch-size probe: 1/8/64-state outputs had identical discrepancies for all five
seeds, so batching did not explain restoration. Probe equivalent BatchRenorm
arithmetic orders (left-associated, combined scale, division, affine) and a
float64 reference. These are diagnostic implementations of the same real-valued
equations; record every variant, do not select a changed policy to fit fixtures.

The left-associated formula with JAX parameter/statistic leaves and one JIT
graph reproduced all first-64-state fixture outputs exactly for all five
seeds (zero maximum error). The prior reconstruction used NumPy leaves and
an additional nested JIT boundary. Restore the Flax leaf types and one graph,
retain the original formula, and expand the final gate to all 512 stored
states per seed before generating the N=32768 mixtures. No gate relaxation.

All 512 states in all five Hopper fixtures now reproduce exactly (mu, log_std,
latent and alternate-batch errors all zero). The six N=32768 mixtures have
been generated. Run run_validation_queue.py on GPU0 for case indices 0,1,2
and GPU1 for indices 3,4,5, with independent queues. Each runs legacy then
verified analysis on exactly the same saved CUDA-generated M=32768 actions.
Do not start N/M convergence trials until all six validation cases pass.

## Phase C pilot implementation

Use convergence.json for the predeclared six-case 2x2 pilot. extract_convergence
checks all six original-N/M pass files before writing any new inputs. Generate
one N=65536 latent bank per action dimension with key 12345; N=32768 uses its
prefix. This differs from the original shape-generated N=32768 bank and must
not be confused with a pure M-change relative to the original table.

For each state generate 65536 component indices and conditional uniforms once
using CUDA seed 271828+case_index. Reduce indices modulo N (both N are powers
of two) to couple the N conditions; all marginal samples remain uniform-weight
mixture samples. M=32768 is the first half of the M=65536 bank. Run M=65536
first, then reuse its per-point converged endpoints for M=32768, verifying
input/sample hashes, residuals and ascent density again. Rebuild the mode
catalog, exploration points, graph and counts separately for each M. A larger
M's additional modes are never inserted into the smaller M catalog.

Queue GPU0 case indices 0,1,2,4 and GPU1 indices 3,5. Preserve every outcome,
including failures. Report exact tau=.5,1,2 and all four cutoffs for each pilot
state; do not present pilot averages as a new five-seed result.

## GPU occupancy handoff requested by user

User explicitly requested continuous GPU 0/1 utilization between TDA jobs on
2026-11-12. tda_executor.py runs tda_runtime.py while its queue is empty,
terminates only its own reservation child before a committed experiment,
and immediately restores the reservation after completion/failure. It does
not terminate other users' jobs. Keep source commit, child PIDs, commands and
timestamps in status/job sidecars. The reservation is an explicitly requested
resource-holding workload, not a scientific result or a training experiment.

## Final candidate-root verification before the N/M gate

All six initial runs had no unresolved assignments or negative persistence.
Inspection found one Humanoid control peak represented twice, separated by
1.585e-7 despite equal density, because both approximate endpoints met the
1e-8 gradient tolerance but differed by more than the 1e-7 equality tolerance.
Its K(.5/1/2) was unchanged (the duplicate had zero persistence). Refine only
already-converged candidate maxima by local fp64 Newton steps (goal scaled
gradient 1e-11; final acceptance 1e-10), then deduplicate at the unchanged
1e-7 coordinate tolerance. Reject nonnegative curvature or a Newton step >1e-3
instead of jumping. Sample ascent and its tolerances are unchanged.

Revalidate all six cases under validation-final-gpu0/1 by reusing the identical
converged endpoints with verified hashes and residuals, rebuilding all mode
catalogs and graphs. Preserve the original endpoint archives and write derived
metadata-bearing caches alongside the new results. Add a regression fixture
for two converged approximations of one root. Gate the N/M pilot on these final
six results, not the earlier unpolished candidate catalogs.

The final six validations passed under source 215d04b, and the N/M pilot was
launched from that immutable revision. Subsequent analysis-only additions do
not change that source attribution. inspect_stopping.py examines the saved
legacy endpoints with one fp32/fp64 EM step and continues a predetermined
subset (16 largest density violations, first 16 far, first 16 overall) in fp64
on CPU. This distinguishes premature stopping from the mode-catalog issue;
it does not replace or filter the complete paired validation results.
