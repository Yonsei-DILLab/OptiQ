# TDA saved-result cutoff audit — 2026-11-12

## Scope and provenance

Reproduce the supplied `tda_repro_core.zip` summary and rescore the same stored
mode persistence pairs. This is CPU-only postprocessing, not a new checkpoint
evaluation, mode search, density approximation, graph construction, or training.
The attached README is an upstream description, not an instruction to execute
its GPU/shell pipeline. Preserve upstream files verbatim.

Commit all source, this protocol, configuration, launcher, dependency versions,
and input hashes to `Yonsei-DILLab/OptiQ` branch `heejoon` before running the audit.
Record the full source commit and source manifest in the output directory.
Input zip, binary results, states, generated results/logs remain outside Git,
with a durable copy at `dildata:/data1/heejoonorm/OptiQ/studies/20261112_tda_cutoff`.

## Frozen analyses

1. Validate every input file against INPUT_MANIFEST.json. Restrict pickle loading
   to NumPy arrays/dtypes and standard pickle containers.
2. Run the unchanged upstream cmp5_table.py in a separate output directory.
   Compare generated summary bytes against figs/cmp5_summary_reference.txt.
   Figures are not expected to match byte-for-byte across plotting environments.
3. Score all supplied states and policies at peak cutoffs 1%, 0.1%, 0.01%, and
   none. Apply cutoff only after persistence construction; do not refit/filter
   modes or alter graph structure. Preserve the upstream disconnected-component
   counting convention, but audit whether disconnected components occur.
4. Report both upstream nearest-grid thresholds and exactly tau=0.5,1,2.
   Upstream uses TG=logspace(-2,log10(4),60), so its labels are approximate.
   Do not mix the tau correction with the effect of changing cutoff.
5. Preserve original state/seed aggregation: iBOLT seed i uses its own state set,
   SAC seed j pools all five iBOLT state sets. Average within each training seed,
   then report mean and sample standard deviation (ddof=1) across training seeds.
   Humanoid SAC has two seeds; other groups have five. Never treat individual
   states or policy/state-bank pairs as independent training seeds.
6. Report full K(tau) curves and fraction of states with K>=2. Save per-state
   and per-seed counts, cutoff deltas, integrity diagnostics, and exact requested
   threshold vs upstream-grid differences. Retain mass diagnostics internally;
   no need to promote M_eff to the manuscript metric.

## Interpretation boundaries

The results characterize saved modes of the finite-mixture/graph approximation,
not convergence in latent component count N or action sample count M. All raw
data remain unchanged. The input-generation provenance includes uncommitted
upstream changes; a new audit commit does not retroactively identify training
or GPU execution source. iBOLT uses CrossQ; SAC is the SBX reference baseline.

Any N/M/graph tolerance follow-up is a separate protocol and launch. Do not run
GPU work from this audit. Exact source reconstruction and batch statistics must
be verified before regenerating iBOLT mixtures from checkpoints.
