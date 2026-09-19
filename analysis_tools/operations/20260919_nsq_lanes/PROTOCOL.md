# Separate slow exact transport from fast non-stationary comparisons

Authorized September19 after runtime diagnosis. Operational change only; preserve
all6,080 final comparisons,8,528 nodes, numerical sources, seeds, temperatures,
updates, metrics and checkpoints. Original numerical commit cc11f537af330e23e1cc77cb94a9426660b55ebb;
legacy closed-loop correction91cb9c3d32f9d3c7f43ff886d34b6199080be811.

- Fast lane:18 GPU workers, all GMM/Sinkhorn/legacy,1D Exact OT, and small16×64 Exact OT.
- Slow lane:2 GPU workers, Exact OT with dim>1 and N×M>1024 only.
- One GPU,two CPUs,32GB per worker. No numerical solver/thread/precision change.
- The two revisions are combined by logical task identity; original legacy closed
  runs are excluded and corrected ones replace them. No duplicate comparisons.
- Resume interrupted tasks first. Among new ready tasks, prioritize source,
  mass/split branches, replay, prefixes, then closed-loop runs. Preserve original
  size/dimension/seed preference within these groups. Only each task's own prefix
  and critic-source dependency is enforced, never all-method completion gates.
- A shared operational claim lock and valid Slurm snapshots prevent duplicate
  leases. Completed and failed tasks are never silently rerun. Do not treat a
  failed Slurm query as an empty scheduler. All source validation gates remain.

Before launch, commit/push these operational scripts. Record the full operational
SHA beside both numerical deployment records and submitted job IDs. Freeze the
ops directory; do not edit running numerical roots or relabel their checkpoints.

Drain only nsq-resume-main and nsq-resume-legacy workers. Cancel pending automatic
replacements first, then signal TERM to the batch shells of live workers. Their
existing trap forwards the stop to the numerical child, which finishes its
current update and saves actor, optimizer, RNG, critic and replay state. Do not
kill a live numerical process or delete its lease/checkpoint. Verify all old
workers exited and every interrupted run has a matching readable checkpoint
whose saved step is at least its pre-stop progress. Then submit fast0-17%18 and
exact0-1%2 with no cross-lane dependencies. Unrelated jobs and held studies stay
untouched. New lanes never overlap old workers.

Early verification: ensure both lane types actually start on GPUs, inspect
checkpoint resume positions and new mass/split progress, verify unchanged source
manifests, and confirm dildata receives ops/provenance/checkpoints. Stop after
early healthy execution; do not wait for the full campaign.
