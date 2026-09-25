# Frozen teacher Monte Carlo diagnostic

The active reward/hyperparameter goal requires successful multimodal retention,
which remains unproved on v3/v4. The fixed64 and env32 screens failed; reducing
conditional sigma also did not improve early acquisition. Before another
training setting, measure the existing teacher's finite-candidate uncertainty.
This diagnostic is not a new training run or a change to N/M defaults.

Use the completed v3 normalized-geodesic/T3/teacherfloor1 checkpoint on180
(d25930197ee9ba060a3aa84c3b6fea419dfe856d) and v4 original-geodesic/T1/floor.5
checkpoint on199 (555bb7e3c0101ee939545cc35d4d0729291781ec), both258304 steps.
Preserve current sigma-screen training, all frozen files and all primary results.

Select eight actual replay observations: nearest samples to seven specified XY
anchors including the origin and both sides, plus one seeded uniform replay
sample. Save indices, requested/actual XY and discrepancy; an unvisited anchor
must never be presented as a visited state. This is a small diagnostic cohort,
not an unbiased estimate over the complete visitation distribution.

For each of eight independent latent clouds, keep the same64 actor components
while independently drawing64/256/1024/4096 candidate actions from the unchanged
conditional mixture and teacher floor. Recompute exactly the same log proposal
density, online twin-mean Q, weights softmax(Q/T-logq), and Direct GMM NLL.
Compare ESS, largest weight, weighted action/Q estimates and output gradients
with a separate4096-candidate Monte Carlo reference. The4096-versus4096 result
shows reference uncertainty; it is not a ground-truth expectation. Output
gradients are with respect to component means/log sigmas, not network parameters.
Require the64-candidate result to match the existing frozen teacher_probe.

Load the full checkpoint with SHA256 and exact parameter/optimizer readback;
use explicit diagnostic RNG keys and prove model/optimizer/RNG/checkpoint
preservation. No optimizer step, environment step, resume, external DACER noise,
NovelD, alternate target Q or new policy sampler. All nine pinned computational
files stay identical. Raw candidate/weight/gradient arrays and JSON provenance
are retained separately from training results. Candidate actions are not path
labels: this cannot establish trajectory multimodality or a causal remedy.

Run one bounded nice19 CPU-only supervisor job on each5090 host, four CPU cores,
autostartfalse/autorestartfalse. No4090, GPU allocation, automatic retry or
follow-up training. Commit/push/share/freeze reporting source before execution;
preserve original training-source hashes in all results. Review this evidence
before selecting any candidate-count training ablation.
