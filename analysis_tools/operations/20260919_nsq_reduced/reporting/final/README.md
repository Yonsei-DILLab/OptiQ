# Completed-only report

Frozen cutoff inventory: original research root scope_reduction_20260919/FROZEN_INVENTORY.json. Exclude original legacy closed-loop and use91cb9c3 results only.1823 final trajectories;758 prefixes;16source streams. No incomplete run is treated as final. Full runs/histogram inputs and final figures are on dildata outside Git.

collect_report.py merges eligible roots; build_completed_report.py reuses unchanged16×64 figures and updates all closed-loop panels and completed large-size results. Figure labels are adjusted to the actual matrix size in reporting code only. report.md is the reviewed narrative using generated CSV/findings.json. Render formulas with MathJax SVG, package data-URI images, and validate offline browser decoding. Final report has63 image occurrences and5 formulas; no external requests. CSV files preserve per-seed data and incomplete group sizes.

Rendering scripts use the documented local workspace tools; markdown-it/mathjax-full live in /tmp/optiq-offline-html/node_modules and build_offline.py supplies existing shared report styling. No numerical source is imported by report generation.

Numerical run provenance remains a08517ef9ed5fb8743252132997638b00feb5d75. This reporting commit does not relabel active experiment code. The new snapshot requires the existing external POT0.9.5 runtime; its per-file hashes are in RUNTIME_DEPENDENCIES.json on central storage. This dependency is not part of the source archive and must be restored on future deployments before multidimensional Exact OT validation.
