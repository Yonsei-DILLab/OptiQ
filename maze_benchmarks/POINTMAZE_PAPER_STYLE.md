# Confirmed PointMaze figure style — 2026-09-26

Use this style for future experiment-result replacements. The user approved
the red-failure version after correcting the goal border to lie inside the box.

- Medium on the first row, Hard on the second. Sans-serif algorithm labels;
  name our method **iBOLT**, bold only that algorithm name.
- Successful trajectories: blue `#2065d1`; failed trajectories: red `#e74747`.
  Always keep failed episodes in the main result figure. The gray-failure and
  success-only variants are comparisons, not the selected default.
- Plot all 500 saved evaluation trajectories. Use width 1.8 points, alpha 0.2
  for the primary image and alpha 0.1 for the alternate.
- Goals reached at least once are blue; goals never reached are red. Determine
  this from saved terminal goal IDs, not proximity or trajectory appearance.
- White goal border: 0.7 points entirely INSIDE the original goal square
  (thinned from 1.1 points at the user's request).
  Clip the double-width stroke to the square after adding it to the axes.
  Never enlarge the goal or cover neighboring gray walls.
- No white trajectory stroke, halo, or underlay. No inactive optional-obstacle
  placeholders. Actual walls remain unchanged.
- No bottom "Figure 4: PointMaze" caption. Keep the outcome legend.
- iBOLT uses fresh random latent z and mu-only actions, conditional sigma OFF.
  Baselines retain their documented native sampling. Current selected iBOLT is
  T=5; a future explicitly selected run changes the input, not the visual style.
- Preserve input hashes, sampling labels, training provenance and truthful
  pending cells. Never substitute a different algorithm's implementation.

The defaults of `python -m maze_benchmarks.paper_pointmaze_figure` implement
this style. PNG is the user-facing deliverable; SVG is retained for publication.
Existing generated figures and raw evaluations are not overwritten.

Approved reference within the artifact directory:
`pointmaze_paper_figure4_t5/output/images/inset_goals_v2_red/figure4_pointmaze_6methods_alpha02_outcomes.png`.
