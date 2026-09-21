# Reduced non-stationary queue, 2026-09-19

User explicitly authorized preserving completed results and checkpointing/stopping excluded conditions. Previous arrays2290511/2290512 were signalled once; all20 checkpoints were verified after exit. Do not restart those arrays or the old8528-task queue.

New numerical tasks2912 =832 prefixes+2080 comparisons; only256×1024 and512×512. The16 original source trajectories are imported read-only with their original numerical IDs, never recomputed/relabelled. Completed16×64 results are retained externally. Epsilon list contains six values. See numerical PROTOCOL.md and tasks.json.

Deploy the committed numerical tree to a fresh root and freeze its file hashes before adding any generated run/log metadata. Add DEPLOYMENT.json and read-only Q-stream symlinks. Commit SHA must precede validation/learning. Validate both normal and legacy-TD paths in4dimensions. Only then launch18fast+2Exact workers. Their queue contains no old size or removed epsilon task. Internal prefix/common-source dependencies only. Infrastructure failure exits without replacing numerical failures; full checkpoints save every200steps and on signal. Record SUBMISSIONS, early progress and dildata backup location. Never modify old numerical snapshots.
