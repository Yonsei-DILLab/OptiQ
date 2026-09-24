# Evaluation starts match training

Latest user clarification2026-09-24: v1 remains random because training is random;
v2/v3/v4 evaluations use the original fixed origin, posture and velocity.
Original upstream physics, training resets, policies and rewards stay unchanged.

Current source defaults and training registration profiles use eval_starts=upstream.
This means v1 XY uniform[-2,2] (original pose/velocity), v2-v4 original full state
at XY[0,0]. v2-v4 primary output is policy-fixed; v1 remains policy-natural.
Final evaluation avoids duplicate/random resets for v2-v4. Explicit historical
random/fixed options remain available, but are not the current defaults.
No new training or learner restart is authorized by this evaluation correction.

Frozen running processes cannot change their in-memory evaluator. Preserve them
and their outputs, labeling their v2-v4 random-start metrics supplementary. A
CPU-only supervisor watches their immutable evaluation checkpoints, evaluates the
latest at registration plus every newly saved checkpoint, and final full state.
40episodes/mode intermediate,100 final, original policy/native samplers. All v2-v4
episodes and both modes share the original complete physical state; action RNG
continues independently across episodes. Direct policy includes conditional sigma;
native OptiQ is mu-only; no external DACER noise or intrinsic reward in evaluation.
This correction changes primary reporting, not the actor or collected experience.

Active targets: new geodesic-no-step v2/v3/v4; old geodesic-no-bonus v3/v4 on199;
old v2 geodesic bonusON/OFF onvast1. v1 needs no reevaluation/reset override.
At most2 CPU evaluators/host,4CPUs each,nice10,noGPU. Failure holds evaluation
queue without affecting training. Do not automatically restart failed evaluation.
Watchers are supervisor-managed and exit after all target training/evaluations end.

The evaluation script imports model and environment code from each training commit,
restores all serialized policy/critic/optimizer values exactly, checks checkpoint
SHA before/after and unchanged parameters after rollout, verifies identical full
initial states and XY[0,0], and recomputes progress returns. Separate provenance
records evaluation versus training source. Never relabel existing random or
first-random-sample-fixed data as original-origin evaluation. Results live under
antmaze-matched-start-eval-20260924, separate from all training sources/logs.
