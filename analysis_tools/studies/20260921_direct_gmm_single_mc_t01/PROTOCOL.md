# DirectGMM SingleQ MC64: temperature 0.1 priority extension

User request 2026-09-21: queue Humanoid-v4 and HalfCheetah-v4, seeds 0–3, ahead of all currently pending jobs. Do not interrupt active training.

## Numerical configuration

Copy the verified immutable numerical snapshot from heejoon commit `9ac88463efc44cf2a084458c322f6f6ae130d6f9`. Only actor temperature changes from 0.25 to 0.1; group/run labels and output paths distinguish this study. Same Direct GMM learned-sigma actor, reward-only single critic, full stochastic 64-action Monte Carlo TD backup, N=M=K=64, batch256, gamma0.99, target Polyak0.005, Adam3e-4, 5K warmup, 1M steps per run. Proposal sigma floor0.05 is teacher-only. No OT, twin-min or entropy reward.

Eight new independent runs: Humanoid seeds0,1,2,3 then HalfCheetah seeds0,1,2,3 as scheduling preferences, without completion dependencies. New runs precede the old T0.25 Ant pending seeds. Active T0.25 runs keep their PIDs, source commit, W&B run IDs and output paths.

## Queue handoff

The original dispatcher keeps pending work in memory. Place PAUSE_QUEUE in the old root before handoff. After source/config verification, the new dispatcher stops only the old scheduling process with SIGSTOP, retaining its terminal and training children. It adopts active workers by PID plus Linux process start identity and tracks actual process exit before freeing a GPU slot. Completed markers verify 1M steps. Training children are never signaled. Once all adopted original children have exited, terminate the suspended old dispatcher. Its pending work is managed by the new controller, with original numerical source and commit.

A single combined queue allows at most two workers per GPU across both studies, on four RTX3090 GPUs. All new and old pending runs are eligible; priority determines which is started in each free slot. Failure is recorded and does not block another environment. No automatic retraining of failed/deferred experiments. Existing outputs are never overwritten. Old STATUS.json is maintained with a handoff pointer to the new controller.

## Validation and provenance

Commit this exact config, immutable snapshot builder, launch commands, controller, and protocol to heejoon before registration/launch. Verify the base source hashes, assert numerical files unchanged, resolve all eight Hydra configurations, and check the existing online W&B project before controller takeover. A pure scheduler test verifies new-before-old priority, no oversubscription, correct same-GPU slots, and process identity matching. Prior three-environment GPU smoke remains applicable: the numerical implementation is unchanged and only temperature differs. No duplicate GPU smoke while all slots are occupied.

W&B entity OptiQ, project DirectGMM_heejoon, group 20260921_DirectGMM_SingleQ_MC64_N64_M64_T01. Full commit and commands are stored with each launch and source manifest. Source and run data stay under the existing dildata-backed remote legacy_monge tree; no credentials in Git or logs.
