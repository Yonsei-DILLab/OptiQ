# DIPO implementation audit — 2026-09-26

Running learner source: `4b46c1bfb1f8bb9d93ec0b15e891edf5d9de8c6f`.
Upstream: BellmanTimeHut/DIPO, commit `c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc`.

All six pinned upstream agent source files were SHA-256 compared against a fresh clone of the official repository; all matched exactly (DiPo, diffusion, model, helpers, replay_memory, vae). The maze adapter imports the actual DiPo learner and calls its train method. Training uses reverse diffusion noise; evaluation uses fresh initial Gaussian noise and reverse noise disabled, matching upstream main.py. Terminal masks are gamma*(1-done), consistent with the original loop.

## Meaningful behavioral difference

The vector adapter overrides DiffusionMemory.replace with advanced-index assignment. Upstream calls np.copyto(self.best_actions[idxs], best_actions), which writes into the temporary array produced by advanced indexing. A CPU test using the exact upstream class confirmed that stored [0.1,0.2] remains [0.1,0.2] after upstream replace([0], [[0.8,0.9]]), whereas the current adapter stores [0.8,0.9]. Therefore the adapter repeatedly refines previously improved actions, while the pinned upstream execution re-starts sampled actions from their stored behavior actions. This was implemented as an intended bug fix, but is a substantive difference from literal upstream behavior. These experiments must be described as an official DIPO core with an action-memory writeback modification, not an unmodified upstream reproduction.

This difference is a plausible contributor to exploitation of critic errors, but no controlled counterfactual training has established it as the cause of the observed collapse. The verified Simple failure is success100% at200704,0% at401408 and600064, along with impossible critic values above the maximum episodic return100. See checkpoint_q_audit.json. Actor and critic checkpoints are finite and optimizer counts increase. Training TD losses and reward visitation were not logged, limiting identification of the first critic-error trigger.

The user-approved fast data profile also differs from native DIPO: 2048 environments,batch4096,128 learner updates per collection,8192 total warmup (4 steps per environment). The PointMaze OptiQ profile has the same UTD1/16 and batch4096; UTD alone is not an established cause. No running source, configuration, job or checkpoint was changed by this read-only audit.
