# Humanoid fixed64 Direct GMM campaign

Humanoid-v4; seeds 0-3; 1M environment steps each; no early stopping. v5 Direct GMM NLL, T=0.1, N=M=64, fixed latent codebook seed20260911 (64x17). Preserve RL defaults: actor/critic256x2, batch256, UTD1, Adam3e-4, mean initialization1e-4, warmup5000, plainTD.

Paired mu-only evaluation retains the v5 independent policy RNG and paired environment seeds: fixed_z is codebook[0], stochastic_z uniformly draws from the SAME training codebook per action. No newly sampled Gaussian latent support; epsilon=0 in both modes. For finite policies eval/mean_reward and final_eval_return alias stochastic_z. Continuous policies preserve zero_z/stochastic_z and zero_z alias unchanged.

Commit code/config/protocol before launch. Launch from frozen clean source on vast-heechan-180 GPU0-3 with supervisor and GPU locks. Fresh output paths/IDs; W&B OptiQ/v5-heechan-gmm explicitly. Existing Ant workers/frozen source remain unchanged.
