# User-requested fixed z64 / M64 addition
Same target, actor, original Direct GMM loss/update, optimizer and 5000-update budget as quick protocol; four seeds 0..3.
Training latent bank z: 64 independent N(0,1) values from PRNGKey(70000+seed), frozen for all updates. Each update draws 64 IID mixture proposals from that fixed bank. Teacher weights detached; sigma floor .05.
Primary evaluation uniformly samples components from the same fixed bank plus fresh Gaussian noise. Save separate fresh-z policy evaluation to distinguish fixed-mixture fitting from generalization to unseen latent values. Evaluate the same six steps, 32768 actions, 256 bins. Save all 64 means/sigmas and latent identities for visual tracking.
This changes both latent resampling and N/M relative to the 2048 baseline; do not attribute differences to fixing z alone.
Launch: python experiments/quick_three_mode/launch_fixed64.py OUTPUT_DIRECTORY
