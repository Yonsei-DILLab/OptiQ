# Latest heejoon 3-mode reproduction and initialization control

User requested latest heejoon, N=M=64, batch 32, and asked which initialization succeeds.
Upstream source is frozen at 6a06acabf03005d1a15edb646f59d6278821e98e using its existing source freezer.
Use the unchanged `experiments.gmm_mode_gradient_batch32.core.engine(64,64,'baseline')`.
Seeds 0,1,2,3; 20,000 updates initially. Resume to 100,000 only if needed to distinguish delay from persistent failure.
Pair default mean-head variance scale 1e-4 with 1.0; all other parameters, initial sigma .5, Adam3e-4,
teacher density correction, fresh normal latents, RNG construction and batch semantics stay unchanged.
The large-init actor uses the same architecture and initialization seed, changes only the mean-head initializer.
For equal seeds both variants use the same training RNG keys, but actions naturally change with learned parameters.
Evaluation matches upstream: fresh policy samples 32768 and a fixed independent bank of 2048 latents.
No model repair, no clipping, no optimizer changes, no fitting to known mode labels.
Omit costly counterfactual gradient-routing diagnostics (which do not update training state).
Save checkpoints, raw samples, latent-conditioned parameters, mode mass and histogram TV.
Evaluate density by averaging conditional tanh-Gaussian densities, with no KDE.
Record source and audit commits. Keep results outside Git. Do not push generated results.

Original validation failed on exact float32 equality across JIT boundaries at a max absolute error
of 1.12e-10. A focused baseline validation at 64x64 checks explicit 32-group gradient averaging,
microbatch 1/4/32, one Adam step, matching RNG and single-group gradient at atol3e-6/rtol3e-4.
Run training only after that baseline validation passes. Preserve the original failure in the report.
