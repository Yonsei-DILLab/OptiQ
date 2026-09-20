# Quick three-mode Direct GMM illustration
Use unchanged heejoon experiments.gmm_gradient_interference.core imports.
Static 1D target: equal modes at -0.6, 0, 0.6 with Gaussian std .1, truncated to [-1,1]. Q=.25 log f, temperature .25.
N=M=2048, seeds 0,1,2,3; 5000 updates per seed; original 256x256 actor and Adam3e-4, learned sigma, no OT.
Evaluate at 0,100,500,1000,2000,5000. Independent fixed evaluation RNG, 32768 samples, 256-bin raw histogram, no KDE. Basin boundaries -.3,.3; coverage of basins is distinct from matching narrow peaks. Specialist latent means conditional basin probability >=.8.
This is a quick fixed-Q visualization, not a full training benchmark. Preserve exact source commit before running. Results outside git; copy artifacts to local machine.
Runtime: existing /home/heechan/.venv-optiq-mujoco, GPU 0-3, JAX float32 default precision. Existing frozen core is unchanged.
Run launch.py from repository root with output directory argument.
