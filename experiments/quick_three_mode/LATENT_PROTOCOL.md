# Fixed versus fresh z matched comparison
N=M64, batch32, T1/Qlogf, mean-head init variance scale1, Adam3e-4, 100000 updates, seeds0/1. Same architecture, initialization and proposal random-key sequence.
Fixed: all64 latents from original bank PRNGKey70000+seed every step, broadcast to32 independent teacher clouds.
Fresh: each teacher cloud draws64 independent N(0,1) latents each step, using fold_in(proposal_key,429); conditional mixture and original loss computed with the same current z.
Evaluate both at all registered steps with identical fixed-bank and fresh-normal-z draws. Primary deployed-policy TV is fixed-bank TV for fixed method and fresh-z TV for fresh method. Also report fresh-z TV for both as common evaluation. Distinguish finite64 prior from continuous Gaussian prior, not a pure variance comparison over an identical policy class.
First observed TV<.1 checkpoint is a diagnostic threshold, not mathematical convergence. No wall-time claims; compare actor update counts. Preserve original code and results. Commit before launch.
