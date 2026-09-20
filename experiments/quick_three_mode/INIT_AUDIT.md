# User-requested T1, batch32 audit with mean initialization control
Q=log f, T=1 explicitly, fixed z64, N=M=64, batch32 independent fresh teacher clouds, 100000 updates, seeds0/1.
Original: same model, Adam3e-4, teacher sigma floor .05, initial sigma .5, mu output variance init1e-4, latent skip0, original fixed bank.
init1: only mu output variance init changes from1e-4 to1. Same network architecture, no latent skip.
Original T.25 toy used Q=.25 log f; target ratio Q/T remains log f. Outputs and earlier runs preserved. No screenshot-replication claim without exact source.
