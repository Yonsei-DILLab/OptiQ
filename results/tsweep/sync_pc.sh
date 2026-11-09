#!/usr/bin/env bash
# Copy the local WSL runs' eval dirs to the 5090 (~/OptiQ/outputs_pc) every 60 s for the live page.
# Run on Windows (Git Bash): bash results/tsweep/sync_pc.sh
# Packing happens in a WSL-side script (~/OptiQ/pack_evals.sh); quotes inside wsl.exe arguments get mangled.
while true; do
  wsl.exe -d Ubuntu -- bash -lc 'bash ~/OptiQ/pack_evals.sh' 2>/dev/null \
    | ssh -o BatchMode=yes -o ConnectTimeout=20 vast-5090 'mkdir -p ~/OptiQ/outputs_pc && tar xf - -C ~/OptiQ/outputs_pc' 2>/dev/null
  sleep 60
done
