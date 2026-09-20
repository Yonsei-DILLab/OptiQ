#!/usr/bin/env bash
set -euo pipefail
root="$1"
cd "$root/repo"
exec python3 -m experiments.gmm_gradient_interference.queue --root "$root"
