#!/usr/bin/env bash
# Pull the temperature-sweep evals (Humanoid, HalfCheetah) from both servers, plot on the 5090, bring PNGs here.
# Usage: bash results/tsweep/show.sh   -> results/tsweep/tsweep_curves.png, tsweep_halfcheetah.png
# Humanoid references come from /tmp/cqplot/cq (results/crossq_compare/show.sh fills it).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
$SSH -n vast-5090 'rm -rf /tmp/tsplot && mkdir -p /tmp/tsplot/ts /tmp/tsplot/hcref' 2>/dev/null
for h in vast-4090 vast-5090; do
  $SSH -n $h 'cd ~/OptiQ/outputs_tsweep && tar cf - $(ls -d humanoid-* halfcheetah-* 2>/dev/null | sed "s|$|/eval|")' 2>/dev/null \
    | $SSH vast-5090 'tar xf - -C /tmp/tsplot/ts' 2>/dev/null
  # HalfCheetah references: SBX CrossQ and paper-critic iBOLT T=0.25 (host prefix keeps duplicate seeds apart)
  $SSH -n $h 'cd ~/OptiQ/outputs && tar cf - $(ls -d halfcheetah-crossq-q* halfcheetah-ibolt-T0.25-s* 2>/dev/null | sed "s|$|/eval|")' 2>/dev/null \
    | $SSH vast-5090 "mkdir -p /tmp/tsplot/hcref/$h && tar xf - -C /tmp/tsplot/hcref/$h && cd /tmp/tsplot/hcref && for d in $h/*; do mv \$d ./\$(basename \$d)-$h; done && rmdir $h" 2>/dev/null
done
scp -q "$HERE/plot_tsweep.py" vast-5090:/tmp/tsplot/
$SSH -n vast-5090 'cd /tmp/tsplot && echo "== humanoid" && ~/OptiQ/.venv/bin/python plot_tsweep.py ts /tmp/cqplot/cq tsweep_curves.png humanoid && echo "== halfcheetah" && ~/OptiQ/.venv/bin/python plot_tsweep.py ts hcref tsweep_halfcheetah.png halfcheetah' 2>/dev/null
scp -q vast-5090:/tmp/tsplot/tsweep_curves.png "$HERE/tsweep_curves.png"
scp -q vast-5090:/tmp/tsplot/tsweep_halfcheetah.png "$HERE/tsweep_halfcheetah.png"
echo "PNG: $HERE/tsweep_curves.png $HERE/tsweep_halfcheetah.png"
