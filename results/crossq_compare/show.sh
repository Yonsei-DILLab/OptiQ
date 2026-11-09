#!/usr/bin/env bash
# Pull the latest CrossQ-comparison evaluations from both servers, plot on the 5090
# server (this PC has no numpy), and bring the PNG back here.
# Usage: bash results/crossq_compare/show.sh      -> results/crossq_compare/crossq_curves.png
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REF="$HERE/../mujoco_3mode_s0-9/eval"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
$SSH -n vast-5090 'rm -rf /tmp/cqplot && mkdir -p /tmp/cqplot/cq /tmp/cqplot/ref' 2>/dev/null
for h in vast-4090 vast-5090; do
  $SSH -n $h 'cd ~/OptiQ/outputs && tar cf - $(ls -d *crossq* 2>/dev/null | sed "s|$|/eval|")' 2>/dev/null \
    | $SSH vast-5090 'tar xf - -C /tmp/cqplot/cq' 2>/dev/null
done
tar cf - -C "$REF" $(cd "$REF" && ls -d humanoid-* ant-* walker2d-*) | $SSH vast-5090 'tar xf - -C /tmp/cqplot/ref' 2>/dev/null
scp -q "$HERE/plot_crossq.py" vast-5090:/tmp/cqplot/
$SSH -n vast-5090 'cd /tmp/cqplot && ~/OptiQ/.venv/bin/python plot_crossq.py cq ref crossq_curves.png' 2>/dev/null
scp -q vast-5090:/tmp/cqplot/crossq_curves.png "$HERE/crossq_curves.png"
echo "PNG: $HERE/crossq_curves.png"
