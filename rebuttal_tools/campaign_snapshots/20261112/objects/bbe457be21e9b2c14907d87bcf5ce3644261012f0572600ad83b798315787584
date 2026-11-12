#!/usr/bin/env bash
set -euo pipefail
root=/workspace/heechan-adroit
mkdir -p "$root"
if [[ ! -x "$root/venv/bin/python" ]]; then
  uv venv --python /workspace/iBOLT/.venv/bin/python "$root/venv"
fi
uv pip install --python "$root/venv/bin/python" -r "$root/source/adroit_rebuttal/requirements.txt"
uv pip check --python "$root/venv/bin/python"
uv pip freeze --python "$root/venv/bin/python" > "$root/requirements.lock.txt"
