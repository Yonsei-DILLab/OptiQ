#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Vast's root-owned uv defaults are not writable by an ordinary SSH user.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR_OVERRIDE:-$HOME/.local/share/uv/python}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR_OVERRIDE:-$HOME/.local/bin}"
export UV_CACHE_DIR="${UV_CACHE_DIR_OVERRIDE:-$HOME/.cache/uv}"
export UV_NO_CACHE=false
uv python install 3.11
if [[ ! -d .venv ]]; then
    uv venv --python 3.11 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
profile=requirements-blackwell.txt
if [[ -f requirements-blackwell.lock.txt ]]; then
    profile=requirements-blackwell.lock.txt
fi
uv pip install --python .venv/bin/python -r "$profile"
uv pip check --python .venv/bin/python
printf 'Ready. Run: source scripts/activate-experiments.sh\n'
