#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${OPTIQ_VENV:-$(dirname "$repo_root")/.venv-optiq-no-anchor}"
command -v uv >/dev/null || { echo "Install uv first" >&2; exit 1; }
if [[ ! -x "$venv_path/bin/python" ]]; then
  uv venv --python 3.11 "$venv_path"
fi
uv pip sync --python "$venv_path/bin/python" "$repo_root/requirements-no-anchor.lock" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
printf 'Environment ready: %s\n' "$venv_path"
