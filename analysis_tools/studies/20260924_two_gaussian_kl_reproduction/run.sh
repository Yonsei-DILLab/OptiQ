#!/usr/bin/env bash
set -euo pipefail
# Usage: PYTHON=/path/to/python bash run.sh /absolute/path/to/new-run
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$src" rev-parse --show-toplevel)
rel=${src#"$repo"/}
commit=$(git -C "$repo" rev-parse HEAD)
if [[ $(git -C "$repo" branch --show-current) != heejoon ]]; then
  echo 'Run from the heejoon branch.' >&2; exit 1
fi
if [[ -n $(git -C "$repo" status --porcelain -- "$rel") ]]; then
  echo 'Commit this study before launching.' >&2; exit 1
fi
out=${1:?Supply a fresh absolute output directory}
[[ "$out" = /* ]] || { echo 'Output path must be absolute.' >&2; exit 1; }
mkdir "$out"
mkdir "$out/source"
git -C "$repo" archive "$commit:$rel" | tar -xf - -C "$out/source"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/optiq-two-gaussian-matplotlib}"
"${PYTHON:-python3}" "$out/source/reproduce.py" --output "$out/results" \
  --source-commit "$commit" --run-id "$(basename "$out")"
