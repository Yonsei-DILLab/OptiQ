#!/usr/bin/env bash
set -euo pipefail
# Usage: PYTHON=/path/to/python bash run_snis.sh /absolute/new-run /absolute/baseline/results
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$src" rev-parse --show-toplevel)
rel=${src#"$repo"/}
commit=$(git -C "$repo" rev-parse HEAD)
[[ $(git -C "$repo" branch --show-current) = heejoon ]] || { echo 'Use heejoon.' >&2; exit 1; }
[[ -z $(git -C "$repo" status --porcelain -- "$rel") ]] || { echo 'Commit before launch.' >&2; exit 1; }
out=${1:?Supply a fresh absolute output directory}
baseline=${2:?Supply the original baseline results directory}
[[ "$out" = /* && "$baseline" = /* ]] || { echo 'Use absolute paths.' >&2; exit 1; }
test -f "$baseline/snapshots.json"
test -f "$baseline/PROVENANCE.json"
mkdir "$out"
mkdir "$out/source" "$out/inputs"
git -C "$repo" archive "$commit:$rel" | tar -xf - -C "$out/source"
cp "$baseline/snapshots.json" "$out/inputs/baseline_snapshots.json"
cp "$baseline/PROVENANCE.json" "$out/inputs/baseline_provenance.json"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/optiq-two-gaussian-matplotlib}"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python3}" "$out/source/snis_column.py" --output "$out/results" \
  --baseline-inputs "$out/inputs" --source-commit "$commit" --run-id "$(basename "$out")"
