#!/bin/sh
set -eu
# Usage: PYTHON=/path/to/python sh run.sh /input/core /output/run FULL_SOURCE_COMMIT
task_source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MPLCONFIGDIR="$2/mpl-cache"
"${PYTHON:-python3}" "$task_source_dir/audit.py" --input "$1" --output "$2" --source-commit "$3"
