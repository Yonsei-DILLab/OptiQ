#!/usr/bin/env bash
# Vast supervisor wrapper; each worker runs its queue in the foreground.
set -eo pipefail
utils=/opt/supervisor-scripts/utils
if [[ -f "$utils/logging.sh" ]]; then
  . "$utils/logging.sh" ""
  . "$utils/environment.sh"
fi
set -u
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$repo_root/scripts/run_no_anchor.sh" "$@"
