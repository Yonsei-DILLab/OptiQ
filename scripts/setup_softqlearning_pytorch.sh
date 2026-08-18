#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="${1:-$repo_root/.third_party/softqlearning-pytorch}"
commit="5d6bd4105b276b431d89173910c78be8d54dbe95"
patch_file="$repo_root/third_party/softqlearning_pytorch_fair.patch"

if [ ! -d "$checkout/.git" ]; then
  mkdir -p "$(dirname "$checkout")"
  git clone https://github.com/ChienFeng-hub/softqlearning-pytorch.git "$checkout"
fi

if git -C "$checkout" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "MEow SQL fairness patch is already applied at $checkout"
  exit 0
fi
if ! git -C "$checkout" diff --quiet || ! git -C "$checkout" diff --cached --quiet; then
  echo "Refusing to modify a dirty SQL checkout: $checkout" >&2
  exit 1
fi
git -C "$checkout" fetch --quiet origin "$commit"
git -C "$checkout" checkout --quiet --detach "$commit"
git -C "$checkout" apply "$patch_file"
echo "Prepared MEow SQL baseline at $checkout ($(git -C "$checkout" rev-parse --short HEAD) + fairness patch)"
