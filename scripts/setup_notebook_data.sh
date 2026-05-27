#!/usr/bin/env bash
# Create the symlinks notebooks need to reach sibling Kalisio repos.
set -euo pipefail

THIS_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
ROOT_DIR=$(dirname "$THIS_DIR")
WORKSPACE_DIR=$(dirname "$ROOT_DIR")

REPOS=(kano kdk crisis kapp skeleton dok)
DATA_DIR="$ROOT_DIR/docs/experiments/data"

mkdir -p "$DATA_DIR"

if ! [ -e "$ROOT_DIR/data" ] && ! [ -L "$ROOT_DIR/data" ]; then
    ln -s docs/experiments/data "$ROOT_DIR/data"
fi

for repo in "${REPOS[@]}"; do
    link="$DATA_DIR/$repo"
    if [ -e "$link" ] || [ -L "$link" ]; then
        continue
    fi
    if [ ! -d "$WORKSPACE_DIR/$repo" ]; then
        echo "missing: $repo" >&2
        continue
    fi
    ln -s "../../../../$repo" "$link"
done
