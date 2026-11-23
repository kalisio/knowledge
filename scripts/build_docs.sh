#!/usr/bin/env bash
set -euo pipefail
# set -x

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
ROOT_DIR=$(dirname "$THIS_DIR")
WORKSPACE_DIR="$(dirname "$ROOT_DIR")"

. "$THIS_DIR/kash/kash.sh"

# kash's build_docs detects the package manager from the repository root and,
# for pnpm, runs `pnpm install && pnpm docs:build` in the current directory.
# knowledge is a Python project: its only Node package is the documentation,
# so the build runs from docs/, whose package.json defines docs:build.
get_node_package_manager() { echo "pnpm"; }

## Parse options
##

PUBLISH=false
CI_STEP_NAME="Build docs"
while getopts "pr:" OPT; do
    case $OPT in
        p) # publish doc
            PUBLISH=true
            ;;
        r) # report outcome to slack
            CI_STEP_NAME=$OPTARG        
            load_env_files "$WORKSPACE_DIR/development/common/SLACK_WEBHOOK_DOCS.enc.env"
            trap 'slack_ci_report "$ROOT_DIR" "$CI_STEP_NAME" "$?" "$SLACK_WEBHOOK_DOCS"' EXIT
            ;;
        *)
            ;;
    esac
done

## Build docs
##

cd "$ROOT_DIR/docs"
build_docs "$ROOT_DIR" "kalisio/knowledge" "$PUBLISH"
