#!/usr/bin/env bash
set -euo pipefail
# set -x

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
ROOT_DIR=$(dirname "$THIS_DIR")
WORKSPACE_DIR="$(dirname "$ROOT_DIR")"

. "$THIS_DIR/kash/kash.sh"

## Parse options
##

CI_STEP_NAME="Run tests"
RUN_SONAR=false
while getopts "sr:" option; do
    case $option in
        s) # enable SonarQube analysis and publish code quality & coverage results
            RUN_SONAR=true
            ;;
        r) # report outcome to slack
            CI_STEP_NAME=$OPTARG
            load_env_files "$WORKSPACE_DIR/development/common/SLACK_WEBHOOK_SERVICES.enc.env"
            trap 'slack_ci_report "$ROOT_DIR" "$CI_STEP_NAME" "$?" "$SLACK_WEBHOOK_SERVICES"' EXIT
            ;;
        *)
            ;;
    esac
done


## Init workspace
##

. "$WORKSPACE_DIR/development/workspaces/services/services.sh" knowledge

## Clear the optional settings

unset INDEXED_REPOSITORIES KLI_ORGANIZATION KLI_WORKSPACE
unset SUPPORTED_FILE_EXTENSIONS MAX_FILE_SIZE
unset IGNORED_DIRECTORIES IGNORED_FILENAMES IGNORED_FILE_PATTERN
unset CHUNK_SIZE CHUNK_OVERLAP CODE_CHUNK_SIZE CODE_CHUNK_OVERLAP
unset QDRANT_COLLECTION_FILES QDRANT_VECTOR_SIZE_COLLECTION_CODE
unset QDRANT_VECTOR_COLLECTION_METADATA QDRANT_LAST_INGESTION_KEY
unset EMBEDDING_BATCH_SIZE LOG_LEVEL
unset TOP_K MAX_CONTEXT_CHARS MAX_ANSWER_TOKENS HOST PORT LLM_PROMPT
unset KNOWLEDGE_AUTH_ENABLED APP_SECRET JWT_ALGORITHM JWT_AUDIENCE JWT_ISSUER

## Start Qdrant
##

"$WORKSPACE_DIR/development/scripts/k-qdrant"

## Setup python env
##

# As setup_python_env create a virtual environment with uv and
# run_python_lib_tests runs the tests without using "uv run"
# (to keep compatibility with micromamba environments), we need
# to activate the environment (-a). uv syncs the project in the
# current directory, hence the cd.
cd "$ROOT_DIR"
setup_python_env -a -g "test"

## Run tests
##

run_python_lib_tests "$ROOT_DIR" "$RUN_SONAR"