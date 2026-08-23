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
unset COMMIT_HISTORY_MAX_AGE_DAYS COMMIT_HISTORY_MIN_COMMITS COMMIT_HISTORY_DEPTH
unset QDRANT_COLLECTION_FILES QDRANT_VECTOR_SIZE_COLLECTION_CODE
unset QDRANT_VECTOR_COLLECTION_METADATA QDRANT_LAST_INGESTION_KEY
unset EMBEDDING_BATCH_SIZE LOG_LEVEL
unset TOP_K MAX_CONTEXT_CHARS MAX_ANSWER_TOKENS HOST PORT LLM_PROMPT
unset KNOWLEDGE_AUTH_ENABLED APP_SECRET JWT_ALGORITHM JWT_AUDIENCE JWT_ISSUER

## Start Qdrant
##

"$WORKSPACE_DIR/development/scripts/k-qdrant"

## Setup micromamba env
##

setup_micromamba_env "knowledge-test" "$ROOT_DIR/environment.yml" true

## Run tests
##

run_python_lib_tests "$ROOT_DIR" "$RUN_SONAR"