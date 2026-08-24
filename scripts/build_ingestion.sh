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

PUBLISH=false
CI_STEP_NAME="Build ingestion"
while getopts "pr:" option; do
    case $option in
        p) # publish app
            PUBLISH=true
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

## Same shape as the api image (knowledge-api): the project name plus the
## component the image runs.
NAME="$(get_toml_value "$ROOT_DIR/pyproject.toml" 'project.name')-ingestion"
VERSION=$(get_toml_value "$ROOT_DIR/pyproject.toml" 'project.version')

echo "About to build $NAME v$VERSION ..."

load_env_files "$WORKSPACE_DIR/development/common/kalisio_dockerhub.enc.env"

## Stage the Kalisio dev tooling into the build context
##
## The image ships k-clone and the rest of the `development` repository: the
## workspace it fills is an empty volume, so the tooling cannot live there.
## The repository is private, but the CI already checked it out next to this
## one, so it is staged from that copy -- no token is needed to build, and
## none can end up in a layer of what is a public image.
##
## Only what git tracks is copied: a developer's decrypted `.dec.` files stay
## on their machine.

TOOLING_STAGE="$ROOT_DIR/.build"

begin_group "Staging the dev tooling from $WORKSPACE_DIR/development ..."

rm -rf "$TOOLING_STAGE"
mkdir -p "$TOOLING_STAGE/development"
git -C "$WORKSPACE_DIR/development" ls-files --recurse-submodules -z \
    | tar -C "$WORKSPACE_DIR/development" --null --files-from - -cf - \
    | tar -C "$TOOLING_STAGE/development" -xf -

end_group "Staging the dev tooling from $WORKSPACE_DIR/development ..."

## Build container
##

IMAGE_NAME="$KALISIO_DOCKERHUB_URL/kalisio/$NAME"
IMAGE_TAG=latest

begin_group "Building container $IMAGE_NAME:$IMAGE_TAG ..."

decrypt_stdout "$WORKSPACE_DIR/development/common/KALISIO_DOCKERHUB_PASSWORD.enc.value" | docker login --username "$KALISIO_DOCKERHUB_USERNAME" --password-stdin "$KALISIO_DOCKERHUB_URL"

# DOCKER_BUILDKIT is here to be able to use Dockerfile specific dockerginore (app.Dockerfile.dockerignore)
DOCKER_BUILDKIT=1 docker build \
    -f ingestion.Dockerfile \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    "$ROOT_DIR"

if [ "$PUBLISH" = true ]; then
    docker push "$IMAGE_NAME:$IMAGE_TAG"
fi

docker logout "$KALISIO_DOCKERHUB_URL"

rm -rf "$TOOLING_STAGE"

end_group "Building container $IMAGE_NAME:$IMAGE_TAG ..."
