#!/usr/bin/env bash
set -euo pipefail
# set -x

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
ROOT_DIR=$(dirname "$THIS_DIR")
cd "$ROOT_DIR"

PUBLISH=false
while getopts "p" option; do
    case $option in
        p) PUBLISH=true ;;
        *) ;;
    esac
done

NAME=$(python - <<'PY'
import tomllib
print(tomllib.load(open("pyproject.toml", "rb"))["project"]["name"])
PY
)
VERSION=$(python - <<'PY'
import tomllib
print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])
PY
)

REGISTRY=${KALISIO_DOCKERHUB_URL:-}
IMAGE_NAME=${REGISTRY:+$REGISTRY/}kalisio/${NAME}-api
IMAGE_TAG=${IMAGE_TAG:-latest}

if git -C "$ROOT_DIR" describe --tags --exact-match >/dev/null 2>&1; then
    IMAGE_TAG=$VERSION
fi

echo "Building $IMAGE_NAME:$IMAGE_TAG ..."
DOCKER_BUILDKIT=1 docker build -f "$ROOT_DIR/api.Dockerfile" -t "$IMAGE_NAME:$IMAGE_TAG" "$ROOT_DIR"

if [ "$PUBLISH" = true ]; then
    docker push "$IMAGE_NAME:$IMAGE_TAG"
fi
