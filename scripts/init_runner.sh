#!/usr/bin/env bash
set -euo pipefail
# set -x

JOB_ID=${1:-unknown}

echo "Init runner for $JOB_ID"
python --version
docker --version

