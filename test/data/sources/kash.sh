#!/usr/bin/env bash
set -euo pipefail

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")

# Add ~/.local/bin to PATH: this is where the tools below are installed.
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

# Install yq in ~/.local/bin
# Arg1: a writable folder where to write downloaded files
install_yq() {
  local DL_PATH="$1"
  local YQ_VERSION=4.40.5
  curl -OLsS "https://github.com/mikefarah/yq/releases/download/v$YQ_VERSION/yq_linux_amd64"
  mv yq_linux_amd64 ~/.local/bin/yq
  chmod a+x ~/.local/bin/yq
}

# Call this to ensure yq is available
ensure_yq() {
  if ! command -v yq >/dev/null 2>&1; then
    install_yq "$TMP_DIR"
  fi
}

### Secrets

# Load the variables of one or more .enc.env files into the environment,
# decrypting them with sops on the fly. Nothing is written to disk.
# Arg1..N: the files to load
load_env_files() {
  for FILE in "$@"; do
    if [ -f "$FILE" ]; then
      set -a
      # shellcheck disable=SC1090
      . <(sops --decrypt "$FILE")
      set +a
    fi
  done
}

# Report the outcome of a CI step to a slack channel.
# Arg1: the repository root
# Arg2: the step name
# Arg3: the exit code
# Arg4: the webhook url
slack_ci_report() {
  local ROOT_DIR="$1"
  local STEP="$2"
  local STATUS="$3"
  local WEBHOOK="$4"
  local COLOR="good"
  if [ "$STATUS" -ne 0 ]; then COLOR="danger"; fi
  curl -sS -X POST -H 'Content-type: application/json' \
    --data "{\"attachments\":[{\"color\":\"$COLOR\",\"text\":\"$STEP: $STATUS\"}]}" \
    "$WEBHOOK"
}
