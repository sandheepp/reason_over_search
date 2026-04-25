#!/usr/bin/env bash
# Open an interactive shell on the vast.ai remote, or run a one-off command.
# Usage:
#   scripts/ssh_remote.sh                      # interactive
#   scripts/ssh_remote.sh "nvidia-smi"         # run command, exit
#   scripts/ssh_remote.sh "cd evaluation_search_r1 && ls"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy .env.example to .env and fill it in." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${REMOTE_HOST:?REMOTE_HOST not set in .env}"
: "${REMOTE_PORT:?REMOTE_PORT not set in .env}"
: "${REMOTE_USER:?REMOTE_USER not set in .env}"
: "${REMOTE_PATH:?REMOTE_PATH not set in .env}"

if [[ $# -eq 0 ]]; then
  ssh -p "$REMOTE_PORT" -t "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && exec \$SHELL -l"
else
  ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && $*"
fi
