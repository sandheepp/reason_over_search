#!/usr/bin/env bash
# Push local repo code to the vast.ai remote.
# - never deletes anything on the remote (no --delete)
# - skips large artifacts: datasets, indexes, models, results, caches, git internals
# Usage:
#   scripts/sync_to_remote.sh           # do the sync
#   scripts/sync_to_remote.sh -n        # dry-run preview

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

DRY=()
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
  DRY=("--dry-run")
  echo "[dry-run] no files will be transferred"
fi

echo "Sync: $REPO_ROOT/  ->  $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/  (port $REMOTE_PORT)"

rsync -avz "${DRY[@]}" \
  -e "ssh -p $REMOTE_PORT" \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.venv/' --exclude 'venv/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.idea/' --exclude '.vscode/' \
  --exclude 'node_modules/' \
  --exclude 'data/' \
  --exclude 'evaluation_search_r1/results/' \
  --exclude 'evaluation_search_r1/search_r1_base_model/' \
  --exclude 'evaluation_search_r1/search_r1_instruct_model/' \
  --exclude 'local_retriever/corpus/' \
  --exclude 'local_retriever/indexes/' \
  --exclude 'local_retriever/models/' \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.gguf' \
  --exclude '*.pt' --exclude '*.pth' \
  --exclude '*.parquet' \
  --exclude '*.index' --exclude '*.index.gz' \
  --exclude '*.jsonl.gz' --exclude '*.tar' --exclude '*.tar.gz' \
  "$REPO_ROOT"/ \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
