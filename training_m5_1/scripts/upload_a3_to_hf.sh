#!/usr/bin/env bash
# Single-repo HF uploader for M5.1-prod-a3 (B200 Spheron run).
#
# Polls the production checkpoint dir + logs dir every 60 s. Uploads any new
# step_N/ atomic-rename directories, the live console log, and any per-step
# rollout JSONLs to ONE HuggingFace model repo.
#
# Decoupled from training:
#   - Runs in a separate process tree from the training loop
#   - Never sends signals to or shares file locks with the training process
#   - On upload failure: logs, retries on next poll cycle
#   - On its own crash: only stops the uploader; training continues
#   - Idempotent across restarts via .uploaded_artifacts state file
#
# Repo layout produced (target):
#   <repo>/
#     README.md            (created externally; not touched by this script)
#     config_snapshot.yaml (created externally)
#     step_50/  step_100/  ... step_600/    (NeMo-RL ckpt dirs, 1:1 with disk)
#     logs/
#       prod.log              (uploaded with overwrite each cycle)
#       train_data/           (per-step rollout JSONLs)
#       wandb_run.txt         (W&B run id; written by training_init)
#     timings.csv           (per-step timings; appended by training_init)
#     final_metrics.json    (created externally at end)
#
# Authored 2026-05-14 for the M5.1-prod-a3 launch. New file (not a modification
# of upload_ckpts_watcher.sh, which uses the per-step-repos pattern).

set -uo pipefail

# ---- config (env-overridable) ----
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
HF_REPO_ID="${HF_REPO_ID:-pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/results/grpo/m5_prod/seed42}"
PROD_LOG="${PROD_LOG:-/workspace/prod.log}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/reason_over_search/logs}"
STATE_FILE="${STATE_FILE:-/workspace/.uploaded_artifacts}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
WATCHER_LOG="${WATCHER_LOG:-/workspace/uploader.log}"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$WATCHER_LOG"
}

is_uploaded() {
  grep -qxF "$1" "$STATE_FILE"
}

mark_uploaded() {
  echo "$1" >> "$STATE_FILE"
}

# Python helper for HF upload using huggingface_hub API.
# Uses miniforge3 base python (hf_hub installed by bootstrap on first prod launch).
# Fallback to nemo venv if base is missing it.
HF_PY="${HF_PY:-/opt/miniforge3/bin/python}"
hf_upload_folder() {
  local local_path="$1"
  local repo_path="$2"
  "$HF_PY" - <<EOF 2>&1 | tail -10
from huggingface_hub import HfApi
import os, sys
api = HfApi(token="$HF_TOKEN")
try:
    api.upload_folder(
        folder_path="$local_path",
        path_in_repo="$repo_path",
        repo_id="$HF_REPO_ID",
        repo_type="model",
        commit_message="auto: upload $repo_path",
    )
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF
}

hf_upload_file() {
  local local_path="$1"
  local repo_path="$2"
  "$HF_PY" - <<EOF 2>&1 | tail -5
from huggingface_hub import HfApi
import os, sys
api = HfApi(token="$HF_TOKEN")
try:
    api.upload_file(
        path_or_fileobj="$local_path",
        path_in_repo="$repo_path",
        repo_id="$HF_REPO_ID",
        repo_type="model",
        commit_message="auto: upload $repo_path",
    )
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF
}

log "uploader started (repo=$HF_REPO_ID, poll=${POLL_INTERVAL}s)"

while true; do
  # 1. Upload any new step_N/ checkpoint dirs (skip tmp_step_N).
  if [[ -d "$CHECKPOINT_DIR" ]]; then
    for ckpt in "$CHECKPOINT_DIR"/step_*; do
      [[ -d "$ckpt" ]] || continue
      step_name="$(basename "$ckpt")"
      key="ckpt:$step_name"
      if is_uploaded "$key"; then continue; fi
      log "uploading checkpoint $step_name ..."
      if result=$(hf_upload_folder "$ckpt" "$step_name"); then
        if echo "$result" | grep -q "^OK"; then
          mark_uploaded "$key"
          log "  ✓ $step_name uploaded"
        else
          log "  ✗ $step_name FAILED: $result"
        fi
      else
        log "  ✗ $step_name upload error (will retry)"
      fi
    done
  fi

  # 2. Upload latest prod.log (overwrites each cycle).
  if [[ -f "$PROD_LOG" ]]; then
    log_size=$(stat -c%s "$PROD_LOG" 2>/dev/null || echo 0)
    last_size_file="${STATE_FILE}.log_size"
    last_size=$(cat "$last_size_file" 2>/dev/null || echo 0)
    if [[ "$log_size" != "$last_size" ]] && [[ "$log_size" -gt 0 ]]; then
      result=$(hf_upload_file "$PROD_LOG" "logs/prod.log")
      if echo "$result" | grep -q "^OK"; then
        echo "$log_size" > "$last_size_file"
        log "  ✓ prod.log uploaded (${log_size} bytes)"
      else
        log "  ✗ prod.log upload FAILED: $result"
      fi
    fi
  fi

  # 3. Upload any new per-step rollout JSONLs in logs/exp_*/
  if [[ -d "$ROLLOUT_DIR" ]]; then
    for jsonl in "$ROLLOUT_DIR"/exp_*/train_data_step*.jsonl; do
      [[ -f "$jsonl" ]] || continue
      relpath="${jsonl#$ROLLOUT_DIR/}"
      key="rollout:$relpath"
      if is_uploaded "$key"; then continue; fi
      result=$(hf_upload_file "$jsonl" "logs/train_data/$relpath")
      if echo "$result" | grep -q "^OK"; then
        mark_uploaded "$key"
        log "  ✓ $relpath uploaded"
      else
        log "  ✗ $relpath upload FAILED: $result"
      fi
    done
  fi

  # 4. Upload timings.csv if present and changed.
  TIMINGS_FILE="/workspace/timings.csv"
  if [[ -f "$TIMINGS_FILE" ]]; then
    t_size=$(stat -c%s "$TIMINGS_FILE" 2>/dev/null || echo 0)
    t_last_file="${STATE_FILE}.timings_size"
    t_last=$(cat "$t_last_file" 2>/dev/null || echo 0)
    if [[ "$t_size" != "$t_last" ]] && [[ "$t_size" -gt 0 ]]; then
      result=$(hf_upload_file "$TIMINGS_FILE" "timings.csv")
      if echo "$result" | grep -q "^OK"; then
        echo "$t_size" > "$t_last_file"
        log "  ✓ timings.csv uploaded"
      else
        log "  ✗ timings.csv upload FAILED: $result"
      fi
    fi
  fi

  sleep "$POLL_INTERVAL"
done
