#!/usr/bin/env bash
# Resilient launcher for M5.1-prod-a3 — auto-restarts from latest checkpoint
# if the training process exits non-zero. Designed for unattended multi-day
# B200 runs on Spheron spot (spot preemption + the rare exception both trigger
# restart).
#
# Restart logic:
#   - On exit != 0, find the latest step_N/ in CHECKPOINT_DIR
#   - NeMo-RL's run_grpo.py auto-resumes from the latest checkpoint when
#     checkpoint_dir contains step_N/ subdirs (verified in smoke; the
#     train_dataloader.pt state lives there)
#   - Wait 60 s before relaunch (lets stale GPU memory clear, avoids tight loops)
#   - Cap at MAX_RESTARTS to prevent unbounded restart on persistent failure
#
# Exit codes:
#   0  — training completed normally (exit 0 from run.sh)
#   1  — training failed after MAX_RESTARTS attempts
#   2  — wrapper itself errored (config missing, etc.)
#
# All output appended to PROD_LOG so the upload watcher can mirror it to HF.
#
# Authored 2026-05-14 for the M5.1-prod-a3 launch. Sibling to (not a
# replacement of) training_m5_1/scripts/run.sh — calls into it.

set -uo pipefail

PROD_LOG="${PROD_LOG:-/workspace/prod.log}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/results/grpo/m5_prod/seed42}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
SLEEP_BEFORE_RESTART="${SLEEP_BEFORE_RESTART:-60}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$(dirname "$PROD_LOG")"

stamp() { date -u +%FT%TZ; }

log() {
  printf '[%s] [wrapper] %s\n' "$(stamp)" "$*" | tee -a "$PROD_LOG"
}

# Find latest step_N (returns empty if none)
latest_step() {
  ls -d "$CHECKPOINT_DIR"/step_* 2>/dev/null \
    | grep -oE 'step_[0-9]+$' \
    | sed 's/step_//' \
    | sort -n \
    | tail -1
}

attempt=0
while [[ $attempt -le $MAX_RESTARTS ]]; do
  step=$(latest_step)
  if [[ -z "$step" ]]; then
    log "attempt $((attempt+1))/$((MAX_RESTARTS+1)) — fresh launch (no prior checkpoint)"
  else
    log "attempt $((attempt+1))/$((MAX_RESTARTS+1)) — resuming (latest ckpt: step_${step})"
  fi

  # Launch training. run.sh auto-detects existing step_N/ ckpts via
  # checkpoint_dir contents and resumes.
  bash training_m5_1/scripts/run.sh --mode prod --seed 42 >> "$PROD_LOG" 2>&1
  exit_code=$?

  if [[ $exit_code -eq 0 ]]; then
    log "TRAINING COMPLETED CLEANLY (exit=0)"
    exit 0
  fi

  log "TRAINING FAILED (exit=$exit_code), attempt $((attempt+1))/$((MAX_RESTARTS+1))"
  attempt=$((attempt+1))

  if [[ $attempt -gt $MAX_RESTARTS ]]; then
    log "MAX_RESTARTS exhausted; giving up"
    exit 1
  fi

  log "waiting ${SLEEP_BEFORE_RESTART}s before restart..."
  sleep "$SLEEP_BEFORE_RESTART"
done
