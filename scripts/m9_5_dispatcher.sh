#!/usr/bin/env bash
# 16-way parallel eval dispatcher: 4 GPUs × 4 slots/GPU = 16 worker processes,
# each pulling tasks atomically from a shared queue file via flock.
#
# Queue task format (one per line):   <RUN_TAG>:<STEP>
# Example queue line:                  m5_1_s43:230
#
# Per task: dispatcher invokes scripts/run_m9_5_ckpt.sh <RUN_TAG> <STEP> <GPU> <SLOT>
# Worker stays on its (GPU, SLOT) until the queue is empty.
#
# Concurrency model:
#   - 16 worker processes (4 per GPU × 4 GPUs)
#   - Each worker holds GPU+SLOT for its full lifetime (no migration)
#   - Queue access serialised by flock; no in-flight task duplication
#   - vLLM/proxy port allocation:
#         VLLM_PORT  = 8000 + GPU*4 + SLOT
#         PROXY_PORT = 9000 + GPU*4 + SLOT
#   - gpu-memory-utilization 0.22 (~21 GB per vLLM on the 96 GB Blackwell Max-Q;
#     4 slots/GPU * 21 GB = 84 GB used, ~12 GB headroom)
#
# Usage:
#   bash scripts/m9_5_dispatcher.sh [build|run|status]
#     build  - regenerate the queue file from RUN_TAG specs (141 tasks)
#     run    - launch 16 workers (queue must exist; resume-safe if rerun)
#     status - print queue length + active worker PIDs
#
# Resume behaviour: workers skip a task if results/<RUN_TAG>/step_<STEP>/done.flag
# already exists. Re-running 'build' regenerates the queue; re-running 'run'
# without rebuild picks up where left off.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
QUEUE="${QUEUE:-$REPO_ROOT/logs/m9_5/queue.txt}"
LOCK="${LOCK:-$REPO_ROOT/logs/m9_5/queue.lock}"
DONE_DIR="${DONE_DIR:-$REPO_ROOT/logs/m9_5/done}"
DISPATCH_LOG="${DISPATCH_LOG:-$REPO_ROOT/logs/m9_5/dispatcher.log}"
NUM_GPUS="${NUM_GPUS:-4}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
# GPU_OFFSET shifts the worker GPU ids: with GPU 0 dedicated to a GPU-FAISS
# retriever, set NUM_GPUS=3 GPU_OFFSET=1 so workers land on physical GPUs 1,2,3.
GPU_OFFSET="${GPU_OFFSET:-0}"
# 2 slots/GPU = 8 workers; tested 2026-05-27 as the sweet spot. 4-slot needs
# --enforce-eager + --max-model-len 4096 to fit memory-wise but the cold-start
# collision (5 min per slot) wiped out the parallelism gain (105s/ckpt amortized
# vs 67s/ckpt at 2-slot in the bamboogle smoke).
WRAPPER="${WRAPPER:-$REPO_ROOT/scripts/run_m9_5_ckpt.sh}"

mkdir -p "$(dirname "$QUEUE")" "$DONE_DIR"

# RUN_TAG -> highest step in the run (every ckpt at step 10, 20, ..., MAX_STEP)
declare -A MAX_STEP=(
  [m5_1_s43]=230   # 23 ckpts
  [m5_5_s43]=250   # 25 ckpts
  [m5_6_s44]=310   # 31 ckpts
  [m5_1_s44]=310   # 31 ckpts
  [m5_5_s44]=310   # 31 ckpts
)

build_queue() {
  : > "$QUEUE"
  local n=0
  for tag in "${!MAX_STEP[@]}"; do
    local max=${MAX_STEP[$tag]}
    for ((s=10; s<=max; s+=10)); do
      # Skip if marked done already (resume support)
      if [[ ! -f "$DONE_DIR/${tag}_step${s}.done" ]]; then
        echo "${tag}:${s}" >> "$QUEUE"
        n=$((n+1))
      fi
    done
  done
  # Shuffle so adjacent slots don't hammer the same source ckpt dir / HF cache
  shuf "$QUEUE" -o "$QUEUE"
  echo "built queue: $n tasks -> $QUEUE"
}

next_task() {
  local task=""
  (
    flock -x 200
    if [[ -s "$QUEUE" ]]; then
      task=$(head -n 1 "$QUEUE")
      tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    fi
    printf '%s' "$task"
  ) 200>"$LOCK"
}

worker() {
  local gpu=$1 slot=$2
  local wid="gpu${gpu}_slot${slot}"
  echo "[$(date -u +%H:%M:%S)] worker $wid online" >> "$DISPATCH_LOG"
  while :; do
    local task
    task=$(next_task)
    if [[ -z "$task" ]]; then
      echo "[$(date -u +%H:%M:%S)] worker $wid queue empty, exiting" >> "$DISPATCH_LOG"
      break
    fi
    local tag=${task%:*} step=${task#*:}
    local flag="$DONE_DIR/${tag}_step${step}.done"
    if [[ -f "$flag" ]]; then
      echo "[$(date -u +%H:%M:%S)] worker $wid skip done $task" >> "$DISPATCH_LOG"
      continue
    fi
    local t0=$(date +%s)
    echo "[$(date -u +%H:%M:%S)] worker $wid START $task" >> "$DISPATCH_LOG"
    if bash "$WRAPPER" "$tag" "$step" "$gpu" "$slot" >> "$DISPATCH_LOG" 2>&1; then
      touch "$flag"
      local dt=$(($(date +%s) - t0))
      echo "[$(date -u +%H:%M:%S)] worker $wid DONE  $task in ${dt}s" >> "$DISPATCH_LOG"
    else
      local dt=$(($(date +%s) - t0))
      echo "[$(date -u +%H:%M:%S)] worker $wid FAIL  $task after ${dt}s" >> "$DISPATCH_LOG"
      # No flag written -> a future dispatcher run will pick it up after build
    fi
  done
}

run_workers() {
  : > "$DISPATCH_LOG"
  echo "[$(date -u +%H:%M:%S)] dispatcher start: $NUM_GPUS GPUs * $SLOTS_PER_GPU slots = $((NUM_GPUS*SLOTS_PER_GPU)) workers" \
    | tee -a "$DISPATCH_LOG"
  local pids=()
  for ((g=0; g<NUM_GPUS; g++)); do
    for ((s=0; s<SLOTS_PER_GPU; s++)); do
      worker "$((g + GPU_OFFSET))" "$s" &
      pids+=($!)
    done
  done
  echo "[$(date -u +%H:%M:%S)] worker pids: ${pids[*]}" | tee -a "$DISPATCH_LOG"
  wait "${pids[@]}"
  echo "[$(date -u +%H:%M:%S)] dispatcher complete" | tee -a "$DISPATCH_LOG"
}

status() {
  local left=0
  [[ -f "$QUEUE" ]] && left=$(wc -l < "$QUEUE")
  local done=$(ls "$DONE_DIR" 2>/dev/null | wc -l)
  echo "queue remaining: $left"
  echo "done flags:      $done"
  echo "active vLLMs:"
  pgrep -af 'vllm.entrypoints' | head -20
  echo "active proxies:"
  pgrep -af 'sgl_to_vllm_proxy' | head -20
}

case "${1:-status}" in
  build)  build_queue ;;
  run)    run_workers ;;
  status) status ;;
  *) echo "Usage: $0 {build|run|status}" >&2; exit 1 ;;
esac
