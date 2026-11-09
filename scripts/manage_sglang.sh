#!/usr/bin/env bash
# Start / stop / wait-for-ready helpers for the SGLang server(s).
# Single-instance usage:
#   scripts/manage_sglang.sh stop
#   scripts/manage_sglang.sh start  base|instruct|qwen_25_3b_instruct
#   scripts/manage_sglang.sh wait   [timeout] [port]
#   scripts/manage_sglang.sh switch base|instruct|qwen_25_3b_instruct   # = stop + start + wait
# Fleet usage (8 GPUs by default; override with FLEET_SIZE):
#   scripts/manage_sglang.sh start_fleet base
#   scripts/manage_sglang.sh wait_fleet  [timeout]
#   scripts/manage_sglang.sh stop_fleet
#
# Notes:
#   * base/instruct load from evaluation_search_r1/search_r1_{base,instruct}_model;
#     qwen_25_3b_instruct loads Qwen/Qwen2.5-3B-Instruct from the HF hub.
#   * Single-instance default port 3000 (override with PORT env).
#   * Fleet uses ports 3000..3000+FLEET_SIZE-1, GPUs 0..FLEET_SIZE-1.
#   * Each fleet member's stdout/stderr → /tmp/sglang_<variant>_p<port>.log.
#   * Fleet pids tracked in /tmp/sglang_fleet.pids.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_search_r1"
HOST=127.0.0.1
PORT="${PORT:-3000}"
PY="${PY:-/venv/evaluation_search_r1/bin/python}"
FLEET_SIZE="${FLEET_SIZE:-8}"
FLEET_PIDFILE="${FLEET_PIDFILE:-/tmp/sglang_fleet.pids}"

# Launch one SGLang server in the background. Echoes the pid on stdout.
# Usage: _launch_one <variant> <gpu_id> <port> [served_name_suffix]
_launch_one() {
  local variant="$1" gpu_id="$2" port="$3" suffix="${4:-}"
  local model_path served logfile
  case "$variant" in
    base)
      model_path="$EVAL_DIR/search_r1_base_model"
      served="search_r1_base${suffix}"
      ;;
    instruct)
      model_path="$EVAL_DIR/search_r1_instruct_model"
      served="search_r1_instruct${suffix}"
      ;;
    qwen_25_3b_instruct)
      model_path="$EVAL_DIR/qwen_25_3b_instruct"
      served="qwen_25_3b_instruct${suffix}"
      ;;
    *)
      echo "[sglang] unknown variant: $variant" >&2
      return 2
      ;;
  esac
  logfile="/tmp/sglang_${variant}_p${port}.log"
  echo "[sglang] launching $variant on GPU $gpu_id port $port (log: $logfile)" >&2
  cd "$EVAL_DIR"
  CUDA_VISIBLE_DEVICES="$gpu_id" nohup "$PY" -m sglang.launch_server \
    --served-model-name "$served" \
    --model-path "$model_path" \
    --tp 1 \
    --context-length 8192 \
    --enable-metrics \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$port" \
    --trust-remote-code \
    > "$logfile" 2>&1 &
  echo $!
}

stop_sglang() {
  local pids
  pids="$(pgrep -f 'sglang.launch_server' || true)"
  if [[ -n "$pids" ]]; then
    echo "[sglang] stopping pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 5
    pids="$(pgrep -f 'sglang.launch_server' || true)"
    if [[ -n "$pids" ]]; then
      echo "[sglang] force kill: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  else
    echo "[sglang] no running server"
  fi
  rm -f "$FLEET_PIDFILE"
  sleep 5
}

start_sglang() {
  local variant="$1"
  local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
  _launch_one "$variant" "$gpu_id" "$PORT" > /dev/null
}

wait_ready() {
  local timeout="${1:-300}"
  local port="${2:-$PORT}"
  local elapsed=0
  echo "[sglang] waiting up to ${timeout}s for ${HOST}:${port}/get_model_info..."
  while (( elapsed < timeout )); do
    if curl -sS --max-time 2 "http://${HOST}:${port}/get_model_info" >/dev/null 2>&1; then
      echo "[sglang] ready on port $port"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[sglang] timeout after ${timeout}s on port $port" >&2
  return 1
}

switch() {
  stop_sglang
  start_sglang "$1"
  wait_ready 600
}

start_fleet() {
  local variant="$1"
  : > "$FLEET_PIDFILE"
  for ((i=0; i<FLEET_SIZE; i++)); do
    local pid
    pid="$(_launch_one "$variant" "$i" "$((3000 + i))" "_g$i")"
    echo "$pid" >> "$FLEET_PIDFILE"
  done
  echo "[sglang] fleet of $FLEET_SIZE launched ($variant); pids in $FLEET_PIDFILE"
}

wait_fleet() {
  local timeout="${1:-600}"
  for ((i=0; i<FLEET_SIZE; i++)); do
    wait_ready "$timeout" "$((3000 + i))" || return 1
  done
  echo "[sglang] fleet of $FLEET_SIZE ready"
}

stop_fleet() {
  if [[ ! -f "$FLEET_PIDFILE" ]]; then
    echo "[sglang] no fleet pidfile at $FLEET_PIDFILE; falling back to stop_sglang"
    stop_sglang
    return 0
  fi
  local pids
  pids="$(tr '\n' ' ' < "$FLEET_PIDFILE")"
  echo "[sglang] stopping fleet pids: $pids"
  kill $pids 2>/dev/null || true
  sleep 5
  local remaining=""
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then
      remaining="$remaining $p"
    fi
  done
  if [[ -n "$remaining" ]]; then
    echo "[sglang] force kill:$remaining"
    kill -9 $remaining 2>/dev/null || true
  fi
  rm -f "$FLEET_PIDFILE"
  sleep 5
  echo "[sglang] fleet stopped"
}

case "${1:-}" in
  stop)        stop_sglang ;;
  start)       start_sglang "${2:?missing variant}" ;;
  wait)        wait_ready "${2:-300}" "${3:-$PORT}" ;;
  switch)      switch "${2:?missing variant}" ;;
  start_fleet) start_fleet "${2:?missing variant}" ;;
  wait_fleet)  wait_fleet "${2:-600}" ;;
  stop_fleet)  stop_fleet ;;
  *)
    cat >&2 <<EOF
usage:
  $0 stop
  $0 start  <base|instruct|qwen_25_3b_instruct>
  $0 wait   [timeout] [port]
  $0 switch <base|instruct|qwen_25_3b_instruct>
  $0 start_fleet <variant>      # FLEET_SIZE=$FLEET_SIZE servers on ports 3000..3000+FLEET_SIZE-1
  $0 wait_fleet  [timeout]
  $0 stop_fleet
EOF
    exit 2
    ;;
esac
