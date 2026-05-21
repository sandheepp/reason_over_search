#!/usr/bin/env bash
# M4 autonomous orchestrator: option C (4 configs × 7 datasets × n=1000)
# then option A (locked configs × full data, both variants).
#
# Phases (resumable — set START_PHASE):
#   1) hybrid C (qwen35_minimal + qwen35_minimal_no_system) — assumes SGLang has hybrid loaded
#   2) switch SGLang to base
#   3) base C (qwen35_minimal + qwen35_minimal_no_system)
#   4) (still on base) base A: full data with locked default (qwen35_minimal_no_system)
#   5) switch SGLang to hybrid
#   6) hybrid A: full data with locked default (qwen35_minimal)
#
# Locked-best per-variant defaults (from §6-§7 in RESULTS_SMOKE_m4):
#   hybrid -> qwen35_minimal
#   base   -> qwen35_minimal_no_system

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
LOGS="$REPO_ROOT/logs"
mkdir -p "$LOGS"

PY=/venv/evaluation_search_r1/bin/python
SGLANG_PY=/venv/evaluation_search_r1/bin/python
HYBRID_PATH="$REPO_ROOT/eval/qwen3.5_0.8b"
BASE_PATH="$REPO_ROOT/eval/qwen3.5_0.8b_base"

DATASETS=(bamboogle nq triviaqa popqa hotpotqa 2wikimultihopqa musique)
START_PHASE="${START_PHASE:-1}"

stop_sglang() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! pgrep -f sglang.launch_server >/dev/null; then break; fi
    sleep 1
  done
  if pgrep -f sglang.launch_server >/dev/null; then
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
  fi
}

start_sglang() {
  local model_path="$1"
  local logf="$2"
  echo "[orch] starting SGLang model=$model_path log=$logf"
  export SGLANG_DISABLE_CUDNN_CHECK=1
  nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$model_path" \
    --host 127.0.0.1 --port 3000 \
    --tp 1 --context-length 8192 --dtype bfloat16 \
    --trust-remote-code > "$logf" 2>&1 &
  disown
  until curl -sS --max-time 2 http://127.0.0.1:3000/get_model_info >/dev/null 2>&1; do sleep 5; done
  echo "[orch] SGLang ready: $(curl -sS http://127.0.0.1:3000/get_model_info | grep -oE '\"model_path\":\"[^\"]+\"')"
}

# Run all 7 datasets for one (variant, prompt_mode) at the given n.
run_sweep() {
  local variant="$1"  # qwen3.5_0.8b or qwen3.5_0.8b_base
  local mode="$2"     # qwen35_minimal or qwen35_minimal_no_system
  local n_arg="$3"    # 1000 or empty for full
  for ds in "${DATASETS[@]}"; do
    echo "=== $(date +%H:%M:%S) $variant/$mode/$ds n=${n_arg:-FULL} ==="
    PROMPT_MODE="$mode" PY="$PY" \
      bash "$REPO_ROOT/scripts/run_m4.sh" "$variant" "$ds" 1 "$REPO_ROOT/data" $n_arg 2>&1 \
      | grep -E "^\{|\[skip\]|^Inference: 100\%|Error|Traceback"
  done
}

echo "[orch] $(date) START_PHASE=$START_PHASE"

# Phase 1: hybrid C (both modes)
if (( START_PHASE <= 1 )); then
  echo "[orch] === PHASE 1: hybrid C ==="
  for mode in qwen35_minimal qwen35_minimal_no_system; do
    run_sweep qwen3.5_0.8b "$mode" 1000
  done
fi

# Phase 2-3: switch to base, run base C (both modes)
if (( START_PHASE <= 3 )); then
  echo "[orch] === PHASE 2: switch SGLang to base ==="
  stop_sglang
  start_sglang "$BASE_PATH" "$LOGS/sglang_base_orch.log"
  echo "[orch] === PHASE 3: base C ==="
  for mode in qwen35_minimal qwen35_minimal_no_system; do
    run_sweep qwen3.5_0.8b_base "$mode" 1000
  done
fi

# Phase 4: base A (full data, locked default mode = qwen35_minimal_no_system)
if (( START_PHASE <= 4 )); then
  echo "[orch] === PHASE 4: base A (full data, locked qwen35_minimal_no_system) ==="
  run_sweep qwen3.5_0.8b_base qwen35_minimal_no_system ""
fi

# Phase 5-6: switch to hybrid, run hybrid A (full data, locked qwen35_minimal)
if (( START_PHASE <= 6 )); then
  echo "[orch] === PHASE 5: switch SGLang to hybrid ==="
  stop_sglang
  start_sglang "$HYBRID_PATH" "$LOGS/sglang_hybrid_orch.log"
  echo "[orch] === PHASE 6: hybrid A (full data, locked qwen35_minimal) ==="
  run_sweep qwen3.5_0.8b qwen35_minimal ""
fi

echo "[orch] $(date) ALL PHASES DONE"
