#!/usr/bin/env bash
# One-shot: start retriever (16w), wait, kick orchestrator START_PHASE=6.
set -uo pipefail
cd /workspace/reason_over_search
LOGS=/workspace/reason_over_search/logs
mkdir -p "$LOGS"

echo "[finish] $(date) starting"

# Start retriever (foreground inside this script; the script itself is setsid-detached)
cd local_retriever
setsid /venv/retriever/bin/python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 16 --port 3005 \
  > "$LOGS/retriever_finish.log" 2>&1 < /dev/null &
RPID=$!
echo "[finish] retriever pid=$RPID"
cd /workspace/reason_over_search

# Wait for /health
for i in $(seq 1 90); do
  if curl -sS --max-time 2 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy; then
    echo "[finish] retriever ready after ${i}x5s"
    break
  fi
  sleep 5
done
curl -sS http://127.0.0.1:3005/health
echo

if ! curl -sS --max-time 2 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy; then
  echo "[finish] FATAL: retriever not ready"
  exit 1
fi

# Run orchestrator from Phase 6 (it manages SGLang)
echo "[finish] launching orchestrator START_PHASE=6"
START_PHASE=6 bash scripts/orchestrate_C_then_A.sh
echo "[finish] $(date) DONE"
