#!/usr/bin/env bash
# Run the two missing hybrid FULL cells (2wiki + musique) directly,
# using the SGLang + retriever already running.
set -uo pipefail
cd /workspace/reason_over_search
LOGS=/workspace/reason_over_search/logs

echo "[run] $(date) start"

# Pre-flight
curl -sS http://127.0.0.1:3005/health > /dev/null || { echo "FATAL retriever down"; exit 1; }
curl -sS http://127.0.0.1:3000/get_model_info > /dev/null || { echo "FATAL sglang down"; exit 1; }
echo "[run] retriever + sglang up"

# Clean leftover empty dirs from the failed attempts
rm -rf evaluation_qwen35/results/2wikimultihopqa/2wikimultihopqa_2026_05_10_14_35*
rm -rf evaluation_qwen35/results/musique/musique_2026_05_10_14_35*

for ds in 2wikimultihopqa musique; do
  echo "[run] $(date) === $ds FULL ==="
  PROMPT_MODE=qwen35_minimal PY=/venv/evaluation_search_r1/bin/python \
    bash scripts/run_m4.sh qwen3.5_0.8b "$ds" 1 /workspace/reason_over_search/data \
    2>&1 | tee -a "$LOGS/m4_finish_two_cells.log" | grep -E "^\[run\]|^\[skip\]|^\{|Error|Traceback|^em:|^f1:|^acc:" || true
done

echo "[run] $(date) DONE"
