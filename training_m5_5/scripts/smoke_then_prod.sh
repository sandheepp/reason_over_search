#!/usr/bin/env bash
# Run M5.5 smoke (4 steps, seq=4096); if it passes end-to-end, launch the
# requested prod mode in the same shell. Both runs share a tmux session so
# the user can detach and reattach without losing either phase.
#
# Invoked from training_m5_5/scripts/start_b300.sh --smoke-first. Not meant
# to be called directly by humans — start_b300.sh handles pre-flight,
# bootstrap, retriever, and W&B/.env loading first.
#
# Usage (called by start_b300.sh):
#   bash training_m5_5/scripts/smoke_then_prod.sh <seed> <prod_mode>
# Args:
#   seed       — GRPO seed (forwarded to both runs)
#   prod_mode  — prod|prod_2xa100|prod_b300|prod_b300_2xgpu

set -uo pipefail

SEED="${1:?missing seed arg}"
PROD_MODE="${2:?missing prod_mode arg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p /root/logs
TS="$(date -u +%Y%m%dT%H%MZ)"
SMOKE_LOG="/root/logs/m5_5_smoke_seed${SEED}_${TS}.log"
PROD_LOG="/root/logs/m5_5_${PROD_MODE}_seed${SEED}_${TS}.log"

banner() { printf '\033[1;36m[smoke→prod]\033[0m %s\n' "$*"; }

banner "Phase 1: SMOKE (4 steps, seq=4096) → ${SMOKE_LOG}"
banner "Phase 2 (after smoke passes): PROD ${PROD_MODE} → ${PROD_LOG}"
banner "started at $(date -u)"
echo

# ---------- Phase 1: smoke ----------
bash training_m5_5/scripts/run.sh --mode smoke --seed "${SEED}" 2>&1 | tee "${SMOKE_LOG}"
SMOKE_RC="${PIPESTATUS[0]}"

if [[ "${SMOKE_RC}" -ne 0 ]]; then
    banner "❌ SMOKE FAILED (exit=${SMOKE_RC}). NOT starting prod."
    banner "   inspect: ${SMOKE_LOG}"
    banner "   common fixes are in docs/setup/B300_RUNBOOK.md"
    exit "${SMOKE_RC}"
fi

# Belt-and-suspenders: confirm we actually saw all 4 smoke steps + a clean
# exit log, not just a 0 exit code from some partial run.
if ! grep -qE "Step\s+4/4|train_data_step4" "${SMOKE_LOG}"; then
    banner "❌ smoke exit=0 but no Step 4/4 marker in log — refusing to launch prod"
    banner "   inspect: ${SMOKE_LOG}"
    exit 1
fi

# Lesson from B300 run 2026-05-16 (RESULTS_M5_5_B300_seed42.md): a smoke can
# reach "Step 4/4" with the retriever completely broken (FastAPI /health
# returning 200 while FAISS workers are OOM-killed). Every <tool_response>
# in those rollouts was "Errno 111 Connection refused". Model trained on
# "this tool is permanently broken" and the production run learned a
# zero-tool policy. Catch this BEFORE handing off to prod by scanning the
# smoke's rollout JSONLs for retriever error strings.
banner "validating smoke retriever payloads (catch /health-lies-while-workers-dead)..."
SMOKE_EXP_DIR="$(ls -dt /root/reason_over_search/logs/exp_* 2>/dev/null | head -1)"
ROLLOUT_FILE=""
if [[ -n "${SMOKE_EXP_DIR}" ]]; then
    ROLLOUT_FILE="$(ls -t ${SMOKE_EXP_DIR}/train_data_step*.jsonl 2>/dev/null | head -1)"
fi
if [[ -n "${ROLLOUT_FILE}" && -s "${ROLLOUT_FILE}" ]]; then
    # Count tool_response error markers in smoke's most recent rollout. If
    # ≥10% of all tool_responses are connection-refused style errors, the
    # retriever is broken and prod must not launch.
    ERR_COUNT="$(grep -oE 'Max retries exceeded|Errno 111|Connection refused|Retriever failed' "${ROLLOUT_FILE}" 2>/dev/null | wc -l | tr -d ' ')"
    TR_COUNT="$(grep -oE '<tool_response>' "${ROLLOUT_FILE}" 2>/dev/null | wc -l | tr -d ' ')"
    banner "  rollout: $(basename ${ROLLOUT_FILE})  tool_responses=${TR_COUNT}  retriever_errors=${ERR_COUNT}"
    if [[ "${TR_COUNT}" -gt 0 ]]; then
        # Anything above ~10% error rate means the retriever is broken enough
        # that prod will degrade to no-tool policy. Bail out.
        ERR_PCT=$(( 100 * ERR_COUNT / TR_COUNT ))
        if [[ "${ERR_PCT}" -gt 10 ]]; then
            banner "❌ smoke retriever validation FAILED: ${ERR_PCT}% of tool_responses are connection errors"
            banner "   This means /batch_search is broken even though /health may be lying."
            banner "   Common cause: FAISS worker OOM during vLLM warmup."
            banner "   Fixes: drop num_retriever to 2, check dmesg | grep -i killed, lower gpu_memory_utilization"
            banner "   Refusing to launch prod with broken retriever."
            exit 1
        fi
        banner "  ✓ retriever payloads look healthy (${ERR_PCT}% error rate, threshold 10%)"
    else
        banner "⚠ no <tool_response> tags in smoke rollout — model may not have called search at all (check prompt)"
    fi
else
    banner "⚠ couldn't locate smoke rollout JSONL for payload validation — proceeding without check"
fi

banner "✅ SMOKE PASSED at $(date -u)"
echo

# ---------- Ray cleanup between phases ----------
# NeMo-RL spins up its own Ray cluster per run; the smoke's cluster won't be
# torn down cleanly. Kill stragglers + clear /tmp/ray before prod launches.
banner "cleaning up Ray state from smoke"
pkill -9 -f "ray::"   2>/dev/null || true
pkill -9 -f raylet    2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
sleep 3
rm -rf /tmp/ray 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo

# ---------- Phase 2: prod ----------
banner "Phase 2: launching PROD ${PROD_MODE} at $(date -u)"
bash training_m5_5/scripts/run.sh --mode "${PROD_MODE}" --seed "${SEED}" 2>&1 | tee "${PROD_LOG}"
PROD_RC="${PIPESTATUS[0]}"
banner "prod exited rc=${PROD_RC} at $(date -u)"
exit "${PROD_RC}"
