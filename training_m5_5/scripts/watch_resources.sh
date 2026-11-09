#!/usr/bin/env bash
# Background resource watcher for M5.5 training runs.
#
# Why this exists: B300 run 2026-05-16 had the FAISS retriever workers
# silently OOM-killed during vLLM warmup. Free RAM dropped from ~140 GB
# to ~5 GB during prod_b300 launch but nothing alerted on it; we only
# discovered it post-mortem from rollout error strings. This watcher
# polls free RAM + disk every 30 s, logs to the chain log, and emits
# loud WARN/CRIT lines when thresholds are crossed. It also probes the
# retriever's /batch_search to catch the "/health-lies-while-workers-dead"
# failure mode mid-run.
#
# With save_period=10 (changed from 50 in commit on 2026-05-16 so a crash
# never costs more than 10 steps), 311 prod steps generates up to 31
# checkpoints × 3.2 GB = ~99 GB of ckpt data. That can fill disk on
# 200-256 GB boxes. The watcher's prune_old_ckpts() fires when disk hits
# WARN: deletes step_N/ directories under <ckpt_dir>/m5_5_*/seed*/
# except the KEEP_LAST_N_CKPTS most recent (default 5 = ~16 GB kept).
#
# The watcher itself does NOT kill the training process or the retriever
# — it just makes failures visible and prunes ckpts. Designed to run as a
# detached background process via `disown`.
#
# Usage:
#   bash training_m5_5/scripts/watch_resources.sh [--log <path>] [--poll <seconds>]
#                                                  [--ram-warn <GB>] [--ram-crit <GB>]
#                                                  [--disk-warn <GB>] [--disk-crit <GB>]
#                                                  [--retriever-url <url>]
#                                                  [--ckpt-dir <path>] [--keep-ckpts <N>]
#                                                  [--trace-every <steps>] [--trace-log <path>]
#
# Defaults (tuned for Verda B300, 270 GB RAM / 193 GB disk):
#   --log              /root/logs/m5_5_resources_<TS>.log
#   --trace-log        /root/logs/m5_5_traces_<TS>.log
#   --poll             30 s
#   --ram-warn         30 GB free (early warning)
#   --ram-crit         10 GB free (OOM imminent)
#   --disk-warn        15 GB free (triggers ckpt prune; rollouts will fill / soon)
#   --disk-crit         5 GB free (Ray plasma about to fail; emergency prune)
#   --retriever-url    http://127.0.0.1:3005
#   --ckpt-dir         /root/reason_over_search/results/grpo
#   --keep-ckpts       5 (per seed; older ones deleted on disk WARN)
#   --trace-every      10 steps (invokes check_trace.py periodically)

set -uo pipefail

POLL_S=30
RAM_WARN_GB=30
RAM_CRIT_GB=10
DISK_WARN_GB=15
DISK_CRIT_GB=5
RETRIEVER_URL="http://127.0.0.1:3005"
TRACE_EVERY_N_STEPS=10           # auto-analyze rollouts every N steps
ROLLOUT_DIR=""                   # auto-detect /root/reason_over_search/logs/exp_*
CKPT_DIR="/root/reason_over_search/results/grpo"  # parent of m5_5_<mode>/seed<N>/step_M
KEEP_LAST_N_CKPTS=5              # when pruning fires, keep this many newest
TS="$(date -u +%Y%m%dT%H%MZ)"
LOG="/root/logs/m5_5_resources_${TS}.log"
TRACE_LOG="/root/logs/m5_5_traces_${TS}.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log) LOG="$2"; shift 2 ;;
        --trace-log) TRACE_LOG="$2"; shift 2 ;;
        --poll) POLL_S="$2"; shift 2 ;;
        --ram-warn) RAM_WARN_GB="$2"; shift 2 ;;
        --ram-crit) RAM_CRIT_GB="$2"; shift 2 ;;
        --disk-warn) DISK_WARN_GB="$2"; shift 2 ;;
        --disk-crit) DISK_CRIT_GB="$2"; shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --trace-every) TRACE_EVERY_N_STEPS="$2"; shift 2 ;;
        --rollout-dir) ROLLOUT_DIR="$2"; shift 2 ;;
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        --keep-ckpts) KEEP_LAST_N_CKPTS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$(dirname "${LOG}")"
exec >> "${LOG}" 2>&1

emit() {
    local level="$1"; shift
    printf '%s [%s] %s\n' "$(date -u +%H:%M:%S)" "${level}" "$*"
}

probe_retriever() {
    local resp
    resp="$(curl -sS --max-time 5 -X POST "${RETRIEVER_URL}/batch_search" \
        -H 'Content-Type: application/json' \
        -d '{"query":["test"],"top_n":1,"return_score":false}' 2>&1 || true)"
    [[ "${resp}" == *'"contents"'* ]]
}

emit INFO "watcher start  poll=${POLL_S}s  thresholds: ram_warn=${RAM_WARN_GB}G ram_crit=${RAM_CRIT_GB}G disk_warn=${DISK_WARN_GB}G disk_crit=${DISK_CRIT_GB}G  retriever=${RETRIEVER_URL}  trace_every=${TRACE_EVERY_N_STEPS}_steps  keep_last=${KEEP_LAST_N_CKPTS}_ckpts"
emit INFO "  trace log: ${TRACE_LOG}"
emit INFO "  ckpt dir:  ${CKPT_DIR} (prune fires when disk hits WARN, keeps last ${KEEP_LAST_N_CKPTS} per seed)"
trap 'emit INFO "watcher stop (signal)"; exit 0' INT TERM

LAST_RAM_LEVEL=""
LAST_DISK_LEVEL=""
LAST_RETRIEVER_LEVEL=""
LAST_ANALYZED_STEP=0
ITER=0

# Locate check_trace.py + parent venv (it needs transformers for tokenizer)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_TRACE="${SCRIPT_DIR}/check_trace.py"
TRACE_PY="${SCRIPT_DIR}/../nemo_rl/.venv/bin/python"
[[ -x "${TRACE_PY}" ]] || TRACE_PY="$(command -v python3)"

run_trace_check() {
    local step="$1"
    {
        echo
        echo "════════════════════════════════════════════════════════════════════════════"
        echo "  $(date -u +%H:%M:%S)  triggered by watch_resources.sh @ step ${step}"
        echo "════════════════════════════════════════════════════════════════════════════"
        if [[ -n "${ROLLOUT_DIR}" ]]; then
            "${TRACE_PY}" "${CHECK_TRACE}" --rollouts "${ROLLOUT_DIR}" --step "${step}" 2>&1
        else
            "${TRACE_PY}" "${CHECK_TRACE}" --step "${step}" 2>&1
        fi
    } >> "${TRACE_LOG}" 2>&1
    # Surface the headline in the main resource log too
    if grep -q "🚩 RED FLAGS" "${TRACE_LOG}" 2>/dev/null; then
        local last_flags="$(grep -A 5 "step ${step} " "${TRACE_LOG}" 2>/dev/null | grep "🚩\|•" | head -5 | tr '\n' '|')"
        emit WARN "trace check @ step ${step}: ${last_flags:-flags raised, see ${TRACE_LOG}}"
    else
        emit INFO "trace check @ step ${step}: all health signals OK"
    fi
}

find_latest_rollout_step() {
    # Look in the configured/detected rollout dir; if not set, auto-detect
    # the most-recently-modified /root/reason_over_search/logs/exp_*/.
    local dir="${ROLLOUT_DIR}"
    if [[ -z "${dir}" || ! -d "${dir}" ]]; then
        dir="$(ls -dt /root/reason_over_search/logs/exp_* 2>/dev/null | head -1)"
    fi
    if [[ -z "${dir}" || ! -d "${dir}" ]]; then
        echo 0; return
    fi
    ls -t "${dir}"/train_data_step*.jsonl 2>/dev/null | head -1 | \
        sed -E 's|.*train_data_step([0-9]+)\.jsonl|\1|' || echo 0
}

prune_old_ckpts() {
    # When disk goes WARN, delete all step_N/ directories under
    # ${CKPT_DIR}/m5_5_*/seed*/ EXCEPT the ${KEEP_LAST_N_CKPTS} highest-N
    # ones. Atomic: NeMo-RL's checkpoint.py writes tmp_step_N/ first and
    # renames on flush, so deleting non-tmp dirs while training is running
    # is safe (mid-write dirs are excluded by name).
    #
    # Each ckpt is ~3.2 GB. Keeping last 5 = ~16 GB. Frees ~15-50 GB on
    # a run that's accumulated 10-20 saves.
    local freed=0 deleted=0
    local seed_dirs
    seed_dirs="$(ls -d ${CKPT_DIR}/m5_5_*/seed* 2>/dev/null || true)"
    [[ -z "${seed_dirs}" ]] && { emit INFO "  prune: no ckpt dirs under ${CKPT_DIR}/m5_5_*/seed*"; return; }
    for seed_dir in ${seed_dirs}; do
        # List step_N dirs sorted by N descending, skip the first KEEP_LAST_N,
        # delete the rest.
        local to_delete
        to_delete="$(ls -d "${seed_dir}"/step_* 2>/dev/null \
            | grep -v '/tmp_step_' \
            | sed -E 's|.*/step_([0-9]+)$|\1 &|' \
            | sort -rn \
            | tail -n +$((KEEP_LAST_N_CKPTS + 1)) \
            | awk '{print $2}')"
        for d in ${to_delete}; do
            local sz; sz="$(du -sm "$d" 2>/dev/null | awk '{print $1}')"
            rm -rf "$d" && {
                freed=$((freed + ${sz:-0}))
                deleted=$((deleted + 1))
            }
        done
    done
    if [[ "${deleted}" -gt 0 ]]; then
        emit INFO "  prune: deleted ${deleted} old ckpts, freed ~$((freed / 1024)) GB. Kept last ${KEEP_LAST_N_CKPTS} per seed."
    fi
}

while true; do
    ITER=$((ITER + 1))

    # ---- RAM ----
    # `free -g` rounds aggressively; use --kilo for finer granularity then convert.
    RAM_FREE_KB="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    RAM_FREE_GB=$(( RAM_FREE_KB / 1024 / 1024 ))
    RAM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 1)"
    RAM_TOTAL_GB=$(( RAM_TOTAL_KB / 1024 / 1024 ))

    if [[ "${RAM_FREE_GB}" -lt "${RAM_CRIT_GB}" ]]; then RAM_LEVEL="CRIT"
    elif [[ "${RAM_FREE_GB}" -lt "${RAM_WARN_GB}" ]]; then RAM_LEVEL="WARN"
    else RAM_LEVEL="OK"
    fi

    # ---- Disk (/) ----
    DISK_FREE_KB="$(df --output=avail / | tail -1 | tr -d ' ')"
    DISK_FREE_GB=$(( DISK_FREE_KB / 1024 / 1024 ))

    if [[ "${DISK_FREE_GB}" -lt "${DISK_CRIT_GB}" ]]; then DISK_LEVEL="CRIT"
    elif [[ "${DISK_FREE_GB}" -lt "${DISK_WARN_GB}" ]]; then DISK_LEVEL="WARN"
    else DISK_LEVEL="OK"
    fi

    # ---- GPU ----
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
    else
        GPU_INFO="n/a"
    fi

    # ---- Retriever liveness (probe every 4th iteration = ~2 min default) ----
    RETRIEVER_LEVEL="?"
    if [[ $((ITER % 4)) -eq 1 ]]; then
        if probe_retriever; then RETRIEVER_LEVEL="OK"
        else RETRIEVER_LEVEL="DOWN"
        fi
    fi

    # ---- Emit ----
    # Always log a heartbeat line at INFO once per minute (every 2nd iteration at default 30s).
    if [[ $((ITER % 2)) -eq 0 ]]; then
        emit INFO "ram_free=${RAM_FREE_GB}/${RAM_TOTAL_GB}G[${RAM_LEVEL}]  disk_free=${DISK_FREE_GB}G[${DISK_LEVEL}]  gpu=${GPU_INFO}  retriever=${RETRIEVER_LEVEL}"
    fi

    # Loud alerts on threshold crossings (avoid spam: only when level changes)
    if [[ "${RAM_LEVEL}" != "${LAST_RAM_LEVEL}" ]]; then
        case "${RAM_LEVEL}" in
            CRIT) emit CRIT "RAM free dropped to ${RAM_FREE_GB} GB (<${RAM_CRIT_GB} GB threshold). OOM killer imminent. Consider: drop num_retriever, lower gpu_memory_utilization, kill non-essential procs." ;;
            WARN) emit WARN "RAM free dropped to ${RAM_FREE_GB} GB (<${RAM_WARN_GB} GB threshold). Watch for FAISS worker death." ;;
            OK)   [[ -n "${LAST_RAM_LEVEL}" ]] && emit INFO "RAM recovered to ${RAM_FREE_GB} GB free" ;;
        esac
        LAST_RAM_LEVEL="${RAM_LEVEL}"
    fi
    if [[ "${DISK_LEVEL}" != "${LAST_DISK_LEVEL}" ]]; then
        case "${DISK_LEVEL}" in
            CRIT) emit CRIT "Disk free ${DISK_FREE_GB} GB (<${DISK_CRIT_GB} GB). Ray plasma spill will fail. Pruning ckpts now and freeing tmp; check rollout JSONLs + /root/.cache/uv/git-v0/checkouts manually."
                  prune_old_ckpts
                  ;;
            WARN) emit WARN "Disk free ${DISK_FREE_GB} GB (<${DISK_WARN_GB} GB). Pruning ckpts older than the ${KEEP_LAST_N_CKPTS} newest per seed."
                  prune_old_ckpts
                  ;;
            OK)   [[ -n "${LAST_DISK_LEVEL}" ]] && emit INFO "Disk recovered to ${DISK_FREE_GB} GB free" ;;
        esac
        LAST_DISK_LEVEL="${DISK_LEVEL}"
    fi
    if [[ "${RETRIEVER_LEVEL}" == "DOWN" && "${LAST_RETRIEVER_LEVEL}" != "DOWN" ]]; then
        emit CRIT "Retriever /batch_search NOT returning docs. Workers may be OOM-killed (check dmesg | tail -50 | grep -i killed). Training rollouts will get connection-refused and learn no-tool policy."
        LAST_RETRIEVER_LEVEL="DOWN"
    elif [[ "${RETRIEVER_LEVEL}" == "OK" && "${LAST_RETRIEVER_LEVEL}" == "DOWN" ]]; then
        emit INFO "Retriever recovered: /batch_search returning docs"
        LAST_RETRIEVER_LEVEL="OK"
    elif [[ "${RETRIEVER_LEVEL}" == "OK" && -z "${LAST_RETRIEVER_LEVEL}" ]]; then
        LAST_RETRIEVER_LEVEL="OK"
    fi

    # ---- Periodic trace-check: every N steps, analyze the rollout for ----
    # tool-collapse / retriever errors / floor dominance / generic answers.
    # Only triggers when a NEW step lands AND step % N == 0 AND we haven't
    # already analyzed it.
    LATEST_STEP="$(find_latest_rollout_step)"
    LATEST_STEP="${LATEST_STEP:-0}"
    if [[ "${LATEST_STEP}" -gt "${LAST_ANALYZED_STEP}" \
          && "${LATEST_STEP}" -gt 0 \
          && $((LATEST_STEP % TRACE_EVERY_N_STEPS)) -eq 0 ]]; then
        if [[ -x "${CHECK_TRACE}" || -f "${CHECK_TRACE}" ]]; then
            run_trace_check "${LATEST_STEP}"
            LAST_ANALYZED_STEP="${LATEST_STEP}"
        fi
    fi

    sleep "${POLL_S}"
done
