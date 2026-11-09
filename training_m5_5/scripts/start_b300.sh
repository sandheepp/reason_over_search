#!/usr/bin/env bash
# Start M5.5 GRPO training on B300 — single command end-to-end.
#
# Invariant: this script is the "start training in B300" trigger. Run on a
# fully-provisioned Verda B300 box (see docs/vast/SETUP_VAST.md §10 for
# provisioning). It pre-flight-checks every dependency, brings up the
# retriever if it's not already serving, and launches the GRPO loop under
# tmux so the SSH session can drop without killing it.
#
# Usage (from /root/reason_over_search):
#   bash training_m5_5/scripts/start_b300.sh                       # 1× B300, seed 42
#   bash training_m5_5/scripts/start_b300.sh --seed 7              # different seed
#   bash training_m5_5/scripts/start_b300.sh --dry-run             # check pre-flight only
#   bash training_m5_5/scripts/start_b300.sh --mode smoke          # smoke config
#   bash training_m5_5/scripts/start_b300.sh --mode prod_b300_2xgpu  # 2× B300 TP=2
#
# What it does, in order:
#   1. Pre-flight (fail loud if anything is missing):
#        - venv at training_m5_5/nemo_rl/.venv/
#        - data/training/musique/train.parquet
#        - src/prompts/m5_qwen35_user.txt
#        - local_retriever/{corpus,indexes,models}/ assets
#        - training_m5_5/.env with WANDB_API_KEY
#        - CUDA visible and reports B300
#   2. Retriever bring-up:
#        - if 127.0.0.1:3005/health is healthy → skip
#        - else launch retriever_serving.py in tmux session 'retriever' with
#          num_retriever=8 workers, wait until /health responds
#   3. Training launch:
#        - tmux session 'train' runs run.sh --mode prod_b300 --seed N
#        - log streams to /root/logs/m5_5_b300_<TS>.log
#        - returns immediately; tail the log to watch
#
# Idempotent: re-running while training is alive will refuse to start a
# second session (tmux 'train' already exists). Kill the session first if
# you want to rerun.

set -euo pipefail

MODE="prod_b300"
SEED="42"
DRY_RUN=0
SMOKE_FIRST=0
# Auto-pick num_retriever based on host RAM. Each FAISS worker mmaps ~16 GB
# of IVF index resident. 8 workers (128 GB) is fine on a 270 GB B300 box
# but OOM-killed during prod warmup on a 200 GB H200 box (RESULTS_M5_5_B300
# post-mortem covers the exact failure mode). Override with NUM_RETRIEVER=N.
if [[ -z "${NUM_RETRIEVER:-}" ]]; then
    HOST_RAM_GB="$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 0)"
    if   [[ "${HOST_RAM_GB}" -ge 256 ]]; then NUM_RETRIEVER=8   # B300-class
    elif [[ "${HOST_RAM_GB}" -ge 128 ]]; then NUM_RETRIEVER=4   # H200-class
    else                                       NUM_RETRIEVER=2  # A100/4090-class
    fi
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --smoke-first)  SMOKE_FIRST=1; shift ;;
        -h|--help)      sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1 (run with --help)" >&2; exit 2 ;;
    esac
done

# --smoke-first chains smoke (4 steps, ~5-10 min) → prod (--mode) in one tmux
# session. Smoke acts as a tripwire: if it doesn't reach Step 4/4, prod
# does not launch. Useful as the "single command, end-to-end" entry point.
if [[ "${SMOKE_FIRST}" -eq 1 && "${MODE}" == *smoke* ]]; then
    echo "error: --smoke-first is for chaining smoke → a prod mode. Got --mode ${MODE}." >&2
    echo "       try: --smoke-first --mode prod_b300   (or prod_b300_2xgpu)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; exit 1; }
info() { printf '\033[1;34m[info]\033[0m %s\n' "$*"; }

info "B300 launcher — mode=${MODE} seed=${SEED} dry_run=${DRY_RUN}"

# -------- 0. auto-bootstrap if prereqs missing --------
#
# If anything the launcher needs is absent (venv, V2 worker venv, retriever
# assets, MuSiQue parquet, etc.), invoke bootstrap_b300.sh once. The bootstrap
# is idempotent — re-running with everything in place exits in seconds.

VENV_PY="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/python"
V2_VENV_PY="${REPO_ROOT}/training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python"
ASSET_CORPUS="${REPO_ROOT}/local_retriever/corpus/wiki18_100w.jsonl"
ASSET_INDEX="${REPO_ROOT}/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index"
ASSET_MODEL="${REPO_ROOT}/local_retriever/models/e5-base-v2/config.json"
MUSIQUE_PARQUET="${REPO_ROOT}/data/training/musique/train.parquet"
PROMPT_FILE="${REPO_ROOT}/training_m5_5/src/prompts/m5_qwen35_user.txt"
QWEN_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"

NEEDS_BOOTSTRAP=0
[[ -x "${VENV_PY}"        ]] || NEEDS_BOOTSTRAP=1
[[ -x "${V2_VENV_PY}"     ]] || NEEDS_BOOTSTRAP=1
[[ -f "${ASSET_CORPUS}"   ]] || NEEDS_BOOTSTRAP=1
[[ -f "${ASSET_INDEX}"    ]] || NEEDS_BOOTSTRAP=1
[[ -f "${ASSET_MODEL}"    ]] || NEEDS_BOOTSTRAP=1
[[ -f "${MUSIQUE_PARQUET}" ]] || NEEDS_BOOTSTRAP=1
[[ -f "${PROMPT_FILE}"    ]] || NEEDS_BOOTSTRAP=1
[[ -d "${QWEN_CACHE_DIR}/snapshots" ]] || NEEDS_BOOTSTRAP=1

if [[ ${NEEDS_BOOTSTRAP} -eq 1 ]]; then
    warn "missing prerequisites detected — running bootstrap_b300.sh (one-time, ~30-45 min cold)"
    bash "${SCRIPT_DIR}/bootstrap_b300.sh"
else
    ok   "all prereqs satisfied (venvs + assets + model cache)"
fi

# -------- 1. pre-flight --------
info "pre-flight checks..."

# venv (re-check post-bootstrap)
[[ -x "${VENV_PY}" ]] || fail "missing venv: ${VENV_PY} (bootstrap did not produce it)."
ok   "venv:           ${VENV_PY}"

# V2 worker venv (DTensorPolicyWorkerV2)
[[ -x "${V2_VENV_PY}" ]] || fail "missing V2 worker venv: ${V2_VENV_PY} (build via bootstrap_b300.sh)"
ok   "v2 venv:        ${V2_VENV_PY}"

# training data
[[ -f "${MUSIQUE_PARQUET}" ]] \
    || fail "missing MuSiQue parquet. Run: python training_m5_5/scripts/prep_musique.py"
ok   "musique parquet: data/training/musique/train.parquet"

# prompt
[[ -f "${PROMPT_FILE}" ]] \
    || fail "missing prompt. Run: python training_m5_5/scripts/sync_m4_prompts.py --mode qwen35_minimal"
ok   "prompt:         training_m5_5/src/prompts/m5_qwen35_user.txt"

# retriever assets
[[ -f "${ASSET_CORPUS}" ]] || fail "missing retriever corpus: ${ASSET_CORPUS}"
[[ -f "${ASSET_INDEX}"  ]] || fail "missing retriever index:  ${ASSET_INDEX}"
[[ -f "${ASSET_MODEL}"  ]] || fail "missing retriever model:  ${ASSET_MODEL}"
ok   "retriever assets present (corpus 14G, index 15G, encoder)"

# Qwen3.5-0.8B in HF cache (avoid slow anonymous re-download at vLLM warmup)
if [[ -d "${QWEN_CACHE_DIR}/snapshots" ]]; then
    QWEN_SAFETENSORS="$(find "${QWEN_CACHE_DIR}/snapshots" -name 'model.safetensors*' 2>/dev/null | head -1)"
    if [[ -n "${QWEN_SAFETENSORS}" && -s "${QWEN_SAFETENSORS}" ]]; then
        ok "qwen3.5-0.8b:   cached at ${QWEN_SAFETENSORS#${HOME}/}"
    else
        warn "Qwen3.5-0.8B cache dir exists but weights are missing — vLLM will re-download at launch"
    fi
else
    warn "Qwen3.5-0.8B not pre-cached — vLLM will download on first launch (slow if anonymous)"
fi

# .env (WANDB_API_KEY required)
ENV_FILE="${REPO_ROOT}/training_m5_5/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    fail "missing ${ENV_FILE}. Create with: WANDB_API_KEY=<key>"
fi
if ! grep -q '^WANDB_API_KEY=' "${ENV_FILE}"; then
    fail "${ENV_FILE} has no WANDB_API_KEY (required for logger.wandb.enabled=true)"
fi
ok   ".env:           ${ENV_FILE} (WANDB_API_KEY set)"

# CUDA
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Are we on a GPU host?"
fi
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
ok   "gpu:            ${GPU_COUNT}× ${GPU_LINE}"

# Mode↔GPU sanity checks
if [[ "${MODE}" == "prod_b300_2xgpu" && "${GPU_COUNT}" -lt 2 ]]; then
    fail "--mode prod_b300_2xgpu requires 2 GPUs but only ${GPU_COUNT} visible. Use --mode prod_b300 for 1 GPU, or check CUDA_VISIBLE_DEVICES."
fi
if [[ "${MODE}" == "prod_b300" && "${GPU_COUNT}" -ge 2 ]]; then
    warn "${GPU_COUNT} GPUs visible but --mode prod_b300 only uses GPU 0. For both, re-run with --mode prod_b300_2xgpu (~1.7× speedup)."
fi
# Refuse to launch prod_b300* on too-small GPUs (B300 yaml is 100+ GB peak; <80 GB will OOM).
# Smoke is fine on anything ≥20 GB. Use the smoke path for validation on 4090/A6000/etc.
if [[ "${MODE}" == prod_b300* && "${GPU_VRAM_MB}" -lt 80000 ]]; then
    fail "--mode ${MODE} needs ≥80 GB VRAM (B300 config peaks ~100 GB at micro=4/seq=8192). Got ${GPU_VRAM_MB} MB. For cheaper validation use --mode smoke."
fi
if [[ "${MODE}" == "prod_h200" && "${GPU_VRAM_MB}" -lt 120000 ]]; then
    fail "--mode prod_h200 needs ≥120 GB VRAM (H200 config peaks ~127 GB with vLLM at gpu_mem=0.5). Got ${GPU_VRAM_MB} MB. Use --mode prod for ≤80 GB cards or --mode smoke for validation."
fi
if [[ "${MODE}" != *smoke* && "${GPU_LINE}" != *B300* && "${GPU_LINE}" != *B200* ]]; then
    warn "${MODE} is tuned for Blackwell-class GPUs; running on ${GPU_LINE} may underperform or OOM. Smoke first."
fi

# tmux for backgrounding
command -v tmux >/dev/null 2>&1 || fail "tmux not installed (apt-get install -y tmux)"
ok   "tmux:           $(tmux -V)"

# -------- 2. retriever bring-up --------
#
# Lesson from B300 run 2026-05-16 (RESULTS_M5_5_B300_seed42.md): /health
# was returning 200 while the FAISS worker processes were already dead
# (OOM-killed during prod vLLM warmup). Training proceeded with every
# /batch_search returning "Errno 111 Connection refused", and the model
# learned to stop using search entirely. /health is NOT a real liveness
# check — must actually hit /batch_search with a known query.

probe_retriever() {
    # POST a real query and confirm the response is a list-of-list of docs
    # (anything else means workers are dead even if FastAPI is up).
    local resp
    resp="$(curl -sS --max-time 10 -X POST http://127.0.0.1:3005/batch_search \
        -H 'Content-Type: application/json' \
        -d '{"query":["What is the capital of France?"],"top_n":3,"return_score":false}' 2>&1 || true)"
    # Real response shape: [[{"id":"...","contents":"..."},...]]
    if [[ "${resp}" == *'"contents"'* ]]; then
        return 0
    fi
    return 1
}

info "retriever check..."
if probe_retriever; then
    ok   "retriever:      /batch_search returns real docs on :3005"
elif curl -sS --max-time 3 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy; then
    warn "retriever /health healthy but /batch_search not returning docs — workers may be dead. Restarting."
    tmux kill-session -t retriever 2>/dev/null || true
    pkill -9 -f retriever_serving 2>/dev/null || true
    sleep 2
fi

if ! probe_retriever; then
    if [[ ${DRY_RUN} -eq 1 ]]; then
        warn "retriever not running (dry-run; would launch with ${NUM_RETRIEVER} workers)"
    else
        info "starting retriever (num_retriever=${NUM_RETRIEVER}, tmux session 'retriever')..."
        RETRIEVER_VENV="${REPO_ROOT}/local_retriever/.venv_cpu/bin/python"
        [[ -x "${RETRIEVER_VENV}" ]] || fail "retriever venv missing: ${RETRIEVER_VENV}. See docs/setup/SETUP_INSTANCE.md §10c."
        if tmux has-session -t retriever 2>/dev/null; then
            tmux kill-session -t retriever
        fi
        # Tee retriever stdout/stderr to the chain log path so it's preserved
        # alongside the training log when logs are pulled. Also keep the
        # separate retriever.log for backwards-compat.
        tmux new-session -d -s retriever -c "${REPO_ROOT}/local_retriever" \
            "${RETRIEVER_VENV} retriever_serving.py --port 3005 --num_retriever ${NUM_RETRIEVER} 2>&1 | tee /root/logs/retriever.log"
        info "waiting for /batch_search to return real docs (up to 8 min on cold start)..."
        # Cold-start needs time: 8 workers each load 16 GB FAISS index into mmap.
        # We probe /batch_search directly (not /health) since /health lies when
        # workers are dead.
        for i in $(seq 1 160); do
            if probe_retriever; then
                ok "retriever:      /batch_search returns real docs after ${i}×3 s"
                break
            fi
            sleep 3
            if [[ $i -eq 160 ]]; then
                fail "retriever /batch_search did not return real docs after 8 min. Inspect: tmux attach -t retriever (or /root/logs/retriever.log). Check dmesg for OOM-killed workers: dmesg | tail -50 | grep -i killed"
            fi
        done
    fi
fi

# -------- 3. training launch --------
if [[ ${DRY_RUN} -eq 1 ]]; then
    ok "dry-run complete; pre-flight + retriever path validated. Re-run without --dry-run to launch training."
    exit 0
fi

mkdir -p /root/logs
TS="$(date -u +%Y%m%dT%H%MZ)"

if tmux has-session -t train 2>/dev/null; then
    fail "tmux session 'train' already exists. Inspect: tmux attach -t train. Kill it (tmux kill-session -t train) and re-run if you want a fresh launch."
fi

# ---- Resource + trace watcher (RAM/disk/retriever + step-10 trace digest) ----
# Detached background process; survives SSH drops. Logs to two files so they
# don't compete with the chain/training log. Replaces any stale watcher.
RESOURCE_LOG="/root/logs/m5_5_resources_seed${SEED}_${TS}.log"
TRACE_LOG="/root/logs/m5_5_traces_seed${SEED}_${TS}.log"
pkill -f "watch_resources.sh" 2>/dev/null || true
nohup bash "${SCRIPT_DIR}/watch_resources.sh" \
    --log "${RESOURCE_LOG}" \
    --trace-log "${TRACE_LOG}" \
    --trace-every 10 \
    > /dev/null 2>&1 & disown
ok "resource watcher launched (pid $!)"
ok "  resource log: ${RESOURCE_LOG}    (RAM/disk/retriever every 30 s)"
ok "  trace log:    ${TRACE_LOG}    (rollout digest every 10 steps)"

if [[ "${SMOKE_FIRST}" -eq 1 ]]; then
    CHAIN_LOG="/root/logs/m5_5_chain_seed${SEED}_${TS}.log"
    info "launching SMOKE → ${MODE} chain (tmux session 'train'; chain log: ${CHAIN_LOG})"
    tmux new-session -d -s train -c "${REPO_ROOT}" \
        "bash training_m5_5/scripts/smoke_then_prod.sh ${SEED} ${MODE} 2>&1 | tee ${CHAIN_LOG}"
    ok "smoke→prod chain launched."
    echo
    echo "  smoke validates the loop (4 steps, ~5-10 min)"
    echo "  if smoke passes: ${MODE} starts automatically"
    echo "  if smoke fails:  prod does NOT launch; chain exits"
    echo
    echo "  tail chain:    tail -f ${CHAIN_LOG}"
    echo "  tail watcher:  tail -f ${RESOURCE_LOG}     (heartbeats + alerts every 30 s)"
    echo "  tail traces:   tail -f ${TRACE_LOG}       (rollout health digest every 10 steps)"
    echo "  ad-hoc trace:  python training_m5_5/scripts/check_trace.py [--step N]"
    echo "  attach:        tmux attach -t train       (Ctrl-b d to detach)"
    echo "  stop:          tmux kill-session -t train"
    echo "  retriever:     tmux attach -t retriever"
else
    TRAIN_LOG="/root/logs/m5_5_b300_${MODE}_seed${SEED}_${TS}.log"
    info "launching training (tmux session 'train'; log: ${TRAIN_LOG})..."
    tmux new-session -d -s train -c "${REPO_ROOT}" \
        "bash training_m5_5/scripts/run.sh --mode ${MODE} --seed ${SEED} 2>&1 | tee ${TRAIN_LOG}"
    ok "training launched."
    echo
    echo "  tail log:      tail -f ${TRAIN_LOG}"
    echo "  tail watcher:  tail -f ${RESOURCE_LOG}    (heartbeats + alerts every 30 s)"
    echo "  tail traces:   tail -f ${TRACE_LOG}      (rollout health digest every 10 steps)"
    echo "  ad-hoc trace:  python training_m5_5/scripts/check_trace.py [--step N]"
    echo "  attach:        tmux attach -t train       (Ctrl-b d to detach)"
    echo "  stop:          tmux kill-session -t train"
    echo "  retriever:     tmux attach -t retriever"
    echo
    echo "First step typically lands in ~15-20 min (vLLM warm-up + first generation pass)."
fi
