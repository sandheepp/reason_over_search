#!/usr/bin/env bash
# M5.5 GRPO (F1+format reward variant) training launcher for ALICE (Apptainer SIF + 1x A100-80GB).
#
# Dual-use design: the SBATCH headers below are bash comments, so the same
# file runs in two modes:
#   1) Submitted via sbatch  -> reads the headers, queues on gpu-a100-80g.
#   2) Invoked via bash      -> headers ignored, body runs in foreground;
#                              used inside an srun for end-to-end smoke
#                              verification before submitting the prod job.
#
# Examples:
#   # Production (1x A100-80GB, 7-day walltime, prod config + seed 42):
#   sbatch training_m5_5/scripts/sbatch_m5_5.sh
#
#   # Production with a different seed:
#   sbatch training_m5_5/scripts/sbatch_m5_5.sh --seed 7
#
#   # In-srun smoke verification (must already be in an srun shell):
#   bash training_m5_5/scripts/sbatch_m5_5.sh --mode smoke
#
# Pre-requisites (idempotent; mirror what training_m5_5/scripts/bootstrap_alice.sh
# does, with the ALICE-side symlink layout):
#   1. Apptainer SIF present at $SIF_PATH
#   2. training/nemo_rl/.venv built (or symlinked from a sibling clone)
#   3. data/training/musique/train.parquet present (real, not LFS stub;
#      training_m5_5/scripts/prep_musique.py rebuilds it)
#   4. local_retriever/{indexes,models,corpus}/ populated (symlinks are fine)
#   5. training_m5_5/.env with WANDB_API_KEY (+ optional ENTITY/PROJECT)
#
# Apptainer / networking notes:
#   - apptainer exec uses the host's network namespace by default. The
#     retriever bound to 127.0.0.1:$RETRIEVER_PORT in one exec is reachable
#     from a second exec running the trainer (both see the host's loopback).
#   - Two execs instead of one keeps the retriever's conda env (retriever,
#     faiss-cpu) cleanly separated from the trainer's uv venv at
#     training/nemo_rl/.venv (torch 2.10 + vllm 0.17).

#SBATCH --job-name=m5_5_grpo
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=240g
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/m5_5_%j_%x.out
#SBATCH --error=logs/m5_5_%j_%x.err

set -euo pipefail

# ---- args ------------------------------------------------------------------
MODE="prod"
SEED="42"
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --) shift; EXTRA_OVERRIDES=("$@"); break ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- paths -----------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

SIF_PATH="${SIF_PATH:-/zfsstore/user/s4374886/apptainer/reason-over-search-v1.sif}"
HF_HOME_HOST="${HF_HOME_HOST:-/zfsstore/user/s4374886/hf_cache}"
UV_CACHE_HOST="${UV_CACHE_HOST:-/zfsstore/user/s4374886/uv_cache}"
RETRIEVER_PORT="${RETRIEVER_PORT:-3005}"
RETRIEVER_INDEX="${RETRIEVER_INDEX:-./indexes/wiki18_100w_e5_ivf4096_sq8.index}"
RETRIEVER_NUM_WORKERS="${RETRIEVER_NUM_WORKERS:-8}"
RETRIEVER_HEALTH_TIMEOUT_S="${RETRIEVER_HEALTH_TIMEOUT_S:-5400}"   # 90 min; cold zfsstore on ALICE measured at ~40 MB/s; 8x retriever boot worst-case ~30-40 min so we keep ~50% safety margin

# Differentiate checkpoint dir from M5.1 baseline so the reward-ablation
# triad does not collide on results/grpo/m5_prod/seed42/.
export CHECKPOINT_DIR_BASE="${CHECKPOINT_DIR_BASE:-results/grpo_m5_5}"

[ -f "$SIF_PATH" ]                                 || { echo "SIF not found: $SIF_PATH" >&2; exit 1; }
[ -e "local_retriever/${RETRIEVER_INDEX#./}" ] || [ -e "local_retriever/$RETRIEVER_INDEX" ] \
  || { echo "Retriever index missing: local_retriever/${RETRIEVER_INDEX#./}" >&2; exit 1; }
[ -s "data/training/musique/train.parquet" ]        || { echo "MuSiQue parquet missing/empty (run training_m5_5/scripts/prep_musique.py)" >&2; exit 1; }
[ -f "training_m5_5/.env" ]                         || { echo "training_m5_5/.env missing (WANDB_API_KEY)" >&2; exit 1; }
[ -d "training_m5_5/nemo_rl/.venv" ]                || { echo "training venv missing at training_m5_5/nemo_rl/.venv (bootstrap_alice.sh or setup.sh; symlink to shared venv is fine)" >&2; exit 1; }

mkdir -p logs

# Extra bind: expose /zfsstore/user/<uid> identically inside the container so
# host-absolute symlinks (e.g. local_retriever/corpus -> /zfsstore/.../flash-rag/...,
# local_retriever/indexes/...index -> /zfsstore/.../reason_over_search/...) resolve.
ZFS_USER_ROOT="${ZFS_USER_ROOT:-/zfsstore/user/$(id -un)}"

RUN_ID="${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
# Apptainer SIF's rootfs has /scratchdata as a read-only mount-point that Ray
# (via NeMo-RL's init_ray) tries to mkdir on init. Bind a writable per-job dir.
RAY_SCRATCH_HOST="${REPO_ROOT}/logs/ray_scratch_${RUN_ID}"
mkdir -p "${RAY_SCRATCH_HOST}"
BIND="${REPO_ROOT}:/workspace/reason_over_search,${HF_HOME_HOST}:/workspace/hf_cache,${UV_CACHE_HOST}:/.uv/cache,${ZFS_USER_ROOT}:${ZFS_USER_ROOT},${RAY_SCRATCH_HOST}:/scratchdata"
RETRIEVER_LOG="logs/m5_5_${RUN_ID}_retriever.log"
TRAIN_LOG="logs/m5_5_${RUN_ID}_train.log"

ts() { date -u +%FT%TZ; }

echo "[$(ts)] mode=${MODE} seed=${SEED}"
echo "[$(ts)] repo=${REPO_ROOT}"
echo "[$(ts)] retriever log: ${RETRIEVER_LOG}"
echo "[$(ts)] train log:     ${TRAIN_LOG}"
echo "[$(ts)] host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')"

# ---- start retriever (background apptainer exec) ---------------------------
echo "[$(ts)] Starting retriever (port=${RETRIEVER_PORT}, workers=${RETRIEVER_NUM_WORKERS}, index=${RETRIEVER_INDEX})"
# OMP threads-per-worker = 4 matches the M4 eval pipeline (sbatch_m4.sh) which
# runs the same 8-worker retriever shape under prod-rate query load.
apptainer exec --nv --bind "$BIND" --env "HF_HOME=/workspace/hf_cache" --env "OMP_NUM_THREADS=4" "$SIF_PATH" \
  bash -lc "
    source /opt/miniforge3/etc/profile.d/conda.sh && conda activate retriever
    cd /workspace/reason_over_search/local_retriever
    exec python -u retriever_serving.py \
      --config retriever_config.yaml \
      --num_retriever ${RETRIEVER_NUM_WORKERS} \
      --index ${RETRIEVER_INDEX} \
      --port ${RETRIEVER_PORT}
  " > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
echo "[$(ts)] Retriever wrapper PID: ${RETRIEVER_PID}"

HF_UPLOADER_PID=""
cleanup() {
  echo "[$(ts)] cleanup"
  if [ -n "${HF_UPLOADER_PID:-}" ] && kill -0 "$HF_UPLOADER_PID" 2>/dev/null; then
    kill "$HF_UPLOADER_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$HF_UPLOADER_PID" 2>/dev/null || true
  fi
  if [ -n "${RETRIEVER_PID:-}" ] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
    kill "$RETRIEVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$RETRIEVER_PID" 2>/dev/null || true
  fi
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- wait for /health ------------------------------------------------------
echo "[$(ts)] Waiting for retriever /health (up to ${RETRIEVER_HEALTH_TIMEOUT_S} s)..."
DEADLINE=$((SECONDS + RETRIEVER_HEALTH_TIMEOUT_S))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
    echo "[$(ts)] Retriever healthy (waited $((SECONDS))s)"
    break
  fi
  if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
    echo "ERROR: retriever wrapper exited before /health became reachable" >&2
    tail -50 "$RETRIEVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null \
  || { echo "ERROR: /health never reachable within ${RETRIEVER_HEALTH_TIMEOUT_S}s" >&2; tail -50 "$RETRIEVER_LOG" >&2; exit 1; }
curl -sS "http://127.0.0.1:${RETRIEVER_PORT}/health"
echo

# ---- start HF Hub upload watcher (decoupled background, optional) ----------
if grep -Eq '^HF_TOKEN=[^[:space:]]+' "training_m5_5/.env" 2>/dev/null; then
  echo "[$(ts)] Starting HF Hub upload watcher (mode=${MODE} seed=${SEED})"
  HF_UPLOADER_LOG="logs/m5_5_${RUN_ID}_hf_uploader.log"
  apptainer exec --bind "$BIND" --env "HF_HOME=/workspace/hf_cache" --env "CHECKPOINT_DIR_BASE=${CHECKPOINT_DIR_BASE}" "$SIF_PATH" \
    bash -lc "
      cd /workspace/reason_over_search
      bash training_m5_5/scripts/upload_ckpts_watcher.sh --mode '${MODE}' --seed '${SEED}'
    " > "$HF_UPLOADER_LOG" 2>&1 &
  HF_UPLOADER_PID=$!
  echo "[$(ts)] HF uploader PID: ${HF_UPLOADER_PID} (log: ${HF_UPLOADER_LOG})"
else
  echo "[$(ts)] HF_TOKEN unset in training_m5_5/.env — skipping HF Hub uploader"
fi

# ---- launch training (foreground apptainer exec) ---------------------------
echo "[$(ts)] Launching training (mode=${MODE} seed=${SEED})${EXTRA_OVERRIDES:+ overrides=${EXTRA_OVERRIDES[*]}}"
EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES[*]:-}"
set +e
apptainer exec --nv --bind "$BIND" --env "HF_HOME=/workspace/hf_cache" --env "CHECKPOINT_DIR_BASE=${CHECKPOINT_DIR_BASE}" "$SIF_PATH" \
  bash -lc "
    cd /workspace/reason_over_search
    bash training_m5_5/scripts/run.sh --mode '${MODE}' --seed '${SEED}' ${EXTRA_OVERRIDES_STR:+-- ${EXTRA_OVERRIDES_STR}}
  " 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(ts)] training exit=${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
