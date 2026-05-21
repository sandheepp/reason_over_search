#!/usr/bin/env bash
# M5.6 GRPO (EM-only reward) training launcher for ALICE — 2x A100-80GB variant.
#
# Differences from sbatch_m5_6.sh (1x A100):
#   - SBATCH --gres=gpu:a100:2
#   - default --mode is prod_2xa100 (loads m5_6_research_paper_2xa100.yaml,
#     which adds vllm tensor_parallel_size=2 + cluster.gpus_per_node=2)
#   - CHECKPOINT_DIR_BASE defaults to results/grpo_m5_6_2xa100 so the 1x and 2x
#     runs do not collide in the checkpoint dir
#   - retriever shape is identical (1x retriever process serving HTTP from
#     CPU FAISS; only the trainer needs 2 GPUs)
#
# Dual-use design (SBATCH headers are bash comments):
#   sbatch training_m5_6/scripts/sbatch_m5_1_2xa100.sh
#   bash   training_m5_6/scripts/sbatch_m5_1_2xa100.sh --mode smoke      # inside srun
#
# See sbatch_m5_6.sh for the pre-req checklist (SIF, venv, parquet, .env).

#SBATCH --job-name=m5_6_grpo_2xa100
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=240g
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/m5_6_%j_%x.out
#SBATCH --error=logs/m5_6_%j_%x.err

set -euo pipefail

# ---- args ------------------------------------------------------------------
MODE="prod_2xa100"
SEED="42"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
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

# Differentiate checkpoint dir from the 1x run.
export CHECKPOINT_DIR_BASE="${CHECKPOINT_DIR_BASE:-results/grpo_m5_6_2xa100}"

[ -f "$SIF_PATH" ]                                 || { echo "SIF not found: $SIF_PATH" >&2; exit 1; }
[ -e "local_retriever/${RETRIEVER_INDEX#./}" ] || [ -e "local_retriever/$RETRIEVER_INDEX" ] \
  || { echo "Retriever index missing: local_retriever/${RETRIEVER_INDEX#./}" >&2; exit 1; }
[ -s "data/training/musique/train.parquet" ]        || { echo "MuSiQue parquet missing/empty (run training_m5_6/scripts/prep_musique.py)" >&2; exit 1; }
[ -f "training_m5_6/.env" ]                         || { echo "training_m5_6/.env missing (WANDB_API_KEY)" >&2; exit 1; }
[ -d "training_m5_6/nemo_rl/.venv" ]                || { echo "training venv missing at training_m5_6/nemo_rl/.venv (symlink to shared venv is fine)" >&2; exit 1; }
[ -f "training_m5_6/configs/m5_6_research_paper_2xa100.yaml" ] || { echo "2x A100 config missing at training_m5_6/configs/m5_6_research_paper_2xa100.yaml" >&2; exit 1; }

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
RETRIEVER_LOG="logs/m5_6_${RUN_ID}_retriever.log"
TRAIN_LOG="logs/m5_6_${RUN_ID}_train.log"

ts() { date -u +%FT%TZ; }

echo "[$(ts)] mode=${MODE} seed=${SEED}"
echo "[$(ts)] repo=${REPO_ROOT}"
echo "[$(ts)] retriever log: ${RETRIEVER_LOG}"
echo "[$(ts)] train log:     ${TRAIN_LOG}"
echo "[$(ts)] host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')"

# ---- start retriever (background apptainer exec) ---------------------------
echo "[$(ts)] Starting retriever (port=${RETRIEVER_PORT}, workers=${RETRIEVER_NUM_WORKERS}, index=${RETRIEVER_INDEX})"
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
if grep -Eq '^HF_TOKEN=[^[:space:]]+' "training_m5_6/.env" 2>/dev/null; then
  echo "[$(ts)] Starting HF Hub upload watcher (mode=${MODE} seed=${SEED})"
  HF_UPLOADER_LOG="logs/m5_6_${RUN_ID}_hf_uploader.log"
  apptainer exec --bind "$BIND" --env "HF_HOME=/workspace/hf_cache" --env "CHECKPOINT_DIR_BASE=${CHECKPOINT_DIR_BASE}" "$SIF_PATH" \
    bash -lc "
      cd /workspace/reason_over_search
      bash training_m5_6/scripts/upload_ckpts_watcher.sh --mode '${MODE}' --seed '${SEED}'
    " > "$HF_UPLOADER_LOG" 2>&1 &
  HF_UPLOADER_PID=$!
  echo "[$(ts)] HF uploader PID: ${HF_UPLOADER_PID} (log: ${HF_UPLOADER_LOG})"
else
  echo "[$(ts)] HF_TOKEN unset in training_m5_6/.env — skipping HF Hub uploader"
fi

# ---- launch training (foreground apptainer exec) ---------------------------
echo "[$(ts)] Launching training (mode=${MODE} seed=${SEED})"
set +e
apptainer exec --nv --bind "$BIND" --env "HF_HOME=/workspace/hf_cache" --env "CHECKPOINT_DIR_BASE=${CHECKPOINT_DIR_BASE}" "$SIF_PATH" \
  bash -lc "
    cd /workspace/reason_over_search
    bash training_m5_6/scripts/run.sh --mode '${MODE}' --seed '${SEED}'
  " 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(ts)] training exit=${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
