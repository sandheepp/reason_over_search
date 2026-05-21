#!/usr/bin/env bash
# Idempotent one-shot bootstrap for Alice HPC using Apptainer.
#
# Why Apptainer: NeMo-RL pins torch==2.10.0 from the pytorch-cu129 index
# and TransformerEngine to CUDA 12.9+. Alice's highest available CUDA module
# is 12.4.0, so native conda/uv installs won't produce a working training env.
# The Docker image pantomiman/reason-over-search-v1:v2 (CUDA 12.9.1) is the
# only supported runtime; Apptainer lets us run it on SLURM without root.
#
# What this script does (all idempotent):
#   1. Pull the Docker image as a SIF file (→ SIF_DIR)
#   2. Git LFS pull for training parquets + eval jsonls
#   3. Run training/scripts/bootstrap.sh inside the container:
#        - uv sync → training/nemo_rl/.venv
#        - download v2/automodel venv from HF (or compile, needs GPU)
#        - download Qwen3.5-2B-Base + Qwen3.5-2B → HF_HOME
#   4. Download retriever assets inside the container:
#        - intfloat/e5-base-v2 → local_retriever/models/
#        - IVF-SQ8 index (~16 GB) → local_retriever/indexes/
#        - corpus (~5 GB) → local_retriever/corpus/
#
# Usage:
#   bash training/scripts/bootstrap_alice.sh
#   SKIP_V2_BUILD=1 bash ...    # skip v2 venv (safe if you only need retriever/eval)
#   V2_BUILD_FROM_SOURCE=1 bash ...  # force compile instead of HF tarball download
#
# Paths (all on ZFS, accessible from any compute node):
#   SIF:      /zfsstore/user/s4374886/apptainer/reason-over-search-v1.sif
#   HF cache: /zfsstore/user/s4374886/hf_cache
#   Repo:     /zfsstore/user/s4374886/omega/reason_over_search  (this repo)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; RED=$'\033[0;31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
log()  { echo "${BOLD}${GREEN}▶${RESET} $*"; }
warn() { echo "${BOLD}${YELLOW}⚠${RESET} $*"; }
err()  { echo "${BOLD}${RED}✗${RESET} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Config: edit these if paths change
# ---------------------------------------------------------------------------
SIF_DIR="/zfsstore/user/s4374886/apptainer"
SIF_PATH="${SIF_DIR}/reason-over-search-v1.sif"
DOCKER_IMAGE="docker://pantomiman/reason-over-search-v1:v2"

HF_HOME="${HF_HOME:-/zfsstore/user/s4374886/hf_cache}"
UV_CACHE_PATH="${UV_CACHE_PATH:-/zfsstore/user/s4374886/uv_cache}"
mkdir -p "$HF_HOME" "$SIF_DIR" "$UV_CACHE_PATH"

# Bind mounts: maps ZFS paths into the container's expected /workspace layout.
# UV_CACHE_PATH → /.uv/cache: the SIF bakes a pre-warmed wheel cache at /.uv/cache
# but Apptainer makes the container read-only. We copy it out to ZFS once
# (see step below) and bind-mount the writable copy back so uv sync can lock/write.
BIND="${REPO_ROOT}:/workspace/reason_over_search,${HF_HOME}:/workspace/hf_cache,${UV_CACHE_PATH}:/.uv/cache"

# Helper: run a command inside the container
apptainer_exec() {
  apptainer exec \
    --nv \
    --bind "${BIND}" \
    --env "HF_HOME=/workspace/hf_cache" \
    "${SIF_PATH}" \
    "$@"
}

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------
command -v apptainer >/dev/null 2>&1 || err "apptainer not found on PATH"
log "Apptainer: $(apptainer --version)"

# ---------------------------------------------------------------------------
# 1. Pull Docker image → SIF  (~20-30 GB, one-time, ~15 min on fast link)
# ---------------------------------------------------------------------------
if [[ -f "${SIF_PATH}" ]]; then
  log "SIF already exists at ${SIF_PATH}; skip pull"
else
  log "Pulling ${DOCKER_IMAGE} → ${SIF_PATH} (one-time, ~20-30 GB)…"
  apptainer pull "${SIF_PATH}" "${DOCKER_IMAGE}"
fi

# ---------------------------------------------------------------------------
# 1b. Extract pre-warmed uv wheel cache from SIF → writable ZFS copy
# The SIF bakes /.uv/cache (~13 GB of pre-resolved torch/vLLM wheels).
# Apptainer makes the SIF read-only, so uv can't acquire locks there.
# We extract once and bind-mount the ZFS copy back over /.uv/cache.
# ---------------------------------------------------------------------------
if [[ ! -f "${UV_CACHE_PATH}/.sif-extracted" ]]; then
  log "Extracting pre-warmed uv wheel cache from SIF (~13 GB, one-time)…"
  apptainer exec "${SIF_PATH}" bash -c "cd /.uv/cache && tar -cf - ." \
    | tar -xf - -C "${UV_CACHE_PATH}"
  touch "${UV_CACHE_PATH}/.sif-extracted"
  log "uv cache extracted: $(du -sh "${UV_CACHE_PATH}" | cut -f1)"
else
  log "uv wheel cache already extracted at ${UV_CACHE_PATH}; skip"
fi

# ---------------------------------------------------------------------------
# 2. Git LFS: pull training parquets and eval jsonls
# ---------------------------------------------------------------------------
if git lfs version >/dev/null 2>&1; then
  if [[ ! -s data/training/nq_hotpotqa_train/train.parquet ]] || \
     [[ $(wc -c < data/training/nq_hotpotqa_train/train.parquet) -lt 1000 ]]; then
    log "Pulling Git LFS objects…"
    git lfs install >/dev/null
    git lfs pull
  else
    log "Git LFS data already present; skip"
  fi
else
  warn "git-lfs not found; skipping LFS pull. Install git-lfs if data files are stubs."
fi

# ---------------------------------------------------------------------------
# 3. Run bootstrap.sh inside the container
#    - creates training/nemo_rl/.venv (uv sync vllm+automodel)
#    - downloads v2 venv from HF or compiles from source
#    - downloads Qwen3.5-2B-Base + Qwen3.5-2B
#
# SKIP_RETRIEVER=1: don't try to start the retriever server here (login node).
# Pass V2_BUILD_FROM_SOURCE / SKIP_V2_BUILD through if set.
# ---------------------------------------------------------------------------
BOOTSTRAP_ENV="SKIP_RETRIEVER=1"
[[ "${SKIP_V2_BUILD:-0}" == "1" ]]        && BOOTSTRAP_ENV+=" SKIP_V2_BUILD=1"
[[ "${V2_BUILD_FROM_SOURCE:-0}" == "1" ]] && BOOTSTRAP_ENV+=" V2_BUILD_FROM_SOURCE=1"

log "Running bootstrap.sh inside container…"
apptainer_exec bash -c \
  "cd /workspace/reason_over_search && ${BOOTSTRAP_ENV} bash training/scripts/bootstrap.sh"

# ---------------------------------------------------------------------------
# 4. Retriever assets
# Corpus and e5-base-v2 are symlinked from existing copies in flash-rag/.
# Only the IVF-SQ8 index is new and needs to be downloaded from HF.
# ---------------------------------------------------------------------------
mkdir -p \
  "${REPO_ROOT}/local_retriever/models" \
  "${REPO_ROOT}/local_retriever/indexes" \
  "${REPO_ROOT}/local_retriever/corpus"

# e5-base-v2: symlink from flash-rag (already downloaded)
FLASH_RAG_E5="/zfsstore/user/s4374886/omega/flash-rag/models/e5-base-v2"
if [[ ! -e "${REPO_ROOT}/local_retriever/models/e5-base-v2" ]]; then
  log "Symlinking e5-base-v2 from flash-rag…"
  ln -sfn "${FLASH_RAG_E5}" "${REPO_ROOT}/local_retriever/models/e5-base-v2"
else
  log "e5-base-v2 already present; skip"
fi

# Corpus: symlink from flash-rag (14 GB, already downloaded)
FLASH_RAG_CORPUS="/zfsstore/user/s4374886/omega/flash-rag/corpus/wiki18_100w.jsonl"
if [[ ! -e "${REPO_ROOT}/local_retriever/corpus/wiki18_100w.jsonl" ]]; then
  log "Symlinking corpus from flash-rag…"
  ln -sfn "${FLASH_RAG_CORPUS}" "${REPO_ROOT}/local_retriever/corpus/wiki18_100w.jsonl"
else
  log "Corpus already present; skip"
fi

# IVF-SQ8 index (~16 GB): download from HF (only new asset needed)
if [[ ! -f "${REPO_ROOT}/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index" ]]; then
  log "Downloading IVF-SQ8 index from pantomiman/reason-over-search (~16 GB)…"
  apptainer_exec bash -lc "
    source /opt/miniforge3/etc/profile.d/conda.sh && conda activate retriever
    cd /workspace/reason_over_search
    hf download pantomiman/reason-over-search \
      retriever/wiki18_100w_e5_ivf4096_sq8.index \
      --repo-type dataset \
      --local-dir local_retriever/indexes
  "
  # HF preserves repo directory structure; flatten to the path retriever_config.yaml expects
  mv "${REPO_ROOT}/local_retriever/indexes/retriever/wiki18_100w_e5_ivf4096_sq8.index" \
     "${REPO_ROOT}/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index"
  rmdir "${REPO_ROOT}/local_retriever/indexes/retriever" 2>/dev/null || true
else
  log "IVF-SQ8 index already present; skip"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "${BOLD}Bootstrap complete.${RESET}"
echo
echo "SIF:           ${SIF_PATH}"
echo "HF cache:      ${HF_HOME}"
echo "Repo:          ${REPO_ROOT}"
echo ""
echo "To run on a compute node, add this to your sbatch / srun command:"
echo ""
echo "  apptainer exec \\"
echo "    --nv \\"
echo "    --bind ${REPO_ROOT}:/workspace/reason_over_search,${HF_HOME}:/workspace/hf_cache \\"
echo "    --env HF_HOME=/workspace/hf_cache \\"
echo "    ${SIF_PATH} \\"
echo "    bash /workspace/reason_over_search/training/scripts/run_grpo_1xa100.sh \\"
echo "      --variant base --seed 42 --arm qwen_native \\"
echo "      -- \\"
echo "      grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20 \\"
echo "      policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true \\"
echo "      policy.train_micro_batch_size=2"
echo ""
echo "To start the retriever on a compute node:"
echo ""
echo "  apptainer exec \\"
echo "    --nv \\"
echo "    --bind ${REPO_ROOT}:/workspace/reason_over_search,${HF_HOME}:/workspace/hf_cache \\"
echo "    ${SIF_PATH} \\"
echo "    bash -lc 'source /opt/miniforge3/etc/profile.d/conda.sh && conda activate retriever &&"
echo "      cd /workspace/reason_over_search/local_retriever &&"
echo "      python retriever_serving.py --config retriever_config.yaml --num_retriever 8 --port 3005'"
