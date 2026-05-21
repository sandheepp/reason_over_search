#!/usr/bin/env bash
# SLURM batch script: run all 7 M4 datasets for ONE Qwen3.5-0.8B variant.
# Submit (full sweep):
#   sbatch scripts/sbatch_m4.sh qwen3.5_0.8b
#   sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base
# Submit (quick smoke, 100 random items per dataset):
#   sbatch scripts/sbatch_m4.sh qwen3.5_0.8b 100
#   sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
#
# Resource sizing identical to sbatch_m3.sh (IVF-SQ8 × 8 retriever workers,
# A100-80GB), with `--time=01:00:00` for the smoke variant since 100 items × 7
# datasets at 0.6 s/item ≈ 7 min on a warm pipeline.

#SBATCH --job-name=m4_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160g
#SBATCH --time=04:00:00
#SBATCH --output=logs/m4_%j_%x.out
#SBATCH --error=logs/m4_%j_%x.err

set -euo pipefail

VARIANT="${1:?Usage: sbatch scripts/sbatch_m4.sh <qwen3.5_0.8b|qwen3.5_0.8b_base> [test_sample_num]}"
TEST_SAMPLE_NUM="${2:-}"

# Use SLURM_SUBMIT_DIR (per CODE_SETUP_m3 #12) so the script resolves correctly under sbatch.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/zfsstore/user/s4374886/omega/reason_over_search}"
IVF_INDEX="$REPO_ROOT/indexes/wiki18_100w_e5_ivf4096_sq8.index"
FLAT_INDEX="/zfsstore/user/s4374886/omega/re-search/assets/indexes/wiki18_100w_e5_flat_inner.index"

# Initialise lmod explicitly. Two reasons this matters under sbatch:
#   1. /etc/profile.d/lmod.sh has an early `return` if $SLURM_NODELIST is set
#      (it's "noop under known resource manager"), so it WON'T define `module`
#      inside an sbatch shell.
#   2. When sbatch is submitted from a non-interactive ssh ("ssh alice 'sbatch …'"),
#      the submitter's environment doesn't have `module` defined either, so
#      --export=ALL doesn't help. Submitting from an interactive ssh login
#      session does work (lmod is initialised by the user's login shell), which
#      is why scripts/sbatch_m3.sh works without this line — submission path
#      matters.
# Init the lmod runtime directly. Path is the standard OpenHPC location.
# /etc/profile.d/lmod.sh exports MODULEPATH=/trinity/shared/modulefiles AND
# sources init/bash, but it early-returns under SLURM, so we replicate both
# steps explicitly. Without MODULEPATH set, `module load ALICE/default` fails
# with "MODULEPATH is undefined" / "unknown module" (observed on srun 2150751).
if ! command -v module >/dev/null 2>&1; then
  export MODULEPATH="${MODULEPATH:-/trinity/shared/modulefiles}"
  . /opt/ohpc/admin/lmod/lmod/init/bash >/dev/null 2>&1 || true
fi
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0

cd "$REPO_ROOT"

# ── Retriever ────────────────────────────────────────────────────────────────
mkdir -p logs
RETRIEVER_LOG="logs/m4_${SLURM_JOB_ID:-local}_retriever.log"
export OMP_NUM_THREADS=4
if [[ -f "$IVF_INDEX" ]]; then
  echo "Using IVF-SQ8 index (× 8 workers): $IVF_INDEX"
  /home/s4374886/.conda/envs/retriever/bin/python -u local_retriever/retriever_serving.py \
    --config local_retriever/retriever_config.yaml \
    --num_retriever 8 \
    --index "$IVF_INDEX" \
    --port 3005 > "$RETRIEVER_LOG" 2>&1 &
else
  echo "IVF index not found; falling back to flat IP (× 2 workers)"
  /home/s4374886/.conda/envs/retriever/bin/python -u local_retriever/retriever_serving.py \
    --config local_retriever/retriever_config.yaml \
    --num_retriever 2 \
    --index "$FLAT_INDEX" \
    --port 3005 > "$RETRIEVER_LOG" 2>&1 &
fi
RETRIEVER_PID=$!
echo "Retriever PID: $RETRIEVER_PID  log: $RETRIEVER_LOG"

# 1200 s wait — same budget as sbatch_m3.sh (post-cliff calibration).
echo "Waiting for retriever to load index (up to 1200 s) ..."
for i in $(seq 1 240); do
  sleep 5
  if curl -sf http://127.0.0.1:3005/health > /dev/null 2>&1; then
    echo "Retriever healthy after $((i*5)) s"
    break
  fi
  if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
    echo "ERROR: retriever process died after $((i*5)) s — last log lines:" >&2
    tail -30 "$RETRIEVER_LOG" >&2
    exit 1
  fi
done
if ! curl -sf http://127.0.0.1:3005/health > /dev/null 2>&1; then
  echo "ERROR: retriever still not healthy after 1200 s — last log lines:" >&2
  tail -30 "$RETRIEVER_LOG" >&2
  kill "$RETRIEVER_PID" 2>/dev/null || true
  exit 1
fi
curl -sS http://127.0.0.1:3005/health

# ── SGLang server ────────────────────────────────────────────────────────────
case "$VARIANT" in
  qwen3.5_0.8b)
    MODEL_PATH="${QWEN35_0_8B_HYBRID_PATH:-$REPO_ROOT/eval/qwen3.5_0.8b}"
    ;;
  qwen3.5_0.8b_base)
    MODEL_PATH="${QWEN35_0_8B_BASE_PATH:-$REPO_ROOT/eval/qwen3.5_0.8b_base}"
    ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model path not found: $MODEL_PATH" >&2
  echo "Set QWEN35_0_8B_HYBRID_PATH / QWEN35_0_8B_BASE_PATH or download via scripts/m4_download_models.sh." >&2
  kill "$RETRIEVER_PID" 2>/dev/null || true
  exit 1
fi

SGLANG_LOG="logs/m4_${SLURM_JOB_ID:-local}_sglang.log"
# Bypass SGLang's PyTorch 2.9.1 + CuDNN<9.15 nn.Conv3d bug check — we serve a
# text LLM with no Conv3d ops, so the warned-against perf/memory regression
# (https://github.com/pytorch/pytorch/issues/168167) doesn't apply. Without
# this, SGLang refuses to launch on the conda env's CuDNN 9.10. Observed on
# sbatch 2150757 (node870, 2026-05-08).
export SGLANG_DISABLE_CUDNN_CHECK=1

# Prepend the conda env's bin to PATH so flashinfer's JIT compile subprocess
# can find `ninja`. Same issue as CODE_SETUP_v2 #2 / "setsid + nohup + PATH"
# wrapper note. Triggered for the first time on L4 (sm_89) sbatch 2151009
# because that GPU has no cached flashinfer binaries; A100 (sm_80) had them
# cached so the JIT never ran. We invoke Python by absolute path elsewhere,
# but flashinfer's child shell inherits the system PATH unless we widen it.
export PATH="/home/s4374886/.conda/envs/evaluation_search_r1/bin:$PATH"
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 3000 \
  --tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "SGLang PID: $SGLANG_PID  log: $SGLANG_LOG"

echo "Waiting for SGLang to load model (up to 600 s) ..."
for i in $(seq 1 120); do
  sleep 5
  if curl -sf http://127.0.0.1:3000/health > /dev/null 2>&1; then
    echo "SGLang healthy after $((i*5)) s"
    break
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "ERROR: SGLang process died after $((i*5)) s — last log lines:" >&2
    tail -30 "$SGLANG_LOG" >&2
    exit 1
  fi
done
if ! curl -sf http://127.0.0.1:3000/health > /dev/null 2>&1; then
  echo "ERROR: SGLang still not healthy after 600 s — last log lines:" >&2
  tail -30 "$SGLANG_LOG" >&2
  kill "$SGLANG_PID" 2>/dev/null || true
  exit 1
fi

# ── Eval loop ─────────────────────────────────────────────────────────────────
for DATASET in bamboogle nq triviaqa popqa hotpotqa 2wikimultihopqa musique; do
  bash scripts/run_m4.sh "$VARIANT" "$DATASET" 1 "$REPO_ROOT/data" "$TEST_SAMPLE_NUM"
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true
wait "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true

echo "All 7 datasets done for variant: $VARIANT (sample_num=${TEST_SAMPLE_NUM:-FULL})"
