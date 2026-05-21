#!/usr/bin/env bash
# SLURM batch script: run all 7 M3 datasets for ONE model variant.
# Submit with:
#   sbatch scripts/sbatch_m3.sh qwen3_0.6b
#   sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0
#
# Resource sizing (IVF-SQ8 × 8 workers):
#   RAM:  8 retrievers × ~16 GB each + corpus + encoder ≈ 134 GB resident
#         → 160 GB requested for headroom
#   CPU:  8 retriever workers × 4 OMP threads = 32 FAISS threads
#         + SGLang tokeniser/scheduler + uvicorn overhead ≈ 40 threads peak
#         → 40 CPUs requested
#   GPU:  Qwen3-0.6B at bf16 ≈ 1.2 GB VRAM; A100 80GB is heavily under-utilised
#         but required for the partition
#   Time: ~1.5–2 h per variant (estimated; update after first validated run)

#SBATCH --job-name=m3_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160g
#SBATCH --time=04:00:00
#SBATCH --output=logs/m3_%j_%x.out
#SBATCH --error=logs/m3_%j_%x.err

set -euo pipefail

VARIANT="${1:?Usage: sbatch scripts/sbatch_m3.sh <qwen3_0.6b|qwen3_0.6b_v0|qwen3_0.6b_v0_no_ex>}"
# Under sbatch, SLURM copies the script to /var/spool/slurmd/<jobid>/slurm_script,
# so BASH_SOURCE-based REPO_ROOT resolves wrong. Use SLURM_SUBMIT_DIR (set by SLURM
# to the directory where `sbatch` was invoked) with a hardcoded fallback for direct execution.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/zfsstore/user/s4374886/omega/reason_over_search}"
IVF_INDEX="$REPO_ROOT/indexes/wiki18_100w_e5_ivf4096_sq8.index"
FLAT_INDEX="/zfsstore/user/s4374886/omega/re-search/assets/indexes/wiki18_100w_e5_flat_inner.index"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0

cd "$REPO_ROOT"

# ── Retriever ────────────────────────────────────────────────────────────────
mkdir -p logs
RETRIEVER_LOG="logs/m3_${SLURM_JOB_ID:-local}_retriever.log"
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

# First-time HF arrow build of wiki18_100w.jsonl (~21M passages) can take 5–10 min;
# subsequent loads (cache warm) are seconds. 1200 s = 20 min total wait (bumped from
# 600 s after sbatch 2134663 on node873 hit the cold-cache cliff: empty log at 600 s,
# retriever Python process alive throughout — index mmap + arrow loading were just
# slower than the prior 570 s reference; doubling gives a margin without affecting
# warm-cache performance, since the loop breaks on /health=200 the moment it's ready).
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
  qwen3_0.6b)            MODEL_PATH="$REPO_ROOT/eval/qwen_3_0.6b" ;;
  qwen3_0.6b_v0)         MODEL_PATH="$REPO_ROOT/eval/qwen_3_0.6b_v0" ;;
  qwen3_0.6b_v0_no_ex)   MODEL_PATH="$REPO_ROOT/eval/qwen_3_0.6b_v0_no_ex" ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

SGLANG_LOG="logs/m3_${SLURM_JOB_ID:-local}_sglang.log"
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
  bash scripts/run_m3.sh "$VARIANT" "$DATASET" 1
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true
wait "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true

echo "All 7 datasets done for variant: $VARIANT"
