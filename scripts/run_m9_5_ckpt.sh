#!/usr/bin/env bash
# Per-ckpt eval wrapper: convert NeMo native ckpt -> HF -> vLLM serve
# + SGL->vLLM proxy -> eval 7 datasets -> HF push results -> cleanup.
#
# vLLM is used instead of SGLang because sgl_kernel has no sm120 (Blackwell)
# prebuilt binary; the proxy translates flashrag's SGLang /generate protocol
# to vLLM's /v1/completions so the existing eval pipeline works unchanged.
#
# Usage:
#   bash scripts/run_m9_5_ckpt.sh <RUN_TAG> <STEP> <GPU> [SLOT]
# SLOT (0..3, default 0) multiplexes up to 4 vLLM instances per GPU when paired
# with `gpu-memory-utilization 0.22` (each gets ~21 GB on the 96 GB Blackwell).
# Port allocation: VLLM=8000+GPU*4+SLOT, PROXY=9000+GPU*4+SLOT.
# Where RUN_TAG in: m5_1_s43 | m5_5_s43 | m5_6_s44 | m5_1_s44 | m5_5_s44
set -euo pipefail

# Raise FD limit. Default ulimit -n 1024 is way too low for the
# vLLM + httpx + flashrag stack under 8-way concurrent load; the first 8-way
# sweep (2026-05-27) silently lost all 7-dataset metric files for ~88 ckpts
# because run_eval.py crashed mid-eval with OSError: Too many open files
# (errno 24) after writing config.yaml but before metric_score.txt. The
# wrapper's `|| echo "WARN: $ds failed"` swallowed the error and the
# dispatcher marked ckpts DONE. Hard limit is 1M so 65536 is safe.
ulimit -n 65536 || echo "WARN: could not raise ulimit -n"

RUN_TAG="${1:?Usage: run_m9_5_ckpt.sh <RUN_TAG> <STEP> <GPU> [SLOT]}"
STEP="${2:?Usage: run_m9_5_ckpt.sh <RUN_TAG> <STEP> <GPU> [SLOT]}"
GPU="${3:-0}"
SLOT="${4:-0}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$REPO_ROOT/logs/m9_5"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_TAG}_step${STEP}_gpu${GPU}_slot${SLOT}.log"

# Load HF + WANDB env vars
set -a; source "$REPO_ROOT/training_m5_5/.env"; set +a

case "$RUN_TAG" in
  m5_1_s43) SRC="$REPO_ROOT/results/grpo/m5_1_prod_dp4_4xblackwell/seed43/step_${STEP}" ;;
  m5_5_s43) SRC="$REPO_ROOT/results/grpo/m5_5_prod_dp4_4xblackwell/seed43/step_${STEP}" ;;
  m5_6_s44) SRC="$REPO_ROOT/results/grpo/m5_prod_dp4_4xblackwell/seed44/step_${STEP}" ;;
  m5_1_s44) SRC="$REPO_ROOT/results/grpo/m5_1_prod_dp4_4xblackwell/seed44/step_${STEP}" ;;
  m5_5_s44) SRC="$REPO_ROOT/results/grpo/m5_5_prod_dp4_4xblackwell/seed44/step_${STEP}" ;;
  *) echo "ERROR: unknown RUN_TAG: $RUN_TAG" >&2; exit 1 ;;
esac

HF_OUT="$REPO_ROOT/eval/m9_5/${RUN_TAG}/step_${STEP}_hf"
BASE_DIR="$REPO_ROOT/eval/qwen3.5_0.8b"
VENV="$REPO_ROOT/training_m5_5/nemo_rl/.venv"
PY="$VENV/bin/python"
RESULTS_REPO="pantomiman/reason-over-search-eval-results"

VLLM_PORT=$((8000 + GPU*4 + SLOT))
PROXY_PORT=$((9000 + GPU*4 + SLOT))
RET_PORT=3005
# Default 0.40 = 2 vLLMs/GPU on the 96 GB Blackwell. vLLM v1 sizes KV cache from
# *free* not *total*, so 4-slot/GPU needs --enforce-eager + --max-model-len 4096
# to fit; tested 2026-05-27 and the cold-start collision (5 min per slot) wiped
# out the parallelism gain, making 4-slot slower per ckpt than 2-slot. Bump to
# 0.85 for single-slot mode.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENFORCE_EAGER_FLAG="${ENFORCE_EAGER_FLAG:-}"

echo "[$(date -u +%H:%M:%S)] === M9.5 ckpt eval: RUN=$RUN_TAG STEP=$STEP GPU=$GPU SLOT=$SLOT vllm=$VLLM_PORT proxy=$PROXY_PORT mem_util=$GPU_MEM_UTIL ===" | tee -a "$LOG"

[[ -d "$SRC" ]] || { echo "ERROR: source ckpt not found: $SRC" | tee -a "$LOG"; exit 2; }

# 1. Convert NeMo native -> HF
if [[ ! -d "$HF_OUT" ]]; then
  echo "[$(date -u +%H:%M:%S)] convert $SRC -> $HF_OUT" | tee -a "$LOG"
  mkdir -p "$(dirname "$HF_OUT")"
  "$PY" "$REPO_ROOT/scripts/convert_m5_ckpt_to_hf.py" \
    --ckpt-dir "$SRC" \
    --base-dir "$BASE_DIR" \
    --out-dir  "$HF_OUT" >> "$LOG" 2>&1
fi

# 2. Verify retriever running on port 3005
if ! curl -sf "http://127.0.0.1:${RET_PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: retriever not running on port $RET_PORT" | tee -a "$LOG"
  exit 3
fi

# 3. Start vLLM OpenAI-compat server on GPU
VLLM_LOG="$LOG_DIR/${RUN_TAG}_step${STEP}_gpu${GPU}_slot${SLOT}_vllm.log"
echo "[$(date -u +%H:%M:%S)] start vLLM on GPU $GPU slot $SLOT port $VLLM_PORT" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$HF_OUT" \
  --port "$VLLM_PORT" \
  --host 127.0.0.1 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  $ENFORCE_EAGER_FLAG \
  --no-enable-log-requests > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "  vllm pid=$VLLM_PID log=$VLLM_LOG" | tee -a "$LOG"

# Wait for vllm (up to 300s; cold start ~60-90s, but be safe)
VLLM_READY=0
for i in $(seq 1 60); do
  sleep 5
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] vllm ready in $((i*5)) s" | tee -a "$LOG"
    VLLM_READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vllm died; last log lines:" | tee -a "$LOG"
    tail -50 "$VLLM_LOG" | tee -a "$LOG"
    exit 4
  fi
done
[[ "$VLLM_READY" == "1" ]] || { echo "ERROR: vllm not ready after 300 s" | tee -a "$LOG"; kill -KILL "$VLLM_PID" 2>/dev/null; exit 4; }

# 4. Start SGL->vLLM proxy
PROXY_LOG="$LOG_DIR/${RUN_TAG}_step${STEP}_gpu${GPU}_slot${SLOT}_proxy.log"
echo "[$(date -u +%H:%M:%S)] start sgl->vllm proxy on port $PROXY_PORT" | tee -a "$LOG"
"$PY" "$REPO_ROOT/scripts/sgl_to_vllm_proxy.py" \
  --proxy-port "$PROXY_PORT" \
  --vllm-url "http://127.0.0.1:${VLLM_PORT}" \
  --model "$HF_OUT" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "  proxy pid=$PROXY_PID log=$PROXY_LOG" | tee -a "$LOG"

# Wait for proxy
for i in $(seq 1 20); do
  sleep 1
  if curl -sf "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] proxy ready in $i s" | tee -a "$LOG"
    break
  fi
done

# 5. Run 7 datasets sequentially
EVAL_DIR="$REPO_ROOT/evaluation_qwen35"
SAVE_NOTE="m9_5_${RUN_TAG}_step${STEP}_seed1"
SGL_REMOTE_URL="http://127.0.0.1:${PROXY_PORT}"
RETRIEVER_REMOTE_URL="http://127.0.0.1:${RET_PORT}"
DATASETS=(nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle)

N_PASS=0
N_FAIL=0
for ds in "${DATASETS[@]}"; do
  echo "[$(date -u +%H:%M:%S)] eval $ds for $RUN_TAG step_$STEP" | tee -a "$LOG"
  cd "$EVAL_DIR"
  if "$PY" run_eval.py \
    --method_name search-r1 \
    --config_path flashrag/config/basic_config.yaml \
    --dataset_name "$ds" \
    --data_dir "$REPO_ROOT/data" \
    --save_dir "$EVAL_DIR/results" \
    --save_note "$SAVE_NOTE" \
    --sgl_remote_url "$SGL_REMOTE_URL" \
    --remote_retriever_url "$RETRIEVER_REMOTE_URL" \
    --generator_model "$HF_OUT" \
    --apply_chat true \
    --enable_thinking false \
    --prompt_mode qwen35_m5_train >> "$LOG" 2>&1; then
    # Post-check: verify metric_score.txt was actually produced. run_eval.py
    # can write config.yaml early then crash without erroring nonzero on
    # certain code paths; require the metric file as the success marker.
    if compgen -G "$EVAL_DIR/results/${ds}_*${SAVE_NOTE}*/metric_score.txt" > /dev/null; then
      N_PASS=$((N_PASS+1))
    else
      echo "WARN: $ds returned 0 but no metric_score.txt found" | tee -a "$LOG"
      N_FAIL=$((N_FAIL+1))
    fi
  else
    echo "WARN: $ds eval exited nonzero" | tee -a "$LOG"
    N_FAIL=$((N_FAIL+1))
  fi
done
echo "[$(date -u +%H:%M:%S)] eval summary: pass=$N_PASS fail=$N_FAIL" | tee -a "$LOG"
# Fail the wrapper if NONE passed: dispatcher will mark FAIL and re-queue can
# pick it up on a later pass; previously a 0-pass ckpt was logged DONE which
# made the queue look complete while producing zero useful data.
if [[ "$N_PASS" -eq 0 ]]; then
  echo "ERROR: no datasets produced metric_score.txt; aborting ckpt" | tee -a "$LOG"
  kill -TERM "$PROXY_PID" 2>/dev/null || true
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  sleep 3
  kill -KILL "$PROXY_PID" 2>/dev/null || true
  kill -KILL "$VLLM_PID" 2>/dev/null || true
  rm -rf "$HF_OUT"
  exit 5
fi

# 6. Push results to HF (per ckpt, all 7 datasets at once)
echo "[$(date -u +%H:%M:%S)] push results to HF: $RESULTS_REPO m9_5/${RUN_TAG}/step_${STEP}/" | tee -a "$LOG"
"$PY" -c "
import os, glob, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
ds_list = ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']
save_note = '$SAVE_NOTE'
repo = '$RESULTS_REPO'
prefix = 'm9_5/$RUN_TAG/step_$STEP'
n_pushed = 0
for d in ds_list:
    # flashrag writes results flat as results/<dataset>_<YYYY>_<MM>_<DD>_<HH>_<MM>_<save_note>/
    # (no per-dataset subdir); pick the newest match by ctime in case the save_note
    # somehow matches multiple runs.
    cands = sorted(glob.glob(f'$EVAL_DIR/results/{d}_*{save_note}*'), key=os.path.getctime)
    if not cands:
        # Fallback: per-dataset-subdir layout some older runs used.
        cands = sorted(glob.glob(f'$EVAL_DIR/results/{d}/{d}_*{save_note}*'), key=os.path.getctime)
    if not cands:
        print(f'WARN: no results for {d}')
        continue
    rdir = cands[-1]
    for fname in ['metric_score.txt','intermediate_data.json','config.yaml']:
        local_f = os.path.join(rdir, fname)
        if os.path.exists(local_f):
            api.upload_file(path_or_fileobj=local_f, path_in_repo=f'{prefix}/{d}/{fname}', repo_id=repo, repo_type='dataset')
            n_pushed += 1
print(f'pushed {n_pushed} files to {repo}/{prefix}/')
" 2>&1 | tee -a "$LOG" || echo "WARN: HF push failed" | tee -a "$LOG"

# 7. Cleanup
echo "[$(date -u +%H:%M:%S)] cleanup: kill vllm $VLLM_PID + proxy $PROXY_PID + rm HF ckpt" | tee -a "$LOG"
kill -TERM "$PROXY_PID" 2>/dev/null || true
kill -TERM "$VLLM_PID" 2>/dev/null || true
sleep 5
kill -KILL "$PROXY_PID" 2>/dev/null || true
kill -KILL "$VLLM_PID" 2>/dev/null || true
rm -rf "$HF_OUT"
echo "[$(date -u +%H:%M:%S)] === done $RUN_TAG step_$STEP GPU $GPU SLOT $SLOT ===" | tee -a "$LOG"
