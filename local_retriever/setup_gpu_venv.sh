#!/usr/bin/env bash
# Set up the GPU retriever venv at local_retriever/.venv with faiss-gpu-cu12 + torch.
# Idempotent: re-running just upgrades/installs missing packages.
#
# Usage:
#   bash local_retriever/setup_gpu_venv.sh
#
# After it completes, run the retriever in GPU mode with:
#   ./.venv/bin/python retriever_serving.py \
#       --config retriever_config.yaml --num_retriever 1 --port 3005 \
#       --gpu --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/requirements-gpu.txt"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

# pip extracts wheels under TMPDIR. The root filesystem on this box is small;
# /workspace has plenty of space, so use it.
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup-gpu-venv] creating venv at $VENV_DIR (python: $PYTHON_BIN)"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup-gpu-venv] reusing existing venv at $VENV_DIR"
fi

echo "[setup-gpu-venv] upgrading pip"
"$VENV_DIR/bin/pip" install --upgrade pip

echo "[setup-gpu-venv] installing $REQUIREMENTS (TMPDIR=$TMPDIR)"
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo "[setup-gpu-venv] verifying GPU bindings"
"$VENV_DIR/bin/python" - <<'PY'
import faiss, torch
print(f"faiss {faiss.__version__}  GPUs visible to FAISS: {faiss.get_num_gpus()}")
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  device count: {torch.cuda.device_count()}")
assert faiss.get_num_gpus() >= 1, "FAISS sees no GPU — install or driver issue"
assert torch.cuda.is_available(), "torch reports no CUDA"
print("OK")
PY

echo
echo "[setup-gpu-venv] done. Run the retriever with:"
echo "  cd $SCRIPT_DIR"
echo "  ./.venv/bin/python retriever_serving.py \\"
echo "      --config retriever_config.yaml --num_retriever 1 --port 3005 \\"
echo "      --gpu --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index"
