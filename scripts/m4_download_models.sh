#!/usr/bin/env bash
# Download Qwen3.5-0.8B base + hybrid into eval/ for SGLang local-path serving.
# Run once on ALICE (login node or interactive shell with /home write access):
#   bash scripts/m4_download_models.sh
#
# After this, scripts/sbatch_m4.sh + run_m4.sh resolve to local paths under
# $REPO_ROOT/eval/ instead of streaming from HF inside SGLang (which avoids
# `huggingface_hub` doing a download in the middle of the SLURM time budget).
#
# Disk usage: ~3.4 GB total (1.7 GB hybrid + 1.7 GB base, bf16 safetensors).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/eval"
mkdir -p "$EVAL_DIR"

# Auto-discover python (Vast → ALICE → system fallback). Override via PY=...
if [[ -z "${PY:-}" ]]; then
  for cand in \
    /opt/miniforge3/envs/evaluation_search_r1/bin/python \
    /opt/miniforge3/envs/retriever/bin/python \
    /home/s4374886/.conda/envs/evaluation_search_r1/bin/python \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"; do
    if [[ -x "$cand" ]]; then PY="$cand"; break; fi
  done
fi
if [[ -z "${PY:-}" || ! -x "$PY" ]]; then
  echo "ERROR: no python interpreter found; set PY=/path/to/python" >&2
  exit 1
fi

declare -A REPOS=(
  [qwen3.5_0.8b]=Qwen/Qwen3.5-0.8B
  [qwen3.5_0.8b_base]=Qwen/Qwen3.5-0.8B-Base
)

for variant in "${!REPOS[@]}"; do
  repo="${REPOS[$variant]}"
  dst="$EVAL_DIR/$variant"
  if [[ -f "$dst/config.json" ]]; then
    echo "[skip] $variant already present at $dst"
    continue
  fi
  echo "[get]  $repo -> $dst"
  "$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$repo",
    local_dir="$dst",
    local_dir_use_symlinks=False,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.py"],
)
EOF
done

echo "All M4 models present under $EVAL_DIR/"
ls -la "$EVAL_DIR/"
