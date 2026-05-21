#!/usr/bin/env bash
# Watch a NeMo-RL checkpoint dir and upload every newly-saved step to HF Hub.
#
# Decoupled from training: this is its own process. Failures CANNOT crash
# training (no shared signal handlers, no shared file locks beyond reading the
# read-only ckpt dir). If this script dies, training continues uninterrupted.
#
# Usage:
#   bash training_m5_6/scripts/upload_ckpts_watcher.sh [--mode prod|smoke] [--seed N]
#
# Defaults: --mode prod --seed 42. Reads HF_TOKEN + HF_REPO_PREFIX from
# training_m5_6/.env (gitignored). Each step N is uploaded to:
#   $HF_REPO_PREFIX-m5_$MODE-seed$SEED-step$N    (private model repo)
#
# State file: results/grpo/m5_$MODE/seed$SEED/.uploaded_steps
# Log file:   logs/hf_uploader_m5_$MODE.log
#
# Recovery: re-running picks up uploads from the state file; already-uploaded
# steps are skipped. A repo that already exists on the Hub is detected via
# huggingface_hub.HfApi.repo_info and re-pushed via upload_folder (idempotent).
#
# Design constraints:
#   - poll loop, not inotify (works in containers without inotify support)
#   - one upload at a time (no concurrent uploads → no rate-limit fanout)
#   - skips tmp_step_N (in-progress saves; only step_N is atomic-renamed)
#   - skips already-uploaded steps via state file
#   - on upload failure: log, sleep 30 s, retry; never raise out of the loop

set -uo pipefail   # NOT -e — we never want to exit on a single failed upload

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="prod"
SEED="42"
POLL_SECONDS="60"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --poll) POLL_SECONDS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -f "${REPO_ROOT}/training_m5_6/.env" ]]; then
    set -a; source "${REPO_ROOT}/training_m5_6/.env"; set +a
fi

if [[ -z "${HF_TOKEN:-}" || -z "${HF_REPO_PREFIX:-}" ]]; then
    echo "error: HF_TOKEN and HF_REPO_PREFIX must be set in training_m5_6/.env" >&2
    exit 2
fi

CKPT_BASE="${CHECKPOINT_DIR_BASE:-results/grpo}/m5_${MODE}/seed${SEED}"
STATE_FILE="${CKPT_BASE}/.uploaded_steps"
LOG_FILE="logs/hf_uploader_m5_${MODE}.log"
mkdir -p "logs" "${CKPT_BASE}"
touch "${STATE_FILE}"

VENV_PYTHON="${REPO_ROOT}/training_m5_6/nemo_rl/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "error: venv python not found at ${VENV_PYTHON}" >&2
    exit 1
fi

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}" >&2; }

upload_step() {
    local step_dir="$1"
    local step_n; step_n="$(basename "${step_dir}" | sed 's/^step_//')"
    local repo_name="${HF_REPO_PREFIX}-m5_6_${MODE}-seed${SEED}-step${step_n}"
    log "upload step=${step_n} → ${repo_name}"
    HF_TOKEN="${HF_TOKEN}" \
        "${VENV_PYTHON}" - "${step_dir}" "${repo_name}" "${step_n}" "${MODE}" "${SEED}" <<'PYEOF' 2>>"${LOG_FILE}"
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
src, repo_id, step, mode, seed = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
except Exception as e:
    print(f"create_repo: {e}", file=sys.stderr)
# Upload everything under step_dir/policy/weights/ (the consolidated safetensors)
weights_dir = Path(src) / "policy" / "weights"
tokenizer_dir = Path(src) / "policy" / "tokenizer"
training_info = Path(src) / "training_info.json"
if not weights_dir.exists():
    print(f"no weights dir at {weights_dir}", file=sys.stderr); sys.exit(2)
# README in the model repo so the Hub page documents provenance
readme = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags: [grpo, research-paper, musique, m5.6, qwen3.5]
---

# Qwen3.5-0.8B GRPO checkpoint — M5.1 step {step}

GRPO training checkpoint from `M5.6-prod` (EM-only reward, paper-faithful Search-R1) (paper-faithful ReSearch recipe on MuSiQue, 1× A100-80GB).
Mode: `{mode}`, seed: `{seed}`, step: `{step}`. Uploaded automatically by `upload_ckpts_watcher.sh`.

- Project: [`reason_over_search`](https://github.com/GaurisankarJ/reason_over_search)
- Run config: `training_m5_6/configs/m5_6_research_paper.yaml`
- Reward: EM-only (0/1) on `<answer>...</answer>` content
- Answer wrap: plain `<answer>X</answer>` (M5.1 divergence #2)

See `RESULTS_m5.md` and `RESULTS_SMOKE_m5.md` in the repo for run context.
"""
Path("/tmp/_hf_readme.md").write_text(readme)
api.upload_file(path_or_fileobj="/tmp/_hf_readme.md", path_in_repo="README.md",
                repo_id=repo_id, repo_type="model", token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=str(weights_dir), path_in_repo="weights",
                  repo_id=repo_id, repo_type="model", token=os.environ["HF_TOKEN"])
if tokenizer_dir.exists():
    api.upload_folder(folder_path=str(tokenizer_dir), path_in_repo="tokenizer",
                      repo_id=repo_id, repo_type="model", token=os.environ["HF_TOKEN"])
if training_info.exists():
    api.upload_file(path_or_fileobj=str(training_info), path_in_repo="training_info.json",
                    repo_id=repo_id, repo_type="model", token=os.environ["HF_TOKEN"])
print(f"uploaded step={step}")
PYEOF
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "${step_n}" >> "${STATE_FILE}"
        log "DONE step=${step_n}"
        return 0
    else
        log "FAIL step=${step_n} rc=${rc} (will retry on next poll cycle)"
        return 1
    fi
}

is_uploaded() {
    local step_n="$1"
    grep -qFx "${step_n}" "${STATE_FILE}" 2>/dev/null
}

log "watcher start mode=${MODE} seed=${SEED} ckpt_base=${CKPT_BASE} poll=${POLL_SECONDS}s"
trap 'log "watcher stop (signal)"; exit 0' INT TERM

while true; do
    if [[ -d "${CKPT_BASE}" ]]; then
        for d in "${CKPT_BASE}"/step_*; do
            [[ -d "$d" ]] || continue
            step_n="$(basename "$d" | sed 's/^step_//')"
            if is_uploaded "${step_n}"; then continue; fi
            upload_step "$d" || true
        done
    fi
    sleep "${POLL_SECONDS}"
done
