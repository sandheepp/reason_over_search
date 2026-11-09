#!/usr/bin/env bash
# Watch a NeMo-RL checkpoint dir and upload every newly-saved step to HF Hub.
#
# Convention: single repo with all steps as subfolders (mirrors H200 M5.1
# layout at https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-m5_1-seed42-f1-only).
# The full on-disk step_N/ tree (policy/{weights,tokenizer}/, config.yaml,
# train_dataloader.pt, training_info.json) is uploaded as-is into repo path step_N/.
#
# Decoupled from training: this is its own process. Failures CANNOT crash
# training (no shared signal handlers, no shared file locks beyond reading the
# read-only ckpt dir). If this script dies, training continues uninterrupted.
#
# Usage:
#   bash training_m5_1/scripts/upload_ckpts_watcher.sh \
#       --ckpt-base results/grpo/m5_1_prod_dp4_4xblackwell/seed43 \
#       --rollouts-dir logs/exp_NNN \
#       --repo pantomiman/qwen3.5-0.8b-grpo-musique-blackwell-m5_1-seed43-f1-only
#
# Reads HF_TOKEN from training_m5_1/.env (gitignored).
#
# If --rollouts-dir is given, train_data_stepN.jsonl files in that dir are
# uploaded to repo:rollouts/<basename> (mirrors H200 M5.5 layout). Use the
# `logs/exp_NNN/` dir for the active training run.
#
# State files: $CKPT_BASE/.uploaded_steps + $CKPT_BASE/.uploaded_rollouts
# Log file:    logs/hf_uploader_$(slug repo).log
#
# Recovery: re-running picks up uploads from the state file; already-uploaded
# steps are skipped. The repo is created with exist_ok=True (PUBLIC); per-step
# uploads are idempotent via upload_folder.
#
# Design constraints:
#   - poll loop, not inotify (works in containers without inotify support)
#   - one upload at a time (no concurrent uploads -> no rate-limit fanout)
#   - skips tmp_step_N (in-progress saves; only step_N is atomic-renamed)
#   - skips already-uploaded steps via state file
#   - on upload failure: log, sleep 30 s, retry; never raise out of the loop

set -uo pipefail   # NOT -e - we never want to exit on a single failed upload

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT_BASE=""
REPO_ID=""
ROLLOUTS_DIR=""
POLL_SECONDS="60"
MODE="prod"
SEED="42"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-base) CKPT_BASE="$2"; shift 2 ;;
        --repo) REPO_ID="$2"; shift 2 ;;
        --rollouts-dir) ROLLOUTS_DIR="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --poll) POLL_SECONDS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -f "${REPO_ROOT}/training_m5_1/.env" ]]; then
    set -a; source "${REPO_ROOT}/training_m5_1/.env"; set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "error: HF_TOKEN must be set in training_m5_1/.env" >&2
    exit 2
fi
if [[ -z "${CKPT_BASE}" || -z "${REPO_ID}" ]]; then
    echo "error: --ckpt-base and --repo are required" >&2
    exit 2
fi

STATE_FILE="${CKPT_BASE}/.uploaded_steps"
ROLLOUT_STATE_FILE="${CKPT_BASE}/.uploaded_rollouts"
SLUG="$(echo "${REPO_ID}" | tr '/' '_')"
LOG_FILE="logs/hf_uploader_${SLUG}.log"
mkdir -p "logs" "${CKPT_BASE}"
touch "${STATE_FILE}" "${ROLLOUT_STATE_FILE}"

VENV_PYTHON="${REPO_ROOT}/training_m5_1/nemo_rl/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "error: venv python not found at ${VENV_PYTHON}" >&2
    exit 1
fi

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}" >&2; }

# One-time bootstrap: create repo (PUBLIC) + push README + config snapshot
# (idempotent; safe to re-run).
bootstrap_repo() {
    HF_TOKEN="${HF_TOKEN}" REPO_ID="${REPO_ID}" MODE="${MODE}" SEED="${SEED}" \
    REPO_ROOT="${REPO_ROOT}" \
        "${VENV_PYTHON}" - <<'PYEOF' 2>>"${LOG_FILE}"
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
token = os.environ["HF_TOKEN"]
repo_id = os.environ["REPO_ID"]
mode = os.environ["MODE"]
seed = os.environ["SEED"]
repo_root = Path(os.environ["REPO_ROOT"])
api = HfApi(token=token)
try:
    create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True, token=token)
except Exception as e:
    print(f"create_repo: {e}", file=sys.stderr)
# Flip to public if HF silently overrode create_repo's private=False. Use
# update_repo_settings (modern API name); fall back to update_repo_visibility
# on older huggingface_hub.
try:
    api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model", token=token)
except AttributeError:
    try:
        api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="model", token=token)
    except Exception as e:
        print(f"update_repo_visibility fallback failed: {e}", file=sys.stderr)
except Exception as e:
    print(f"update_repo_settings failed: {e}", file=sys.stderr)
readme = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags: [grpo, research-paper, musique, m5.1, qwen3.5, blackwell, f1-only]
---

# Qwen3.5-0.8B GRPO on MuSiQue - M5.1 (F1-only reward) - 4x Blackwell

Auto-uploaded checkpoints from the M5.1 F1-only run on 4x RTX PRO 6000 Blackwell
(rift, DP=4, async_engine). Mirrors the H200 M5.1 cadence (save_period=10),
one folder per saved step: `step_10/`, `step_20/`, ...

Each `step_N/` contains the full NeMo-RL checkpoint tree:
- `step_N/policy/weights/` - consolidated safetensors
- `step_N/policy/tokenizer/`
- `step_N/config.yaml`, `step_N/training_info.json`, `step_N/train_dataloader.pt`

## Run config
- Mode: `{mode}`, seed: `{seed}`
- Reward: F1-only on `<answer>...</answer>` content (M5.1 divergence #1)
- Answer wrap: plain `<answer>X</answer>` (M5.1 divergence #2)
- Topology: DP=4 + colocated vLLM (TP=1) + async_engine
- Config snapshot: `config_snapshot.yaml` in this repo

## Reproduce
- Project: https://github.com/GaurisankarJ/reason_over_search
- Branch: `training_2_rift`
- Config: `training_m5_1/configs/m5_1_research_paper_dp4_4xblackwell.yaml`

See [`docs/report/RESULTS_M5_1_BLACKWELL.md`](https://github.com/GaurisankarJ/reason_over_search/blob/training_2_rift/docs/report/RESULTS_M5_1_BLACKWELL.md) for results trail.
"""
Path("/tmp/_hf_readme.md").write_text(readme)
api.upload_file(path_or_fileobj="/tmp/_hf_readme.md", path_in_repo="README.md",
                repo_id=repo_id, repo_type="model", token=token)
# Snapshot the live config for provenance
cfg_src = repo_root / "training_m5_1/configs/m5_1_research_paper_dp4_4xblackwell.yaml"
if cfg_src.exists():
    api.upload_file(path_or_fileobj=str(cfg_src), path_in_repo="config_snapshot.yaml",
                    repo_id=repo_id, repo_type="model", token=token)
# Canary
Path("/tmp/_hf_canary.txt").write_text("uploader online\n")
api.upload_file(path_or_fileobj="/tmp/_hf_canary.txt", path_in_repo=".uploader_canary.txt",
                repo_id=repo_id, repo_type="model", token=token)
print("bootstrap ok")
PYEOF
    return $?
}

upload_step() {
    local step_dir="$1"
    local step_n; step_n="$(basename "${step_dir}" | sed 's/^step_//')"
    log "upload step=${step_n} -> ${REPO_ID}:step_${step_n}/"
    HF_TOKEN="${HF_TOKEN}" REPO_ID="${REPO_ID}" STEP_DIR="${step_dir}" STEP_N="${step_n}" \
        "${VENV_PYTHON}" - <<'PYEOF' 2>>"${LOG_FILE}"
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
token = os.environ["HF_TOKEN"]
repo_id = os.environ["REPO_ID"]
src = Path(os.environ["STEP_DIR"])
step_n = os.environ["STEP_N"]
api = HfApi(token=token)
api.upload_folder(
    folder_path=str(src),
    path_in_repo=f"step_{step_n}",
    repo_id=repo_id,
    repo_type="model",
    token=token,
    ignore_patterns=["**/.uploaded_steps"],
)
print(f"uploaded step_{step_n}")
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

is_rollout_uploaded() {
    local basename="$1"
    grep -qFx "${basename}" "${ROLLOUT_STATE_FILE}" 2>/dev/null
}

upload_rollout() {
    local src="$1"
    local basename; basename="$(basename "${src}")"
    log "upload rollout=${basename} -> ${REPO_ID}:rollouts/${basename}"
    HF_TOKEN="${HF_TOKEN}" REPO_ID="${REPO_ID}" SRC="${src}" BN="${basename}" \
        "${VENV_PYTHON}" - <<'PYEOF' 2>>"${LOG_FILE}"
import os, sys
from huggingface_hub import HfApi
token = os.environ["HF_TOKEN"]
repo_id = os.environ["REPO_ID"]
src = os.environ["SRC"]
bn = os.environ["BN"]
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=src,
    path_in_repo=f"rollouts/{bn}",
    repo_id=repo_id,
    repo_type="model",
    token=token,
)
print(f"uploaded rollouts/{bn}")
PYEOF
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "${basename}" >> "${ROLLOUT_STATE_FILE}"
        log "DONE rollout=${basename}"
        return 0
    else
        log "FAIL rollout=${basename} rc=${rc} (will retry on next poll cycle)"
        return 1
    fi
}

log "watcher start ckpt_base=${CKPT_BASE} repo=${REPO_ID} rollouts_dir=${ROLLOUTS_DIR:-<none>} poll=${POLL_SECONDS}s"
bootstrap_repo || log "WARN bootstrap_repo returned non-zero (continuing; per-step uploads will retry)"
trap 'log "watcher stop (signal)"; exit 0' INT TERM

while true; do
    if [[ -d "${CKPT_BASE}" ]]; then
        for d in $(ls -d "${CKPT_BASE}"/step_* 2>/dev/null | sort -t_ -k2 -n); do
            [[ -d "$d" ]] || continue
            step_n="$(basename "$d" | sed 's/^step_//')"
            if is_uploaded "${step_n}"; then continue; fi
            upload_step "$d" || true
        done
    fi
    if [[ -n "${ROLLOUTS_DIR}" && -d "${ROLLOUTS_DIR}" ]]; then
        for f in $(ls "${ROLLOUTS_DIR}"/train_data_step*.jsonl 2>/dev/null | sort -t 'p' -k 3 -n); do
            [[ -f "$f" ]] || continue
            bn="$(basename "$f")"
            if is_rollout_uploaded "${bn}"; then continue; fi
            upload_rollout "$f" || true
        done
    fi
    sleep "${POLL_SECONDS}"
done
