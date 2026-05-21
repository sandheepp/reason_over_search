#!/usr/bin/env bash
# Set up the NeMo-RL venv after `git clone`-ing this repo.
#
# Two contexts:
#
#  1. ON A VAST INSTANCE running `pantomiman/reason-over-search-v1`:
#     - The image's uv wheel cache is pre-warmed for NeMo-RL @ v0.6.0 with
#       the `vllm` extra. Running this script creates the .venv at
#       training/nemo_rl/.venv/ in seconds-to-minutes (no re-download).
#
#  2. ON A LOCAL MAC / LINUX machine, outside docker:
#     - First run: downloads ~5GB of wheels (torch 2.10, vLLM 0.17, cuDNN, etc).
#     - Subsequent runs reuse uv's cache.
#
# Knobs:
#   NEMO_RL_REF       — git ref to re-clone NeMo-RL at; default: "" (use the
#                       committed source under training/nemo_rl/, no re-clone)
#                       Set to bump the pinned NeMo-RL version (also requires
#                       FORCE_RECLONE=1).
#   FORCE_RECLONE     — wipe training/nemo_rl/ and re-clone from upstream
#                       at NEMO_RL_REF (or v0.6.0 if NEMO_RL_REF is empty).
#                       After re-clone you must `git add training/nemo_rl/`
#                       and commit the new tree.
#   UV_EXTRAS         — extras passed to uv sync (default: "vllm")
#                       Common options: "vllm", "vllm,fsdp", "vllm,nemo_gym".
#
# Notes:
#   - training/nemo_rl/ is committed to this repo. setup.sh does not touch it
#     unless FORCE_RECLONE is set. Local edits go directly into the committed
#     tree and into git history alongside our overlay.

set -euo pipefail

NEMO_RL_REF="${NEMO_RL_REF:-}"
UV_EXTRAS="${UV_EXTRAS:-vllm}"
FORCE_RECLONE="${FORCE_RECLONE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_RL_DIR="${SCRIPT_DIR}/nemo_rl"

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }

# 1. Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found — installing via the Astral installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installed but not on PATH. Add ~/.local/bin (or ~/.cargo/bin) to PATH and re-run."
        exit 1
    fi
fi
log "uv version: $(uv --version)"

# 2. Optionally re-clone NeMo-RL (to bump the pinned version)
if [[ -n "${FORCE_RECLONE}" ]]; then
    REF="${NEMO_RL_REF:-v0.6.0}"
    log "FORCE_RECLONE set — wiping ${NEMO_RL_DIR} and re-cloning @ ${REF}"
    rm -rf "${NEMO_RL_DIR}"
    git clone --recursive --branch "${REF}" \
        https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL_DIR}"

    log "removing .git directories so it's not a nested repo"
    find "${NEMO_RL_DIR}" -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true
    log "remember to commit training/nemo_rl/ after a successful re-clone"
fi

if [[ ! -d "${NEMO_RL_DIR}" ]]; then
    echo "ERROR: ${NEMO_RL_DIR} does not exist. The NeMo-RL source is committed to this repo —"
    echo "       did the git clone of reason_over_search succeed? If you intentionally cleared"
    echo "       it, set FORCE_RECLONE=1 (with optional NEMO_RL_REF=...) to re-clone upstream."
    exit 1
fi

# 3. uv venv + uv sync
cd "${NEMO_RL_DIR}"
if [[ ! -d ".venv" ]]; then
    log "creating uv venv at ${NEMO_RL_DIR}/.venv (Python 3.13)"
    uv venv
fi

log "installing NeMo-RL deps with extras: ${UV_EXTRAS}"
EXTRA_ARGS=()
IFS=',' read -r -a EXTRAS <<< "${UV_EXTRAS}"
for e in "${EXTRAS[@]}"; do
    EXTRA_ARGS+=(--extra "${e}")
done
uv sync "${EXTRA_ARGS[@]}"

log "done."
log ""
log "Activate the env:"
log "  source ${NEMO_RL_DIR}/.venv/bin/activate"
log ""
log "Or run via uv (no activation needed):"
log "  cd ${NEMO_RL_DIR}"
log "  uv run python examples/run_grpo.py --help"
