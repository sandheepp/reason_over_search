#!/usr/bin/env bash
# Package the locally-built DTensorPolicyWorkerV2 venv and upload to HF Hub
# as an arch-tagged tarball, so future bootstraps on the same SM hit the
# fast-path instead of recompiling.
#
# Usage:
#   bash training_m5_5/scripts/package_v2_venv.sh [--repo <user>/<repo>]
#                                                 [--name <filename.tar.gz>]
#                                                 [--make-public]
#
# Defaults:
#   --repo  : reads HF username from HF_TOKEN, uses <user>/reason-over-search-venvs
#   --name  : auto-derived from detected GPU SM, e.g. dtensor_policy_worker_v2_sm103.tar.gz
#   private : true (use --make-public to flip)
#
# Requires HF_TOKEN with write access in training_m5_5/.env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Load HF_TOKEN
ENV_FILE="${REPO_ROOT}/training_m5_5/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "error: HF_TOKEN not set (write-scoped). Add to training_m5_5/.env." >&2; exit 1; }

REPO_ID=""
FILE_NAME=""
PRIVATE_FLAG="--private"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)        REPO_ID="$2"; shift 2 ;;
        --name)        FILE_NAME="$2"; shift 2 ;;
        --make-public) PRIVATE_FLAG=""; shift ;;
        -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default REPO_ID: <hf_user>/reason-over-search-venvs
# Parse `hf auth whoami` output carefully — it prints an ANSI-colored "✓ Logged in"
# banner on stdout line 1 and the username on line 2 ("  user: sandheepp").
# Strip ANSI escapes, grep for "user:", take the value.
if [[ -z "${REPO_ID}" ]]; then
    VENV_HF="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/hf"
    [[ -x "${VENV_HF}" ]] || VENV_HF="$(command -v hf)"
    HF_USER="$( (HF_TOKEN="${HF_TOKEN}" "${VENV_HF}" auth whoami 2>/dev/null || true) \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -E '^\s*user:' \
        | head -1 \
        | sed -E 's/^\s*user:\s*//' \
        | tr -d '[:space:]' || true )"
    [[ -n "${HF_USER:-}" ]] || { echo "error: hf auth whoami didn't return a username — token invalid or no network." >&2; exit 1; }
    echo "[pkg] HF user: ${HF_USER}"
    REPO_ID="${HF_USER}/reason-over-search-venvs"
fi

# Default FILE_NAME: dtensor_policy_worker_v2_sm<CC>.tar.gz
if [[ -z "${FILE_NAME}" ]]; then
    DETECTED_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    [[ -n "${DETECTED_CC}" ]] || { echo "error: nvidia-smi unavailable; pass --name explicitly." >&2; exit 1; }
    FILE_NAME="dtensor_policy_worker_v2_sm${DETECTED_CC}.tar.gz"
fi

V2_VENV="${REPO_ROOT}/training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
[[ -x "${V2_VENV}/bin/python" ]] || { echo "error: V2 venv missing at ${V2_VENV}" >&2; exit 1; }

# Sanity-check the venv imports before packaging — never upload a broken venv
echo "[pkg] sanity-checking V2 venv imports..."
"${V2_VENV}/bin/python" -c "import torch, transformer_engine, causal_conv1d, mamba_ssm, deep_ep, nemo_automodel; \
    import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('  V2 imports OK')" \
    || { echo "error: V2 venv imports fail — refusing to package broken venv." >&2; exit 1; }

# Capture the env metadata as a sidecar so consumers can verify compatibility
META="/tmp/v2_venv_meta.txt"
{
    echo "# DTensorPolicyWorkerV2 metadata — generated $(date -u)"
    echo "compute_cap: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    echo "gpu_name:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "os:          $(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"')"
    echo "nvcc:        $(/usr/local/cuda/bin/nvcc --version | tail -1)"
    echo "torch:       $(${V2_VENV}/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "te:          $(${V2_VENV}/bin/python -c 'import transformer_engine; print(transformer_engine.__version__)')"
    echo "nemo_rl_commit: $(git -C ${REPO_ROOT}/training_m5_5/nemo_rl rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "NVTE_CUDA_ARCHS_used: ${NVTE_CUDA_ARCHS:-unknown}"
} > "${META}"
echo "[pkg] metadata:"
sed 's/^/    /' "${META}"

# Tar + compress
TAR_PATH="/tmp/${FILE_NAME}"
echo "[pkg] creating ${TAR_PATH} (~5-10 GB compressed; takes 1-3 min)..."
tar -czf "${TAR_PATH}" -C "$(dirname "${V2_VENV}")" "$(basename "${V2_VENV}")"
ls -lh "${TAR_PATH}"

# Upload (free local tarball as soon as upload succeeds; verify step needs
# another ~5-10 GB of disk for the round-trip download and is rarely useful
# in practice — the `hf upload` exit code + commit URL already tell us if
# it worked).
VENV_HF="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/hf"
[[ -x "${VENV_HF}" ]] || VENV_HF="$(command -v hf)"
echo "[pkg] uploading to HF ${REPO_ID} (private=$([[ -n "$PRIVATE_FLAG" ]] && echo yes || echo no))..."
HF_TOKEN="${HF_TOKEN}" "${VENV_HF}" repo create "${REPO_ID}" --repo-type dataset ${PRIVATE_FLAG} -y 2>&1 | tail -3 || true
if HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN}" \
        "${VENV_HF}" upload "${REPO_ID}" "${TAR_PATH}" "${FILE_NAME}" --repo-type dataset 2>&1 | tail -5; then
    echo "[pkg] ✅ tarball uploaded — deleting local copy to free disk"
    rm -f "${TAR_PATH}"
else
    echo "[pkg] ⚠️  tarball upload failed — keeping local at ${TAR_PATH}"
fi
# Upload metadata sidecar (small, no disk pressure)
HF_TOKEN="${HF_TOKEN}" "${VENV_HF}" upload "${REPO_ID}" "${META}" "${FILE_NAME%.tar.gz}.meta.txt" --repo-type dataset 2>&1 | tail -3 || true
rm -f "${META}"

# Optional verify (disabled by default — needs ~6 GB of free disk to re-download.
# Set VERIFY_UPLOAD=1 to re-enable on boxes with disk headroom.)
if [[ "${VERIFY_UPLOAD:-0}" == "1" ]]; then
    echo "[pkg] verifying upload (re-download)..."
    HF_TOKEN="${HF_TOKEN}" "${VENV_HF}" download "${REPO_ID}" "${FILE_NAME}" --repo-type dataset --local-dir /tmp/_verify >/dev/null 2>&1 \
        && echo "[pkg] ✅ round-trip OK" \
        || echo "[pkg] ⚠️  re-download failed — check HF UI manually"
    rm -rf /tmp/_verify
fi

echo
echo "Future bootstraps on sm_${DETECTED_CC:-?} auto-discover this tarball via your HF_TOKEN's whoami:"
echo "    bootstrap_b300.sh probes ${REPO_ID}:${FILE_NAME} first."
echo "Override manually if needed:"
echo "    V2_VENV_HF_REPO=${REPO_ID} V2_VENV_HF_FILE=${FILE_NAME} bash training_m5_5/scripts/bootstrap_b300.sh"
