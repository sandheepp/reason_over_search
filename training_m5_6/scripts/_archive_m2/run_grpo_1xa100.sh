#!/usr/bin/env bash
# Run Search-R1 GRPO on Qwen3.5-2B, 1× A100 80GB.
#
# Usage:
#   bash training/scripts/run_grpo_1xa100.sh \
#        --variant {base,hybrid} \
#        --seed N \
#        [--arm {qwen_native,paper}]    # default: qwen_native
#
# Example:
#   bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42
#   bash training/scripts/run_grpo_1xa100.sh --variant hybrid --seed 7 --arm paper
#
# Prerequisites: retriever live at 127.0.0.1:3005 (local_retriever/README.md);
# training/.env loaded (W&B); training/nemo_rl/.venv materialized
# (training/setup.sh or `uv sync --extra vllm` inside the docker image).
set -euo pipefail

# ---- defaults ----
VARIANT="base"
SEED="42"
ARM="qwen_native"
EXTRA_OVERRIDES=()

# ---- argparse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --) shift; EXTRA_OVERRIDES=("$@"); break ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$VARIANT" in
    base)   MODEL="Qwen/Qwen3.5-2B-Base" ;;
    hybrid) MODEL="Qwen/Qwen3.5-2B" ;;
    *) echo "--variant must be base or hybrid" >&2; exit 2 ;;
esac

case "$ARM" in
    qwen_native|paper) ;;
    *) echo "--arm must be qwen_native or paper" >&2; exit 2 ;;
esac

# ---- repo root + venv ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_PYTHON="${REPO_ROOT}/training/nemo_rl/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "venv not found at ${VENV_PYTHON} — run training/setup.sh first" >&2
    exit 1
fi

# ---- W&B (optional but recommended) ----
if [[ -f "${REPO_ROOT}/training/.env" ]]; then
    set -a; source "${REPO_ROOT}/training/.env"; set +a
fi

# ---- Hydra overrides ----
# Timestamp suffixed to the W&B run name so repeated launches (e.g. resuming
# after an OOM, or re-running the same seed for sanity) get distinct W&B runs
# while still resuming from the seed's checkpoint dir below.
TS="$(date -u +%Y%m%dT%H%MZ)"
RUN_NAME="qwen3.5-2b-${VARIANT}-search_r1-${ARM}-1xa100-seed${SEED}-${TS}"
# Checkpoint dir is keyed by (variant, arm, seed) — NOT timestamped — so a
# resumed run picks up where it left off.
CKPT_DIR="${CHECKPOINT_DIR_BASE:-results/grpo}/qwen3.5-2b-${VARIANT}/${ARM}/seed${SEED}"

OVERRIDES=(
    "policy.model_name=${MODEL}"
    "grpo.seed=${SEED}"
    "data.train.arm=${ARM}"
    "env.search_r1.arm=${ARM}"
    "logger.wandb.name=${RUN_NAME}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

# Hybrid variant: surface <think> blocks during generation. Use Hydra's `++`
# (force-add) so the key is created even though the config has it as null.
if [[ "${VARIANT}" == "hybrid" ]]; then
    OVERRIDES+=("++policy.tokenizer.chat_template_kwargs.enable_thinking=true")
fi

# Paper arm: swap the qwen_native system prompt out for the paper's user-message
# instruction template. Both prompts live under training/src/prompts/.
if [[ "${ARM}" == "paper" ]]; then
    OVERRIDES+=("data.default.prompt_file=training/src/prompts/search_r1_paper.txt")
    OVERRIDES+=("data.default.system_prompt_file=null")
fi

echo "[run_grpo_1xa100] launching: ${RUN_NAME}"
echo "[run_grpo_1xa100] model=${MODEL} arm=${ARM} seed=${SEED}"
echo "[run_grpo_1xa100] ckpt=${CKPT_DIR}"

exec "${VENV_PYTHON}" training/scripts/run_grpo.py \
    --config=training/configs/grpo_qwen3.5_2b_1xa100.yaml \
    "${OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
