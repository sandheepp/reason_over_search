#!/usr/bin/env bash
# Run Search-R1 GRPO on Qwen3.5-2B, 2× A100 80GB.
#
# Same args + behavior as run_grpo_1xa100.sh — only the config (and thus
# cluster.gpus_per_node, vllm.tensor_parallel_size, train_micro_batch_size)
# differs.
#
# Usage:
#   bash training/scripts/run_grpo_2xa100.sh \
#        --variant {base,hybrid} \
#        --seed N \
#        [--arm {qwen_native,paper}] \
#        [-- HYDRA_OVERRIDE_1 HYDRA_OVERRIDE_2 ...]
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

# ---- W&B ----
if [[ -f "${REPO_ROOT}/training/.env" ]]; then
    set -a; source "${REPO_ROOT}/training/.env"; set +a
fi

# ---- Hydra overrides ----
# See run_grpo_1xa100.sh for the rationale: timestamp on the W&B run name only,
# checkpoint dir stays keyed by (variant, arm, seed) so resumes work.
TS="$(date -u +%Y%m%dT%H%MZ)"
RUN_NAME="qwen3.5-2b-${VARIANT}-search_r1-${ARM}-2xa100-seed${SEED}-${TS}"
CKPT_DIR="${CHECKPOINT_DIR_BASE:-results/grpo}/qwen3.5-2b-${VARIANT}/${ARM}/seed${SEED}"

OVERRIDES=(
    "policy.model_name=${MODEL}"
    "grpo.seed=${SEED}"
    "data.train.arm=${ARM}"
    # NOTE: data.validation.arm is intentionally NOT overridden here. The first-pass
    # config has data.validation: null, and Hydra cannot override a nested key under a
    # null parent (errors with `ConfigCompositionException`). When validation is
    # re-enabled per VALIDATION.md §7, restore this line.
    "env.search_r1.arm=${ARM}"
    "logger.wandb.name=${RUN_NAME}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

if [[ "${VARIANT}" == "hybrid" ]]; then
    # `++` force-adds the key since the base config has chat_template_kwargs: null.
    OVERRIDES+=("++policy.tokenizer.chat_template_kwargs.enable_thinking=true")
fi

if [[ "${ARM}" == "paper" ]]; then
    OVERRIDES+=("data.default.prompt_file=training/src/prompts/search_r1_paper.txt")
    OVERRIDES+=("data.default.system_prompt_file=null")
fi

echo "[run_grpo_2xa100] launching: ${RUN_NAME}"
echo "[run_grpo_2xa100] model=${MODEL} arm=${ARM} seed=${SEED}"
echo "[run_grpo_2xa100] ckpt=${CKPT_DIR}"

exec "${VENV_PYTHON}" training/scripts/run_grpo.py \
    --config=training/configs/grpo_qwen3.5_2b_2xa100.yaml \
    "${OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
