#!/bin/bash
# Training smoke for ALICE 2126456: 1 step on Qwen3.5-2B-Base, smallest viable shape.
set -e
SIF=/zfsstore/user/s4374886/apptainer/reason-over-search-v1.sif
REPO=/zfsstore/user/s4374886/omega/reason_over_search
HF=/zfsstore/user/s4374886/hf_cache
FLASH=/zfsstore/user/s4374886/omega/flash-rag
UV=/zfsstore/user/s4374886/uv_cache
mkdir -p "$UV"

apptainer exec --nv \
  --bind "$REPO:/workspace/reason_over_search,$HF:/workspace/hf_cache,$FLASH:$FLASH,$UV:/.uv/cache" \
  --env HF_HOME=/workspace/hf_cache \
  --env WANDB_MODE=disabled \
  --env CHECKPOINT_DIR_BASE="$REPO/logs/smoke_2126456/ckpt" \
  --env RAY_TMPDIR=/tmp/ray_smoke \
  --env TMPDIR=/tmp \
  "$SIF" \
  bash -c "
    cd /workspace/reason_over_search
    bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 -- \
      grpo.max_num_steps=1 \
      grpo.num_prompts_per_step=2 \
      grpo.val_period=0 \
      grpo.val_at_start=false \
      policy.train_global_batch_size=8 \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=true \
      policy.train_micro_batch_size=2 \
      env.search_r1.retriever_url=http://127.0.0.1:3005/search \
      logger.wandb_enabled=false \
      checkpointing.enabled=false
  "
