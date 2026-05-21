"""Custom TP plan for Qwen3.5 family.

Avoids the FSDP+TP>1 vocab-parallel bug
(reset_sharded_param narrow(length=vocab) > local shard) by NOT including
embed_tokens or lm_head; they fall back to FSDP-only handling (replicated
across TP ranks). Cost: ~0.55 GB extra per GPU for the replicated embed +
lm_head — trivial on 80 GB.

Layer paths use the single-prefix form (model.layers.*) consistent with the
heuristic the parallelizer otherwise applies for Qwen3.5; the qkv_proj +
gate_up_proj fused variants are included so the plan is robust to either
the split or fused projection style.

Mixed-attention layers (Qwen3_5GatedDeltaNet, every Nth layer in Qwen3.5)
don't match any of these patterns — they stay fully replicated. No
speedup on those, but no breakage either.
"""

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def custom_parallel_plan():
    return {
        # Attention layers (Qwen3_5Attention type)
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        # MLP layers (standard)
        "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    }
