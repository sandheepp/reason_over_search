---
title: Open questions
tags: [meta, questions]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Open questions

Running log of research questions and uncertainties to chase down later. One heading per question. When the user says "register this as a question" (or similar phrasing), append a new entry at the bottom following the template below; do not paraphrase the question, capture it as the user states it plus enough context for future-me to pick up cold.

To resolve a question, leave the entry in place and add a `**Resolution**` block beneath it (date, one-paragraph answer, and links to the source / PR / run / commit). Flip `**Status**` to `resolved`. Do not delete resolved entries; they double as a learning trail.

Cross-references: [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) lists the five "key tensions and unknowns" that drive the M2 ablation plan; this file is broader and lower-priority by default (things to look into when there is time, not things blocking a decision).

## Template

```
## Q<n>. <one-line question>

**Status**: open
**Filed**: YYYY-MM-DD
**Tags**: <comma-separated; e.g. systems, training, eval>

<Body: what is unknown, why it matters, and what to chase. Pointers to code paths, papers, runs, or framework versions where relevant.>
```

---

## Q1. Sequence packing in verl and NeMo-RL: works with Qwen3, not Qwen3.5

**Status**: open (preliminary answer filed; wall-clock follow-up pending)
**Filed**: 2026-05-07
**Tags**: systems, training, frameworks

**Question**: What is sequence packing in verl and NeMo-RL, and why does it work with the Qwen3 architecture but break with Qwen3.5?

**Context**: Verl not supporting Qwen3.5 is the load-bearing reason M2 ports to NeMo-RL ([CLAUDE.md "Gotchas"](../../claude/CLAUDE.md)). The 2026-05-06 smoke ran with packing OFF. If the dynamic-batching fallback turns out 1.5x to 2x slower than packing on the recipe-ablation shape, our 11 to 17 day wall-clock and 2 to 3 run budget shift.

**Answer**: Sequence packing concatenates multiple short sequences into one long tensor (no padding) and passes cumulative length offsets (`cu_seqlens`) to FlashAttention's varlen path; each segment only attends to itself. Verl exposes it as `use_remove_padding` ("RMPad"); NeMo-RL exposes it as `policy.sequence_packing`, supported on DTensor + Megatron since [PR #651](https://github.com/NVIDIA-NeMo/RL/pull/651) ([design doc](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html); background: [HF blog](https://huggingface.co/blog/packing-with-FA2)).

Qwen3 dense models (incl. Phase 1's Qwen3-0.6B) are pure self-attention; the standard FA-varlen path handles them. **Qwen3.5 is hybrid**, interleaving Mamba-style GatedDeltaNet `linear_attention` layers with `full_attention` ([vLLM Qwen3.5](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); interleaving recorded in [`training/fix/CHANGES.md:44-46`](../../training/fix/CHANGES.md)). The PyTorch chunked kernel `torch_chunk_gated_delta_rule` does not honour `cu_seqlens`; with packing on it raises `CUDA: illegal memory access` at `policy.get_logprobs()`. Mitigation in the active config: `policy.sequence_packing.enabled: false` + `policy.dynamic_batching.enabled: true` ([`grpo_qwen3.5_2b_1xa100.yaml:271-282`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml); confirmed in [`SMOKE_RESULTS_2026-05-06.md:25-29`](../training/SMOKE_RESULTS_2026-05-06.md)).

Verl stacks two extra blockers on top of the GDN issue: the RMPad allowlist historically excluded Qwen3 (added later, hence Phase 1's Qwen3-0.6B run on `verl_latest` worked; [verl#1305](https://github.com/volcengine/verl/issues/1305), [`RESULTS_m0_b.md:60`](../report/RESULTS_m0_b.md)); and Qwen3.5 is not loadable in verl's bundled transformers, while upgrading transformers breaks the vLLM rollout ([verl#5937](https://github.com/verl-project/verl/issues/5937)). Net: hard error, not silent drift; workaround is dynamic batching; NeMo-RL inherits the same GDN root cause but gives a clean off-switch.

**Open follow-ups**: quantify wall-clock cost of packing-off + dynamic batching on a real ablation shape; watch for an upstream GDN varlen fix ([Megatron-Bridge#1776](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1776) is a related bug to track).
