---
title: Milestone 5.3 — Analyze and improve M5.1 training time
tags: [milestone, m5.3, training, performance, nemo-rl, qwen3.5]
source: internal
created: 2026-05-11
updated: 2026-05-12
status: planning (M5.1 not yet completing)
---

# Milestone 5.3 — Analyze and improve M5.1 training-time

Companion to [`MILESTONE_5.md`](MILESTONE_5.md) (M5 scaffold + M5.1 full run) and [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md) (paper-faithfulness audit). M5.3 is the **post-M5.1 efficiency milestone**: take the observed bottlenecks from M5.1 training and explore source-level fixes that were out of scope during the original launch.

**Status (2026-05-12):** Planning. M5.1 is currently **not running** — a1 crashed at step 50 (config bug), a2 was killed at step 15 (misdiagnosis); see [`../report/RESULTS_SMOKE_m5.md` §7 / §7.8 / §7.8.1](../report/RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11). a3 awaits user authorization. M5.3 work begins once M5.1 lands a clean ckpt or when an interim ckpt is good enough to validate a patch against.

The training-phase analysis below remains accurate: a1's 49-step trajectory and a2's 15-step trajectory both confirm the 71-72% training-phase share at production shape. The `micro=1` ceiling and the `model_utils.py:1378` fp32-cast root cause are unchanged.

---

## 1. Why M5.3 exists

The M5.1 production run uses `train_micro_batch_size=1` because:
- v7 baseline OOM'd at `micro=2` with seq=8192 (off by 0.62 GiB).
- Root cause: NeMo-RL's TP=1 logprob path casts the full `[B, S, V] = [2, 8192, 248320]` logits tensor to fp32 **before any chunking can apply** at [`distributed/model_utils.py:1378`](../../training_m5_1/nemo_rl/nemo_rl/distributed/model_utils.py#L1378). 16.3 GiB allocation; we had 13.11 GiB free.
- `logprob_chunk_size` is wired to `LogprobsPostProcessor` (logprob phase) but not to `LossPostProcessor` (training phase). Setting it does not help the training OOM.

Consequence: per-step wall is dominated by training (71-72% of step, vs 58% at smoke v6 shape). At ~30 min/step steady state, the full 622-step run is **~10-15 days**. Unlocking `micro=2` would roughly halve the training-phase wall-clock → projected **~6-8 day** full run.

Per-step phase share at v7-baseline & live (M5.1 step 8):

| Phase | Smoke v6 | M5.1 step 8 |
|---|---:|---:|
| `policy_training` | 58.4% | **71.6%** |
| `policy_and_reference_logprobs` | 18.7% | 20.5% |
| `generation` (vLLM async) | 10.6% | 6.9% |
| `prepare_for_generation` | 7.0% | 0.3% |

Training is the bottleneck and growing as the model produces shorter rollouts (generation work shrinks → training share grows).

---

## 2. Scope

M5.3 is **explicitly limited to throughput / memory levers that do not change training math** vs the M5.1 paper-faithful config. The same paper-faithfulness boundary from M5.2 applies (see [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md) §8 for the locked knobs).

Out of scope:
- Anything that changes GRPO group size, KL term, reward shape, advantage estimator, or sequence-length budget.
- Multi-node training (we own 1× A100 by design).
- Algorithm-side changes (DAPO / Dr. GRPO / drop-KL) — these live in [`PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md), not here.

---

## 3. Lever menu (research targets)

### 3.1 Chunked fp32-cast in TP=1 training path — **primary target**

**Lever**: patch [`distributed/model_utils.py:1378`](../../training_m5_1/nemo_rl/nemo_rl/distributed/model_utils.py#L1378) so the non-TP path (the one our `tensor_parallel_size: 1` config hits) chunks the `next_token_logits.to(torch.float32)` cast along the sequence dimension. Mirror the chunking pattern that already exists in `ChunkedDistributedLogprobWithSampling` (TP>1 path) for the TP=1 case.

**Expected wins**:
- Unlocks `train_micro_batch_size=2` at seq=8192 → halves training-phase wall-clock → 622 steps × ~15-20 min = **6.5-9 days**.
- Frees up gen budget headroom for future R4-style `gpu_memory_utilization` bumps.

**Risks**:
- Numerical equivalence vs the unchunked path needs validation. Loss must match within `seq_logprob_error_threshold` over a sequence-length sweep.
- Kernel launch overhead per chunk; tune chunk size (probably 1024 or 2048 tokens).

**Effort**: 1-2 days source mod + smoke validation. Mostly a port of the chunked TP>1 path to TP=1.

### 3.2 Decouple `logprob_batch_size` from `train_micro_batch_size`

**Lever**: the current config wires `logprob_batch_size: ${policy.train_micro_batch_size}`. After 3.1 unblocks `train_micro_batch_size=2`, the logprob phase (still 20% of step) might tolerate a higher batch (less memory-bound since `logprob_chunk_size` works there). Test `logprob_batch_size=4` or `8` independently.

**Expected wins**: 10-20% on the 20% logprob slice → 2-4% on full step.

**Risks**: Low. logprob phase already supports chunking; just need to verify peak memory at the higher batch.

**Effort**: 1 evening — config-only change after 3.1.

### 3.3 `gpu_memory_utilization` bump (revisit R4 once micro=2 lands)

**Lever**: M5.2 skipped R4 (0.5 → 0.6) because v7 OOM was 0.62 GiB into the red. With micro=2 unlocked, the training peak drops; vLLM may have headroom to grow.

**Expected wins**: vLLM gen phase already only 4-7% of step at production shape, so the absolute gain is small — but if the rollout phase grows in importance after micro=2 lands (since training shrinks), R4 might be worth another look.

**Risks**: OOM during gen-train colocation swap; protected by smoke iteration.

**Effort**: 1 smoke iteration.

### 3.4 `torch.compile` on the policy DTensor worker

**Lever**: NeMo-RL DTensor backend doesn't expose `enable_torch_compile`; only SGLang does ([`generation/sglang/config.py:45`](../../training_m5_1/nemo_rl/nemo_rl/models/generation/sglang/config.py#L45)). Adding the knob in `dtensor_cfg` would let us compile the forward+backward.

**Expected wins**: 1.10-1.25× on the training phase per [`RUNTIME_EFFICIENCY.md` §O5](../research/RUNTIME_EFFICIENCY.md#5-optimizer--training-phase) → at 72% of step = **~7-18%** wall-clock cut.

**Risks**:
- Qwen3.5's hybrid GatedDeltaNet + attention layers may not compile cleanly (compile graph breaks).
- 5-10 min initial compile cost (amortized over 622 steps = negligible).
- The reduction from `cudagraph` may interact with Ray remote actor lifecycle.

**Effort**: 2-3 days — needs NeMo-RL source mod to plumb the config + Qwen3.5 graph-break debugging.

### 3.5 Sequence packing (BLOCKED — model swap)

**Lever**: `sequence_packing.enabled=true` packs trajectories of different lengths into the same micro-batch, eliminating padding waste. NeMo-RL supports it natively.

**Blocker**: Qwen3.5's GatedDeltaNet kernel crashes with packed sequences ([`training/fix/CHANGES.md §5`](../../training/fix/CHANGES.md)). Two unblock paths:
- Patch the GDN kernel upstream (real maintainer-level work).
- Switch base model to **Qwen2.5-1.5B-Instruct** (no GDN). Packing then works. Documented as defensible per [`PARADIGM_REVIEW.md §8`](../research/PARADIGM_REVIEW.md#8-search-r1-empirical-study).

**Expected wins**: 30-60% on training + logprob phases (RUNTIME_EFFICIENCY §O6) — but only if we model-swap.

**Risks**: Model swap is a recipe-level decision, not an efficiency knob. Would need new prompt + reward validation.

**Effort**: 3-4 days for the model swap variant; indefinite for the GDN kernel patch.

### 3.6 8-bit AdamW (bitsandbytes)

**Lever**: optimizer state in fp32 currently uses ~6.4 GB. 8-bit halves to ~3.2 GB. Frees memory that 3.1 or 3.3 can claim.

**Blocker**: NeMo-RL doesn't ship bnb integration for the DTensor backend (RUNTIME_EFFICIENCY §O2 — "PR-level effort").

**Expected wins**: Direct gain ~0; enabling gain ~10-20% via 3.3 + larger micro-batch.

**Risks**: Optimizer-state precision drift; rare on RL post-training.

**Effort**: 3-5 days source mod.

---

## 4. Plan of attack (ordering)

| Priority | Lever | Why first | Estimated effort | Expected step-time cut |
|---:|---|---|---|---:|
| 1 | 3.1 — chunked fp32-cast | Biggest win; the bottleneck the live data identifies | 1-2 d | ~50% on training phase |
| 2 | 3.2 — decoupled logprob batch | Cheap follow-up once 3.1 lands; tests memory headroom | 0.5 d | 2-4% on step |
| 3 | 3.4 — `torch.compile` for DTensor | Independent of 3.1; could land in parallel | 2-3 d | 7-18% on step |
| 4 | 3.3 — R4 revisit | Needs 3.1's headroom; small absolute gain | 0.5 d | 1-2% on step |
| 5 | 3.5 / 3.6 — model swap or 8-bit AdamW | Only if a-priori 1-4 didn't get us under 5 days | 3-5 d each | model swap: 30-60% / 8-bit: 5-15% via 3.3 |

Each lever lands on its own branch (cut from `research_v2`), with a 10-step production-shape smoke validating per-step wall-clock and (for 3.1) numerical equivalence vs the baseline.

Targets:
- **After 3.1 alone**: ~6-8 d full run.
- **After 3.1 + 3.4**: ~5-6 d.
- **Stretch (3.1 + 3.4 + model swap)**: ~3-4 d.

---

## 5. Validation strategy

Each lever must clear three gates before merging into a production yaml:

1. **No OOM** across a 10-step production-shape smoke at full `num_prompts_per_step=64`.
2. **Numerical equivalence** vs the M5.1 baseline. Concretely: train_data jsonls from step 1 should produce loss values within `seq_logprob_error_threshold` of the baseline run with the same seed + data slice. For 3.1 this is critical (chunked vs unchunked log-softmax should be bitwise-equivalent in fp32; in bf16 a small numerical drift is acceptable if reward trajectory is unchanged).
3. **Per-step wall-clock improvement** of the predicted magnitude. If 3.1 yields <30% on training-phase wall-clock vs baseline, the patch is suspect.

The M5.1 final checkpoint serves as the **continuation point** for M5.3 runs: rather than re-training from scratch to validate, we can resume from a mid-training M5.1 ckpt and compare per-step times directly. NeMo-RL `checkpointing` supports `get_latest_checkpoint_path` + dataloader state restore (`grpo.py:323-371`), so resume is straightforward.

---

## 6. Open research questions

These need literature / upstream-issue research before/during implementation:

1. **Is there an upstream PR for chunked TP=1 logprob in NeMo-RL?** Search [NVIDIA-NeMo/RL issues + PRs](https://github.com/NVIDIA-NeMo/RL). If yes, cherry-pick instead of patching.
2. **What's the right chunk size for the fp32 cast in 3.1?** TP>1 path uses `chunk_size` from logprob_chunk_size; that's typically 1024 or 2048. Validate the trade-off (smaller → less memory, more kernel launches).
3. **Does `torch.compile` work on Qwen3.5 hybrid arch in DTensor?** Check NVIDIA test suite / discussion forums for hybrid-arch compile bugs. May need `fullgraph=False` and per-module graph wrapping.
4. **Sequence-packing alternatives**: are there alternative kernels for GatedDeltaNet that support variable-length / packed inputs? Check Mamba2 / flash-mamba codepaths.
5. **What is the actual marginal cost of running `train_micro_batch_size=2` vs `1` at seq=8192 in NeMo-RL?** Quantify: kernel launches, all-reduce overhead, memory allocator behavior. The 1.7× wall-clock overhead from micro=1 we observed is suspicious; we should understand it before deciding whether 3.1 actually buys 50%.

These map to a research/ doc to be authored alongside the implementation work; reference [`RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md) for the broader systems context and [`PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md) for the algorithm-side companion.

---

## 7. Out-of-scope levers (recorded for completeness)

These were considered in M5.2 and rejected on paper-faithfulness or hardware grounds:

| Lever | Why out of scope |
|---|---|
| `G=5 → 4` (group size) | Changes GRPO baseline statistics; user-locked. |
| `max_total_sequence_length=8192 → 3072` | Increases truncation rate; user-locked. |
| `β=0` (drop KL) | Training math change; covered in [`PARADIGM_REVIEW.md §3`](../research/PARADIGM_REVIEW.md#3-removing-or-replacing-the-kl-term). |
| `DAPO dynamic sampling` | Training math change. |
| `LoRA r≥64` | Capacity ceiling at <2B per Plasticity vs Rigidity. |
| `async_grpo.enabled` | Slight off-policy correction; tested in M2 but not in M5.1. |
| EAGLE-3 spec-dec | Rollout phase already <7% of step at production shape — not the bottleneck. |
| 2× A100 decolocation | Owns 1× A100 by design. |

If the M5.3 lever stack still doesn't get under 5 days for the full run, revisit some of these (especially `async_grpo` and EAGLE-3) under a fresh divergence-budget.

---

## 8. Pointers

- M5 milestone narrative: [`MILESTONE_5.md`](MILESTONE_5.md)
- M5 paper-faithfulness audit: [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md)
- M5 smoke iteration log (where the v7 OOM is documented): [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md)
- M5 code-setup audit: [`../report/CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md)
- Runtime-efficiency menu (full systems-lever catalogue): [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)
- Paradigm review (algorithm-side companion): [`../research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md)
- NeMo-RL upstream: [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
- The bottleneck site: [`training_m5_1/nemo_rl/nemo_rl/distributed/model_utils.py:1378`](../../training_m5_1/nemo_rl/nemo_rl/distributed/model_utils.py#L1378)
