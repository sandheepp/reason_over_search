---
title: GPU MEMORY
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# What's in GPU memory during GRPO training

> Educational deep-dive on the memory layout of a colocated GRPO step. Concrete numbers for our 1× A100 80GB config (Qwen3.5-2B, seq=4096, group=5). Reading this is *not* required to run training; it's here for when something OOMs and you need to know what to evict.

## The high-level picture

GRPO is a two-phase loop:

1. **Rollout** — the policy *generates* multi-turn traces via vLLM (think → search → tool_response → think → answer).
2. **Training** — forward / backward / optimizer step on those traces.

NeMo-RL's default is **colocated mode** (`policy.generation.colocated.enabled: true`) — both phases share the same GPU. That means VRAM has to hold both vLLM's needs *and* training's needs simultaneously. `gpu_memory_utilization=0.6` is the budget split: vLLM gets up to 60% of VRAM, training gets the rest.

## Always resident (≈ 40 GB on our 1× A100)

These never leave the GPU during the whole 1005-step run:

| What | Size | Notes |
|---|---|---|
| **Training policy weights (bf16)** | **4 GB** | 2B params × 2 bytes. The thing we're training. |
| **Adam optimizer state** | **24 GB** | Two fp32 moments (`m`, `v`) at 8 GB each = 16 GB, plus an fp32 *master copy* of the weights (8 GB). Mixed-precision training keeps fp32 master so weight updates don't lose precision. |
| **Gradients (bf16)** | **4 GB** | One bf16 grad per param. With `grad_reduce_in_fp32: false` in our DDP config (default), grads stay bf16. |
| **Reference policy (bf16)** | **4 GB** | Frozen copy of the base Qwen3.5-2B used to compute KL(π‖π_ref) for `reference_policy_kl_penalty=0.001`. No optimizer state — never trained. Lives separately because we need its logprobs every step. |
| **vLLM weight copy (bf16)** | **4 GB** | A *separate* copy of the policy laid out for vLLM's inference kernels. This is the "duplicate weights" cost of colocated rollout. After each gradient step, NeMo-RL **refits** these from the training policy so vLLM samples from the latest checkpoint. |

So **~40 GB of an 80 GB A100 is gone before activations or KV cache enter the picture.**

This is why people complain about RL fine-tuning memory: a 2B model that you could comfortably fine-tune in fp16 with 16 GB suddenly needs 40 GB just to sit there because of (a) Adam state, (b) the reference model, (c) the vLLM weight duplication.

## During rollout (vLLM generating traces)

vLLM holds the **KV cache** — the running keys & values from every transformer layer, for every concurrent prompt being generated.

- Per-token KV size for Qwen3.5-2B (~28 layers, GQA with 8 KV heads, 128 dim, bf16):
  `2 (K+V) × 28 layers × 8 heads × 128 dim × 2 bytes ≈ 114 KB/token`
- vLLM's allocation under `gpu_memory_utilization=0.6`: ~48 GB total, minus 4 GB for vLLM weights = **~44 GB KV-cache budget**.
- That's ~390k tokens, so vLLM can hold **~95 concurrent prompts** at the full 4096 max_seq each, or many more if they're shorter.

For our run: 102 prompts/step × 5 generations = 510 concurrent rollouts. They don't all fit in KV cache simultaneously, so vLLM batches them in waves of ~95. Each wave runs in parallel using `generation_batch_size=32` as a soft scheduler hint.

While rollout runs, the training-side allocations (Adam state etc.) are *idle* — sitting in their reserved regions but not being read or written. They still consume VRAM.

The retriever is **off-GPU entirely** — runs as a separate CPU process at `127.0.0.1:3005`, holds the FAISS index in 65 GB of host RAM, queried over HTTP from inside the env actor.

## During training (forward / backward / step)

vLLM's KV cache is freed (the rollout finished and produced complete traces). Now training-side memory grows.

| What | Size | Lifetime |
|---|---|---|
| **Activations (with `activation_checkpointing=true`)** | **~4 GB** | One micro-batch's worth, kept until backward. With checkpointing only ~1 in 5 layers' activations are stored; the rest get recomputed on backward (5× memory savings, ~30% compute cost). Without checkpointing this would be ~20 GB. |
| **Logits / logprobs (transient)** | **0–4 GB** | The full logits tensor is `(micro_batch, seq, vocab) = (4, 4096, 151k) × 2 bytes ≈ 5 GB`. NeMo-RL chunks along the sequence dim via `logprob_chunk_size` to avoid materializing the whole thing. |
| **Reference logprobs** | **~0.1 GB** | Just the chosen-token logprob per position: `(micro_batch, seq) ≈ 16k floats`. Tiny — but computing them requires running the ref policy forward, which transiently uses its own activations (~4 GB, checkpointed). |
| **Gradient accumulators** | already counted above | 4 GB persistent. Each micro-batch's backward writes into them; cleared after `optimizer.step()`. |

For one training step: the dataloader hands us 510 trajectories; we chunk into 510 / 4 = **128 micro-batches**; each micro-batch does forward + ref-forward + backward; gradients accumulate; one final `optimizer.step()` updates weights and clears grads.

After the step finishes, **refit** copies the new weights into the vLLM allocation, the rollout phase starts again, and we go around the loop.

## Numerical sanity check

Worst case during training (everything resident + activations live):

```
Persistent (training side)
  policy bf16             :  4 GB
  Adam m+v fp32           : 16 GB
  master weights fp32     :  8 GB
  gradients bf16          :  4 GB
  reference policy bf16   :  4 GB
                          ──────
                           36 GB

Persistent (vLLM side)
  vLLM weight copy        :  4 GB

Transient training (peak)
  activations (ckpt'd)    :  4 GB
  ref forward activations :  4 GB
  logits chunk            :  1 GB
                          ──────
                            9 GB

Reserved for vLLM (idle during training)
  KV/buffers reserve      : 28 GB  (= 0.6 × 80 − 4 GB vLLM weights − overhead)

TOTAL                     : ~77 GB / 80 GB
```

Tight. That's why our config bumped `train_micro_batch_size` from upstream's 4 (at seq=512) down for our seq=4096, and turned `activation_checkpointing` ON.

## OOM recovery ladder

If we hit OOM, the recovery sequence — in order of preference, since each later step costs more wall-clock — is documented in [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md):

1. `policy.train_micro_batch_size=2` → halves activation memory in step (most common fix).
2. `policy.generation.vllm_cfg.gpu_memory_utilization=0.55` → reduce vLLM's reserve, more for training.
3. `policy.dtensor_cfg.cpu_offload=true` → moves Adam state to RAM (slow but always wins).

## Why colocated is worth it

The alternative (`colocated.enabled: false`) puts vLLM on dedicated GPUs and training on others. That eliminates the dance for memory budget but doubles GPU count. For 2B on 80 GB A100, colocated is the right trade — there's *just enough* VRAM if we're disciplined.

For 2× A100 (`grpo_qwen3.5_2b_2xa100.yaml`), we use `vllm.tensor_parallel_size=2` so vLLM weights split across the two GPUs. Per-GPU vLLM weight footprint drops from 4 GB to 2 GB, freeing some KV cache budget — that's why the 2× config bumps `gpu_memory_utilization` from 0.6 to 0.7. Training stays DDP at TP=1; per-GPU training memory is identical to the 1× config.

## What the GPU is doing across one full step

```
Time →

Rollout phase (~70% of step time)
  GPU: vLLM kernels  ← KV cache fills, attention runs, tokens emitted
  Training-side weights idle in their reserved region

  Env actors parse <tool_call>, hit /batch_search on CPU
  Retriever (CPU) returns docs, env wraps as <tool_response>
  vLLM continues from the new context for the next turn

  Repeat up to max_rollout_turns=4 per trajectory, for all 510 trajectories

Training phase (~30% of step time)
  GPU: training kernels  ← forward × 128 micro-batches, ref forward, backward,
                            gradient accumulation, optimizer.step()
  vLLM weights idle but resident
  vLLM KV cache freed (waiting for next rollout)

Refit (<1% of step time)
  GPU: copy training policy → vLLM weight slot
  Both sides see fresh weights

Repeat for 1005 steps.
```

Multiply by ~5–10 minutes per step → 80–170 hours total wall clock projection on 1× A100. See [`PAPER_VS_OURS_TRAINING.md §7`](../training/PAPER_VS_OURS_TRAINING.md#7-compute) for the derivation.
