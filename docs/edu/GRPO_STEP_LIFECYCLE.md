---
title: GRPO STEP LIFECYCLE
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# What happens in one GRPO training step

> Educational deep-dive on the per-step compute flow during GRPO training. Concrete numbers for our 1× A100 80GB config (Qwen3.5-2B, group=5, global batch=512). Companion to [GPU_MEMORY.md](GPU_MEMORY.md) — that doc covers *what's in VRAM*; this one covers *what runs and when*.

## The high-level picture

One training step has two forward-pass surfaces, not one:

1. **Rollout phase** — vLLM does forward passes only (inference). No gradients.
2. **Training phase** — the policy model (separate PyTorch copy with autograd) does forward + backward + one optimizer step.

The vLLM model and the training policy are **two copies of the same weights**. They go in and out of sync: vLLM is "stale" during rollout, then gets refreshed with the trained weights after the optimizer step.

## Per-step lifecycle (5 phases)

### 1. Rollout phase — vLLM, no gradients

- Sample 102 prompts (= `train_global_batch_size / group_size = 512 / 5`).
- For each prompt, generate **5 trajectories** in parallel → **510 trajectories total**.
- Each trajectory is multi-turn: `<think> → <tool_call>search → <tool_response>docs → <think> → <answer>`. Up to `max_rollout_turns=4` retrieval calls per trajectory (Search-R1's setting).
- Every token vLLM emits is a forward pass. Multi-turn ≠ multi-batch — vLLM uses KV-cache continuation, so the cost grows ~linearly with total tokens, not quadratically.
- **No autograd, no gradients, no backward**. This is pure inference.
- vLLM is using weights from **the end of the previous step** (or step 0 for the first iteration).

### 2. Reward + advantage computation — CPU/GPU mix, ~1s

- For each of the 510 trajectories: extract the `<answer>X</answer>` span, EM-score against `golden_answers` → reward ∈ {0, 1}.
- Within each group of 5 trajectories from the same prompt, normalize:
  ```
  advantage_i = (reward_i − group_mean) / group_std
  ```
  Or the leave-one-out variant: `advantage_i = reward_i − mean(other 4)`. Both are roughly equivalent in expectation; NeMo-RL uses leave-one-out by default.
- Build a per-token `loss_mask`: `1` for tokens the model *generated* (assistant role), `0` for tokens it *consumed* (user, tool/retrieval-response). Only assistant tokens contribute to the policy gradient.

### 3. Training phase — policy model, with autograd

This is where backward passes happen.

- The 510 trajectories are split into **micro-batches**. With `train_micro_batch_size=4`, that's `510 / 4 ≈ 128 micro-batches`.
- For each micro-batch:
  1. **Forward** through the policy model (PyTorch with autograd) → get log-probs of every token.
  2. **Forward** through the reference model (frozen Qwen3.5-2B-Base) → get reference log-probs (for KL term).
  3. Compute the PPO loss:
     ```
     ratio = exp(logp_new - logp_old)
     surr1 = ratio * advantage
     surr2 = clip(ratio, 1−ε, 1+ε) * advantage     # ε=0.2
     pg_loss = -min(surr1, surr2)
     kl_term = β * KL(π_new || π_ref)               # β=0.001 (k3 estimator)
     loss = (pg_loss + kl_term) * loss_mask
     ```
  4. **Backward** → `loss.backward()` accumulates gradients into `param.grad`.
- After all 128 micro-batches: **one** `optimizer.step()` (Adam) updates all params, then `optimizer.zero_grad()` clears the accumulator.
- That's **one** gradient update per training step. GRPO does a single PPO inner epoch by default — if you increased that to 2, you'd loop steps 3.1–3.4 over the same 510 trajectories twice with two optimizer steps.

### 4. Weight refit — vLLM gets the new weights

- The new policy weights (in `param.data`) are streamed/copied into vLLM's loaded model.
- For a 2B model, this is a few seconds of host↔GPU transfer.
- vLLM is now ready to roll out from the updated policy.

### 5. Logging + checkpoint (every 100 steps)

- Push metrics to W&B: `train/reward_mean`, `train/kl`, `train/policy_loss`, `train/clip_fraction`, etc.
- Every `checkpointing.save_period=100` steps: save the policy + optimizer state to disk.
- Every `grpo.val_period=100` steps: pause training, run a validation rollout against held-out NQ + HotpotQA, log `val/em` etc.

Then loop back to step 1 with the new vLLM weights. **1005 outer steps total.**

## Where mental models often go wrong

| Common belief | What's actually happening |
|---|---|
| "510 forward passes per step" | ~510 trajectories in rollout (each is *many* token forward passes through vLLM, no backward). Then **128 forward+backward micro-batches** through the *training* policy. Two different forward-pass surfaces, two different model copies. |
| "Backward pass per trajectory" | Backward happens per **micro-batch** (4 trajectories at a time on our config), not per trajectory. The micro-batches accumulate gradients into the same buffer; the optimizer steps only after all micro-batches are done. |
| "Gradients accumulate forever" | Only across the micro-batches *within one optimizer step*. After `optimizer.step()`, `optimizer.zero_grad()` resets to zero. Each of the 1005 outer steps is one full forward+backward+update cycle. |
| "vLLM is the model being trained" | vLLM holds a **read-only copy** of the policy used purely for rollout. Training happens on a separate PyTorch model. After each optimizer step, the trained weights are copied into vLLM (the "refit"). |
| "Training updates use the latest weights to score" | The trajectories were generated by π_old (vLLM's weights at rollout time). The PPO loss uses the *importance ratio* `π_new / π_old` to correct for this — see below. |

## The importance ratio (and why ε=0.2 matters)

The PPO loss includes a ratio:

```
ratio = exp(logp_new - logp_old)
```

- `logp_old` = the log-probability of the trajectory's tokens **at rollout time**, recorded by vLLM as it generated them.
- `logp_new` = the same log-probability **right now**, computed by the training model (after step 3.1's forward pass).

Why this exists: the trajectories were sampled from π_old, but we want to update π_new. Without the ratio, gradient estimates would be biased — we'd be evaluating "what π_new would have done" using samples that reflect "what π_old did do".

The `clip(ratio, 1−ε, 1+ε)` step (ε=0.2 for Search-R1) bounds how much the new policy can drift from the old one in a single update. If ratio > 1.2 (new policy gives this token *much* higher probability) or ratio < 0.8 (much lower), the gradient gets clipped — preventing one batch from yanking the policy too far.

In Search-R1's GRPO with **single PPO inner epoch**:
- π_new is exactly one optimizer step ahead of π_old
- Ratios stay near 1.0
- `clip_fraction` (the fraction of tokens where clipping fires) is typically <5%

If you bumped `ppo.num_inner_epochs` to 4 (looping steps 3.1–3.4 four times over the same trajectories), π_new would drift further from π_old between epochs, ratios would diverge, and clipping would matter much more. That's the PPO/GRPO knob that controls "how much do we squeeze out of each batch of rollouts before generating new ones".

## Phase-by-phase wall-clock (estimated, 1× A100 80GB, Qwen3.5-2B)

These are guesses pre-first-run; we'll fill in real numbers from W&B once a training run completes.

| Phase | Estimated wall-clock | Notes |
|---|---|---|
| Rollout (510 trajectories × ~4 turns × ~500 tokens) | 60–120s | Bound by vLLM throughput. Largest single chunk of step time. |
| Reward + advantage | <1s | EM scoring + group normalisation, mostly CPU. |
| Training (128 micro-batches × forward+backward) | 30–60s | Bound by policy model FLOPs + grad accumulation. |
| Optimizer step | ~1s | Adam update on 2B params. |
| vLLM weight refit | 2–5s | Host→device weight copy. |
| **Total per step** | **~100–200s** | At 1005 steps × 150s avg ≈ **42 hours** for one full training run on 1× A100. |

The rollout phase dominates because it's autoregressive (no parallelism across tokens within a trajectory). Increasing `num_generations_per_prompt` (group size) trades GPU utilisation against trajectory diversity — bigger groups fill the rollout batch better but reduce the prompt diversity per step.

## Cross-references

- [GPU_MEMORY.md](GPU_MEMORY.md) — what occupies VRAM during each of these phases (the memory side of the same picture)
- [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) §6 — every hyperparameter referenced above (group size, KL coef, clip ratio, micro-batch size, etc.)
- [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) §3 — GRPO algorithm knobs in NeMo-RL config language
- [`docs/training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md) §2 — the `low_var_kl ≡ k3` proof + state-masking / loss-mask explanation
