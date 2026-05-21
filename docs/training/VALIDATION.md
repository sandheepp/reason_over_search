---
title: VALIDATION
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-06
---

# Validation Set & Cadence

In-loop validation during GRPO training. Mirrors Search-R1's validation setup so the training dynamics are comparable to the paper.

Sources: [Search-R1 arXiv 2503.09516](https://arxiv.org/abs/2503.09516), [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1).

> ⚠️ **DISABLED for first-pass training** (current state). Both YAML configs at [`training/configs/`](../../training/configs/) have `grpo.val_period: 0`, `grpo.val_at_start: false`, `grpo.val_at_end: false`, `data.validation: null`. The first-pass run is mechanics verification only — getting the GRPO loop, retriever HTTP, env actor, and W&B logging to all line up cleanly. Once that's confirmed, re-enable per the steps in §7 below.

> **Status (paper-side)**: paper-side numbers and conventions are settled. `test_freq=100` and `total_training_steps=1005` are confirmed against upstream [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) — the EM-only baseline that produced the published GRPO checkpoints we evaluated in Milestone 1. We're matching this once we re-enable.

---

## 1. Training corpus (for context)

Search-R1 trains on a **mix of NQ-train and HotpotQA-train**, both from the standard FlashRAG / Search-R1 distribution. We use the same files in [`data/`](../../data/) (already present from Milestone 1). Training corpus committed to LFS at [`data/training/nq_hotpotqa_train/`](../../data/training/nq_hotpotqa_train/).

Verified row counts (`pyarrow.parquet.read_metadata`):

| Split | Rows |
|---|---:|
| `train` | 169,615 |
| `test` (in-loop val + final eval) | 51,713 |

**Paper / verl regime.** With `train_batch_size=512` × `n_agent=5` (= 2560 trajectories/step) × 1005 steps, the loop sees `512 × 1005 = 514,560` prompts — ~3.03× over the 169,615-row corpus (roughly 3 epochs). (Paper text says "500 steps" but the published-checkpoint verl run is 1005; see [`PAPER_VS_OURS_TRAINING.md`](PAPER_VS_OURS_TRAINING.md) §5.)

**Our (NeMo-RL) regime.** With `num_prompts_per_step=102` × `num_generations_per_prompt=5` (= 510 trajectories/step; 5× fewer than verl) × 1005 steps, the loop sees `102 × 1005 = 102,510` prompts — **~0.604× over the corpus, well under 1 epoch**. Smaller rollout footprint per step is the price we pay to fit on 1× A100 80GB; full breakdown in [`PAPER_VS_OURS_TRAINING.md §7`](PAPER_VS_OURS_TRAINING.md#7-compute) and [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md).

## 2. Validation dataset

**Paper:** Search-R1 logs in-loop validation on **the same NQ + HotpotQA splits** they use for the final test (held out from training). Specifically the Milestone 1 jsonl files at:

- [`data/nq.jsonl`](../../data/nq/test.jsonl) (NQ-test)
- [`data/hotpotqa.jsonl`](../../data/hotpotqa/dev.jsonl) (HotpotQA-dev)

These are the in-distribution benchmarks; out-of-distribution datasets (Bamboogle, MuSiQue, 2Wiki, PopQA, TriviaQA) are eval-time only — *not* used during training validation.

**Ours:** match the paper. Use the same two jsonl files. To avoid overfitting validation choice to a single sample, we'll use a **deterministic 1k subsample** ([`data_subsample/nq_1k.jsonl`](../../data_subsample/nq/test.jsonl), [`data_subsample/hotpotqa_1k.jsonl`](../../data_subsample/hotpotqa/dev.jsonl)) for fast in-loop validation; the full sets are reserved for post-training eval.

## 3. Cadence

**Paper / verl:** `save_freq=100` and `test_freq=100` (validation co-runs with each checkpoint) — confirmed in [`scripts/nq_hotpotqa/v0.2/train_grpo.sh:68-69`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh). Plus `+trainer.val_before_train=true` runs validation at step 0 as a baseline.

**Ours (planned, after re-enable):** match — `checkpointing.save_period: 100`, `grpo.val_period: 100`, `grpo.val_at_start: true`. **First-pass (current):** all three off.

With verl's 1005-step training, the planned cadence gives **10 validation points** per run plus step 0 = **11 total**.

## 4. Metrics logged to W&B

At every validation step:

Emitted by [`SearchR1Env.global_post_process_and_metrics`](../../training/src/environments/search_r1_env.py) (mirroring `MathEnvironment`'s shape so checkpointing can select on `val:accuracy`):

| Metric | Source |
|---|---|
| `val/accuracy` | Mean reward (zero-masked for not-properly-ended rollouts) — the canonical `metric_name` for `keep_top_k` checkpointing. For paper arm with full format match this is in [0, 1.0]; for qwen_native the format-failure penalty caps at ~0.8 even on correct EM, so `accuracy` is a reward-mean, not a strict EM rate. |
| `val/em_hit_rate` | Fraction of rollouts with reward ≥ 0.8 (proxy for "got the answer", arm-agnostic). |
| `val/fraction_of_samples_properly_ended` | Fraction of rollouts that emitted a real stop token (not max-tokens-truncated). Should be ≥ 0.95 by step ~200; lower means model isn't learning the format. |
| `val/generation_lengths` | Mean total response length in tokens. |
| `val/prompt_lengths` | Mean prompt length (post chat-template) in tokens. |
| `val/num_problems_in_batch` | Sanity counter — should match `grpo.max_val_samples`. |

Per-step training metrics (already standard in NeMo-RL):

- `train/reward_mean`, `train/reward_std`
- `train/kl`, `train/policy_loss`, `train/clip_fraction`
- `train/lr`, `train/grad_norm`
- `train/rollout_throughput` (tokens/s)

## 5. Stopping criteria

**Paper / verl:** fixed schedule of 1005 steps (`total_training_steps=1005`, capped before the 15-epoch nominal); no early stopping.

**Ours (when validation is re-enabled):** match. Run all 1005 steps. Save the best-by-`val/accuracy` checkpoint via NeMo-RL's `keep_top_k: 3` mechanism; that's the candidate we feed into the Milestone 1 eval pipeline. **First-pass (current)**: validation off, no checkpointing, just run all 1005 steps and verify training mechanics from W&B `train/*` metrics.

If reward is collapsing or training is unstable (NaN loss, KL spike), abort manually rather than auto-stop — we want the failure mode visible in W&B for diagnosis, not silently truncated.

## 6. Open questions

1. **Validation subsample size.** 1k per validation point × 11 validations × {NQ, HotpotQA} × 5 GRPO rollouts (group=5) = 110k validation rollouts per run. If this dominates wall-clock, drop to 500-question subsamples.
2. **Validation rollout sampling.** Paper does not state whether validation rollouts use temp=0 or temp=1. Eval is greedy (Milestone 1 confirmed); training rollouts are temp=1. We default validation to **temp=1, single sample** to match the *training* distribution (the point of in-loop val is to track what the policy actually does during training). Final post-training eval uses the Milestone 1 pipeline at temp=0.

## 7. Re-enabling validation (planned, not active)

When the first-pass training run is confirmed mechanically sound, restore validation by editing both [`training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`](../../training/configs/):

```yaml
grpo:
  val_period: 100        # was 0
  val_at_start: true     # was false
  val_at_end: true       # was false
  max_val_samples: 1000  # already set; effective once val_period > 0

data:
  validation:            # was null
    dataset_name: search_r1
    data_path: data/training/nq_hotpotqa_train/test.parquet
    arm: qwen_native     # must match data.train.arm

checkpointing:
  enabled: true          # was false
```

After flipping these, the run will produce 11 validation points (step 0 + every 100 steps over 1005 steps) and keep the top-3 checkpoints by `val:accuracy`. The metric name and env hooks are already wired — see [`SearchR1Env.global_post_process_and_metrics`](../../training/src/environments/search_r1_env.py).

No code changes needed for the re-enable; it's a config flip.
