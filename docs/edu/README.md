---
title: README
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# Educational deep-dives

Explainers that go deeper than the operational docs — for the moments where you want to understand *why* a particular knob exists, not just what to set it to.

These are not load-bearing for running the pipeline. The runbook + configs are the source of truth for "what to do"; these docs are "what's actually happening".

## Index

| File | Topic |
|---|---|
| [RNG.md](RNG.md) | What a random number generator actually is — pseudo-random vs true random, why they're stateful, the four+ separate RNGs in our stack, and why "fully reproducible" is approximate. Read first if the term `RNG` is unfamiliar. |
| [SEED.md](SEED.md) | What `--seed` controls, where it propagates in the pipeline, what it does *not* fix, and why we run 3 seeds × 2 variants in Phase 2. |
| [GPU_MEMORY.md](GPU_MEMORY.md) | What lives in VRAM during GRPO training — always-resident state vs. rollout-phase vs. training-phase, with concrete numbers for our 1× A100 80GB config. |
| [BATCH_MATH.md](BATCH_MATH.md) | The three batch-size knobs (`gbs`, `num_prompts_per_step`, `num_generations_per_prompt`) and how they interact. Why the convention `gbs == prompts × gen` exists, what `force_on_policy_ratio` enforces, and the ~10× gradient-update gap between our setup and the paper's verl-style multi-update-per-step. |
| [GRPO_STEP_LIFECYCLE.md](GRPO_STEP_LIFECYCLE.md) | What runs (and when) inside one GRPO training step — the two forward-pass surfaces (vLLM rollout vs. training policy), micro-batch gradient accumulation, weight refit, and the importance ratio. Companion to GPU_MEMORY.md. |

## See also

- [`docs/training/README.md`](../training/README.md) — overlay architecture + step-5 audit summary
- [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) — every paper hyperparameter cross-checked
- [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) — concrete run sequence
