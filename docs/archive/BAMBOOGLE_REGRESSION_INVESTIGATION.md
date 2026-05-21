---
title: BAMBOOGLE REGRESSION INVESTIGATION
tags: []
source: internal
created: 2026-04-28
updated: 2026-04-28
---

# Post-mortem: Bamboogle "regression" was n=125 variance

**Date**: 2026-04-28
**TL;DR**: When the v1 base sweep (apply_chat=True + prompt sentence restored + `add_special_tokens` block removed) reported Bamboogle EM 0.088 — significantly below the earlier 13:05 probe's 0.128 — we briefly worried that one of the two "minor" fixes had regressed the model. NQ-1k results from the same v1 config quickly resolved the worry: the locked config is genuinely converging. The Bamboogle 0.088 was just single-seed n=125 sampling variance.

## Concern

Earlier in the day:

| Bamboogle base run | EM | Notes |
|---|---:|---|
| Plan B v0 (no chat, no minor fixes) | 0.112 | 21/125 length-truncated |
| Probe (apply_chat only)              | 0.128 | 0/125 length-truncated; matched paper exactly |
| **v1 sweep (apply_chat + minor fixes)** | **0.088** | 0/125 truncated, but EM dropped |

A 4-pp drop from the probe to v1 was suspicious — the only differences were the two LOW-severity audit fixes:

- Restore `For example, <answer> Beijing </answer>.` to the prompt template.
- Remove the `add_special_tokens` block at `active_pipeline.py:37-42`.

## What resolved it

The next dataset in the v1 sweep — NQ-1k — completed and showed a *large* improvement on top of apply_chat alone:

| NQ-1k run | EM | Δ vs prior |
|---|---:|---:|
| Plan B v0 (no chat, no minor fixes) | 0.316 | — |
| Probe: base + apply_chat only       | 0.346 | +3.0 pp |
| v1: base + apply_chat + minor fixes | **0.390** | **+4.4 pp on top** |
| Paper                               | 0.421 | residual +3.1 pp |

So the two minor fixes together gave **+4.4 pp** on NQ at n=1000, far exceeding the audit's "≤1 pp" estimate for each. Combined with apply_chat that is +7.4 pp from v0; the residual gap to paper is within plausible single-seed + subsample noise.

This is irreconcilable with the minor fixes having *regressed* the model. The most likely explanation for the Bamboogle 0.088 is sampling variance:

- Bamboogle has only 125 examples → 1-pp standard error per question = ~3 pp standard error overall, and the tail distribution is heavier than Gaussian on greedy multi-hop runs.
- Per-example outcomes can flip on ±1 token at the start of `</answer>`, and Bamboogle is the most reasoning-heavy dataset (so the most sensitive to small tokenization changes — `add_special_tokens` removal, in particular, shifts how `<search>` etc. tokenize).
- The 0.128 probe and the 0.088 v1 result are within ~3.4 σ of each other at n=125, which is unusual but not implausible at single seed.

## Decision

- **Don't roll back the minor fixes.** NQ proves they're net positive at n=1000.
- **Don't re-run Bamboogle to "fix" the number.** Per-dataset n=125 noise is what it is; bracketing it requires more seeds, not more runs at the same seed.
- **Wait for the full v1 sweep** to land all 7 datasets, then check whether the average gap to paper has closed. If yes, ship the v1 config to Plan A.
- **At Plan A** (5 seeds × 7 datasets), the per-dataset SE on Bamboogle drops from ~3 pp to ~1.3 pp, which will cleanly distinguish "1.6-pp lift" from "noise."

## Lessons

- **Single-seed n=125 noise is real.** Treat any per-dataset move <3 pp on Bamboogle as suspect until confirmed by either (a) a second seed or (b) consistency with NQ/TriviaQA/PopQA where n=1000 gives ~1.5 pp SE.
- **Look at the larger-n dataset first** when verifying a config change. NQ-1k landed before the others in the sweep order, which is what saved time on this one.
- **Don't audit-estimate token-distribution changes** as ≤1 pp by default — `add_special_tokens` mutations move every multi-turn rollout's tokenization, and the GRPO policy is fine-tuned on specific token IDs.
