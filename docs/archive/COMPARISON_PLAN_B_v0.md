---
title: COMPARISON PLAN B
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Plan B v0 vs. Search-R1 paper — gap analysis

> **Status (2026-04-28): resolved.** This document captures the **v0** Plan B numbers and the gap analysis that led to identifying the load-bearing miss (`apply_chat=False` for base, see [../eval/PAPER_VS_OURS_AUDIT.md D1](../eval/PAPER_VS_OURS_AUDIT.md)). The fix is applied in code; the **v1** sweep is converging (see [../milestone_1/MILESTONE_1.md#status-2026-04-28](../milestone_1/MILESTONE_1.md#status-2026-04-28)). Sections retained below for the historical baseline + probe evidence; the obsolete suspect-ranking and next-steps moved to [COMPARISON_PLAN_B_SUSPECTS.md](COMPARISON_PLAN_B_SUSPECTS.md).

Side-by-side of [RESULTS_PLAN_B_v0.md](RESULTS_PLAN_B_v0.md) (1 seed × 7 datasets × 2 variants, factoid/multihop subsampled to 1 k, Bamboogle/MuSiQue full) against the GRPO numbers from Search-R1 v5 (Appendix F / Table 3). Paper targets and the 10 fixes already applied are tracked in [../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md).

## Per-dataset EM, ours vs paper

| Dataset | Variant | Plan B EM | Paper EM | Δ (pp) |
|---|---|---:|---:|---:|
| NQ              | base     | 0.316 | 0.421 | **−10.5** |
| NQ              | instruct | 0.399 | 0.397 | +0.2 |
| TriviaQA        | base     | 0.421 | 0.583 | **−16.2** |
| TriviaQA        | instruct | 0.539 | 0.565 | −2.6 |
| PopQA           | base     | 0.309 | 0.413 | **−10.4** |
| PopQA           | instruct | 0.412 | 0.391 | +2.1 |
| HotpotQA        | base     | 0.201 | 0.297 | **−9.6** |
| HotpotQA        | instruct | 0.354 | 0.331 | +2.3 |
| 2WikiMultiHopQA | base     | 0.207 | 0.274 | **−6.7** |
| 2WikiMultiHopQA | instruct | 0.353 | 0.310 | +4.3 |
| MuSiQue         | base     | 0.034 | 0.066 | −3.2 |
| MuSiQue         | instruct | 0.149 | 0.124 | +2.5 |
| Bamboogle       | base     | 0.112 | 0.128 | −1.6 |
| Bamboogle       | instruct | 0.360 | 0.232 | **+12.8** |

Averages:

| Variant  | Plan B | Paper | Δ |
|---       |---:    |---:   |---:|
| base     | 0.229  | 0.312 | **−8.3 pp** |
| instruct | 0.367  | 0.336 | +3.1 pp |

## Headline pattern

The two variants behave very differently against the paper:

- **Base** is below paper on **all 7 datasets** (−1.6 to −16.2 pp). The bias is one-sided and systematic — far beyond the ~1.5 pp subsample SE on 1 k-row factoid samples. Something is wrong with the base run, not just noise.
- **Instruct** is within ±5 pp on 6 of 7 datasets and overshoots Bamboogle by +12.8 pp. The +3.1 pp average lift is consistent with the Bamboogle smoke observation in [../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md#smoke-validation): instruct cleanly closes `</answer>`, while base length-truncates 17 % of the time and loses those examples.

So the instruct→paper gap is plausibly noise + a small systematic edge from cleaner stop behaviour. The base→paper gap is not. Plan A will not fix the base variant — it will just give us tighter error bars on a wrong number.

## What is the same as the paper

Confirmed by audit of the codebase against the official `PeterGriffinJin/Search-R1`:

| Knob | Ours | Paper |
|---|---|---|
| Models | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo` | same checkpoints |
| Underlying base | `Qwen/Qwen2.5-3B` | same |
| Retriever encoder | E5-base-v2 | same |
| Corpus | wiki18_100w | same |
| Index | FAISS Flat IP, 768-d | same |
| Top-k | 3 | 3 |
| Max search turns | 4 | 4 |
| Per-step token limit | 500 | 500 |
| Retrieval query max len | 256 | 256 |
| Observation truncation | 500 tokens | 500 tokens |
| Passage format | `Doc i(Title: …) <text>` | byte-identical |
| `<information>` whitespace | `\n\n` outside, none inside | byte-identical |
| Prompt template | `SEARCH_R1_TEMPLATE` (`flashrag/search_r1/templates.py`) | byte-identical to `make_prefix(template_type='base')` |
| Chat template (instruct) | `tokenizer.apply_chat_template()` | same |
| Stop tokens | `</search>`, `</answer>`, `<\|im_end\|>`, `<\|endoftext\|>` | same |
| `<search>` regex | first-match | same |
| Invalid-search corrective text | matches official wording | same |
| EM metric | SQuAD-canonical normalize → exact equality (`flashrag/search_r1/answer_utils.py`) | same |
| F1, ACC | token-level F1, sub-EM | same |

All 10 audit divergences listed in [../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md#divergences-fixed) have been applied.

## Resolution (2026-04-28)

The −8.3 pp base-variant gap was the load-bearing question this analysis tried to answer. The audit ([../eval/PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md)) traced it to three divergences:

- **D1 (HIGH)**: `apply_chat=False` for base in `scripts/run_one.sh:35` — fixed.
- **D-prompt-micro (LOW)**: missing `For example, <answer> Beijing </answer>.` in `templates.py` — fixed.
- **D8 (LOW)**: `add_special_tokens` runtime additions in `active_pipeline.py:37-42` — removed.

NQ-1k v1 result (locked config = all three fixes): EM **0.390**, +7.4 pp from v0's 0.316, leaving ~3.1 pp residual to paper. See [../milestone_1/MILESTONE_1.md](../milestone_1/MILESTONE_1.md) for the converged status.

The earlier suspect ranking and step-by-step next-actions in this document are preserved in [docs/archive/COMPARISON_PLAN_B_SUSPECTS.md](../archive/COMPARISON_PLAN_B_SUSPECTS.md) for the historical record.

## Probe: base + `apply_chat=True` on Bamboogle (2026-04-28)

Hypothesis: the base GRPO checkpoint also benefits from the chat scaffold the instruct variant uses, even though `run_one.sh:35` hard-codes `apply_chat=False` for base. The base model's tokenizer config ([`search_r1_base_model/tokenizer_config.json`](../../evaluation_search_r1/search_r1_base_model/tokenizer_config.json)) ships with a Qwen2.5 chat template (`"You are a helpful assistant."`), so `apply_chat=True` is well-defined for it.

Re-ran Bamboogle (test split, n=125, seed=1) with `--apply_chat True --generator_model search_r1_base_model`, save_note `search_r1_base_applychat_seed1`. Everything else unchanged.

| Run | EM | F1 | ACC | closes `</answer>` | mean turns | paper EM |
|---|---:|---:|---:|---:|---:|---:|
| Plan B base, no chat        | 0.112 | 0.172 | 0.120 | **105/125 (84 %)** | 0.95 | 0.128 |
| **Base + `apply_chat=True`** | **0.128** | **0.207** | **0.136** | **125/125 (100 %)** | 1.01 | 0.128 |
| Plan B instruct (chat)      | 0.360 | 0.451 | 0.376 | 125/125 (100 %)    | 1.94 | 0.232 |

What this shows:
- **Trace hygiene is fully fixed.** 20 of 125 samples in the no-chat base run never produced an `</answer>` close — they length-truncated mid-trace and lost their EM credit. With chat scaffolding the rate is 0/125.
- **EM hits the paper number on the nose** (0.128). The +1.6 pp lift on Bamboogle is small (within noise on n=125), but it is in the right direction and the mechanism (truncation rate) clearly improved.
- **Mean turns barely moved** (0.95 → 1.01), well below instruct's 1.94. Chat scaffolding fixes *how the trace ends*, not how aggressively the base model uses search. So the lift on multi-hop datasets (HotpotQA, 2Wiki) — which need >1 turn — may be smaller than on factoid; conversely, factoid datasets that need ~1 turn are exactly where the truncation losses dominated and where the lift should be largest.

Caveat: 16 % truncation rate is dataset-dependent. NQ/TriviaQA prompts are shorter than Bamboogle's reasoning-heavy ones, so the without-chat base could either be cleaner there (smaller lift) or just as broken (similar lift). Need to measure.

For the current open work (format-validity tabulation, full-data scale check, Plan A, write-up), see [../milestone_1/MILESTONE_1.md#whats-left](../milestone_1/MILESTONE_1.md#whats-left).
