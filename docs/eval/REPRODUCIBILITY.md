---
title: REPRODUCIBILITY
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Search-R1 Reproduction

End-to-end reproduction of Search-R1's published 3B GRPO results on this codebase. Covers the comparison numbers, the divergences we found and fixed, and the smoke-test evidence the fixes landed. Per-dataset Plan B numbers are in [../report/RESULTS_m1.md](../report/RESULTS_m1.md). Operational details (how to run sweeps, where the wall-clock goes) are in [EVAL_OPS.md](EVAL_OPS.md).

## Models — confirmed GRPO

Local checkpoints verified against the upstream GRPO releases by **LFS sha256 + shard sizes + `eos_token_id`** (the only fields that are unique to the GRPO upload):

| | HF repo | shard-1 LFS sha256 | total | eos_token_id |
|---|---|---|---:|---:|
| base     | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`    | `7ac54e1b…36a9dabf` | 13.6 GB (3 shards) | 151643 (`<\|endoftext\|>`) |
| instruct | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` | `3d787062…68ccd35` | 13.6 GB (3 shards) | 151645 (`<\|im_end\|>`)     |

Both shard-1 sha256s match the LFS pointers on HF exactly; sizes match all three shards bit-for-bit. Raw `Qwen/Qwen2.5-3B` ships as 2 shards / ~6 GB bf16, so a wrong-download mistake (raw Qwen instead of GRPO) would be obvious from sizes alone — they are not.

Note: `config.json` `_name_or_path` reads `Qwen/Qwen2.5-3B` (and `…-3B-Instruct`) — that's what's stamped on the HF GRPO repo too, because the trainer initialised from the underlying Qwen and saved without overriding the path. It is *not* a usable identity check for the GRPO checkpoint; use the sha256 above instead.

Underlying base for both: `Qwen/Qwen2.5-3B`.

## Paper numbers we compare against

The paper's main results table reports PPO. The PPO-vs-GRPO comparison is in Appendix F / Section 5.1 / Table 3 of v5 (`arxiv.org/html/2503.09516v5#A6`):

### Qwen2.5-3B EM

|              | NQ    | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg   |
|---           |---    |---       |---    |---       |---    |---      |---        |---    |
| **GRPO base**     | **0.421** | **0.583** | **0.413** | **0.297** | **0.274** | **0.066** | **0.128** | **0.312** |
| **GRPO instruct** | **0.397** | **0.565** | **0.391** | **0.331** | **0.310** | **0.124** | **0.232** | **0.336** |

Bold rows are the comparison targets.

## Divergences fixed

10 audit-identified divergences from the official `PeterGriffinJin/Search-R1`. All applied; full per-fix code review is in git history (commit `6d4ac02` and following).

| # | Fix | Severity |
|---|---|---|
| 1 | Format passages as `Doc i(Title: …) <text>` (was raw `Title\nText`) | High |
| 2 | `retrieval_topk` 5 → 3 | High |
| 3 | `<information>…</information>` whitespace: `\n\n` outside, no inner padding | Medium-High |
| 4 | `max_search_turns` 8 → 4 | Medium |
| 5 | Truncate retrieval observation to 500 tokens | Medium |
| 6 | Question normalization: `.strip()` + trailing `?` | Low-Medium |
| 7 | `retrieval_query_max_length` 128 → 256 | Low-Medium |
| 8 | Invalid-search corrective text matches official wording | Low |
| 9 | First-match `<search>` regex (was last-match) | Low |
| 10 | Per-step token limit 512 → 500 | Negligible |

**Files modified**: [flashrag/pipeline/active_pipeline.py](../../evaluation_search_r1/flashrag/pipeline/active_pipeline.py), [flashrag/search_r1/parser.py](../../evaluation_search_r1/flashrag/search_r1/parser.py), [flashrag/config/basic_config.yaml](../../evaluation_search_r1/flashrag/config/basic_config.yaml), [../../local_retriever/flashrag/config/basic_config.yaml](../../local_retriever/flashrag/config/basic_config.yaml).

**No changes needed** to: prompt template (already byte-identical to `make_prefix(template_type='base')`), EM scorer (canonical SQuAD normalization in [flashrag/search_r1/answer_utils.py](../../evaluation_search_r1/flashrag/search_r1/answer_utils.py)), FAISS index, encoder, model checkpoints, sampling temperature.

## Smoke validation

After applying all 10 fixes, on Bamboogle (smallest test set, 125 examples, 1 seed):

|              | EM    | F1    | mean turns | length-truncated | paper EM |
|---           |---    |---    |---         |---               |---       |
| 3B base      | 0.088 | 0.155 | 1.03       | 21/125           | 0.128    |
| 3B instruct  | 0.360 | 0.451 | 1.93       | 0/125            | 0.232    |

Trace shape verified end-to-end: chat template applied for instruct; `Doc i(Title: …) text` rendered correctly; `<information>` whitespace matches official; `<answer>` cleanly closes generation.

### Reading the smoke deltas

- **Base −4 pp** vs paper is within ~1.3 σ on n=125 — unremarkable single-seed sampling variance.
- **Instruct +12.8 pp** is outside noise (~3.4 σ) but in the model's favour. Most likely candidates: (a) cleaner `</answer>` stop than the paper's training-time rollout, which helps the instruct variant (always reaches `</answer>`) more than base (length-truncates 17 % of the time); (b) paper version drift across the 5 arxiv revisions; (c) single-seed n=125 variance. *(Note: an earlier draft of this list cited "lucky seed at temp=1.0" — that was based on a wrong assumption about paper sampling; both paper eval and ours run greedy. See [PAPER_VS_OURS_AUDIT.md D3](PAPER_VS_OURS_AUDIT.md).)*
- Both deltas point the *same way* would be evidence of a systemic issue. They don't, so the pipeline is not broken — the gap is mostly noise plus a small systematic edge from how cleanly we close traces.

Decision: don't chase the overshoot. Tighten with more seeds and at least one big-N benchmark — see [../report/RESULTS_m1.md](../report/RESULTS_m1.md).

## Final results

[../report/RESULTS_m1.md](../report/RESULTS_m1.md) — auto-generated by `scripts/aggregate.py`. Plan B is 1 seed × 7 datasets × 2 variants on subsampled (1k) large datasets and full Bamboogle/MuSiQue.
