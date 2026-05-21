---
title: DISCARDED ABLATIONS
tags: []
source: internal
created: 2026-04-28
updated: 2026-05-01
---

# Discarded ablations — autoresearch loop

History of pipeline ablations tried on `experiment_ros/<tag>` branches that did **not** survive the keep/discard threshold (improvement < ~2 pp EM and within single-seed noise on n=125 Bamboogle/instruct).

Source of truth at the time the loop ran: `results.tsv` and `program.md` at the repo root. Both files were untracked in commit `3ed0b11` once the loop wound down — the table below is the surviving record.

Baseline used as reference: **instruct / Bamboogle / seed 1 / temp=0**. EM 0.336 (commit `ab0220f`, `temperature=0` was the only change kept on top of `09822ca`).

## Ablation log (oldest → newest)

| Commit  | EM    | F1    | Status   | Change |
|---      |---:   |---:   |---       |---|
| `09822ca` | 0.320 | 0.393 | baseline | instruct/Bamboogle/seed 1, on `index_modifications` |
| `ab0220f` | 0.336 | 0.406 | **kept** | `temperature=0` (greedy) for deterministic eval |
| `caf2a9e` | 0.320 | 0.433 | discard  | retrieval_topk 3 → 5 (F1 ↑, EM flat — within noise) |
| `98ca7b0` | 0.296 | 0.379 | discard  | max_obs_length 500 → 750 |
| `91ac011` | 0.328 | 0.411 | discard  | max_search_turns 4 → 5 |
| `060e9e2` | 0.304 | 0.402 | discard  | topk 3→5 **and** max_obs_length 500→833 |
| `dfed93f` | 0.272 | 0.373 | discard  | multi-query retrieval (model query + question, interleave-dedup) |
| `f44f417` | 0.328 | 0.398 | discard  | fix partial `</search>` / `</answer>` stop tokens (real bug, but greedy noise net −1 pp) |
| `d658376` | 0.216 | 0.318 | discard  | query expansion (append question to retrieval query) |
| `ea8dff6` | 0.328 | 0.416 | discard  | drop default system message in chat template |
| `46d80c2` | 0.320 | 0.399 | discard  | fix partial `</answer>` stop only (run noise overwhelmed +1–3 pp gain) |
| `f2c3f43` | 0.312 | 0.401 | discard  | serialize inference (max_workers=1) — "true" greedy actually lower than batched |
| `62b0ec0` | 0.312 | 0.391 | discard  | repetition_penalty 1.05 |

## Patterns

- **`topk 3→5`, `max_obs_length` ↑, `max_search_turns` ↑** — all expanded the retrieval/observation surface. None lifted EM. Hypothesis: the GRPO policy was trained on the original budgets and learned to rely on them.
- **Multi-query / query expansion** — both *hurt*, suggesting the policy's emitted query is what it expects to see; adding noise to the retrieval input degrades.
- **Stop-token corrections** (`f44f417`, `46d80c2`) — fixed real parsing bugs but the EM signal at n=125 was within noise. Worth revisiting at larger n.
- **Drop default system message** — neutral; the chat template's "You are a helpful assistant." default doesn't shift the GRPO policy meaningfully.
- **Serialize inference** — confirms the parallel/batched path is **not** distorting EM at temp=0; nondeterminism is small.
- **`repetition_penalty 1.05`** — base Qwen2.5 doesn't need it at greedy.

## Caveats

- Single seed × n=125 Bamboogle has ~3 pp standard error. Anything below that band is noise. Some "discards" might survive with 3 seeds; not worth re-running unless we have a stronger prior.
- All of the above tested on **instruct/Bamboogle**. The base-variant gap (the bigger problem per [../archive/COMPARISON_PLAN_B_v0.md](../milestone_1/../archive/COMPARISON_PLAN_B_v0.md)) was not exercised by these ablations.
- The audit in [PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md) identified the load-bearing miss (base `apply_chat=False`) **after** the ablation loop above; that fix lives outside this discarded-list.
