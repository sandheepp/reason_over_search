---
title: README
tags: [report, index]
source: internal
created: 2026-05-04
updated: 2026-05-08
---

# `docs/report/`

Thesis-facing record. Files at the top of this list are the canonical entry points; everything below is supporting detail.

## Start here

| File | Purpose |
|---|---|
| [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](./SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) | Canonical chapter-closing summary for the 2026-05-07 meeting (heavy rebuild 2026-05-07; § 4 refactored to parallel M3 + M3.1 reporting on 2026-05-08). § 0 TL;DR · § 1 Phase-1 (29 ALICE runs) · § 2 M1 Search-R1 baseline · § 3 M2 NeMo-RL pivot · **§ 4 M3 + M3.1 parallel evals** (closed 2026-05-07 / 2026-05-08: simple-mean EM 0.102 → 0.155 (M3) → 0.169 (M3.1)) · § 5 compute reality + reframed RQ · § 6 next-steps recipe (E2H + S-GRPO + MC-GRPO + JustRL control). PDF alongside. |
| [`RESULTS_m3.md`](./RESULTS_m3.md) | M3 + M3.1 numerical record: per-dataset EM/ACC/F1 for pre-GRPO + post-GRPO (z7kcxfof, M3) + post-GRPO (el6s2d2h, M3.1) on full Plan A (51,713 items / variant); training-side context; wall-clock; reproduction. § 1–13 are M3; § 14 is M3.1 (parallel headline + per-dataset + training-curve panel). HF checkpoints: [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0) (M3) and [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (M3.1). |
| [`PROGRESS_REPORT_01_m0_to_3.md`](./PROGRESS_REPORT_01_m0_to_3.md) | TL;DR + concise progress report covering 2026-04-23 to 2026-05-08 (window updated post-M3.1). |
| [`CONVERSATION_CONTEXT.md`](./CONVERSATION_CONTEXT.md) | Living snapshot of project status, decisions, compute, and a doc inventory. Update when something material changes. |

## Phase-1 results (Qwen3-0.6B, ALICE, frozen)

| File | What |
|---|---|
| [`RESULTS_m0_a.md`](./RESULTS_m0_a.md) | First ablation block (W&B project `research`). 14 runs: prompt sensitivity on hybrid model + base-model attempts. 17 PNGs embedded; full prompt strings recovered from W&B notes. CSVs at `archive/m0_a/csv/`. |
| [`RESULTS_m0_b.md`](./RESULTS_m0_b.md) | Second ablation block (W&B project `research_revamp`). 15 runs: new in-distribution `<tool_call>` prompt format, fresh base-model attempts, reward-function probing. 21 PNGs embedded; CSVs at `archive/m0_b/csv/`. |

Storyline plot used in the supervisor doc lives at `supervisor_assets/storyline_v0_v1.png`.

## Setup and codebase

| File | What |
|---|---|
| [`CODE_SETUP_m0.md`](./CODE_SETUP_m0.md) | What changed in the ReSearch port vs. paper code: Qwen2.5-7B → Qwen3-0.6B, verl pin → verl_latest, faiss-gpu → faiss-cpu, 4-GPU → 1-GPU profiles, prompt and reward rewrites. Hyperparameter matrices side-by-side with paper. |
| [`CODE_SETUP_m2.md`](./CODE_SETUP_m2.md) | What changed from v0 to the M2 NeMo-RL pipeline: Qwen3.5-2B, NeMo-RL replaces verl, EM-only reward, two prompt arms, 6 smoke-run bugs fixed, Qwen3.5 architecture constraints (sequence packing off, enable_thinking), IVF-SQ8 retriever default, bootstrap.sh env setup. Quick-summary format. |
| [`CODE_SETUP_m3.md`](./CODE_SETUP_m3.md) | M3 + M3.1 eval pipeline: 14 alignment fixes between clone-and-run and the first clean comparison; per-mode dispatch (`prompt_mode=qwen3*` vs `search_r1`); ReSearch comparison; reproduction. § 13.5 documents the additive M3.1 deltas (new prompt template, family-prefix matching, verl-FSDP → HF checkpoint conversion, `/health` wait-window bumps). |

## Original plan (frozen, historical)

| File | What |
|---|---|
| [`ORIGINAL_PLAN_A.md`](./ORIGINAL_PLAN_A.md) | Abstract / introduction draft (Feb 2026): three research threads. |
| [`ORIGINAL_PLAN_B.md`](./ORIGINAL_PLAN_B.md) | Problem statement: extend RLVR to non-verifiable domains via tool-use. RQ1 to RQ4. |

The reframing of the research questions is in `SUPERVISOR_MEETING_2026-05-07_m0_to_3.md` § 5 ("Compute reality and the reframed research question").

## Literature (moved)

The literature-review and survey strand (`LITERATURE_REVIEW.md`, `SURVEY.md`, `SURVEY_FOCUSED.md`, `SURVEY_OVERFLOW.md`) lives in [`docs/research/`](../research/) alongside the algorithm + systems research. See [`docs/research/README.md`](../research/README.md) and [`docs/research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md).

## Assets

- `archive/m0_a/`: 17 PNGs (3 combined + 14 per-run) and 14 CSVs.
- `archive/m0_b/`: 21 PNGs (6 combined + 15 per-run) and 15 CSVs.
- `supervisor_assets/`: 1 storyline PNG.

All 29 W&B run histories are saved as CSV under `archive/m0_a/csv/` and `archive/m0_b/csv/` so analyses are reproducible without W&B access.
