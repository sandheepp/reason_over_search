---
title: README
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Archive — discarded experiments and historical snapshots

Index of everything we tried that did **not** make it into the locked Plan B v1 config (see [../report/CODE_SETUP_m1.md](../report/CODE_SETUP_m1.md)) or that is preserved for the historical record.

Treat anything in this directory as **not load-bearing** for current work. None of these settings, hypotheses, or numbers should be used as a reference for new experiments — go to the parent docs/ for the live truth.

## Pre-v1 historical snapshots

- [RESULTS_PLAN_B_v0.md](RESULTS_PLAN_B_v0.md) — frozen aggregate of the **pre-fix** Plan B sweep (single seed × 7 datasets × 2 variants, before the apply_chat / prompt-sentence / special-tokens fixes). Base avg EM 0.229 (paper 0.312 → −8.3 pp gap), instruct avg 0.367 (paper 0.336 → +3.1 pp). v0 result directories are at `evaluation_search_r1/results/_archive_v0/` (committed in `cf9e2fb`). 13 runs are archived; the bamboogle/instruct row in `RESULTS_PLAN_B_v0.md` (EM 0.360) is the smoke-test number documented in [../eval/REPRODUCIBILITY.md#smoke-validation](../eval/REPRODUCIBILITY.md#smoke-validation), recorded before the formal v0 sweep started — its run dir was not preserved.
- [COMPARISON_PLAN_B_SUSPECTS.md](COMPARISON_PLAN_B_SUSPECTS.md) — suspect ranking + recommended-next-steps section as it stood **before** the v1 sweep landed. Useful as a record of how we reasoned to the three audit fixes; superseded by [../report/RESULTS_m1.md](../report/RESULTS_m1.md) and [../eval/PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md).

## Archived setup docs

- [VAST_INSTANCE_SETUP_v0.md](VAST_INSTANCE_SETUP_v0.md) — original M1-era manual-download Vast.ai walkthrough (corpus + flat IP index + GRPO checkpoints + eval datasets). **Archived 2026-05-09**. Successor: [`docs/vast/SETUP_VAST.md`](../vast/SETUP_VAST.md), which uses [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) for one-shot setup, defaults to IVF-SQ8 (flat IP retired), and covers M2 training + M4 Qwen3.5-0.8B eval rather than the M1 GRPO checkpoint reproduction path.
- [BOOTSTRAP_NEW_INSTANCE_v0.md](BOOTSTRAP_NEW_INSTANCE_v0.md) — original M1-era host-agnostic manual-download walkthrough (parallel content to `VAST_INSTANCE_SETUP_v0.md`, framed for in-house GPU + cloud VM hosts beyond Vast). **Archived 2026-05-09**. Successors: [`docs/vast/SETUP_VAST.md`](../vast/SETUP_VAST.md) for Vast; [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../setup/BOOTSTRAP_NEW_INSTANCE.md) (rewritten thin) for non-Vast hosts — `docker pull`/`run` recipe + delegate to SETUP_VAST.md.

## Hypotheses that turned out wrong

- [TEMPERATURE_HYPOTHESIS_WRONG.md](TEMPERATURE_HYPOTHESIS_WRONG.md) — I (Claude) hypothesized paper eval used `temperature=1.0`, citing paper Appendix B.2 + verl rollout YAML. Wrong: verl `_validate()` hard-codes `do_sample=False` and `vllm_rollout.py:162-171` overrides to `temperature=0, top_p=1.0` whenever that flag is set. Paper eval is greedy; our `temperature: 0.0` is correct. **Do not change `temperature` or `top_p`.**
- [BAMBOOGLE_REGRESSION_INVESTIGATION.md](BAMBOOGLE_REGRESSION_INVESTIGATION.md) — Bamboogle base v1 reported EM 0.088 vs the earlier probe's 0.128. We briefly worried the minor fixes had regressed the model. NQ-1k landed +4.4 pp from the same fixes the next morning, confirming the Bamboogle drop was n=125 single-seed variance. **Lesson: treat any per-dataset move <3 pp on n=125 Bamboogle as suspect until confirmed by a larger-n dataset or a second seed.**

## Discarded ablations from the autoresearch loop

- [DISCARDED_ABLATIONS.md](DISCARDED_ABLATIONS.md) — 11 ablations tried on the `experiment_ros/apr27` branch in late April. Only **`temperature=0`** (commit `ab0220f`) was kept. Discarded list includes:
  - `retrieval_topk` 3 → 5 (EM flat, F1 +)
  - `max_obs_length` 500 → 750 / 833 (EM dropped)
  - `max_search_turns` 4 → 5 (EM dropped)
  - Multi-query retrieval (model query + question, dedup) (EM dropped)
  - Stop-token corrections for partial `</search>` / `</answer>` matches (real bug, but greedy noise net-negative on n=125)
  - Query expansion (append question to retrieval query) (EM dropped sharply)
  - Drop default chat-template system message (neutral)
  - Serialize inference (`max_workers=1`) (slightly worse than batched)
  - `repetition_penalty=1.05` (neutral)

  Pattern: the released GRPO checkpoints are tightly tuned to the original retrieval/observation budgets (top-k=3, max_search_turns=4, max_obs_length=500). Expanding any of them hurts. None of these should be revisited without a stronger prior than "let's see if it helps."

- The autoresearch loop's working files (`program.md`, `results.tsv`) are no longer tracked in git; they were the loop's runtime state and are not needed for reproducibility.

## What is *not* archived (and why)

- Three audit fixes that **were** kept and made it into v1 — see [../eval/PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md) (D1, D-prompt-micro, D8) and [../report/CODE_SETUP_m1.md](../report/CODE_SETUP_m1.md). These are load-bearing.
- The `apply_chat=True` Bamboogle/NQ probes that led to v1 — captured inline in [../milestone_1/../archive/COMPARISON_PLAN_B_v0.md](../milestone_1/../archive/COMPARISON_PLAN_B_v0.md) ("Probe" section) and [../report/RESULTS_m1.md](../report/RESULTS_m1.md). Probes that pointed to the right answer aren't "discarded" — they're part of the v1 derivation.
