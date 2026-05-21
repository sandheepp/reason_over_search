---
title: Conversation Context (thesis-writing snapshot)
tags: [report, snapshot]
source: internal
created: 2026-05-04
updated: 2026-05-08
---

# Conversation Context

> Living snapshot of project status, decisions, and pointers to the canonical docs. Update this when something material changes; keep historical sections frozen below.

**Last updated**: 2026-05-08 (M3 + M3.1 closed; chapter wrap-up)

> **Doc reorganisation 2026-05-06**: The literature-review strand (`LITERATURE_REVIEW.md`, `SURVEY.md`, `SURVEY_FOCUSED.md`, `SURVEY_OVERFLOW.md`) moved to [`docs/research/`](../research/) so this folder is purely thesis-writing. See [`docs/research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) for the research-strand snapshot. The supervisor-meeting brief is now date-suffixed: `SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`.

---

## 1. Status (one paragraph)

Phase-1 of the thesis (Qwen3-0.6B prompt + base-model ablations on 1× A100-40GB, ALICE; 29 runs across v0 + v1 blocks) is closed and documented in `RESULTS_m0_a.md` and `RESULTS_m0_b.md`. The Search-R1 evaluation baseline (M1, Qwen2.5-3B) is reproduced (`report/RESULTS_m1.md`, ±2.5 pp of paper). The Phase-2 (M2) NeMo-RL training pipeline was built around Qwen3.5-2B and is launch-ready (15 parity tests pass; smoke-tested on Vast.ai 1× A100-80GB at ~57 s/step); the Qwen3.5 small-model family (released 2026-03-02) is **0.8B / 2B / 4B / 9B**, and **Phase-2 will start with Qwen3.5-0.8B** for cheap iteration before extending to 2B if the recipe holds. **M3 (first eval of the v0 GRPO checkpoint, closed 2026-05-07)**: 1046-step `p1_basic_w_ex_z7kcxfof` (the only Phase-1 run that converged on heavy-tool 2-call/4-turn behaviour; 23h 47m 30s wall on A100-40GB) lifted average EM 0.102 → 0.155 (+52 % relative, +0.053 absolute) over the untrained Qwen3-0.6B hybrid across all 7 paper benchmarks at full Plan A (51,713 items / variant on ALICE A100-80GB); 6 / 7 datasets improved; held-out generalisation rules out memorisation. **M3.1 (second eval, closed 2026-05-08)** evaluated the **highest-reward Phase-1 run** `p3_decide_no_ex_el6s2d2h` (no-example + decision-rules prompt; end reward 0.215, 2280 steps) on the same 7 benchmarks at full Plan A; simple-mean EM lifted further to **0.169** (+9 % rel over M3, +66 % over pre-GRPO); ACC/F1 widen the M3.1-vs-M3 gap to +12 %/+14 %, indicating the no-example variant produces higher-quality answers. **Conclusion**: the Phase-1 structural finding that decision-rule scaffolding can substitute for the few-shot example survives held-out evaluation — the no-example variant is a *pareto improvement* (lower compute + higher quality) and recipe transfer to Qwen3.5 is lower-risk. HF checkpoints: [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0) (M3) and [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (M3.1). The eval pipeline is now pinned and reusable for Phase-2 NeMo-RL evaluation. Phase-2 wall-clock at our affordable 0.6-epoch budget (1005 steps × 102 prompts/step ≈ 0.604 epochs of the 169,615-row train corpus) is **11–17 days per run on 1× A100-80GB**; matching the paper's 3-epoch schedule at our batch shape would be ~5× → ~55–85 d / run, which kills both the original reward-function-ablation plan and any paper-faithful sweep. The thesis question is being reframed from "extending RLVR via tool-use" to "what is the optimised training recipe for a small LM under a single-A100 budget"; proposed recipe stacks E2H curriculum + S-GRPO + MC-GRPO on a Search-R1 GRPO baseline with a JustRL plain-GRPO control alongside. Net captured in `SUPERVISOR_MEETING_2026-05-07_m0_to_3.md` (chapter-closing summary); detailed M3 numerical record in `RESULTS_m3.md`.

## 2. Hard timeline

| Date | Milestone |
|---|---|
| 2026-05-07 | Supervisor meeting (`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md` is the brief) |
| 2026-06-10 | Experimentation must be finished |
| 2026-06-15 | Thesis submission |
| ~2026-07-15 | Defense |

## 3. Compute and constraints (confirmed)

- **GPU**: 1× A100-80GB (rented; ALICE retired going forward).
- **Budget**: ~\$1000 USD for training.
- **Implication**: 2 to 3 full Qwen3.5 small-model GRPO runs feasible **at the affordable 0.6-epoch budget** (11 to 17 d / run on 1× A100-80GB; 5 to 8.5 d on 1× H100). Paper-equivalent 3-epoch training would be ~55–85 d / run on 1× A100, infeasible. Phase-2 starts with Qwen3.5-0.8B (cheaper) before extending to 2B. No reward-ablation sweep affordable in either regime.

## 4. Decisions captured

| Decision | When | Rationale |
|---|---|---|
| Use hybrid Qwen3 (not base) | After v0 | Base could not produce structured tool_call format from cold start; 5/5 v1 base attempts had 0 tool calls |
| Switch to `<tool_call>` JSON tags | v1 | In-distribution for Qwen3 chat template; same reward at equal step count as paper `<search>` tags |
| Drop reward-function ablation | After Phase-2 wall-clock | 11 to 17 d / run on 1× A100; multiple paired runs not affordable |
| Pivot from verl to NeMo-RL | Phase-2 setup | verl does not support Qwen3.5 |
| FAISS Flat IP → IVF-SQ8 for training | Phase-2 smoke | Flat IP times out under rollout HTTP load |
| Reframe thesis question | 2026-05-04 | Original RQs (reward modeling, meta-reasoning, curriculum) require a sweep that is not affordable; reframed to "is it feasible + what is the optimised recipe" |
| Proposed training recipe | 2026-05-04 | E2H curriculum + S-GRPO + MC-GRPO on Search-R1 GRPO baseline; all drop-in, all from [`docs/research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md) |

## 5. Working memory

Things that are easy to lose track of:

- **Two repos**: `reason_over_search` (thesis) and `research` (training infrastructure, Qwen3-0.6B port). Both reference each other.
- **Two W&B projects**: `research` (v0, 14 focus runs / 26 in W&B project; 12 noise/aborted excluded — see `RESULTS_m0_a.md` §1) and `research_revamp` (v1, 15 runs). All Qwen3-0.6B; analysis frozen.
- **CSVs of the 29 focus runs (14 v0 + 15 v1)** are saved at `archive/m0_a/csv/` and `archive/m0_b/csv/` so the analysis can be regenerated without W&B access.
- **Reward function gotcha**: Search-R1's GitHub ships `qa_em.py` (paper-faithful EM-only) and `qa_em_format.py` (shaped 6-tier with non-zero defaults). Earlier project docs conflated them; caught in `training/SMOKE_RESULTS_2026-05-06.md` (renamed from _V4). Phase-2 NeMo-RL port uses EM-only.
- **`base_breakthrough` (v1, b8vv0qe2)** showing reward 0.7 is a reward-function-code change artifact, not learning. Configs are identical to `base_state_machine_a` which scored 0.0. Treat as instrumented, not earned.

## 6. Doc inventory

See `README.md` in this directory for the per-file index. The supervisor-facing summary is `SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`. Literature/survey strand lives in [`docs/research/`](../research/).

---

## Historical (frozen)

### Original vision (Plan A and Plan B, Feb 2026)

Three threads in `ORIGINAL_PLAN_A.md`: reward model design, meta-reasoning tags, curriculum + transfer learning. Core question: "Does domain adaptation mean learning a way to *interact with* the domain (e.g. tool use) rather than internalising the domain?"

Plan B (`ORIGINAL_PLAN_B.md`) RQs:
- RQ1: Current approaches for domain expansion?
- RQ2: How to model reward functions?
- RQ3: Can meta-reasoning improve training?
- RQ4: Is curriculum-based training feasible?

Original timeline: Feb (lit) → Mar (pipeline) → Apr (experiments) → May (analysis) → Jun (writing). Slipped: pipeline + experiments compressed into Apr to early May. The Phase-2 compute reality (Section 3 above) forced the reframing in Section 4.

### The pivot to ReSearch + Search-R1

Picked as the entry papers because they are the cleanest concrete instance of "tool-use as the surrogate objective" in the lit review. ReSearch flagged as "EXACTLY WHAT I WANTED TO DO" in [`docs/research/LITERATURE_REVIEW.md`](../research/LITERATURE_REVIEW.md). The "non-verifiable domains" framing collapsed into "exact-match QA over Wikipedia" (verifiable via tool); the reframing in Section 4 above resolves this loose end by changing the question from "extend RLVR" to "find the optimised recipe".
