---
title: CONVERSATION CONTEXT
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-07
---

# Conversation Context: Research Strand

> Living snapshot of the literature + algorithm + systems strand that feeds the M2 ablation plan. Update this when a load-bearing decision changes; freeze older state below the line.
>
> Sibling snapshots: [`../report/CONVERSATION_CONTEXT.md`](../report/CONVERSATION_CONTEXT.md) (thesis writing + Phase-1 results), [`../training/CONVERSATION_CONTEXT.md`](../training/CONVERSATION_CONTEXT.md) (Phase-2 training pipeline), [`../internship/CONVERSATION_CONTEXT.md`](../internship/CONVERSATION_CONTEXT.md) (Alstom internship — gitignored, lives on the user's own machines per `claude/CLAUDE.md`).

**Last updated**: 2026-05-07 (M3 closed; Phase-2 NeMo-RL is the next chapter)

---

## 1. Status (one paragraph)

The research strand has two arms that are now both ready to drive Phase-2 ablations. The **literature arm** (96-card `SURVEY.md`, project-focused `SURVEY_FOCUSED.md`, completeness `SURVEY_OVERFLOW.md`, personal `LITERATURE_REVIEW.md`) is frozen as of 2026-05-04 and surfaced the candidate recipe (E2H curriculum + S-GRPO + MC-GRPO + JustRL control) used in [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 6. The **algorithm + systems arm** (`INTEGRATION_GUIDE.md`, `PARADIGM_REVIEW.md` v1→v2→v3, `RUNTIME_EFFICIENCY.md`) is also frozen at v3 (2026-05-03). The big open tension is "tricks may hurt" (JustRL): the recipe must be validated with a JustRL plain-GRPO control or it cannot be claimed to be the optimised recipe. Wall-clock budget allows 2 to 3 full Qwen3.5-2B GRPO runs on 1× A100, so the C / C-minimal pair is the affordable comparison; sweeping individual tricks is not. **M3 (first eval of the v0 GRPO checkpoint) closed 2026-05-07 with EM 0.102 → 0.155 (+52 % rel) across 51,713 items / variant; the eval pipeline is now pinned and reusable for Phase-2 NeMo-RL evaluation.** See [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 4 and [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md).

## 2. Active question

The supervisor-facing reframe of the thesis (see [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 5):

> *Is it feasible to post-train a small LM (Qwen3.5-2B) to Search-R1-level results under realistic resource constraints (1× A100-80GB, ~\$1000), and what is the optimised training recipe?*

Candidate answer (from this folder): a stack of E2H curriculum + S-GRPO + MC-GRPO on a Search-R1 GRPO baseline, with a **JustRL plain-GRPO control** alongside.

## 3. Decisions captured (research-strand)

| Decision | When | Rationale | Source |
|---|---|---|---|
| Drop reward-function ablation; pivot to recipe ablation | 2026-05-04 | Phase-2 wall-clock (11 to 17 d / run on 1× A100) makes paired reward sweeps unaffordable | [`SURVEY_FOCUSED.md`](SURVEY_FOCUSED.md) §9 |
| Recipe = E2H + S-GRPO + MC-GRPO with JustRL control | 2026-05-04 | All three are drop-in additions on a Search-R1 GRPO baseline; JustRL is the required negative control per "tricks may hurt" | [`SURVEY_FOCUSED.md`](SURVEY_FOCUSED.md) §§2.3, 2.5, 5.1; [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md) §17 |
| β anneal 1e-3 → 0 + LLDS regulariser (not pure β=0) | 2026-05-03 | β=0 has LLD Death Spiral on Search-R1 + Qwen2.5-3B; insurance is cheap | [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md) §13 (v2 update) |
| Use Dr. GRPO baseline; swap to λ-GRPO if length collapses | 2026-05-03 | v1 default; if heterogeneous behaviour appears, monitoring + swap | [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md) §§5, 14 |
| Systems-only smoke run before any algorithmic ablation | 2026-05-03 | Halve per-step wall-clock first (R1 + R2 + R4 + R5 + O1); ~57 s/step → target ≤30 s/step | [`RUNTIME_EFFICIENCY.md`](RUNTIME_EFFICIENCY.md); [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) |
| Skip LoRA at 2B on multi-turn | 2026-05-03 | Plasticity-vs-Rigidity shows r<256 fails on reasoning at ≤1.5B; LoRA is a capacity trap here | [`RUNTIME_EFFICIENCY.md`](RUNTIME_EFFICIENCY.md) (LoRA note) |

## 4. Compute and budget (links into thesis snapshot)

- **GPU**: 1× A100-80GB rented on Vast.ai. ALICE retired going forward.
- **Budget**: ~\$1000 USD total training.
- **Per-run wall-clock**: 11 to 17 days on 1× A100 (5 to 8.5 d on 1× H100, 6.5 to 9.5 d on 2× A100). With systems wins R1+R2+R4+R5+O1: 6 to 12 days on 1× A100.
- **Implication**: 2 to 3 full Qwen3.5-2B GRPO runs feasible. The C / C-minimal pair is the affordable comparison; per-trick ablation is not.

Detailed numbers: [`RUNTIME_EFFICIENCY.md`](RUNTIME_EFFICIENCY.md) §1, [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md).

## 5. Key tensions and unknowns

The five live questions from `README.md` § "Key tensions & unknowns", here distilled:

1. **Tricks help or hurt?** v1 stacked, v2 stacked more, v3 (JustRL) showed adding tricks can degrade. Resolution: run C and C-minimal; if C beats C-minimal by ≥2 EM, the stack is justified.
2. **β = 0 or small β?** Pure β=0 collapses on Search-R1 + Qwen2.5-3B (LLD Death Spiral). Belt-and-suspenders: β anneal 1e-3 → 0 + LLDS regulariser.
3. **Dr. GRPO or adaptive?** Default Dr. GRPO; swap to λ-GRPO + monitoring if length collapse shows up.
4. **Curriculum or just train hard?** Curriculum on retrieval difficulty is cheap and well-motivated for multi-hop QA; include it.
5. **How many tricks is too many?** Open. Run C and C-minimal; let the data answer.

## 6. What feeds where

```
literature (SURVEY_FOCUSED.md, LITERATURE_REVIEW.md)
        │
        │ surfaces candidate techniques + JustRL counter-evidence
        ▼
algorithm + systems (INTEGRATION_GUIDE → PARADIGM_REVIEW + RUNTIME_EFFICIENCY)
        │
        │ specifies the C / C-minimal stacks + measured baseline
        ▼
training (training/CONVERSATION_CONTEXT.md, configs/grpo_qwen3.5_2b_1xa100.yaml)
        │
        │ runs the ablation, reports per-step + EM
        ▼
report (report/CONVERSATION_CONTEXT.md, SUPERVISOR_MEETING_2026-05-07_m0_to_3.md)
        │
        │ packages the answer for the supervisor
        ▼
thesis writeup (Jun 2026)
```

The active ablation list (priority order, ≤10 h per run target) lives in [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) and the project [`CLAUDE.md`](../../claude/CLAUDE.md) "Active ablation plan".

## 7. Doc inventory

See [`README.md`](README.md) in this directory for the full per-file index. Quick map:

| Arm | Files |
|---|---|
| Literature | `LITERATURE_REVIEW.md`, `SURVEY.md`, `SURVEY_FOCUSED.md`, `SURVEY_OVERFLOW.md` |
| Algorithm + systems | `INTEGRATION_GUIDE.md` (entry point), `PARADIGM_REVIEW.md`, `RUNTIME_EFFICIENCY.md` |
| Open questions | [`QUESTIONS.md`](QUESTIONS.md) (running register; "register this as a question" entries land here) |

Cross-folder pointers worth knowing:

- [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md): supervisor-facing canonical chapter-closing brief (uses § 6 to propose the recipe).
- [`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`RESULTS_m0_b.md`](../report/RESULTS_m0_b.md): Phase-1 Qwen3-0.6B ablation findings that grounded the literature search.
- [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md): hyperparameter cross-check vs Search-R1 paper.
- [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md): measured per-step numbers feeding `RUNTIME_EFFICIENCY.md`.

## 8. Working memory (research-strand gotchas)

- **`SURVEY.md` vs `SURVEY_FOCUSED.md`**: `SURVEY.md` is the full ~96-card landscape; `SURVEY_FOCUSED.md` is the project-specific subset (35 cards) with the week-by-week plan and budget table. Read `SURVEY_FOCUSED` for decisions; use `SURVEY` only as a reference.
- **`PARADIGM_REVIEW.md` v1 vs v2 vs v3**: v1 (§2-12) is clean math-RLVR recommendations; v2 (§13-15) added counter-evidence on Search-R1 specifically; v3 (§16-17) folded in JustRL "tricks may hurt". The current plan is the v3 consolidated stack, not v1.
- **JustRL is load-bearing**: the C-minimal control is non-optional. Without it the thesis cannot claim "optimised recipe" credibly, only "stack we tried".
- **Smoke baseline anchor**: 57 s/step at 20 traj/step (2026-05-06 smoke); used as the denominator for all wall-clock projections. If smoke shape changes, re-derive.
- **LoRA caveat**: do not bolt LoRA onto Qwen3.5-2B for the recipe runs; the rank floor for multi-turn reasoning at ≤1.5B is r≥256, which negates the VRAM savings.

---

## Historical (frozen)

### Pre-reorg location (until 2026-05-06)

Literature/survey lived under [`../report/`](../report/) until the 2026-05-06 reorganisation. Git history preserves the move via `git mv`.

### How the recipe was selected

The four candidate techniques (E2H, S-GRPO, MC-GRPO, JustRL control) all came from `SURVEY_FOCUSED.md`. The selection criteria: drop-in (no architectural changes), measurable on the 2 to 3-run budget, and addressing the Phase-1 Qwen3-0.6B failure modes (slow learning, low tool-use rate, partial-credit floor masking signal).
