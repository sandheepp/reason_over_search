---
title: MILESTONE 6 — Literature review + experiment planning for publication
tags: [milestone, literature-review, planning, publication, thesis]
source: internal
created: 2026-05-11
updated: 2026-05-11 (Phase-1 salvage + data audit added)
---

# Milestone 6: Literature review and experiment planning for publication

## Context

[M5](../milestone_5/MILESTONE_5.md) is in flight: M5.1 production training launched 2026-05-11 (~01:05 UTC, pid 178440, W&B run `uwbodqgt`); per [`log.md` 2026-05-11](../log.md) the live trajectory through step 10 tightened ETA from ~25 d → ~10-13 d as gen length collapsed. The M5.1 checkpoint will be the baseline against which any M6-derived experiment is compared.

M6 is the **non-training milestone** that runs in parallel with M5.1 wall-clock. The thesis question was reframed on 2026-05-04 (per [`research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) §2) from "extending RLVR via tool-use" to *"what is the optimised training recipe for a small LM under a single-A100 budget"*. M6's job is to translate that reframe into a **publication-grade experimental plan**: which two experiments to run after M5.1 lands, how to position them against the 2026 literature, and what venue to target.

The hard timeline is unchanged ([`report/CONVERSATION_CONTEXT.md`](../report/CONVERSATION_CONTEXT.md) §2): experimentation must close **2026-06-10**, thesis submitted **2026-06-15**, defense **~2026-07-15**. Publication target is downstream of the thesis, but venue framing decisions (NeurIPS main vs ICLR vs Findings vs workshop vs blogpost) shape what the next two experiments need to demonstrate.

## Objective

Two concrete deliverables:

1. **A defended choice of the next two experiments**, with: experiment name, what it ablates, what it controls for, what it costs in wall-clock and dollars on the available hardware, what the success criterion is, and what the *finding* (positive or negative) would mean for the thesis chapter and the publication. Selection must survive a critical comparison against M5.1 as the baseline and against the 2026 competitive landscape (Tree-GRPO ICLR 2026, DIVA-GRPO, JustRL ICLR 2026 blogpost, MC-GRPO 2601.22582, the broader Agentic-RL survey at 2509.02547).

2. **A publication-framing brief** that names: (a) the one-sentence takeaway a reviewer should quote, (b) the realistic venue with acceptance probability, (c) the related-work positioning paragraph, (d) the threats-to-validity a reviewer will hammer on. Without this brief, the two experiments cannot be defended as "the right two" — they only become defensible once the venue and the takeaway sentence are fixed.

## Why this is a milestone and not a todo entry

The M0-M5 work has been pipeline-and-recipe-driven; M6 is question-driven and shapes everything downstream. Three reasons it earns a milestone folder:

- **Decision compounding.** The next two experiments consume ~30-40 % of the remaining \$1000 budget and 60-70 % of the remaining wall-clock window before 2026-06-10. A bad choice here cannot be recovered by adding tricks later — there is no later.
- **Publication ≠ thesis.** The thesis chapter can describe a recipe ablation honestly even if it doesn't publish; the publication needs a *finding* with mechanism and a defended novelty claim. These are different artifacts and the next two experiments need to serve both.
- **Critical positioning required.** Per the 2026-05-11 conversation (see [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) §"Working assumption from 2026-05-11 critical review"), the framing the user instinctively reached for — *"basic paper recreation (Search-R1 + ReSearch) + MC-GRPO efficiency improvement"* — has three problems that need to be resolved on paper before any GPU is rented:
  1. "Search-R1 + ReSearch combined" is not accurate. M5 is ReSearch-paper recipe ported to NeMo-RL; Search-R1's role is the 7-dataset *eval* pipeline only. The framing as a combination of two papers will be caught in review.
  2. "MC-GRPO applied to search-tool" is a genuine gap (the MC-GRPO paper evaluates on math), but Tree-GRPO at ICLR 2026 already addresses the *efficient rollouts for agent RL* axis and will be the reviewer's natural comparator. The mechanistic story for *why* median-baselining helps search-tool specifically (retrieval reward variance, longer trajectories, smaller G under memory pressure) needs to be made explicit, or the work reads as method-transfer.
  3. The "two efficiency improvements stacked, same results, faster" framing is competitive with DIVA-GRPO (2.55× step reduction, 1.76× wall-clock) and Tree-GRPO (1/4 rollout budget). Without a head-to-head against at least one of these, NeurIPS-tier reviewers will reject for incomplete comparison.

## Sources of truth (in order)

1. [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) — the M6-strand snapshot, including the critical review from 2026-05-11 and any updates as M5.1 produces signal.
2. [`LOG.md`](LOG.md) — append-only chronological record specific to M6 (literature finds, decision points, reframings).
3. [`research/SURVEY.md`](../research/SURVEY.md), [`research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md), [`research/LITERATURE_REVIEW.md`](../research/LITERATURE_REVIEW.md), [`research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md), [`research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), [`research/INTEGRATION_GUIDE.md`](../research/INTEGRATION_GUIDE.md) — the frozen literature arm (2026-05-04). M6 extends, does not rewrite.
4. [`report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) — last canonical chapter-closing brief; §5 carries the reframed RQ, §6 the original candidate recipe (E2H + S-GRPO + MC-GRPO with JustRL control).
5. [`milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md) and [`milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) — what M5.1 actually does (the baseline M6 has to position against).
6. M5.1 live results as they land — [`report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md), W&B `uwbodqgt`. The mid-training reward trajectory shapes which efficiency lever or algorithmic ablation is the right "next experiment".
7. **External (verified 2026-05-11)**:
   - JustRL — [arXiv:2512.16649](https://arxiv.org/abs/2512.16649) (ICLR 2026 blogpost track, math reasoning, 1.5B base models).
   - MC-GRPO — [arXiv:2601.22582](https://arxiv.org/abs/2601.22582) (median baseline GRPO, math evaluation).
   - Tree-GRPO — [arXiv:2509.21240](https://arxiv.org/abs/2509.21240) (ICLR 2026, tree-search rollouts for LLM agent RL, built on Search-R1, Qwen2.5-3B).
   - Agentic RL Survey — [arXiv:2509.02547](https://arxiv.org/abs/2509.02547) (last revised 2026-04-17, synthesises 500+ works).
   - ReSearch — [arXiv:2503.19470](https://arxiv.org/abs/2503.19470) (NeurIPS 2025; the recipe M5.1 ports).
   - Awesome-RL-based-Agentic-Search-Papers — [github.com/ventr1c/...](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers) (community survey).

## Scope (what M6 will and will not produce)

**Will**:
- A frozen comparison table of M5.1 vs the 2026 competitive landscape (rows: ReSearch, Search-R1, R1-Searcher, E2H, Tree-GRPO, DIVA-GRPO, JustRL, MC-GRPO; columns: model, task, algorithm, scale, compute, reward, reported metric, source).
- A frozen short-list of 3-5 candidate "next experiment" pairs with cost / wall-clock / success-criterion / failure-mode for each.
- A picked pair (the two experiments) with rationale and the one-sentence reviewer takeaway each is meant to support.
- A publication-framing brief (venue, probability, related-work paragraph, threats-to-validity).
- Updates to [`research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) §"Active question" and [`report/CONVERSATION_CONTEXT.md`](../report/CONVERSATION_CONTEXT.md) reflecting M6's conclusions.

**Will not**:
- Run any GPU training. M6 is purely planning + literature + writing.
- Replace the M5.1 baseline. The picked experiments are *follow-ups* to M5.1, not alternatives.
- Decide the thesis chapter outline (that's a downstream task after the two experiments land).
- Re-do the frozen literature arm. M6 extends with 2026-05 papers that the 2026-05-04 survey missed (Tree-GRPO, JustRL ICLR-blogpost, MC-GRPO, the April 2026 update to the Agentic RL survey).

## Method

Three phases, each producing a frozen artifact in this folder. Order matters — a candidate experiment short-list (Phase 2) cannot be defended without the landscape pass (Phase 1), and the picked pair (Phase 3) cannot be defended without both.

### Phase 1: Literature landscape pass (2026-05-11 → 2026-05-13)

Goal: produce a single frozen table that captures every 2026 paper at the intersection of {GRPO-family, agent / tool / retrieval, small-model or efficiency-focused}. Output: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md).

- Read Tree-GRPO end-to-end. Specifically: rollout budget claim ("1/4"), comparison protocol vs Search-R1, what they ablate. Is the prefix-sharing win composable with MC-GRPO's median baseline, or does prefix-sharing remove the small-G regime MC-GRPO targets?
- Read JustRL end-to-end. The "tricks may hurt" claim, the specific tricks tested, whether any of their negative findings replicate on tool-use.
- Re-read MC-GRPO with the search-tool transfer question in mind: what claims survive when reward is F1-from-retrieval (high variance) instead of math correctness (low variance)?
- Skim the Agentic RL Survey 2026-04 update — identify any new entry that targets the 0.5B-2B scale.
- Skim Awesome-RL-based-Agentic-Search-Papers for any 2026-04 or 2026-05 entry the survey missed.
- Capture in the table: paper, venue, task, algorithm-family, model + size, compute, reward shape, reported metric, the single sentence that defines its contribution. No editorialising in the table; that goes in §5 of the table doc.

Estimated effort: 1-2 working days.

### Phase 2: Candidate experiment short-list (2026-05-13 → 2026-05-15)

Goal: 3-5 candidate "next-experiment pairs" with cost / wall-clock / success-criterion / failure-mode. Output: [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md).

Candidates worth considering (provisional list; M6 will prune and add):

| # | Candidate | What it tests | Defended novelty hook | Risk |
|---|---|---|---|---|
| A | MC-GRPO on M5.1 (median baseline at G=5, then G=2) | Does median-baselining recover the small-G gap when reward variance is from retrieval, not math? | Mechanism: retrieval-induced reward variance | Method transfer; Tree-GRPO does prefix-sharing better |
| B | JustRL-style minimal control vs full M5.1 stack | Does the F1-only + plain GRPO recipe match the M5.1 recipe? Is M5.1's stack helping? | Negative-result hook ("tricks hurt for search-tool too") | If M5.1 already matches a minimal control, the chapter has no recipe contribution |
| C | Reward-shaping ablation (F1 vs F1+format vs partial-credit) | Quantify the "0.1 partial-credit floor masks tool-use signal" Phase-1 finding at 0.8B | Phase-1 finding made publishable — actually novel; nobody has ablated this in the ReSearch family | One run per reward setting; ~3 runs; expensive in budget |
| D | E2H curriculum on M5.1 (NQ → HotpotQA → MuSiQue) | Does curriculum help on RAG-augmented tasks (not closed-book math)? | E2H-for-RAG is unexplored | Data scheduler work; failure mode is "curriculum doesn't help" — that's also a finding |
| E | Tree-GRPO ablation on M5.1 setup | Replicate Tree-GRPO's prefix-sharing claim on Qwen3.5-0.8B at our compute scale | Strong related-work comparison; head-to-head | Requires Tree-GRPO infrastructure (forked Search-R1) — port cost is high |
| F | Scale extrapolation (0.8B → 2B on the M5.1 recipe) | Does the recipe transfer up? Reviewer-killer answer to "does this scale" | Reviewer's predictable objection turned into a contribution | 2B at the 0.6-epoch budget is 2-3× more expensive — eats the whole remaining budget |

For each candidate the docu **must** include: cost on 1× A100 / H100 / B200 per [`setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md), the success criterion (specific number, not "shows improvement"), the failure-mode interpretation (what does a *negative* result mean for the chapter — every candidate must be informative under both outcomes), and the conflict with concurrent work (which 2026 paper does this overlap with, and how).

Estimated effort: 2 working days.

### Phase 3: Pick the pair + publication framing brief (2026-05-16 → 2026-05-18)

Goal: the chosen pair plus the publication-framing brief. Outputs: [`PICKED_PAIR.md`](PICKED_PAIR.md) and [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md).

Selection criteria (in priority order):
1. Each experiment must be informative under both positive and negative outcomes (the failure mode is still a chapter section).
2. The pair must together support one *mechanism-named* takeaway sentence — not a recipe stack.
3. Combined cost ≤ \$400 and combined wall-clock ≤ 18 days on the chosen hardware (whichever B200 / H200 / A100 lane is picked by then).
4. The pair must include at least one experiment that addresses the strongest reviewer objection identified in Phase 1 (currently: "how does this compare to Tree-GRPO").
5. At least one experiment must not require new infrastructure beyond what M5.1 already has — for risk control.

The publication-framing brief must answer:
- The one-sentence reviewer-quotable takeaway.
- The realistic venue (NeurIPS main / ICLR / ACL or EMNLP main / Findings / workshop / ICLR-blogpost) with acceptance probability range from the critical-review estimates (NeurIPS main 5-10 % unless dramatic, Findings 50 %+, blogpost very plausible).
- The 200-word related-work paragraph that positions the work against Tree-GRPO, JustRL, MC-GRPO, ReSearch, Search-R1 — written tightly enough to drop into the introduction.
- The three threats-to-validity a reviewer will hammer: small-scale generalisation, head-to-head with concurrent work, novelty bound by method-transfer framing. Each gets a defended response (or an acknowledged limitation).

Estimated effort: 2-3 working days.

## Deliverables (this folder)

| File | Purpose | Lifecycle |
|---|---|---|
| [`MILESTONE_6.md`](MILESTONE_6.md) | This file — the milestone definition | Updated only on scope change |
| [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) | M6-strand snapshot; updated as decisions land | Living |
| [`LOG.md`](LOG.md) | Append-only chronological record for M6 | Append-only |
| [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) | Three Phase-1 findings reusable as paper motivation, with file:line citations and the limit of each claim | Authored 2026-05-11; revised if a salvage finding gets new evidence |
| [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md) | Deep-dive: what is measured cleanly, what is read-off-heterogeneous-data, what is missing entirely, and what would close each gap | Authored 2026-05-11; revised when the picked pair lands and closes a gap |
| `LANDSCAPE_TABLE_2026-05.md` | Phase 1 output: frozen 2026 competitive landscape table | Frozen once Phase 1 closes |
| `CANDIDATE_EXPERIMENTS.md` | Phase 2 output: 3-5 candidate pairs with cost / criterion / risk | Frozen once Phase 2 closes |
| `PICKED_PAIR.md` | Phase 3 output: the chosen two experiments with defended rationale | Frozen once Phase 3 closes |
| `PUBLICATION_FRAMING.md` | Phase 3 output: the publication-framing brief | Frozen once Phase 3 closes |

### Pre-Phase-1 update (2026-05-11)

Following the 2026-05-11 critical review of the Phase-1 record, two Phase-2-anticipating artifacts were authored ahead of the planned phase order: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) and [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md). The audit elevates **Candidate C (reward-shape ablation: F1+0.1-floor vs F1-only vs EM-only at the M5.1 recipe)** from "provisional candidate" to **"likely pick #1"** — the audit shows it directly closes Finding 1's gap (the 0.1 floor masks tool-use signal, [`PHASE_1_SALVAGE.md` §"Finding 1"](PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)) and is the only candidate that pairs an unfilled-by-the-literature gap with a Phase-1 motivation paragraph the project already owns. The second pick remains undecided pending Phase 1 landscape pass.

## What's left (task table)

| # | Task | Phase | Owner | Blocked on |
|---|---|---|---|---|
| 1 | Create folder + initial three docs (this, `CONVERSATION_CONTEXT.md`, `LOG.md`) | — | — | nothing |
| 2 | Read Tree-GRPO end-to-end and capture in landscape table | 1 | — | (1) |
| 3 | Read JustRL end-to-end and capture in landscape table | 1 | — | (1) |
| 4 | Re-read MC-GRPO with search-tool transfer lens; capture in landscape table | 1 | — | (1) |
| 5 | Skim Agentic-RL survey 2026-04 update + Awesome-Agentic-Search repo for missed 2026-04/05 entries | 1 | — | (1) |
| 6 | Freeze [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md) | 1 | — | (2)-(5) |
| 7 | Draft [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) with cost / criterion / risk per candidate | 2 | — | (6) |
| 8 | Stress-test candidates against the M5.1 mid-training reward trajectory (W&B `uwbodqgt`) | 2 | — | (7), M5.1 step ≥ 100 |
| 9 | Freeze [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) | 2 | — | (8) |
| 10 | Pick the pair; write [`PICKED_PAIR.md`](PICKED_PAIR.md) | 3 | — | (9) |
| 11 | Write [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md) | 3 | — | (9) |
| 12 | Update [`research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) §"Active question" and [`report/CONVERSATION_CONTEXT.md`](../report/CONVERSATION_CONTEXT.md) | 3 | — | (10), (11) |
| 13 | Supervisor meeting brief (extend the 2026-05-07 brief or author 2026-05-1X follow-up) | 3 | — | (12) |

## Pointers

- M5 in-flight (the baseline M6 plans against): [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), live trajectory [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md).
- Frozen literature arm (M6 extends, does not rewrite): [`../research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md), [`../research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md), [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), [`../research/INTEGRATION_GUIDE.md`](../research/INTEGRATION_GUIDE.md).
- Last canonical chapter brief: [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md).
- Hardware + cost lanes for the picked pair: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md).
- Active project TODO (M6 tasks will be added here for visibility): [`../todo/TODO_2026-05-11.md`](../todo/TODO_2026-05-11.md).
