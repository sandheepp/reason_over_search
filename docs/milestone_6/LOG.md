---
title: Milestone 6 Log
tags: [milestone, m6, log]
source: internal
created: 2026-05-11
updated: 2026-05-11
---

# Milestone 6 Log

Append-only chronological record specific to M6 (literature landscape, candidate experiments, picked pair, publication framing). New entries on top (newest first). One `## YYYY-MM-DD` heading per day; bullets within.

Format mirrors the wiki [`../log.md`](../log.md):

```
## YYYY-MM-DD
- Decision: <one-liner>, recorded in <page>
- Learned: <one-liner>, filed in <page>
- Ingested: <source path or URL> -> <pages touched>
- Reframe: <one-liner about a positioning shift>, recorded in <page>
```

This file is **scoped to M6** — broader project events (ingest of unrelated papers, infra changes, M5 events) belong in the top-level [`../log.md`](../log.md). M6 decisions that change the broader project (e.g. "thesis chapter restructured around the picked pair") get cross-posted to both.

---

## 2026-05-11

- Decision: **Phase-1 salvage + data audit authored** (this commit). Two new docs: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) (three Phase-1 findings reusable as paper motivation, with file:line cites + limit-of-claim for each), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md) (deep-dive on what is measured cleanly vs what is missing entirely, per-finding gap-closing experiment with cost). Authored ahead of the planned phase order because the M5.1 wall-clock (~10-13 d) means experiment selection cannot wait for the landscape table. The audit elevates Candidate C (reward-shape ablation) from provisional candidate to likely pick #1 — see [`CONVERSATION_CONTEXT.md` §9-10](CONVERSATION_CONTEXT.md).
- Learned: **The 9-run Phase-1 v0 prompt sweep is the project's strongest data asset.** Three paired comparisons (`p1_basic` / `p1_basic_no_ex`, `p2_basic2` / `p2_basic2_no_ex`, `p3_decide` / `p3_decide_no_ex`) hold rules section fixed and toggle the few-shot example — the *only* clean A/B in the Phase-1 record. The decision-rules variant (`p3_decide_no_ex`) is the pareto winner: best end-of-run reward (0.215), tool-use survives example removal, half the response budget of the with-example heavy-tool variant. Held-out at 51,713 items: +0.014 EM, +12 % ACC, +14 % F1 over `p1_basic_w_ex`. Filed in [`PHASE_1_SALVAGE.md` §Finding 2](PHASE_1_SALVAGE.md#finding-2--the-no-example--decision-rules-prompt-is-a-pareto-improvement).
- Learned: **The 0.1 partial-credit floor finding is observation-grade, not ablation-grade.** All 9 v0 runs used the paper-faithful 3-tier reward; there is **no run where same prompt × seed × {F1+0.1, F1-only, EM-only} were measured side-by-side**. The 3-6 pp tool/no-tool gap is read across cross-prompt comparisons with confounded changes (rules section + example + tag schema). To publish the "floor masks signal" claim, the M6 picked pair must include a clean reward-shape ablation. Filed in [`DATA_AUDIT_PHASE_1.md` §1](DATA_AUDIT_PHASE_1.md#1-finding-1--01-partial-credit-floor-masks-tool-use-signal).
- Learned: **Five cross-cutting gaps affect every salvage finding**: (a) no multi-seed runs anywhere, (b) no per-rollout-level analysis (only batch means), (c) no 0.8B / Qwen3.5 prompt-sweep replication, (d) no direct measurement of format-OK-but-F1=0 rate by run (back-calculated only), (e) no baseline that converges to a different regime than M5.1's shrink-and-improve shape. Filed in [`DATA_AUDIT_PHASE_1.md` §5](DATA_AUDIT_PHASE_1.md#5-cross-cutting-gaps-that-affect-everything).
- Learned: **Qwen3.5-0.8B hybrid (0.060 EM) is uniformly *below* Qwen3-0.6B hybrid (0.102 EM) on the same eval protocol**, all 7 datasets, no crosses, mean Δ −0.042 EM (−41 % rel). Confirmed from [`../report/RESULTS_m4.md` §5](../report/RESULTS_m4.md#L67). Excluded from Bucket-A salvage because no mechanism is attached (could be tokenizer drift 151K → 248K, chat-template change, prompt-mode misalignment, training-distribution drift). Filed in [`DATA_AUDIT_PHASE_1.md` §4a](DATA_AUDIT_PHASE_1.md#4a-qwen35-08b-hybrid--qwen3-06b-hybrid-on-untrained-tool-use-results_m4md-5) as a candidate 1-page negative result if a cheap diagnostic study can attach mechanism.
- Reframe: **The Phase-1 work is "introduction material", not paper-anchor material.** The strongest framing is a single motivation paragraph in the introduction citing the three salvage findings, which earns the small-scale regime as the paper's scope and turns the M6 picked pair into a mechanism-driven follow-up rather than a method-transfer. Recorded in [`PHASE_1_SALVAGE.md` §"How these three findings ladder into the paper"](PHASE_1_SALVAGE.md#how-these-three-findings-ladder-into-the-paper).

- Decision: **Milestone 6 created** as a planning-only, parallel-to-M5.1 milestone. Objective: pick the next two experiments + publication framing before M5.1 lands. Recorded in [`MILESTONE_6.md`](MILESTONE_6.md). Three frozen artifacts planned: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md) (Phase 1), [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) (Phase 2), [`PICKED_PAIR.md`](PICKED_PAIR.md) + [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md) (Phase 3).
- Reframe: **"M5 = Search-R1 + ReSearch combined" is inaccurate** — M5 is the ReSearch-paper recipe alone (GRPO, F1-only, MuSiQue, G=5); Search-R1's role is the 7-dataset eval pipeline. Recorded in [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md). Implication: framing in supervisor brief + any draft must be corrected.
- Reframe: **NeurIPS main is not the realistic target** with the current plan. Honest ranges: NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, ICLR blogpost very plausible. Recorded in [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md). Implication: venue decision precedes experiment selection in Phase 3.
- Learned: **MC-GRPO ([arXiv:2601.22582](https://arxiv.org/abs/2601.22582)) is real and not yet applied to search-tool** — abstract + reported experiments are math (Qwen3-1.7B on GSM8K: 2-rollout 78.90 % → 83.54 % with MC). Algorithm- and model-agnostic across GRPO family. Filed as a Phase 1 read in [`MILESTONE_6.md` §"Method"](MILESTONE_6.md).
- Learned: **Tree-GRPO ([arXiv:2509.21240](https://arxiv.org/abs/2509.21240), ICLR 2026) is the direct competitor** on the "efficient rollouts for agent RL" axis. Built on Search-R1 repo; uses tree-search rollouts with shared prefixes; claims 1/4 chain-based rollout budget; Qwen2.5-3B on multi-hop QA. Reviewer-anticipated comparison. Filed as Phase 1 read in [`MILESTONE_6.md` §"Method"](MILESTONE_6.md).
- Learned: **JustRL ([arXiv:2512.16649](https://arxiv.org/abs/2512.16649)) is math reasoning, ICLR 2026 blogpost track** — not a search-tool result. Findings: simple single-stage GRPO + fixed hyperparameters + no KL + no curriculum beats ProRL-V2's nine-stage pipeline at half the compute on DeepSeek-1.5B / Nemotron-1.5B. Filed in [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md). Implication: usable as control methodology, not as a novelty hook.
- Learned: **The 2026 competitive landscape on agentic-RL is denser than the 2026-05-04 frozen survey captured** — DIVA-GRPO (2.55× step / 1.76× wall-clock), Tree-GRPO (ICLR 2026), JustRL (ICLR 2026 blogpost), Agentic-RL Survey 2509.02547 (April 2026 update synthesises 500+ works). The frozen [`../research/SURVEY.md`](../research/SURVEY.md) is missing these. Phase 1 must catch up.
- Ingested (provisional, for Phase 1): JustRL [arXiv:2512.16649](https://arxiv.org/abs/2512.16649), MC-GRPO [arXiv:2601.22582](https://arxiv.org/abs/2601.22582), Tree-GRPO [arXiv:2509.21240](https://arxiv.org/abs/2509.21240), Agentic RL Survey [arXiv:2509.02547](https://arxiv.org/abs/2509.02547). To be captured in [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md) when Phase 1 starts.
