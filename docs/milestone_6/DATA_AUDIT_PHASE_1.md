---
title: Phase-1 Data Audit — what we have vs what's missing for publication
tags: [milestone, m6, phase1, data-audit, publication]
source: internal
created: 2026-05-11
updated: 2026-05-11
---

# Phase-1 Data Audit: what is in the data, what is not

> Deep-dive audit of the Phase-1 experimental record (29 Qwen3-0.6B v0+v1 runs, M3/M3.1 held-out evals, M4 untrained baseline, M5.1 mid-training snapshot) against the three publication-eligible findings catalogued in [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md). For each finding: what is measured cleanly, what is read-off-heterogeneous-data, what is missing entirely, and what would close the gap.
>
> The "missing" list is **not** a complaint — it is a list of measurements the M6 picked-pair experiments can choose to make. Each gap doubles as a candidate experiment objective.

**Last updated**: 2026-05-11.

---

## 0. What is in the artifact record

| Artifact | Source | Scale | Confidence |
|---|---|---:|---|
| Phase-1 v0 prompt-sweep CSVs | [`../report/archive/m0_a/csv/`](../report/archive/m0_a/csv/) — 9 ablation runs, 1000-3000 steps each | 13 metrics × ~17,000 step rows | High (W&B-backed) |
| Phase-1 v1 base-bootstrap CSVs | [`../report/archive/m0_b/csv/`](../report/archive/m0_b/csv/) — 5+ base attempts | same schema | High |
| M3 held-out eval | [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md) — `p1_basic_w_ex_z7kcxfof` checkpoint | 51,713 items × 7 datasets × single-seed greedy | High |
| M3.1 held-out eval | [`RESULTS_m3.md` §14](../report/RESULTS_m3.md#L338) — `p3_decide_no_ex_el6s2d2h` checkpoint | 51,713 items × 7 datasets × single-seed greedy | High |
| M4 untrained baseline | [`../report/RESULTS_m4.md`](../report/RESULTS_m4.md) — Qwen3.5-0.8B hybrid + base | 51,713 items × 7 datasets × single-seed greedy × 2 variants | High |
| M5.1 mid-training trajectory | [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md), W&B `uwbodqgt` | live; through step 10+ as of 2026-05-11 | High but partial |
| M1 Search-R1 reproduction | [`../report/RESULTS_m1.md`](../report/RESULTS_m1.md) — Qwen2.5-3B base + instruct | 51,713 items × 7 datasets × single-seed greedy × 2 variants | High |

**Per-CSV column schema** (verified 2026-05-11):
```
_step, critic/rewards/mean, critic/score/mean, tool_call_counts/mean,
re_search/search_calls/mean, num_turns/mean, response_length/mean,
actor/loss, actor/kl_loss, actor/grad_norm, actor/pg_loss,
re_search/budget_utilization, response/aborted_ratio
```

What this gives us: per-step trajectory of (a) reward, (b) tool-use rate, (c) response length, (d) optimisation health. What it does **not** give us: per-rollout rewards (only batch means), no F1-vs-format breakdown (the reward column is the combined 3-tier value), no per-prompt accuracy by step (only end-of-run held-out evals).

---

## 1. Finding 1 — "0.1 partial-credit floor masks tool-use signal"

### What is in the data

- **End-of-run reward by prompt**, 9 runs ([`RESULTS_m0_a.md` §10](../report/RESULTS_m0_a.md#L436)).
- **End-of-run tool-call rate by prompt** in the same table — three behavioural modes (heavy-tool 2.0, standard 1.0, collapse ≤0.08).
- **Reward gap between modes**: tool-using ≈ 0.18-0.22, no-tool ≈ 0.16. The 3-6 pp gap claim is read off this cross-prompt comparison.
- **Floor mechanism quantification**: §7 derives that "6 to 12 % of episodes get a non-zero F1 hit on top of the 0.1 baseline" from the 0.16-0.22 band.

### What is *not* in the data

- **No A/B reward-shape test.** All 9 v0 runs use the paper-faithful 3-tier reward. There is **no run where same prompt × same seed × {F1+0.1, F1-only, EM-only} were measured side-by-side.** The "floor masks signal" claim is an *observation from cross-prompt comparison*, not a clean ablation. Reviewer 2's first question: "what happens with no floor?"
- **No per-rollout reward distribution.** Only batch means. We can't show the histogram (e.g., "X % of rollouts at exactly 0.1, Y % above F1=ε, Z % at 0") — only the mean.
- **No second seed.** All 9 v0 runs are single-seed. The 3-6 pp tool-vs-no-tool gap is not bracketed by run-to-run variance.
- **No tool-allowed vs tool-disallowed paired comparison** on the same prompt. The collapse-to-zero-tool prompts (`p2_basic2_no_ex`) and the heavy-tool prompts (`p1_basic_w_ex`) differ in *the prompt itself*, not in a tool-availability switch.

### Gap-closing experiment

**Candidate C from [`MILESTONE_6.md`](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15)** (the reward-shape ablation) is the direct gap-fill. Three runs at the M5.1 prompt × reward ∈ {F1+0.1, F1-only, EM-only}, single seed each, ~3 days each on A100. Bonus: collect per-rollout reward histograms by patching the rollout dump.

### Confidence after gap-fill

- Currently: **observation grade**. Defensible as motivation, fragile as contribution.
- After gap-fill: **ablation grade**. Defensible as the paper's headline reward-design finding *if* the gap-fill produces a clean separation (≥3 pp EM lift from removing the floor).

---

## 2. Finding 2 — "no-example + decision-rules prompt is a pareto improvement"

### What is in the data

- **Three paired comparisons** (`p1_basic` / `p1_basic_no_ex`, `p2_basic2` / `p2_basic2_no_ex`, `p3_decide` / `p3_decide_no_ex`) where the only variable changed is the few-shot example block. Identical rules section within each pair.
- **End-of-run reward, tool-call rate, response length** for each — full table at [`RESULTS_m0_a.md` §8](../report/RESULTS_m0_a.md#L381).
- **Held-out held-out 51,713-item eval** of two of these checkpoints (the with-example heavy-tool `z7kcxfof` and the no-example decision-rules `el6s2d2h`) at full Plan A scale: EM 0.155 vs 0.169, ACC widens to +12 % rel, F1 widens to +14 % rel. ([`RESULTS_m3.md` §14.4-14.5](../report/RESULTS_m3.md#L381)).

### What is *not* in the data

- **Single seed for everything.** Each cell in the 3-pair table is one run; the held-out EM gap of +0.014 (+9 % rel) is one run vs one run. No within-run variance estimate.
- **Bamboogle (n=125) regression on the no-example variant.** [`RESULTS_m3.md` §14.5](../report/RESULTS_m3.md#L412) flags it as the one anti-result, not denting the structural conclusion but worth a second seed. The archived n=125 noise postmortem at [`../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md) documents that single-seed Bamboogle has ~3 pp SE and the tail is heavier than Gaussian — so the Bamboogle move on the no-example variant could be either signal or noise.
- **No 0.8B / Qwen3.5 replication of the prompt sweep.** M4 ran a different prompt-mode lock (`qwen35_minimal` / `qwen35_minimal_no_system` per [`RESULTS_m4.md` §2](../report/RESULTS_m4.md#L24)), not a re-run of the 9-run v0 sweep. We **do not know** whether decision-rules-substitute-for-example is a property of Qwen3-0.6B specifically or transfers to Qwen3.5-0.8B.
- **No held-out reward at the checkpoints.** Held-out evals report EM/ACC/F1 (downstream metrics), not the training reward applied to held-out rollouts. So we cannot say "the +0.025 reward gap closes / widens / stays / inverts on held-out data" — only that the EM gap is +0.014.
- **No mid-training held-out evals.** Both M3 and M3.1 eval the *final* checkpoint. No intermediate checkpoints were evaluated; we don't know when in training the pareto win opens up.

### Gap-closing experiment(s)

- **Two-seed re-run** of the picked prompt pair (`p3_decide_no_ex` vs `p1_basic_w_ex`) on Qwen3.5-0.8B with the M5.1 recipe. Cost: ~2 × 5-day runs on H100. Confirms the Phase-1 cross-family transfer claim and gives a variance bar on Bamboogle.
- **Held-out reward dump** at every M5.1 checkpoint. Zero compute cost on top of M5.1 (just save it). Lets us trace when the pareto opens up.

### Confidence after gap-fill

- Currently: **single-seed observation grade**. The held-out 51,713-item eval gives confidence the EM gap is not partial-credit-floor inflation (because ACC/F1 widen), but single-seed.
- After gap-fill: **two-seed cross-family confirmation grade**. Strong enough to anchor the prompt-design section of the paper. Without the cross-family replication, the finding stays Qwen3-0.6B-specific.

---

## 3. Finding 3 — "Shrink-and-improve" RL regime

### What is in the data

- **M5.1 live trajectory** through step 10+ ([`RESULTS_SMOKE_m5.md` §6.5](../report/RESULTS_SMOKE_m5.md)): tc_mean 8.96 → 3.47, tok_mean 7038 → 2183 (3.2× compression), reward 0.020 → 0.132, truncation 68.4 → 0 %, all simultaneously, all within first 5 % of the run.
- **Wall-clock decomposition** in §6.5 — proves the compression is real wall-clock saving (per-step 57.9 → 21.3 min), not a metric artifact.
- **Comparison framing** in [`RESULTS_m5.md` §4.1](../report/RESULTS_m5.md#L61): contrasts with the long-CoT regime (length grows with reward) seen in math reasoning RL. Citations attached.

### What is *not* in the data

- **One run.** M5.1 alone. No paired comparison against a same-scale run that *doesn't* shrink (e.g. math reasoning baseline or a long-CoT regime on the same hardware).
- **Mid-training snapshot.** As of 2026-05-11, M5.1 is at step ~10-16 of ~622. The shrink-and-improve claim is supported by the first 5 % of the run; the rest of the trajectory could plateau, drift, or invert. The full-run picture isn't available.
- **No mechanistic decomposition.** The §4.1 mechanism ("model shrinks away failure modes — rambling `<think>`, literal function placeholders, max_turns") is descriptive, not measured. No data on the *rate* at which each failure mode disappears (e.g., what fraction of tokens are literal-copy of the prompt's tool template at step 0 vs step 16?).
- **No comparison to the ReSearch published training curves.** Their [Figure 4](../papers/2503.19470_research.md#L37) shows the same shape (rewards rising), but length / tool-call shrinkage is not the focus there. A direct overlay would strengthen the "regime" claim.

### Gap-closing experiment(s)

- **M5.1 full-run completion** is the natural gap-fill. No extra compute.
- **One paired math reasoning GRPO run** on the same hardware / scale would let the "different regime from long-CoT" claim move from observation to comparison. **~1× M5.1 cost**. Not cheap.
- **Rollout-sample analysis** of M5.1 step 0 vs step 16: a manual / automated annotation of failure modes (rambling think, literal template copies, premature termination) in a stratified sample. ~Zero compute, all human / LLM-judge time.

### Confidence after gap-fill

- Currently: **single-run observation grade**. Useful as introduction framing — *"the regime our work characterises"* — not strong enough to be a contribution.
- After M5.1 full-run: **trajectory-supported observation**. Defensible as introduction framing in a Findings paper.
- After paired comparison with long-CoT regime: **two-regime characterisation**. Plausibly its own short paper / workshop submission.

---

## 4. Findings *not* in [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) — audited for completeness

### 4a. Qwen3.5-0.8B hybrid < Qwen3-0.6B hybrid on untrained tool-use ([`RESULTS_m4.md` §5](../report/RESULTS_m4.md#L67))

- **What is measured**: full Plan A (51,713 items / variant × 7 datasets); M3 Qwen3-0.6B hybrid 0.102 EM vs M4 Qwen3.5-0.8B hybrid 0.060 EM. All 7 datasets show degradation, no crosses, mean Δ −0.042 EM (−41 % rel).
- **What is missing**: no mechanism. Possibilities not ruled out: tokenizer drift (Qwen3 → Qwen3.5 changed the vocab from 151K → 248K), chat-template drift, training-distribution drift (Qwen3.5's post-training emphasised different domains), prompt-mode misalignment (M4 used a different prompt-mode lock than M3 — M4.2 minimal vs M3 hybrid system).
- **Why excluded from salvage**: no mechanism attached makes this a *negative observation*, not a publishable finding. A reviewer's first question is "what causes the drop?", and we don't have an answer.
- **Gap-closing**: a tokenizer / chat-template / prompt-mode diagnostic study could turn this into a 1-page finding. Estimated cost: 2-3 days human time, ~zero compute.

### 4b. Base-model cold-start fails at 0.6B ([`RESULTS_m0_b.md` §12.2](../report/RESULTS_m0_b.md))

- **What is measured**: 5/5 v1 base attempts on Qwen3-0.6B-Base, 0 tool calls throughout training; longest 2300 steps. Two "with-example" base runs crashed with response_length → 1 token within 250 steps.
- **Why excluded from salvage**: R1-Searcher's two-stage curriculum is motivated by precisely this failure mode ([`../papers/2503.05592_r1-searcher.md`](../papers/2503.05592_r1-searcher.md#method)). We replicate their motivation, we don't add new evidence. The cleanest publishable use of this data is in the *related-work* paragraph: "consistent with R1-Searcher's observation that base models require a cold-start curriculum, our 5/5 attempts at 0.6B confirm this floor."

### 4c. M3 +52 % rel EM lift from GRPO ([`RESULTS_m3.md`](../report/RESULTS_m3.md))

- **What is measured**: 0.102 → 0.155 EM across 51,713 items / variant, single seed, greedy decode.
- **Why excluded from salvage**: this is a *replication of ReSearch at smaller scale*. ReSearch reports +14-16 pp EM improvements at 7B/32B; we report +0.053 EM (+52 % rel) at 0.6B. The qualitative result transfers, the methods chapter of the thesis can recite it, but it is not a paper.

### 4d. M1 Search-R1 ±2.5 pp reproduction ([`RESULTS_m1.md`](../report/RESULTS_m1.md))

- **Why excluded**: pure replication of Search-R1 eval pipeline. Related-work sentence ("we reproduce the Search-R1 evaluation suite within ±2.5 pp"), not a contribution.

### 4e. 14 train/eval alignment fixes ([`RESULTS_m3.md` §10.5](../report/RESULTS_m3.md))

- **Why excluded**: engineering audit. Methods section, possibly an appendix or a methods workshop submission. Not a research paper.

---

## 5. Cross-cutting gaps that affect everything

These are the missing measurements that limit *all* salvageable findings, not specific to one:

1. **No multi-seed runs anywhere.** Every Phase-1 run is single-seed. Every M3 / M3.1 / M4 eval is single-seed (greedy decode is seed-invariant, but the *training* runs aren't). The Bamboogle n=125 noise postmortem documents that single-seed variance is real and load-bearing for small datasets. **Minimum needed for a credible paper: 2 seeds on at least one of the M6-picked experiments.**

2. **No per-rollout-level reward / behaviour analysis.** All metrics are batch means. We can't show histograms of reward, rollout length, tool-call count by individual rollout. Each W&B run dumps rollout JSONLs, but they aren't extracted into the analysis pipeline. Cost: a day of analysis code; **zero compute**.

3. **No cross-prompt control on Qwen3.5-0.8B.** The Phase-1 prompt sweep was Qwen3-0.6B only. M4 ran one prompt per variant on 0.8B (locked at `qwen35_minimal` / `qwen35_minimal_no_system`). We don't have a 0.8B prompt-sweep equivalent of the 9-run Phase-1 block, so the prompt-design finding doesn't yet generalise across scale.

4. **No "format-OK-but-F1=0" vs "format-broken" rate by run.** Reward CSV records the *combined* 3-tier reward; we never split format-OK / format-broken rates explicitly. The 0.1-floor finding rests on the 6-12 % F1-hit-rate derivation in §7 of `RESULTS_m0_a.md`, which is back-calculated from the mean reward, not directly observed.

5. **No baseline that converges to a different regime.** All Phase-1 + M5.1 runs are in the "shrink-and-improve" regime. We have no contrastive evidence that *another* recipe / task / model would produce the long-CoT regime in our hands. Without that, calling the M5.1 shape a "regime signature" is one-sided.

---

## 6. Summary — gap severity by finding

| Finding | Current grade | Severity of gaps | Cost to close | Worth closing? |
|---|---|---|---:|---|
| 1. Partial-credit floor masks signal | Observation (cross-prompt) | **High** — needs an A/B reward-shape ablation | 3 runs × ~3 days each on A100/H100 | **Yes** — turns Finding 1 into the paper's headline ablation |
| 2. No-example + decision-rules pareto | Single-seed cross-family-untested | **Medium-High** — needs 2 seeds + Qwen3.5-0.8B replication | 2 × 5-day runs on H100 | **Yes** — closes the strongest standalone finding |
| 3. Shrink-and-improve regime | Single-run mid-training | **Medium** — needs M5.1 full + ideally a long-CoT baseline | M5.1 completion (free) + optional 1× M5.1 cost | **M5.1 completion: yes. Long-CoT baseline: only if budget allows.** |
| 4a. Qwen3.5 < Qwen3 cross-family degradation | Negative observation | **High** — needs a mechanism diagnostic | ~2-3 days human, ~zero compute | **Maybe** — could become a 1-page negative result |
| 4b. Base bootstrap fails at 0.6B | Replicates R1-Searcher | **N/A** — known result | Don't replicate further | **No** — cite, move on |
| 4c. M3 +52 % rel EM lift | Replication | **N/A** | — | **No** — thesis methods chapter |
| 4d. M1 ±2.5 pp reproduction | Replication | **N/A** | — | **No** — related-work sentence |
| 4e. 14 alignment fixes | Engineering | **N/A** | — | **No** — appendix |

**Three of the eight rows say "yes, worth closing the gap"**, and they are exactly the three salvage findings + the cross-family-degradation diagnostic. The first three map onto the M6 candidate-experiment list ([`MILESTONE_6.md` §"Phase 2"](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15)): reward-shape ablation (closes Finding 1), prompt-design + scale transfer (closes Finding 2), M5.1 completion (closes Finding 3 partially). The fourth (cross-family diagnostic) is cheap and could be added as a side study.

---

## 7. What this means for the M6 picked pair

After this audit, the strongest two-experiment pair under the \$500-600 / 17-day remaining budget appears to be:

1. **Reward-shape ablation** (Candidate C from [`MILESTONE_6.md`](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15)) — closes Finding 1's gap. Three runs × ~3-5 days each, depending on hardware lane. **Closes the strongest reward-design gap; positions against ReSearch directly.**

2. **Either** (a) **2-seed Qwen3.5-0.8B replication of the picked prompt pair** (closes Finding 2's gap, low-novelty but high-defensibility), **or** (b) **MC-GRPO at small G on M5.1 recipe** (the original second-experiment plan, novelty-driven but higher-risk).

The audit suggests (a) is the more defensible pick for the thesis chapter and for a Findings-grade venue; (b) is the more ambitious pick for a NeurIPS / ICLR submission. **The choice depends on the venue, which depends on Phase 3 of M6.** This audit defers the choice to Phase 3 but flags that the pair *can* be picked from this audit alone if budget pressure forces the call earlier.

## Cross-references

- Findings catalogued: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md).
- Milestone definition: [`MILESTONE_6.md`](MILESTONE_6.md).
- Snapshot: [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md).
- Result docs audited: [`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md), [`../report/RESULTS_m1.md`](../report/RESULTS_m1.md), [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md), [`../report/RESULTS_m4.md`](../report/RESULTS_m4.md), [`../report/RESULTS_m5.md`](../report/RESULTS_m5.md), [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md).
- CSV archive: [`../report/archive/m0_a/csv/`](../report/archive/m0_a/csv/) (9 prompt-ablation runs), [`../report/archive/m0_b/csv/`](../report/archive/m0_b/csv/) (base-bootstrap runs).
- Bamboogle noise postmortem: [`../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md).
