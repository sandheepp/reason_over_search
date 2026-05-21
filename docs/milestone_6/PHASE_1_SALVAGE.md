---
title: Phase-1 Salvage — findings reusable as paper motivation
tags: [milestone, m6, phase1, salvage, publication]
source: internal
created: 2026-05-11
updated: 2026-05-11
---

# Phase-1 Salvage: findings reusable as paper motivation

> Three Phase-1 findings that survive critical scrutiny and can be cited as motivation for the M6-picked-pair paper. Each entry: file:line citations, exact numbers, the mechanism the data supports, and the limit of the claim. Companion doc: [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md) for what is *not* in the data.
>
> Scope: this doc lists what is salvageable, not what is novel. Whether each finding is publishable depends on the venue and the chosen pair of M6 experiments (see [`MILESTONE_6.md`](MILESTONE_6.md) §"Method").

**Last updated**: 2026-05-11.

---

## Why this doc exists

Per the 2026-05-11 critical review (this conversation thread), the Phase-1 work (29 Qwen3-0.6B runs across v0 + v1 blocks) is not a publication on its own — the M3 +52 % rel EM lift is a replication of ReSearch at smaller scale, the M1 ±2.5 pp reproduction is replication, and the 14-fix eval audit is engineering. However, three Phase-1 findings, all with mechanism attached, can serve as the **motivation paragraph** for the M6-picked-pair paper:

1. **The 0.1 partial-credit floor masks tool-use signal** ([RESULTS_m0_a.md §11.4](../report/RESULTS_m0_a.md#L472)).
2. **The "no-example" prompt ablation is a pareto improvement** ([RESULTS_m3.md §14](../report/RESULTS_m3.md#L381)).
3. **The "shrink-and-improve" RL-training regime for multi-hop QA + retrieval** ([RESULTS_m5.md §4.1](../report/RESULTS_m5.md#L61)).

Together they earn the small-scale regime as a legitimate scope for the paper rather than an apology. Each is cited with the exact file:line below.

---

## Finding 1 — The 0.1 partial-credit floor masks tool-use signal

**Source**: [`RESULTS_m0_a.md` §11.4](../report/RESULTS_m0_a.md#L472) ("The 0.1 partial-credit floor masks the tool-use signal"); supporting evidence in [`RESULTS_m0_a.md` §7](../report/RESULTS_m0_a.md#L367) ("The 0.1 floor explains the 0.18-0.22 reward band") and [`RESULTS_m0_a.md` §10 cross-run table](../report/RESULTS_m0_a.md#L436).

### Claim

In the ReSearch paper-faithful 3-tier reward (`0` if format broken, `0.1` if format-OK but F1=0, `F1` if F1>0), the 0.1 floor pulls all training runs into a 0.16-0.22 band regardless of whether the model uses the search tool. Tool-using and no-tool policies separate by **only 3-6 pp of reward** while the prompt-design lever moves end-of-run behaviour from 0 to 2 tool calls and 480 to 2050 token responses (9 pp behavioural swing) — roughly a **2× to 3× signal-to-noise gap** between *what the model is doing* and *what the reward records*.

### Numbers (from the cross-run table at [`RESULTS_m0_a.md` §10](../report/RESULTS_m0_a.md#L436))

| Run | End-of-run reward | Tool calls (last decile) | Response length |
|---|---:|---:|---:|
| `p1_basic_w_ex` (heavy-tool, 2 calls / 4 turns) | 0.190 | 2.00 | 2047 |
| `p1_basic_no_ex` (tool collapses to 0.08) | 0.169 | 0.08 | 478 |
| `p2_basic2_no_ex` (total tool collapse) | 0.159 | 0.00 | 640 |
| `p3_decide_no_ex` (best — tool survives example removal) | 0.215 | 1.00 | 1117 |

Reward gap between *total-tool-collapse* (`p2_basic2_no_ex`) and *heavy-tool* (`p1_basic_w_ex`): 0.190 − 0.159 = **+3.1 pp** despite the rollout-behaviour signature being almost diametrically opposite. The 0.1 floor is the load-bearing knob.

### Mechanism (from §7 of `RESULTS_m0_a.md`)

A model that emits a well-formatted-but-wrong answer sits at 0.1 baseline; end-of-run means of 0.16-0.22 imply only 6-12 % of rollouts get a non-zero F1 hit on top. For a 0.6B model on MuSiQue (hard multi-hop), this is plausible. The implication: a flat 0.18 mean reward is **not** training failure; it is the partial-credit reward saturating a small model. Any tool-use-driven signal lives in the 3-6 pp residual above the floor.

### Limit of the claim

- Measured across single-seed runs with **different prompts**. The 3-6 pp gap is read across heterogeneous prompts with confounded changes (rules section + example + tag schema all varied). It is **not** a clean A/B test of {tool-allowed, tool-disallowed} on the same prompt.
- Phase-1 was paper-faithful reward only. There is **no Phase-1 ablation of {F1+0.1, F1-only, EM-only}**. The "floor masks signal" claim is an *observation*, not an ablation result.

### What makes this publishable

ReSearch never ablates the 0.1 floor — the [paper note §"Takeaways for us"](../papers/2503.19470_research.md#L77) calls this "the load-bearing detail [...] presented in the paper without comment; it is the most ablation-worthy line in the loss." This finding both motivates and frames the **reward-shape ablation** as M6 candidate experiment C (see [`MILESTONE_6.md` §"Phase 2"](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15)).

---

## Finding 2 — The no-example + decision-rules prompt is a pareto improvement

**Source**: training-side prompt ablation in [`RESULTS_m0_a.md` §8 (paired-comparison table)](../report/RESULTS_m0_a.md#L381) + [`RESULTS_m0_a.md` §11.3](../report/RESULTS_m0_a.md#L467); held-out confirmation in [`RESULTS_m3.md` §14.4-14.7](../report/RESULTS_m3.md#L381).

### Claim

In the Phase-1 v0 prompt sweep, three paired comparisons (`p1_basic` / `p1_basic_no_ex`, `p2_basic2` / `p2_basic2_no_ex`, `p3_decide` / `p3_decide_no_ex`) hold rules section fixed and toggle the few-shot example. With 2-sentence rules sections, removing the example collapses tool-use (tool calls 0.08 / 0.00; reward floor). With a 4-sentence rules section containing per-step decision guidance ("decide whether another search is needed" / "if a search result is incomplete, search again"), removing the example **preserves tool-use and produces the best reward of the block (0.215)**. The pareto-dominant variant uses *half* the response budget and one fewer tool call than the heavy-tool with-example variant.

### Numbers

**Training-side (end-of-run, single seed, Qwen3-0.6B on MuSiQue)**:

| Pair | Rules | With example | Without example | Survives `_no_ex`? |
|---|---|---|---|---|
| `p1_basic` | 2 sentences | reward 0.190, 2.00 calls, 2047 tok | reward 0.169, 0.08 calls, 478 tok | **No, collapses** |
| `p2_basic2` | 2 sentences | reward 0.189, 1.00 calls, 1320 tok | reward 0.159, 0.00 calls, 640 tok | **No, collapses** |
| `p3_decide` | 4 sentences + decision guidance | reward 0.190, 1.00 calls, 1150 tok | reward **0.215**, 1.00 calls, 1117 tok | **Yes; best reward** |

**Held-out (M3.1, 51,713 items / variant, greedy decode, full 7-dataset Plan A; from [`RESULTS_m3.md` §14.4-14.5](../report/RESULTS_m3.md#L381))**:

| Variant | End-of-train reward | Held-out EM (simple mean) | Held-out ACC | Held-out F1 |
|---|---:|---:|---:|---:|
| `p1_basic_w_ex_z7kcxfof` (M3 baseline) | 0.190 | 0.155 | (per §14.5) | (per §14.5) |
| `p3_decide_no_ex_el6s2d2h` (M3.1 challenger) | 0.215 | **0.169** | +12 % rel | +14 % rel |
| Δ | +0.025 (+13 % rel) | +0.014 (+9 % rel) | +12 % | +14 % |

ACC and F1 widen the gap relative to EM — consistent with the no-example variant producing higher-quality answers that don't always satisfy EM's strict match. Wins concentrate on PopQA, TriviaQA, HotpotQA (predicted by the hypothesis); Bamboogle regresses (the one anti-result; n=125 sample noise per [archive postmortem](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md)).

### Mechanism (from §8 of `RESULTS_m0_a.md` and §14.6 of `RESULTS_m3.md`)

The four-sentence decision-rules section gives the model an *explicit per-step decision protocol* ("after each search result, decide whether another search is needed"). This substitutes for the example anchor that weak rules sections rely on. The pareto win on response budget is *because* the no-example variant doesn't imprint on the example's hop count (the Hamlet example always does 2 searches), so it adopts the minimum tool use that the rules + task require.

### Limit of the claim

- **Single-seed**, both training and held-out. Bamboogle (n=125) regression flagged in [`RESULTS_m3.md` §14.6](../report/RESULTS_m3.md#L412) as not denting the structural conclusion but worth a second seed.
- **Only Qwen3-0.6B**. Cross-family transfer to Qwen3.5-0.8B is untested; M4 ran a different prompt-mode lock (`qwen35_minimal`/`qwen35_minimal_no_system`) per [`RESULTS_m4.md` §2](../report/RESULTS_m4.md#L24), not a re-run of the v0 prompt sweep.
- **EM gap of +0.014 (+9 % rel) is modest** in absolute terms. The ACC / F1 widening (+12 % / +14 %) is the load-bearing evidence that the finding is not partial-credit-floor inflation.

### What makes this publishable

This is the strongest standalone finding the project has. The mechanism is named ("decision-rules scaffolding substitutes for in-context example"), the comparison is paired (single variable changed in three pairs), the held-out confirmation is at full Plan A scale (51,713 items), and the **direction of the lever runs against the small-model literature's instinct** to add few-shot examples. Frame it as the "prompt-design dominates reward differences at this scale" motivation paragraph.

---

## Finding 3 — Multi-hop tool-use produces a "shrink-and-improve" RL regime

**Source**: [`RESULTS_m5.md` §4.1](../report/RESULTS_m5.md#L61) ("Transferable observation — RL-training dynamic regimes").

### Claim

GRPO on multi-hop QA + retrieval (Search-R1 / ReSearch family) produces a training-dynamic signature that **inverts the long-CoT regime**: rollout length and tool calls *decrease* while reward *increases*. M5.1's first 16 steps (~5 % of the run): `tc_mean` 8.96 → 3.47, `tok_mean` 7038 → 2183 (3.2× compression), reward 0.020 → 0.132 peak, truncation rate 68.4 % → 0 %. Length and tool-call count converge to *task complexity* (~3-4 hops for MuSiQue), not unbounded growth. This is a transferable observation independent of M5.1's final outcome.

### Numbers (from [`RESULTS_m5.md` §4.1](../report/RESULTS_m5.md#L61))

| Step | Wall-clock | reward / mean | tc_mean | tok_mean | truncation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.020 | 8.96 | 7038 | 68.4 % |
| 16 | (M5.1 live) | 0.132 (peak) | 3.47 | 2183 | 0 % |
| Δ | | 6.6× | −5.5 calls | 3.2× compression | full clear |

Phase shares stable across the drop (training ~70 %, logprobs ~20 %, generation ~8 %); the per-step wall-clock fell 57.9 → 21.3 min over 10 steps, with absolute wall dropping proportionally from each phase as rollouts shorten.

### Mechanism (from §4.1)

The model "shrinks away" failure modes — rambling `<think>` blocks, literal function-placeholder copies of the prompt's tool-call template, hitting `max_turns`. The reward function (F1-only on `<answer>X</answer>` in M5.1; partial-credit + F1 in the paper) gives no credit for length or call count, so the policy unwinds these without sacrificing the answer signal. Tool calls converge to the *true hop count* of the task distribution, not the maximum allowed.

### Limit of the claim

- **One run, one task, one model.** The signature is a transferable *observation* (consistent with the ReSearch / Search-R1 training-curve shape published in their papers), not a paired comparison against a long-CoT regime on equivalent compute.
- **Mid-training snapshot.** Final M5.1 numbers not yet available. The "regime" label is supported by the trajectory through step 16; full-run analysis pending.
- **No baseline that *doesn't* shrink.** To call this a *signature*, a paired comparison against a long-CoT-regime run on equivalent compute would be required.

### What makes this publishable

The cleanest version of this finding is not a contribution claim but a **characterisation** that anchors the introduction: *the regime we are in has a specific dynamic that small-rollout, retrieval-induced reward variance changes the trade-offs of, motivating the picked-pair experiments*. The contrast with the long-CoT regime is the wedge that earns small-scale RAG-tool-use a section of its own.

---

## How these three findings ladder into the paper

The intended use in the M6-picked-pair paper's introduction (single paragraph):

> *At sub-1B scale on multi-hop retrieval-augmented QA, three observations from our Phase-1 prompt sweep and M5.1 training run shape what the next ablation should target. First, the ReSearch partial-credit reward floor (`+0.1` if format-OK but F1=0) collapses tool-using and no-tool policies into a 3-6 pp band of training reward, while the prompt-design lever moves rollout behaviour by ~9 pp [Finding 1]. Second, in a controlled three-pair example-ablation sweep, removing the few-shot example pareto-dominates the with-example variant when the rules section provides explicit per-step decision guidance: held-out EM rises +9 % rel (ACC +12 %, F1 +14 %) at half the response budget [Finding 2]. Third, the GRPO training dynamic on multi-hop retrieval is a "shrink-and-improve" regime — rollout length and tool-call count compress 3× toward task complexity as reward grows, inverting the long-CoT regime familiar from math reasoning [Finding 3]. Building on these three observations, we ablate (i) the partial-credit floor under F1-only and EM-only reward variants and (ii) median-baseline GRPO under retrieval-induced reward variance, and find [...].*

This paragraph: (a) names the regime, (b) earns small-scale as the scope, (c) makes the two M6-picked experiments mechanism-driven rather than method-transfer, (d) cites only the project's own measurements, not the literature.

## Cross-references

- Critical review that produced this salvage: see [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review).
- Companion data audit (what is *not* in the data): [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md).
- Original results docs (frozen): [`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md), [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md), [`../report/RESULTS_m4.md`](../report/RESULTS_m4.md), [`../report/RESULTS_m5.md`](../report/RESULTS_m5.md).
- Source-of-truth paper notes: [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md) (ReSearch), [`../papers/2503.09516_search-r1.md`](../papers/2503.09516_search-r1.md) (Search-R1).
- Bamboogle n=125 noise postmortem: [`../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md).
