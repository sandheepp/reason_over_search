---
title: Recipes catalog
tags: [recipe, rl, training, living-doc]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Recipes catalog

Living catalog of RL+search training recipes we have encountered, tested, or plan to test. For each recipe: status, what the algorithm actually does, what is good/bad about it, and exactly what happened when we ran it. Cross-links to paper notes and result files.

Companion documents:
- [RECIPE_COMPARISON.md](RECIPE_COMPARISON.md) - side-by-side table of all three source papers (static snapshot, grounded against paper text).
- Paper notes: [Search-R1](../papers/2503.09516_search-r1.md), [R1-Searcher](../papers/2503.05592_r1-searcher.md), [ReSearch](../papers/2503.19470_research.md).

## How to add a recipe

Copy the template at the bottom of this file; fill in status and every field you can from the paper/code. Leave fields blank rather than guessing. Log the addition in [`../log.md`](../log.md).

---

## Search-R1 shape

**Source paper**: [Search-R1, arXiv 2503.09516v5](../papers/2503.09516_search-r1.md)
**Status**: M1 eval-only (published GRPO checkpoints reproduced) + Phase-2 NeMo-RL training port in progress (smoke-tested).

### What it is

GRPO (or PPO) on a Qwen2.5 model with:
- Tag schema: `<search>...</search>` / `<information>...</information>` / `<answer>...</answer>`.
- Reward: outcome-only EM. Strictly binary: 0 or 1. No format reward, no partial credit.
- Retrieved-token loss masking: yes (the paper's headline ablation; §5.4 reports ~25% avg relative gain from masking).
- Train data: NQ + HotpotQA merged.
- Retriever: E5, top-k=3, Wikipedia 2018.
- Scale (paper): 8× H100, 500 steps, batch=512, max-seq=4096.

### What is good

- **Cleanest reward signal of the three papers.** Binary EM means every non-zero reward is a real answer, not a format consolation prize. The 0.1 floor issue (see Learnings L1) does not exist here.
- **Retrieved-token masking is actually ablated.** Other papers assert it; only Search-R1 measures it (~25% relative gain). If you are deciding whether to implement it, this is the paper to read.
- **Shortest/cheapest training shape**: 500 steps, 4k max seq, batch 512 on 8 H100. Fits closer to our budget than the others.
- **Broadest eval coverage** (7 benchmarks vs 4 for the others). The only paper where you can compare NQ + TriviaQA + PopQA alongside the multi-hop benchmarks.

### What is bad

- **Instruct model performance is lower than base on NQ** (0.397 vs 0.421). Unexplained in the paper; worth investigating before choosing the base/instruct split.
- **PPO vs GRPO** comparison is in §5.1 but the headline GRPO group size G is not clearly stated in the sections we extracted (ablated as 1, 3, 5 in Appendix H). Cannot confirm what G the paper-published checkpoints used without reading the appendix directly.
- **Training data mix is NQ + HotpotQA only** (single + multi-hop). Generalises to 5 OOD benchmarks, but training diversity is narrower than R1-Searcher's difficulty-bucketed approach.

### Our results

**M1 evaluation (published checkpoints, Plan B v1 sweep, 2026-05-05):**

| Variant | Our avg EM | Paper avg EM | Delta |
|---|---:|---:|---:|
| GRPO Base (3B) | 0.292 | 0.312 | -2.0 pp |
| GRPO Instruct (3B) | 0.361 | 0.336 | +2.5 pp |

Within ±2.5 pp paper average. Full benchmark breakdown: [`../report/RESULTS_m1.md`](../report/RESULTS_m1.md).

**Phase-2 smoke test (NeMo-RL port, Qwen3.5-2B, 1× A100 80GB, 2026-05-06):**

| Combo | Mean reward (8 steps) | EM hits / 320 traj | Avg retrieval calls |
|---|---:|---:|---:|
| base × paper template | 0.085 | 2 / 320 | 0.2 |
| hybrid × paper template | 0.059 | 0 / 320 | 0.8 |

Smoke is mechanics-only (8 steps); no convergence signal. Retrieval call rate is low: hybrid × paper at 0.8 calls/traj means the model is not reliably invoking search. Full trace: [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md).

### Learnings in our setting

- M1 reproduction gap on base model (-2.0 pp) was traced to three config divergences (D1: `apply_chat=True` for base; D-prompt-micro: example sentence restored; D8: `add_special_tokens` block removed). Details: [`../eval/REPRODUCIBILITY.md`](../eval/REPRODUCIBILITY.md).
- Phase-2 retrieval call rate is low at smoke scale. Not yet clear if this is a cold-start issue (too few steps) or a prompt mismatch.

---

## ReSearch recipe

**Source paper**: [ReSearch, arXiv 2503.19470v3, NeurIPS 2025](../papers/2503.19470_research.md)
**Status**: Phase-1 training tested (29 runs, Qwen3-0.6B on ALICE). **Closed block.** Results: [`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md) and [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md).

### What it is

GRPO on a Qwen2.5 model with:
- Tag schema: `<think>...</think>` / `<search>...</search>` / `<result>...</result>` + `\boxed{}` for final answer.
- Reward: F1 between predicted and gold when F1 > 0; **0.1 when F1 = 0 but format is correct; 0 otherwise.** The 0.1 partial-credit component is the critical detail.
- Retrieved-token loss masking: yes (asserted; not ablated in paper).
- Train data: MuSiQue training set only (19,938 samples).
- Retriever: E5-base-v2, top-k=5, Wikipedia 2018 (FlashRAG).
- Scale (paper): 64× H800, 2 epochs, batch=256, G=5.

Our Phase-1 implementation used the `re_search` reward manager from the Search-R1 codebase (paper-faithful shape), GRPO (`kl_coef=0.001`), on Qwen3-0.6B instruct, ALICE 1× A100-40GB, batch=4, G=3. Full config: [`../report/RESULTS_m0_a.md §2`](../report/RESULTS_m0_a.md).

In v1 we swapped the tag schema to `<tool_call>{...JSON...}</tool_call>` + `<tool_response>` (in-distribution format for Qwen3-0.6B instruct) while keeping the rest of the recipe unchanged.

### What is good

- **Strong OOD generalisation on one training dataset.** Trained only on MuSiQue; generalises to HotpotQA and 2Wiki (which are different question types). This is the cleanest evidence in the literature that outcome-RL generalises from a small curated set.
- **Both base and instruct models work** (at 7B/32B). At our 0.6B scale, only instruct worked.
- **Accepted at NeurIPS 2025.** Highest peer-review bar of the three papers; results are likely the most carefully validated.

### What is bad

- **The 0.1 partial-credit reward.** When F1 = 0 but the format is correct, the model gets a reward of 0.1. At Qwen3-0.6B scale, this created a noise floor: the model could maintain reward ~0.1 simply by producing well-formatted non-answers, masking whether it was learning to use the retriever. We measured a **3-6 pp gap** between runs that used the tool and those that didn't, all in the 0.14-0.22 band (see L1 below).
- **64× H800 compute.** Not reproducible at our scale without accepting a much longer wall-clock. Their 2-epoch run over 20k samples at G=5 on 64 H800 is the single biggest compute gap between any of the three papers and our setup.
- **Retrieved-token masking not ablated.** The paper asserts it but never measures it. We cannot know how much it matters from this paper.

### Our results

**v0 (W&B `research`, 14 focus runs, Qwen3-0.6B instruct, ALICE):**

Best run: `p3_decide_no_ex` at reward 0.215 (2280 steps). All 9 convergent runs cluster at reward 0.16-0.22. Single biggest behavioral lever: whether the prompt contains a few-shot example. Stripping the example from weak rule-sets collapses tool use to 0.00. Full table: [`../report/RESULTS_m0_a.md §1`](../report/RESULTS_m0_a.md).

**v1 (W&B `research_revamp`, 15 runs, Qwen3-0.6B instruct + base, ALICE):**

Instruct runs with new `<tool_call>` tag schema: reward 0.138-0.179 (slightly lower than v0; smaller prompt + smaller batch). Base-model runs: **5/5 failed**, `tool_call_counts/mean` stuck at 0.00 throughout, longest run 2301 steps. `base_breakthrough` (run `b8vv0qe2`) showed reward 0.700 but with 0 tool calls; traced to a reward-function code edit between 2026-04-17 and 2026-04-18 and treated as instrumented artifact, not learning. Full table: [`../report/RESULTS_m0_b.md §1`](../report/RESULTS_m0_b.md).

### Learnings in our setting

- See L1 (0.1 floor), L2 (base cold-start), L3 (tag schema interchangeable), L4 (prompt drives behaviour), L5 (base_breakthrough artifact) below.

---

## R1-Searcher recipe (2-stage curriculum)

**Source paper**: [R1-Searcher, arXiv 2503.05592v2](../papers/2503.05592_r1-searcher.md)
**Status**: referenced only; not tested. No R1-Searcher-derived training runs exist in this project.

### What it is

Reinforce++ on Qwen2.5-7B-Base / Llama-3.1-8B-Instruct with:
- Tag schema: `<begin_of_query>...</end_of_query>` / `<begin_of_documents>...</end_of_documents>`.
- Two-stage curriculum:
  - Stage 1 (350 samples): reward = 0.5 × (called retriever) + 0.5 × (correct format). Forces the model to use the tool before learning the answer.
  - Stage 2 (8,148 samples): reward = F1 + format. Format penalty is asymmetric: `+0` if correct, `-2` if wrong. Prevents format-collapse late in training.
- RAG-rollout: generation pauses at end-of-query tag; retriever is called; result is inserted before generation continues.
- Retrieved-token loss masking (retrieve-mask loss): yes (asserted; not ablated).
- Train data: HotpotQA + 2Wiki, difficulty-bucketed (easy/medium/hard by rollout count to solve). Hard examples are load-bearing.
- Retriever: BGE-large-en-v1.5, KILT 2019 Wikipedia.
- Scale: 8 GPUs (type not stated), G=16, max-gen=29000.

### What is good

- **Two-stage curriculum solves the cold-start problem.** Stage 1 uses a retrieval reward to force the model to call the tool at all, before Stage 2 switches to answer quality. This directly addresses the failure mode we hit: Qwen3-0.6B base at 0 tool calls for 2301 steps (L2 below). Stage 1 is cheap (350 samples), making it a near-zero-cost drop-in.
- **Asymmetric format penalty** (`+0 / -2`) is more targeted than a format bonus: it only activates when the model breaks format, which is a different failure mode from "correct format but wrong answer". Prevents the format-reward gaming we expect if we add any non-zero format reward.
- **`init_kl_coef=0.0` in script**: no KL anchor on the reference model. The paper trains stably without it. Removing KL frees the reference-model GPU allocation; potentially useful on 1× A100 to reduce memory pressure.
- **Hard examples are explicitly included.** Paper ablates difficulty bucketing: hard examples (>20 rollouts to solve) drove a 3.4 pp CEM gain. Easy examples are removed. This is a principled data selection strategy, not just "use all the data".

### What is bad

- **G=16** rollouts per prompt. Roughly 3x ReSearch and at least 3x what Search-R1 tests. At our 1× A100 budget this dominates rollout time.
- **generate_max_len=29000** is ~4x ReSearch and ~7x Search-R1. Not clear from the paper why this length is needed; at our scale it would push well outside our 4k-ish budget.
- **Retriever is BGE-large-en-v1.5** (different from E5 used by Search-R1 and ReSearch). Switching retrievers mid-project requires re-verifying the retrieval quality; not worth it.
- **Algorithm is Reinforce++, not GRPO.** Our Phase-2 NeMo-RL pipeline is already wired for GRPO (the Search-R1 shape). Switching algorithms requires code changes. The two-stage curriculum is the borrow-worthy part; the algorithm is not.
- **Wall-clock not stated, GPU type not stated.** Cannot compare directly to our budget.

### Borrow-worthy tricks (in priority order)

1. **Stage-1 retrieval reward** (350 samples, 0.5 × tool-use + 0.5 × format). Add as a warm-up phase before the main GRPO run on the Search-R1 shape. Targets L2 (base cold-start).
2. **Asymmetric format penalty** (`+0 / -2`) in Stage 2. Cheaper and more targeted than a format bonus.
3. **Hard-only data selection**. Remove easy examples (those solvable without retrieval or with <10 rollouts) from the training set. Cheap data-side win.
4. **`init_kl_coef=0.0`**. Ablate once we have a stable training run. Compute win more than algorithmic.

### Our results

None yet. If this recipe gets tested, create a new results file in `docs/training/` (or `docs/archive/training/` if discarded) and link it here.

---

## E2H curriculum (Easy-to-Hard data scheduler)

**Source paper**: [E2H Reasoner, arXiv 2506.06632, ICLR 2026](../papers/2506.06632_e2h.md)
**Status**: planned. Item #5 in the active ablation list ([`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)). No runs yet.

### What it is

A **data-side curriculum** that stacks on top of any GRPO (or PPO) recipe; it does not modify the algorithm, only the dataloader's sampling distribution.

- Tag schema, reward, retriever: inherited from the base recipe (in our case the Search-R1 shape).
- Curriculum mechanics:
  - Partition training tasks by difficulty into K levels (k = 0 easiest to k = K-1 hardest).
  - At each step t, sample from level k with probability given by a schedule that walks from "all easy" toward "all hard" over T total steps.
  - Two schedule families: **E2H-Cosine** (cosine interpolation between levels) and **E2H-Gaussian** (Gaussian mixture with progression speed `β` and concentration `σ`).
  - Best paper config: **E2H-Gaussian, β = 0.5, σ = 0.5**.
- Difficulty proxy:
  - Paper: zero-shot pass-rate (MATH, Countdown) or plan length (Blocksworld).
  - Our extension to retrieval QA: hop count. NQ = 0 (1-hop), HotpotQA = 1 (2-hop), MuSiQue = 2 (2 to 4 hop). The other 4 benchmarks stay held-out OOD. Mapping detail: [`DATASET_ANALYSIS.md`](DATASET_ANALYSIS.md).
- Scale (paper, our reading): Qwen 1.5B primary; GRPO with G = 8, KL `β = 0.001`, lr 1e-6, cosine LR. GPU count and total steps not stated.

Implementation cost on top of an existing GRPO trainer: a single dataloader change. The KL term, reward, advantage, and update rule are unchanged.

### What is good

- **Drop-in**: no GRPO code change; only the data sampler. Our NeMo-RL pipeline can adopt it by swapping the dataset adapter.
- **Targets the cold-start regime**: vanilla GRPO at 1.5B to 3B fails on hard reasoning because reward is too sparse (zero advantages across most groups). Easy-first sampling produces non-zero advantage signal early, then progressively hardens. This is structurally the same problem we hit at 0.6B (L2 below): Qwen3-0.6B base never emitted `<tool_call>` cold. E2H is a candidate fix; R1-Searcher Stage-1 is the other (RECIPES.md §"R1-Searcher").
- **Largest paper gains where the easy-to-hard gap is largest**: Blocksworld +11.8 pp, Countdown +22.9 pp on Qwen 1.5B hard-split accuracy vs balanced-sampling GRPO. Hop count for retrieval QA is a similarly crisp ladder (1 / 2 / 2 to 4), so the mechanism should transfer.
- **Stage-based approximation is cheap to test first**: ~300 steps per stage with manual fade is implementable in an hour and gets most of the gain without the schedule-curve plumbing. Our supervisor-meeting brief proposes this version ([`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md` § 6](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md)).
- **Has a theoretical backing**: finite-sample policy-iteration analysis showing curriculum reduces total sample requirement. Most curriculum papers in the RL literature are pure empirical.

### What is bad

- **No retrieval-augmented evaluation in the paper**. All experiments are closed-book MATH / Countdown / Blocksworld. Applying E2H to retrieval-augmented multi-hop QA with live search calls in the GRPO rollout is a thesis novelty; convergence rates and pp gains may differ. This is the single biggest validity gap.
- **Difficulty proxy is domain-specific**. Their zero-shot pass-rate proxy doesn't exist for retrieval QA without an extra eval pass per training example. Hop count is a coarser substitute and may not match model-perceived difficulty (e.g. a long-tail PopQA entity can be harder for a 2B model than a clean 2-hop HotpotQA bridge question).
- **Fade rate is non-trivial**. Paper's central operational insight: easy tasks must be reduced over time, otherwise they dominate and block hard-task learning. Getting fade right at our T = 500 to 1000 steps (vs paper's larger T) is unverified; the schedule curve at our step count may not fully transition.
- **Per-step cost grows with the curriculum**. As MuSiQue (longer chains, more search calls per rollout) becomes the dominant sample late in training, per-step time increases. Wall-clock budgeting must account for this; E2H is not free at our scale even though it's free in the paper's closed-book setting.
- **Compute claim is light**. Paper does not state total wall-clock or GPU type explicitly in our reading. Treat reproduction-cost claims with caution; we have to budget against our own per-step measurements, not the paper's.
- **Validated only at 1.5B**. Effect at our 2B scale is extrapolated, not measured. Plausible but not confirmed.
- **JustRL counter-evidence applies**. Per [arxiv 2512.16649](https://arxiv.org/abs/2512.16649), curriculum stacks can hurt OOD performance; format / partial-credit rewards and curricula degrade the advantage signal in some settings. The Phase-2 ablation plan therefore runs a plain-GRPO C-minimal control alongside the stack ([`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Active ablation plan"); if C-minimal beats the E2H run on val EM, the curriculum is hurting and we strip it.

### Borrow-worthy tricks (in priority order)

1. **Hop-count difficulty bucketing**: NQ → HotpotQA → MuSiQue, with the other 4 benchmarks held out. Cheap (one-line dataset filter) and the mapping is grounded in the v1 base EM gradient (0.390 / 0.263 / 0.055).
2. **Stage-based fade** (3 stages × ~300 steps with manual mixture weights). First-cut implementation; cheaper than the continuous Gaussian and likely captures most of the gain.
3. **E2H-Gaussian with paper defaults** (`β = 0.5, σ = 0.5`). Second-cut; only worth the implementation cost if the stage-based version shows a positive signal vs C-minimal.
4. **Cosine schedule as fallback**: simpler than Gaussian, slightly weaker in paper. Use only if Gaussian implementation hits friction.

### Our results

None yet. Phase-2 ablation #5 will run E2H curriculum on top of the Search-R1 GRPO baseline (with MC-GRPO + S-GRPO already stacked); see the table at [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Active ablation plan". When run, file results in `docs/training/` and link from here.

### Learnings in our setting

None yet. The relevant cross-cutting findings to validate against are L2 (base cold-start: E2H is a candidate fix) and L6 (JustRL counter-evidence: stacks can hurt; need the C-minimal control).

---

## Recipe template

```
## [Recipe name]

**Source paper**: [Title](../papers/<arxiv-id>_<slug>.md)
**Status**: tested / referenced / planned

### What it is

Algorithm; tag schema; reward; data; retriever; scale.

### What is good

### What is bad

### Our results

[Link to result file, run count, hardware, dates, key numbers.]

### Learnings in our setting
```

---

## Cross-cutting learnings

These are findings from our training runs that apply across recipes or explain the gap between paper claims and our observations. Each learning should be filed back to the source result doc.

### L1: 0.1 partial-credit reward creates a noise floor

**Source**: Phase-1 v0/v1 runs following the ReSearch-shaped reward (`re_search` reward manager).
**What happened**: The reward function returns 0.1 when F1 = 0 but the response is well-formatted. At Qwen3-0.6B scale, the model converged to producing plausible-looking but wrong answers, maintaining reward ~0.10-0.14. Tool-using and non-tool-using runs clustered in the same 0.14-0.22 band, making it impossible to distinguish learning from format gaming.
**Measured gap**: 3-6 pp between runs that used the tool and those that did not, at the same reward level.
**Implication**: the reward signal cannot cleanly assign credit to tool use. Fix: use outcome-only binary EM (as Search-R1 does) and remove any partial-credit component.
**Filed in**: [`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md); flagged as the most actionable lever in [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Gotchas".

### L2: Base model fails cold-start without warm-up

**Source**: Phase-1 v1 block, all 5 base runs.
**What happened**: `Qwen3-0.6B-Base` never emitted `<tool_call>` tokens across all 5 runs (longest: 2301 steps). `tool_call_counts/mean` stayed at 0.00 throughout. The base model's prior over the tag tokens is not strong enough for GRPO to lift from zero.
**Attempted mitigations in v1**: varied prompt (state-machine, with-example, without-example); none worked.
**Implication**: at this scale, base models need either SFT warm-start or a Stage-1 retrieval reward (see R1-Searcher, L1 borrow). Instruct models have a usable prior over tool-call structure and can bootstrap from GRPO alone.
**Filed in**: [`../report/RESULTS_m0_b.md §1`](../report/RESULTS_m0_b.md), [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Gotchas".

### L3: Tag schema does not matter at equal step count

**Source**: Phase-1 v0 (`<search>` / `<result>` paper schema) vs v1 (`<tool_call>` / `<tool_response>` in-distribution schema).
**What happened**: Switching from the paper `<search>` tags to Qwen3's in-distribution `<tool_call>` format produced the same reward range (0.14-0.18 in v1 vs 0.16-0.22 in v0; the small gap is attributable to the smaller v1 batch and shorter prompts, not the tag change). Tool-use counts were equivalent.
**Implication**: the model learns structural patterns, not specific token strings. Do not over-engineer the tag choice; match whatever the target model's in-distribution format is, for minimal cold-start friction.
**Filed in**: [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"What's been done", finding 5.

### L4: Prompt content drives early behaviour more than reward shape

**Source**: Phase-1 v0 prompt ablation (9 runs, `run_1_4b` tag).
**What happened**: removing a few-shot example from weak-rule prompts (`p1_basic`, `p2_basic2`) caused tool use to collapse to 0.00-0.08 immediately. The reward function was identical across runs. Stronger rule-sets (`p3_decide`) survived example removal. Best reward (`p3_decide_no_ex`, 0.215) came from a strong rule-set without an example.
**Implication**: in the early phase of training (before GRPO has found a stable policy), the prompt is the primary lever for keeping the model in the tool-use region. Reward shaping is secondary. Invest in the prompt before tuning the reward.
**Filed in**: [`../report/RESULTS_m0_a.md §1`](../report/RESULTS_m0_a.md).

### L5: `base_breakthrough` reward spike is an artifact

**Source**: Phase-1 v1, run `b8vv0qe2` (`base_breakthrough`).
**What happened**: this run showed reward 0.700 at step 2301, identical config to `base_state_machine_a` which finished at 0.0. On inspection: 0 tool calls, ~93-token responses. The reward-function code (`re_search` reward manager) was edited between 2026-04-17 and 2026-04-18. Conclusion: the reward function was relaxed, not the model learning.
**Implication**: reward spikes that are not accompanied by tool-use signal (`tool_call_counts/mean > 0`) and reasonable response lengths should be investigated for reward-function changes before being treated as learning signal.
**Filed in**: [`../report/RESULTS_m0_b.md §1`](../report/RESULTS_m0_b.md), [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Gotchas".

### L6: JustRL counter-evidence (arxiv 2512.16649)

**Source**: literature, not our runs yet.
**Claim**: ["RLVR Is Not RL" (arxiv 2512.16649)](https://arxiv.org/abs/2512.16649) argues that complex reward-shaping tricks can hurt compared to plain GRPO ("JustRL"). The paper's argument is that partial-credit rewards, curriculum, and multi-component reward signals may degrade the advantage signal rather than improve it.
**Implication**: our Phase-2 ablation plan runs a **JustRL plain-GRPO control** alongside the recipe stack (E2H curriculum + S-GRPO + MC-GRPO). If the control beats the stack, the stack is hurting. Per [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Active ablation plan", this is run #2 in the ablation order.
**Status**: planned; no runs yet.

### L7: FAISS flat-IP index times out under training rollout HTTP load

**Source**: Phase-2 smoke test setup.
**What happened**: the flat inner-product FAISS index (used for paper-fidelity eval) timed out when called concurrently from the training rollout HTTP server. The IVF-SQ8 quantised index handles the load.
**Implication**: use IVF-SQ8 for training even though paper-fidelity eval uses flat IP. The two indexes are not interchangeable.
**Filed in**: [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Gotchas", [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md).

### L8: Search-R1 GitHub ships two reward modules

**Source**: code audit during Phase-2 setup.
**What happened**: `PeterGriffinJin/Search-R1` ships `qa_em.py` (paper-faithful EM-only, binary) and `qa_em_format.py` (6-tier shaped reward with non-zero defaults producing visible reward even at EM=0). Our Phase-2 NeMo-RL port uses EM-only (`qa_em.py`). If the wrong module is wired in, the training will appear to converge (positive reward) even with no real learning.
**Implication**: always verify which reward module is wired in before starting a run. Unexpected non-zero rewards at step 0 before any tool-use is a red flag.
**Filed in**: [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) §"Gotchas".
