---
title: PARADIGM REVIEW
tags: []
source: internal
created: 2026-05-03
updated: 2026-05-06
---

# Small-LM RL Training Paradigms — Systematic Review

**Audience**: anyone working on this project (researcher, engineer, or visiting collaborator). Each technique gets a technical description *and* an ELI5 paragraph so the doc reads even if RL post-training isn't your day job.

**Scope**: arXiv 2025-2026 work on RL post-training of small (1-7B) language models, filtered for techniques that fit a single-GPU budget and a search-augmented multi-hop QA task. Output is intended to drive concrete experiment choices on top of [`PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md).

**Status**: v1, drafted 2026-05-03. A v2 deeper-pass (alphaxiv / connected-papers traversal, ablation-number pulls, counter-evidence) is appended below as §10 once the second-round agent returns.

**Companion docs**:
- [`RUNTIME_EFFICIENCY.md`](RUNTIME_EFFICIENCY.md) — systems / engineering levers (vLLM prefix caching, async engine, colocation cost, fused AdamW, dynamic batching, gpu_memory_utilization). Engineering wins compose orthogonally with algorithmic wins.
- [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) — Navigate the three rounds of research (v1 → v2 → v3), decision trees for your specific constraints, failure modes + mitigations, and a complete checklist before running. **Start here if you want a clear path to action.** See this guide for how the v1 recommendations evolved, why, and what changed.

---

## 0. Your setting in one paragraph (the constraint we're optimizing for)

Qwen3.5-2B base (or hybrid with soft-switch reasoning), GRPO via NeMo-RL, leave-one-out advantage, group size G=5, lr=1e-6, β (KL coefficient) = 0.001, ε (clip) = 0.2, 1005 steps × 510 trajectories/step ≈ 515k trajectories per run, pure-EM reward (1.0 / 0.0). Hardware: 1× A100 80GB or 1× H100 80GB SXM. Budget: ~$200-400 per run, ~5-15 days wall-clock. Plan: 3 seeds × 2 variants = 6 runs. Pain points: full-finetune of 2B is tight on 80GB with vLLM colocated for rollout; sequence packing disabled (Qwen3.5 Mamba layers crash); per-step time dominated by rollout (multi-turn search) and the weight-refit between rollout and training mode swaps. See [`SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config) for the per-step measurements this is built on.

---

## 1. Quick primer (ELI5 first, then technical)

### What is GRPO, in plain terms

> **ELI5.** Imagine teaching a kid to answer trivia by giving them 5 tries on each question. After all 5, you tell them which tries got the answer right. The kid then thinks "OK, the things I did on the right ones, I should do more of; the things on the wrong ones, less." That's basically GRPO — Group Relative Policy Optimization. You sample G answers from the model, score each, and the *advantage* of any one answer is just "how much better than the average of this group". No critic, no value network. Cheap. Works.

**Technical.** GRPO replaces PPO's learned value function with a per-prompt empirical baseline (the group mean of rewards across G rollouts of the same prompt). The advantage for trajectory *i* in group *g* is `A_i = (r_i − mean(r_g)) / std(r_g)` (this group-std normalization is one of two things Dr. GRPO challenges — see §3). The policy update is the standard PPO clipped surrogate with KL regularization to a frozen reference policy: `L = E[ min(ratio · A, clip(ratio, 1−ε, 1+ε) · A) ] − β · KL(π || π_ref)`. Group size G is typically 4-8.

### What is RLVR

> **ELI5.** "RL with Verifiable Rewards." Instead of training a separate model to *guess* whether an answer is good (reward model in classic RLHF), you check the answer against a ground-truth: did exact-match hit? did the code pass tests? did the math equation evaluate correctly? Way cheaper, and the reward is honest — no reward-model exploits.

**Technical.** RLVR replaces the learned reward model with a deterministic verifier: string match for QA, unit tests for code, symbolic equality for math, etc. Tulu-3 ([2411.15124](https://arxiv.org/abs/2411.15124)) popularized the framing; DeepSeek-R1-Zero ([2501.12948](https://arxiv.org/pdf/2501.12948)) showed RLVR alone (no SFT, no preference data) can teach long-chain reasoning. Search-R1 is RLVR (the "verifier" is normalized exact match against gold answers).

### What "trajectory" means here

> **ELI5.** One full attempt at answering a question. Not a token, not a sentence — the whole thing from "user asks" to model's final `<answer>` tag. If the model called `<search>` 3 times and then answered, that's one trajectory with 3 search turns.

**Technical.** A sequence of (state, action) pairs: prompt → model tokens → optional tool call → tool response → more model tokens → ... → final answer. For Search-R1, max 4 search turns, max 4096 total tokens. 510 trajectories per step = 102 prompts × 5 group rollouts.

---

## 2. TL;DR — top 5 changes ranked by upside × ease

| # | Change | Source | Upside | Effort |
|---|---|---|---|---|
| 1 | **Drop KL term entirely (β=0)** | DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)), Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)) | Frees ~4-6 GB VRAM (no ref policy in mem during update), small speed win, no quality loss reported on rule-verified rewards at 1.5-7B | 1 config flag |
| 2 | **Hard-example filter the train set** (drop prompts where base model gets 5/5 or 0/5 in a sample) | Hard Examples ([2508.14094](https://arxiv.org/abs/2508.14094)), DAPO dynamic sampling | Up to +47% over random selection at fixed budget | ~0.5 day |
| 3 | **DAPO dynamic sampling** (drop all-correct/all-wrong groups in-batch, oversample to refill) | DAPO | ~50% fewer steps to match perf in DAPO ablation | ~1 day in NeMo-RL |
| 4 | **EAGLE-3 spec-dec for rollout** | NeMo-RL native | 1.41× rollout speedup, zero accuracy delta on 8B reasoning | 1-2 days config |
| 5 | **Rejection-sampled SFT warmup** (1-2 epochs of "the trajectories that hit EM=1") before RL | R1-Searcher ([2503.05592](https://arxiv.org/abs/2503.05592)), RAGEN ([2504.20073](https://arxiv.org/abs/2504.20073)) | Stabilizes early RL, reduces total RL steps ~30-50% | ~1-2 days |

**Skip LoRA** as a primary lever (see §4). **Skip PRMs** (process reward models, see §8).

---

## 3. Removing or replacing the KL term

> **ELI5.** The KL term is the leash that keeps the trained model from drifting too far from the original. Classic intuition: without a leash, the model "hacks" the reward and forgets how to write English. But on rule-verified tasks (right/wrong, no soft preference data), recent papers find the leash is dead weight — it costs you GPU memory (you have to keep a frozen copy of the original model around), it slows the update, and **the reward itself is honest enough that drift is self-correcting**. Modal recommendation in 2025-2026: **drop it.**

**Technical evidence**:

| Paper | Setting | KL choice | Finding |
|---|---|---|---|
| DeepSeek-R1 ([2501.12948](https://arxiv.org/pdf/2501.12948)) | 7B-671B, math/code | β small; R1-Zero variant has no SFT anchor either | Pure RL works without strong KL; long-chain reasoning emerges |
| Qwen2.5-Math ([2409.12122](https://arxiv.org/abs/2409.12122)) | 7B/72B, math RL | β = 1e-3 | Default-not-optimum (same as your current 0.001) |
| Kimi k1.5 ([2501.12599](https://arxiv.org/abs/2501.12599)) | Multi-modal RL | small KL | Stabilizes but not load-bearing |
| **DAPO ([2503.14476](https://arxiv.org/abs/2503.14476))** | Qwen2.5-32B, AIME | **β = 0 explicitly** | Argues KL hurts long-CoT exploration; entropy collapse fixed via clip-higher, not KL |
| **Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783))** | Qwen2.5-Math-7B | β = 0 | SOTA on MATH in 27 GPU-hours, no KL |
| Tina ([2504.15777](https://arxiv.org/abs/2504.15777)) | 1.5B + LoRA | β = 0 | $9 run, 43% AIME |
| REINFORCE++ ([2501.03262](https://arxiv.org/abs/2501.03262)) | 1.5-7B | KL kept, global-batch advantage norm | Argues GRPO group-norm is the instability source, not KL absence |

**Concrete recommendation**: set **β = 0**, take the reference-policy load off-GPU during the update phase. On Qwen3.5-2B bf16 this saves ~4-6 GB. Your current 0.001 already provides almost no regularization — you're paying memory for noise. **Effort: 1 config line. Risk: none observed at 1.5-7B with EM-style rewards in any cited paper.** Caveat: if length-hacking appears (responses grow without quality improvement), apply DAPO's "overlong reward shaping" — a targeted fix that doesn't bring KL back.

---

## 4. LoRA / QLoRA in RL post-training

> **ELI5.** LoRA = "instead of updating all 2 billion weights, train tiny add-on matrices that capture the *delta*". Saves a lot of memory because you only store gradients/optimizer-state for the add-ons. **The catch**: those add-ons have a capacity ceiling — they can only express so much new behavior. For light fine-tunes (style transfer, instruction following), it works great. For deep behavioral changes (learning to reason, learning to use tools), recent results suggest small-rank LoRA can't keep up. Like trying to turn a small car into a truck by adding cargo straps — the underlying chassis is still a car.

**Verdict**: viable as a memory lever but **likely a quality regression for multi-hop search-RL at 2B**. Use only if VRAM still doesn't fit after the §3 stack.

| Paper | Setting | Result |
|---|---|---|
| Tina ([2504.15777](https://arxiv.org/abs/2504.15777)) | DeepSeek-R1-Distill-Qwen-1.5B + LoRA r=4..64 + GRPO | r=16 sweet spot, +20% over base on AIME, $9 cost. **Critically, base was already a distilled-R1 reasoning model** |
| **Plasticity vs Rigidity ([2601.06677](https://arxiv.org/abs/2601.06677))** | ≤1.5B, RLVR + LoRA on a single A40, 24h | r=8 **fails to capture optimization dynamics**; **r=256 needed** for decent results; 40% AIME24 |
| Tulu-3 ([2411.15124](https://arxiv.org/abs/2411.15124)) | Llama-3 8B, RLVR | Full FT used in main; LoRA supported in code but not the headline |

**The capacity-ceiling result** (Plasticity vs Rigidity) is the most relevant: at ≤1.5B, low-rank LoRA *fails entirely* on math reasoning. To recover quality you need r≥256. Tina's success comes from starting from an already-reasoning-distilled checkpoint, where LoRA is fine-tuning behaviors the base already has.

**VRAM math (Qwen3.5-2B, bf16, AdamW)**:
- **Full FT**: 2B × (2 weights + 4 optimizer + 2 grad) ≈ 16 GB params/grads/opt + activations. With vLLM colocated and KV cache, single-A100 80GB is tight but fits when you turn off seq-packing — exactly your situation.
- **LoRA r=16** (target attn + MLP): ~6M trainable, ~0.5 GB optimizer + 4 GB frozen base. **Saves ~10-12 GB**.
- **LoRA r=64**: ~24M trainable, ~2 GB optimizer + 4 GB base = 6-7 GB.
- **QLoRA** (NF4 base + LoRA): base shrinks to ~1.4 GB. Total ~3-4 GB but bf16 forward overhead and ~20-30% slower step. For RL where rollout (full-precision via vLLM) dominates, **trade isn't favorable**.

**Recommendation**: skip LoRA as your first lever. After β=0 + Dr. GRPO + dynamic sampling + spec-dec are landed, if full FT still doesn't fit at desired context, fall back to LoRA r≥64 on attn+MLP. Expect to lose 2-5 EM points relative to full FT based on Tina/Plasticity gaps. **Applicability: medium for VRAM rescue; low for quality.**

---

## 5. GRPO alternatives and simplifications

> **ELI5.** GRPO is good but has known leaks. (1) The "average across the group" baseline is biased when reward variance differs across groups — long answers tend to get dragged down even when correct (Dr. GRPO's finding). (2) When all 5 rollouts in a group score identically (all EM=1 or all EM=0), the gradient is zero — that batch did nothing (DAPO's "dynamic sampling" finding). (3) Per-prompt normalization can over-fit to small data (REINFORCE++'s finding). The 2025-26 follow-ups patch these one by one.

| Algorithm | Paper | 1-line claim | Effort | Verdict |
|---|---|---|---|---|
| **DAPO** | [2503.14476](https://arxiv.org/abs/2503.14476) | Clip-higher + dynamic sampling + token-level loss + overlong shaping; matches DeepSeek-R1-Zero-Qwen-32B in **50% steps** | Moderate (4 distinct changes; verl has a recipe, NeMo-RL doesn't yet). Dynamic sampling alone is ~1 day | **High** — best-validated GRPO++ |
| **Dr. GRPO** | [2503.20783](https://arxiv.org/abs/2503.20783) | Removes length-norm and std-norm bias; fixes "longer-incorrect" pathology | Trivial (~1 day; remove two normalizations) | **High** — pure win, no downside reported |
| **REINFORCE++** | [2501.03262](https://arxiv.org/abs/2501.03262) | Global-batch advantage norm instead of per-prompt group norm; cites GRPO 95%/0% train/test overfit | ~2 days | **High** — your small-data regime is exactly their failure mode |
| **GSPO** | [2507.18071](https://arxiv.org/abs/2507.18071) | Sequence-level importance ratio; backbone of Qwen3 RL | Real port (IS math changes); verl supports | **Medium** — wins are MoE-specific, you have dense Qwen3.5 |
| **RLOO** | [2402.14740](https://arxiv.org/abs/2402.14740) | Leave-one-out baseline | You already use leave-one-out | **Already partially adopted** |
| **RAFT (Minimalist)** | [2504.11343](https://arxiv.org/abs/2504.11343) | Train only on positively-rewarded samples; competitive with GRPO/PPO | Low | Useful as warmup or backup |

> **ELI5 of clip-higher (DAPO)**: classic PPO clip is symmetric — you cut off the gradient if the policy moves too far in *either* direction. DAPO's insight: when you're trying to *grow* low-probability good moves (e.g. exploring a new reasoning style), the upper clip kills you. So they raise the upper clip but keep the lower clip tight. Asymmetric leash.

> **ELI5 of dynamic sampling (DAPO)**: a group of 5 rollouts where all 5 got the answer right is *useless for learning* — the advantage is zero for everyone. Same for all-5-wrong. DAPO drops these batches and over-samples to refill, so every gradient step actually contains learning signal.

> **ELI5 of Dr. GRPO**: GRPO's std normalization makes long correct answers look "less surprising than they should be" because long sequences have higher reward variance. Dr. GRPO removes this and a length-norm — gradient direction unchanged, but no longer biased toward shorter wrong answers.

**Pick to port first**: **Dr. GRPO + DAPO dynamic sampling** combined. Dr. GRPO is ~30 lines. Dynamic sampling is config + an oversample loop. Together they neutralize GRPO's two known pathologies. Run REINFORCE++ as a tiebreak A/B if train-test divergence appears.

---

## 6. Sample-efficient training (<25k rows)

> **ELI5.** RL training is famously data-hungry, but recent work shows that's a myth at small scale on rule-verified tasks. The model can wring a *lot* of signal out of just hundreds of well-chosen examples. The trick is choosing the *right* examples — neither too easy (model already gets them, no signal) nor too hard (model never gets them, also no signal). Examples where the model gets it right *sometimes* — that's where learning happens.

| Paper | Data size | Finding |
|---|---|---|
| **1-Shot RLVR** ([2504.20571](https://arxiv.org/abs/2504.20571)) | **1 example** on Qwen2.5-Math-1.5B | MATH500 36% → 73.6%; matches 1.2k DeepScaleR. "Post-saturation generalization" |
| **Hard Examples** ([2508.14094](https://arxiv.org/abs/2508.14094)) | Top-10% hardest, ~1k examples | +47% over random selection; AIME-2025 OOD wins only on hard-trained |
| Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)) | MATH L3-5 | SOTA in 27 GPU-hours |
| Tina ([2504.15777](https://arxiv.org/abs/2504.15777)) | ~7k DeepScaleR | $9 run, 43% AIME |
| **RAGEN** ([2504.20073](https://arxiv.org/abs/2504.20073)) | Multi-turn agent tasks | Identifies "Echo Trap" — overtraining on small data with low-variance prompts collapses reasoning. Mitigation: filter + diverse init states |

**Curriculum** (easy→hard) is engineering tax for no measurable gain. Hard Examples paper shows just-train-hard wins.

**Pre-filtering** for your Search-R1 mix:
1. Run base Qwen3.5-2B for 5 rollouts on each train prompt before RL starts.
2. Keep prompts with **mixed outcomes** (EM ∈ {1, 2, 3, 4} out of 5).
3. Drop prompts with 5/5 (too easy — gradient is dead) and 0/5 (too hard — gradient is dead).
4. Expect to drop ~30-50% of NQ (single-hop trivia, often too easy or impossible at 2B without retrieval) and ~20% of HotpotQA.
5. Re-filter every ~200 steps as the policy improves (DAPO's dynamic-sampling insight applied at a longer cadence).

**Effort**: 1 day of rollout + ~0 to integrate. **Upside**: ~1.5× effective sample efficiency; same wall-clock budget yields more learning per step.

---

## 7. Rollout-side savings

> **ELI5.** In RL training, the model is constantly running inference (rollout) to generate training samples — this is often more than half the total compute. Anything that speeds up inference without changing the output speeds up training. Speculative decoding is the big one: a tiny draft model proposes 4-8 tokens at once; the big model checks them in a single batched pass. If the draft was right, you got 4-8 tokens for the price of 1.

| Technique | Source | Savings | Effort |
|---|---|---|---|
| **EAGLE-3 spec-dec in NeMo-RL** | Native | **1.41× rollout speedup**, no accuracy delta on 8B reasoning | 1-2 days config |
| **DAS — distribution-aware spec-dec** | [2511.13841](https://arxiv.org/pdf/2511.13841) | Up to **50% generation-time reduction** vs verl baseline, no accuracy delta | Research-grade; verl-only fork |
| **Fully-async rollout/trainer** | verl async recipe | 2.35-2.67× speedup (Qwen2.5-7B, 128 GPUs) | High effort; benefit smaller on single GPU |
| **AReaL** ([2505.24298](https://arxiv.org/pdf/2505.24298)) | Async RL system | Large speedups but multi-node assumption | **Skip** for single-GPU |
| **FlashAttention-3** | vLLM/NeMo-RL native on H100 | ~10-20% throughput | Already on with H100 + recent vLLM |
| **KV-cache reuse across turns** | ReTool ([2504.11536](https://arxiv.org/abs/2504.11536)) | Significant for long multi-turn | Real work (~3-5 days); ReTool's recipe is open |

**Recommendation**: enable EAGLE-3 spec-dec — config switch in NeMo-RL. **Skip async rollout** at single-GPU scale. **KV-cache-reuse-across-turns** (ReTool style) is a real engineering project but specifically targets your "weight refit between rollout and training" pain — worth scoping in v2 if the run cadence justifies it.

---

## 8. Tool-use RL specifically (search, code, calculator)

> **ELI5.** Beyond Search-R1 and ReSearch, the 2025-26 wave of tool-use RL papers converges on a few lessons: (1) format reward (does the model emit the right tags?) helps, but (2) reward for *retrieval quality* (did the retrieved docs contain the answer?) doesn't. (3) Two-stage training (SFT on traces first, then RL) stabilizes everything; cold-start RL works but is brittle on small models. (4) Multi-turn rollouts have their own failure modes — "Echo Trap" (model copies its own previous turn) is real.

| Paper | Tool | Innovation relevant to you |
|---|---|---|
| **R1-Searcher** ([2503.05592](https://arxiv.org/abs/2503.05592)) | Search | Two-stage: SFT cold-start on format → outcome RL. Stable on 7B |
| **R1-Searcher++** ([2505.17005](https://arxiv.org/abs/2505.17005)) | Search + internal-knowledge | Adds bonus for **not searching** when the model already knows; reduces unnecessary tool calls |
| **Search-R1 Empirical Study** ([2505.15117](https://arxiv.org/abs/2505.15117)) | Search | **Format reward helps; intermediate retrieval reward does NOT.** General-purpose base trains more stably than reasoning-specialized base. Stronger search engine improves stability |
| **ReTool** ([2504.11536](https://arxiv.org/abs/2504.11536)) | Code | KV-cache reuse before tool call; outcome-only; matches text-RL with **50% fewer steps** |
| **ToRL** ([2503.23383](https://arxiv.org/abs/2503.23383)) | Code | RL from base, no SFT. 1.5B/7B Qwen2.5-Math; 7B reaches 43% AIME |
| **RAGEN** ([2504.20073](https://arxiv.org/abs/2504.20073)) | Multi-turn agents | StarPO-S: trajectory filtering + critic + gradient stabilization. Identifies "Echo Trap" |
| **Turn-Level Credit Assignment** ([2505.11821](https://arxiv.org/html/2505.11821v1)) | Multi-turn | MDP framing + turn-level advantages + outcome reward |

**Concrete portable techniques**:

1. **Format reward, not retrieval reward**. If you're tempted to add +0.1 for "well-formed `<search>` tag", evidence supports it. **Skip** rewarding "did the retrieved doc contain the answer" — empirically null.

2. **R1-Searcher++ "internal-knowledge" bonus**: small bonus (e.g. +0.05) when the model answers without search and is correct. Reduces over-searching and saves rollout tokens.

3. **ReTool-style KV-cache reuse across turns**: directly targets the "weight-refit between rollout and training" pain. The KV from before `<search>` is recoverable; only tool-output tokens need new attention compute.

4. **RAGEN's StarPO-S filtering**: drop trajectories with reward variance below threshold across the group — same idea as DAPO dynamic sampling but at trajectory level.

**Skip turn-level credit assignment** at 2B. Adds a critic/value net, blows VRAM budget, unclear delta with EM rewards.

---

## 9. Reward design — pure EM vs shaped

> **ELI5.** "Process reward" or "shaped reward" means giving the model partial credit along the way: +0.1 if it called search, +0.2 if it got the format right, +0.5 for actually answering. Sounds intuitive — like grading partial credit on a math test — but in practice on rule-verified tasks at small scale, it routinely backfires. The model learns to game the partial credit (search a lot, format perfectly, never actually answer correctly) instead of getting the answer right. Outcome-only rewards are brutal but honest.

**Strong 2025-26 consensus: keep it pure for verifiable tasks at ≤7B**.

- DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)): outcome-only + 4 algorithmic tweaks beat shaped variants.
- Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)): outcome-only.
- DeepSeek-R1 ([2501.12948](https://arxiv.org/pdf/2501.12948)): rule-based outcome + format only.
- Search-R1 Empirical Study ([2505.15117](https://arxiv.org/abs/2505.15117)): **explicitly tested intermediate retrieval rewards — limited / null impact**. Format reward helps, but it's a hard structural constraint, not a process reward.
- Tulu-3 RLVR ([2411.15124](https://arxiv.org/abs/2411.15124)): verification function, not learned RM. +1.7/3.3/1.3 on MATH/GSM8K/IFEval over DPO baseline.
- PRMs (Process Reward Models, [2502.10325](https://arxiv.org/abs/2502.10325), [2503.21295](https://arxiv.org/abs/2503.21295), [2504.16828](https://arxiv.org/abs/2504.16828)): used for **inference-time guided search** or as a *value function* in PPO, not as RL reward shaping for outcome-verifiable tasks at small scale.

**Recommendation**: stay on pure EM. The one consideration: add a small **format reward** (~0.05) penalizing egregious tag violations (no `<answer>` tag at all). Don't shape mid-trajectory retrieval quality — null evidence.

---

## 10. Skip these (won't help in your setting)

- **PRMs / step-level reward shaping**: 7B+ math results don't transfer to 2B search-QA; cost of training/serving a PRM at single-GPU is prohibitive.
- **Async rollout/trainer split (AReaL, verl async)**: requires multi-machine; single A100 negates the win.
- **GSPO (sequence-level IS)**: main benefit is MoE; your dense 2B doesn't see it.
- **QLoRA**: 4-bit base saves a few GB but slows step ~20-30%; for an RL run dominated by rollout (vLLM uses full-precision copy anyway), the trade isn't favorable.
- **Process reward models for tool use**: Search-R1 empirical study explicitly null result.
- **Curriculum learning (easy→hard)**: Hard Examples paper shows just-train-hard wins.
- **DPO/preference variants on top of GRPO**: orthogonal axis, not memory-saving; skip unless you build a preference dataset.

---

## 11. Suggested experiment plan

You have 3 seeds × 2 variants. Treat the 3 seeds as the noise budget for **one** A/B comparison. Use the 2 variants as your axis.

**Variant A (control)**: current GRPO recipe — β=0.001, group-norm advantages, no filter, no spec-dec.

**Variant B (recommended stack)**:
1. **β = 0** (drop KL term, free ref-policy from GPU)
2. **Dr. GRPO normalizations** (remove length-norm + std-norm in the advantage)
3. **DAPO dynamic sampling** (drop all-1 / all-0 groups, oversample to refill)
4. **EAGLE-3 spec-dec** for rollout

Combined upside estimate: **+2-4 EM points** on HotpotQA (DAPO/R1-Searcher ablation magnitudes) and **~1.4-1.6× wall-clock speedup** from spec-dec + fewer wasted rollouts. Each component is independently reversible by flag if any single one misbehaves. ~3-4 days total port effort.

**Hold for next iteration** (Variant C in v2):
- **Rejection-sampled SFT warmup** before RL — high upside, ~+1 day per run; do it in v2 if Variant B converges but plateaus early.
- **Hard-example pre-filter** — could fold into Variant B; depends on willingness to add a one-off filtering pass before training kicks off.
- **LoRA** only as VRAM-fallback if full FT still doesn't fit after Variant B.

**Do not** make LoRA your first variant. The Plasticity vs Rigidity finding (LoRA r<256 fails on reasoning at ≤1.5B) is the most under-discussed result in this literature.

---

## 12. References

| Paper | arXiv | alphaxiv |
|---|---|---|
| DAPO | [2503.14476](https://arxiv.org/abs/2503.14476) | [link](https://www.alphaxiv.org/abs/2503.14476) |
| DeepSeek-R1 | [2501.12948](https://arxiv.org/pdf/2501.12948) | [link](https://www.alphaxiv.org/abs/2501.12948) |
| Dr. GRPO (Understanding R1-Zero) | [2503.20783](https://arxiv.org/abs/2503.20783) | [link](https://www.alphaxiv.org/abs/2503.20783) |
| Kimi k1.5 | [2501.12599](https://arxiv.org/abs/2501.12599) | [link](https://www.alphaxiv.org/abs/2501.12599) |
| Qwen2.5-Math | [2409.12122](https://arxiv.org/abs/2409.12122) | [link](https://www.alphaxiv.org/abs/2409.12122) |
| Tina (LoRA + GRPO) | [2504.15777](https://arxiv.org/abs/2504.15777) | [link](https://www.alphaxiv.org/abs/2504.15777) |
| Plasticity vs Rigidity | [2601.06677](https://arxiv.org/abs/2601.06677) | [link](https://www.alphaxiv.org/abs/2601.06677) |
| Tulu-3 | [2411.15124](https://arxiv.org/abs/2411.15124) | [link](https://www.alphaxiv.org/abs/2411.15124) |
| REINFORCE++ | [2501.03262](https://arxiv.org/abs/2501.03262) | [link](https://www.alphaxiv.org/abs/2501.03262) |
| RLOO (Back to Basics) | [2402.14740](https://arxiv.org/abs/2402.14740) | [link](https://www.alphaxiv.org/abs/2402.14740) |
| GSPO | [2507.18071](https://arxiv.org/abs/2507.18071) | [link](https://www.alphaxiv.org/abs/2507.18071) |
| Search-R1 | [2503.09516](https://arxiv.org/abs/2503.09516) | [link](https://www.alphaxiv.org/abs/2503.09516) |
| ReSearch | [2503.19470](https://arxiv.org/abs/2503.19470) | [link](https://www.alphaxiv.org/abs/2503.19470) |
| R1-Searcher | [2503.05592](https://arxiv.org/abs/2503.05592) | [link](https://www.alphaxiv.org/abs/2503.05592) |
| R1-Searcher++ | [2505.17005](https://arxiv.org/abs/2505.17005) | [link](https://www.alphaxiv.org/abs/2505.17005) |
| Search-R1 Empirical Study | [2505.15117](https://arxiv.org/abs/2505.15117) | [link](https://www.alphaxiv.org/abs/2505.15117) |
| ReTool | [2504.11536](https://arxiv.org/abs/2504.11536) | [link](https://www.alphaxiv.org/abs/2504.11536) |
| ToRL | [2503.23383](https://arxiv.org/abs/2503.23383) | [link](https://www.alphaxiv.org/abs/2503.23383) |
| RAGEN | [2504.20073](https://arxiv.org/abs/2504.20073) | [link](https://www.alphaxiv.org/abs/2504.20073) |
| Turn-Level Credit Assignment | [2505.11821](https://arxiv.org/abs/2505.11821) | [link](https://www.alphaxiv.org/abs/2505.11821) |
| Hard Examples GRPO | [2508.14094](https://arxiv.org/abs/2508.14094) | [link](https://www.alphaxiv.org/abs/2508.14094) |
| 1-Shot RLVR | [2504.20571](https://arxiv.org/abs/2504.20571) | [link](https://www.alphaxiv.org/abs/2504.20571) |
| RAFT (Minimalist) | [2504.11343](https://arxiv.org/abs/2504.11343) | [link](https://www.alphaxiv.org/abs/2504.11343) |
| Spec-Dec for RL Rollouts | [2604.26779](https://arxiv.org/abs/2604.26779) | [link](https://www.alphaxiv.org/abs/2604.26779) |
| DAS distribution-aware spec-dec | [2511.13841](https://arxiv.org/pdf/2511.13841) | [link](https://www.alphaxiv.org/abs/2511.13841) |
| AReaL async RL | [2505.24298](https://arxiv.org/pdf/2505.24298) | [link](https://www.alphaxiv.org/abs/2505.24298) |
| PRM Lessons | [2501.07301](https://arxiv.org/abs/2501.07301) | [link](https://www.alphaxiv.org/abs/2501.07301) |
| PRMs for Agents | [2502.10325](https://arxiv.org/abs/2502.10325) | [link](https://www.alphaxiv.org/abs/2502.10325) |

---

## 13. (v2) Deeper round — counter-evidence, new techniques, ablation numbers

> **Heads-up.** Several v1 recommendations need revision. Most importantly, **β = 0 has a documented failure mode in Search-R1 + Qwen2.5-3B** (the LLD Death Spiral) that the v1 review didn't account for. The revised experiment plan in §17 is what to actually run.
>
> Tooling note: alphaxiv and connectedpapers were rate-limited / blocked during the deeper pass; coverage came from arxiv listings, Google Scholar, awesome-lists, framework repos, and Hugging Face papers.

### 13.A NEW techniques (not in v1)

#### A1. LUFFY — Off-Policy Guidance for GRPO ([2504.14945](https://arxiv.org/abs/2504.14945), NeurIPS 2025)

**Tech.** Mixed-policy GRPO: each rollout batch contains both on-policy samples and *off-policy demonstrations* (e.g. distilled R1 traces). Adds policy shaping via regularized importance sampling so the model learns from low-probability but high-value off-policy tokens instead of doing surface imitation. Reports +6.4 avg over six math benches and +6.2 OOD over RLVR baselines.

> **ELI5.** Pure GRPO is a chess student who only plays themselves. LUFFY mixes in tournament games from grandmasters; instead of copying every move (SFT), it tells the student "look how often a grandmaster plays this move vs you" and weights the lesson by surprise. The student doesn't lazily imitate the obvious moves — only the surprising-and-better ones.

**Applicability: HIGH.** A 2B model on multi-hop QA has very low pass@k early; off-policy traces from a stronger search model (R1-Searcher-7B, DeepResearcher-7B) give dense gradient signal exactly where pure GRPO sees zero-reward groups. **Effort: medium (1-2 weeks).** **Risk:** if the off-policy tool-call distribution doesn't match your retriever, you get distribution mismatch — gate on "off-policy `<search>` queries are valid against your corpus."

#### A2. ExGRPO / RePO / BAPO — Off-Policy Replay Buffers ([2510.02245](https://arxiv.org/abs/2510.02245), [2506.09340](https://arxiv.org/abs/2506.09340), [2510.18927](https://arxiv.org/abs/2510.18927))

**Tech.** Maintain a buffer of past high-reward rollouts and mix them into each GRPO batch. **RePO** reports +18.4 abs on Qwen2.5-Math-1.5B and +4.1 on Qwen3-1.7B vs vanilla GRPO. **BAPO** adds adaptive clipping bounds that re-balance positive/negative gradient contributions when off-policy drift is large; +12.5% avg over GRPO including in partial-rollout settings.

> **ELI5.** Vanilla GRPO is an athlete who deletes every game tape after one viewing. Replay-GRPO keeps the best tapes and re-watches with fresh eyes — but you discount old footage because the athlete has changed since. BAPO is the version that adjusts the discount automatically as the athlete improves.

**Applicability: HIGH.** Single-GPU GRPO is rollout-bound. Reusing high-reward rollouts is essentially free quality, and on multi-hop QA where reward is sparse, a buffer of historical EM=1 trajectories is the obvious cure for zero-advantage groups. **Effort: medium (1 week).** Use **freshness-decay weighting** ([2604.16918](https://arxiv.org/abs/2604.16918) reports +46% on NQ Search; raw PER without decay *degraded* perf — a real failure-mode signal).

#### A3. CDE — Curiosity-Driven Exploration ([2509.09675](https://arxiv.org/abs/2509.09675))

**Tech.** Two intrinsic-bonus terms on top of GRPO/PPO: actor-side perplexity bonus (penalize over-confidence on wrong answers, reward exploration on novel-but-plausible tokens) and critic-side variance bonus (multi-head value variance as a count-based-exploration proxy). ~+3 AIME points over RLVR baseline.

> **ELI5.** Standard GRPO only cares whether the *final* answer is right. CDE adds a small bonus when the model says "I'm not super sure" mid-trajectory and ends up correct anyway — like rewarding a chess engine for finding non-book lines that still win.

**Applicability: MEDIUM.** Multi-hop QA at 2B has the exact zero-reward-plateau CDE targets, but benchmarks are math; uncertain how perplexity-as-curiosity transfers when retrieved-token perplexity spikes naturally. **Effort: medium-high (1-2 weeks);** critic branch needs a multi-head value head you don't have in value-free GRPO.

#### A4. PREPO — Prompt-perplexity scheduling + entropy-prioritized rollouts ([2511.00794](https://arxiv.org/abs/2511.00794))

**Tech.** Two cheap intrinsic-property tricks: curriculum over prompt perplexity (model's own perplexity grades difficulty); prioritize rollouts within a group by relative entropy. Claims **3× fewer rollouts** for matched math performance.

> **ELI5.** If your model finds a question really easy (low perplexity), don't waste 5 rollouts on it — sample fewer. If a question's rollouts are all looking similar (low entropy), the model is in a rut — prefer rollouts where it's branching out. Both moves cost zero training time but recover signal.

**Applicability: HIGH** for single-GPU. Rollout cost dominates; cutting rollouts 3× is a giant win even at matched accuracy. **Effort: low (3-5 days).** **Risk:** the 3× headline is from abstract — not yet independently verified on multi-hop QA.

#### A5. EDGE-GRPO + GTPO — Entropy-control variants ([2507.21848](https://arxiv.org/abs/2507.21848), [2508.03772](https://arxiv.org/abs/2508.03772))

**Tech.** EDGE-GRPO uses entropy-driven advantage scaling and "guided error correction" to fix advantage collapse when groups are all-correct or all-wrong. GTPO controls gradient conflict and entropy explosion at the *group* level — directly addresses why dynamic-sampling-filtered batches can still collapse.

> **ELI5.** Dynamic sampling drops the obviously-useless batches (all-right, all-wrong). EDGE-GRPO and GTPO go further: even within "useful" batches, some trajectories disagree about *what* to learn (one says "do more X", another says "do less X"). They detect that gradient-conflict and dampen it instead of letting the model thrash.

**Applicability: HIGH.** Entropy collapse is the main GRPO failure mode at small scale. **Effort: low-medium.**

#### A6. ZeroSearch + s3 — Cheap retrieval for RL training ([2505.04588](https://arxiv.org/abs/2505.04588), [2505.14146](https://arxiv.org/abs/2505.14146))

**Tech.** **ZeroSearch** replaces real search APIs during rollouts with an LM-simulated retriever (curriculum from clean → noisy docs); cuts cost from $586.70 → $70.80 per 64k queries with comparable performance. **s3** decouples searcher from generator and trains the searcher with a "Gain Beyond RAG" reward; baseline-beating quality on **2.4k samples vs 70× more for Search-R1**.

> **ELI5.** ZeroSearch: instead of really hitting Wikipedia 510 times per training step (slow, sometimes failing), have a frozen 7B model *pretend* to be the retriever and hand back fake-but-plausible docs. The student doesn't know the difference and learns to use search anyway. s3: separate "the agent that decides what to search" from "the agent that writes the answer", train only the former with RL. Faster, cheaper, often better.

**Applicability: HIGH** for prototyping. ZeroSearch is your dev-loop accelerator; s3's data efficiency suggests massive over-spend in the v1 plan. **Effort: medium for ZeroSearch** (you keep Qwen2.5-7B around as the retriever sim).

#### A7. SDPO / Self-Distilled RLVR ([2601.20802](https://arxiv.org/abs/2601.20802))

**Tech.** On-policy self-distillation: the *current* model conditioned on verifier feedback is treated as a "self-teacher" and distills its feedback-informed next-token predictions back into the policy. Reward shaping by self-distillation rather than scalar reward.

> **ELI5.** Normal RL: model gets a scalar "you got 0.7". Self-distill: model is shown its own would-have-been better answer (the one that got the higher score) and learns to imitate that, instead of just nudging toward it via gradient. Strictly more information per training step.

**Applicability: MEDIUM-HIGH** especially combined with replay (A2): your buffered EM=1 trajectories become the self-teacher.

#### A8. BRIDGE / SASR / ReLIFT — Interleaved SFT-RL ([2509.06948](https://arxiv.org/abs/2509.06948), [2506.07527](https://arxiv.org/abs/2506.07527))

**Tech.** Instead of two-stage (SFT then RL), interleave SFT and RL updates within a single training run. **BRIDGE** reports **44% faster training + 13% perf gain on Qwen2.5-3B** and 14% faster + 10% on Qwen3-8B. **SASR** probabilistically interleaves based on output similarity to expert demos. **ReLIFT** does interleaved fine-tuning specifically on the hardest questions RL fails on.

> **ELI5.** Standard recipe: 2 weeks of SFT (memorize good answers), then 2 weeks of RL (try things, get rewarded). Interleaved: every few hours, mix in a half-hour SFT refresh. Stops the model from forgetting good patterns when RL takes it on a detour.

**Applicability: HIGH at 2B.** v1's "SFT cold-start then RL" framing assumed the standard two-stage; BRIDGE and SASR show interleaving strictly dominates at 3-8B. **Effort: medium-high** (manage two data streams).

#### A9. StepSearch — step-wise PPO for multi-hop search ([2505.15107](https://arxiv.org/abs/2505.15107), EMNLP 2025)

**Tech.** Adds intermediate per-step rewards based on **information gain** and **redundancy penalty** to global outcome reward. 60k MuSiQue-derived corpus of sub-question search keywords. **+11.2 abs (3B), +4.2 abs (7B)** over global-reward search-RL baselines using only **19k training data**.

> **ELI5.** Pure-EM rewards only the final answer; everything before it is "did you reduce uncertainty about the answer?" StepSearch turns that question into an actual reward signal — small bonus when a `<search>` query retrieves something *new* that's relevant, small penalty for redundant queries. Same multi-hop task, denser learning signal per trajectory.

**Applicability: HIGH** — directly your setting (multi-hop QA, small models). Step-level reward partially solves sparse-reward at no extra rollout cost. **Effort: medium** (need a per-step reward computer: information-gain via overlap with gold passage / query novelty against prior turns).

#### A10. Co-rewarding — self-supervised RL without labels ([2508.00410](https://arxiv.org/abs/2508.00410))

**Tech.** Two variants: data-side contrastive agreement across paraphrased questions, or model-side EMA-teacher self-distillation. +3.31% avg over self-rewarding baselines, **+7.49% on Llama-3.2-3B-Instruct**, sometimes matching RLVR-with-ground-truth.

> **ELI5.** If you don't have gold answers, you can still teach the model: paraphrase the question two different ways, ask the model both, and reward consistency. Or keep a slow-moving copy of the model (EMA) as the "teacher" and reward the fast-moving copy for matching it. Self-supervision without a labeled set.

**Applicability: MEDIUM** — useful if your gold-EM set is small; less if you have HotpotQA-scale labels.

#### A11. **LLDS — the direct fix for GRPO collapse in Search-R1** ([2512.04220](https://arxiv.org/abs/2512.04220), Dec 2025) ⭐

**Tech.** Identifies "Lazy Likelihood-Displacement (LLD) Death Spiral": the model's likelihood on *both* correct and incorrect responses systematically drops, gradients explode, then collapse. Adds a lightweight likelihood-preserving regularizer that activates only when a trajectory's likelihood declines. **+37.8% on Qwen2.5-3B and +32.0% on Qwen2.5-7B across 7 open-domain / multi-hop QA benches.**

> **ELI5.** GRPO + Search-R1 has a known death spiral: the model becomes less and less confident in its own answers — including the *correct* ones. Eventually all answers look equally bad to the model, gradients explode, training crashes. LLDS notices this happening (likelihood is dropping) and gently pushes back, *only* when it's dropping. Targeted bandage rather than a tighter leash.

**Applicability: VERY HIGH.** This paper is essentially "the v1 stack on Search-R1 is broken; here's the surgical fix." **Effort: low-medium** (a regularizer term; reference impl on GitHub).

#### A12. Magistral — Mistral's published recipe ([2506.10910](https://arxiv.org/abs/2506.10910))

**Tech.** Mistral-Small 24B reasoning RL recipe: **β=0 (no KL)** but with **Clip-Higher** to compensate, plus careful batch/minibatch separation. Confirms v1's stack at medium scale, with one new observation: **SFT-on-reasoning-traces-then-RL beats RL-from-base** on reasoning quality.

> **ELI5.** A real production team trained an open-weights 24B reasoner with no KL term. Worked for them. They did one thing v1 didn't emphasize: even a small dose of SFT before RL beats jumping straight to RL.

**Applicability: HIGH** as validation of v1 directionally; but note this is 24B, not 2B.

#### A13. Tina v2 / GRPO++ tricks (practitioner-aggregated)

**Tech.** Bag of small tricks consolidated by practitioners ([Cameron Wolfe's GRPO++ post](https://cameronrwolfe.substack.com/p/grpo-tricks)): TIS (token importance sampling) with bf16 rollouts, removing question-level difficulty bias by *not* dividing advantage by group std (consistent with Dr. GRPO), generating diverse group rollouts via temperature sweep (e.g. T = [0.7, 0.9, 1.0, 1.1, 1.3] across the 5 group members instead of all at 1.0), buffer-size-vs-LR tradeoffs.

> **ELI5.** A bag of "practitioner cheap wins" from people running GRPO at scale who got tired of the variance. Most are one-line config changes.

**Applicability: HIGH** for cheap experimentation. Temperature-sweep across the group is essentially free and increases group diversity (more useful gradient signal).

### 13.B Counter-evidence to v1 recommendations

#### B1. **β=0 (KL-free) GRPO can fail on small models in tool-use settings** — STRONG counter-evidence

- **LLDS ([2512.04220](https://arxiv.org/abs/2512.04220), Dec 2025)** documents that on Search-R1 with Qwen2.5-3B, GRPO collapses via the LLD Death Spiral — *not* via reward hacking. β=0 + Search-R1 is the exact setting they study.
- **MO-GRPO ([2509.22047](https://arxiv.org/abs/2509.22047))**: multi-objective settings (which a search agent has implicitly: format + EM) cause GRPO to optimize one objective at the expense of others under low-KL.
- **Practitioner reports (Mukherjee, Qwen2.5-1.5B-Instruct via HF GRPOTrainer):** β=0 yields tokens stuffed with random numbers to game length; adding small KL fixed it.

> **Implication.** v1's Variant B (β=0 + Dr.GRPO + DAPO dynamic sampling) is correct *for math* but at minimum needs the LLDS regularizer for tool-use, or a small β > 0 (~1e-3 to 5e-3) as cheap insurance. **Revise recommendation:** β = 1e-3 to 5e-3 with anneal-to-0, OR β=0 + LLDS. Don't run β=0 naked on Search-R1.

#### B2. Format reward CAN reward-hack on small models

[Reward Hacking Mitigation via Composite Rewards ([2509.15557](https://arxiv.org/abs/2509.15557))] documents that format-reward alone can cause the model to **prematurely insert the answer in the reasoning section** to claim format credit without doing the reasoning. Solution: composite of format + outcome with format as a **multiplicative gate** (returns 0 if format violated AND answer wrong) rather than additive bonus.

> **Implication.** v1 suggested "small +0.05 for format". Fine, but make it gate-style: penalize only egregious violations and don't add positive credit that can be hacked.

#### B3. Dr. GRPO normalizations are not strictly the right call in heterogeneous settings

- **λ-GRPO ([2510.06870](https://arxiv.org/abs/2510.06870))** argues Dr. GRPO's elimination of per-sequence length normalization yields *uniform* training signals but cannot adapt when optimal length preference differs across prompt classes. λ-GRPO learns the token aggregation weighting and outperforms both Dr. GRPO and DAPO in heterogeneous RLVR.
- **DAPO authors themselves** note that **clip-higher, soft overlong punishment, token-level loss, and dynamic sampling all hurt on a 0.5B Qwen2.5-0.5B** ([Lavaee replication](https://alexlavaee.me/projects/reasoning-slms/)). At 2B you're closer to "helpful" than "hurts" but the regime is non-monotonic in scale.

> **Implication.** Dr. GRPO is a defensible default but isn't unambiguously better; if you observe length collapse in either direction, swap to λ-GRPO or vanilla GRPO with explicit length monitoring.

#### B4. Curriculum learning *does* help in multiple controlled studies

- **E2H Reasoner / Curriculum RL Easy-to-Hard ([2506.06632](https://arxiv.org/abs/2506.06632))** — easy-to-hard curriculum yields cleaner generalization than uniform sampling.
- **CLPO ([2509.25004](https://arxiv.org/abs/2509.25004))** — curriculum + policy optimization beats both fixed-difficulty and hard-only.
- **RAG-RL ([2503.12759](https://arxiv.org/abs/2503.12759))** — explicit curriculum on number of distractor passages.

The picture is genuinely mixed; "Hard Examples" (v1) is one data point, not consensus. A **retrieval-difficulty curriculum** (fewer distractors first → more later) is well-motivated for multi-hop QA and cheap.

#### B5. Dynamic sampling itself can introduce a different bias

DAPO drops both all-correct and all-wrong groups. At 2B with HotpotQA, your *initial* policy will produce nearly-all-wrong groups for a long time — exactly the regime where dropping them starves the model of training signal, even though those are the groups you most need to learn from. Combined with B1 (collapse risk): at very low pass-rates, dynamic sampling + β=0 ≈ guaranteed collapse. **BAPO ([2510.18927](https://arxiv.org/abs/2510.18927))** explicitly shows adaptive clipping > static dynamic sampling for stability.

#### B6. SFT cold-start contributes less than headlines claim

[Beyond Two-Stage Training ([2509.06948](https://arxiv.org/abs/2509.06948))] and BRIDGE find: **SFT provides effective initialization and rapid early convergence but contributes little to final convergence performance.** Counter to the strong "SFT cold-start is essential" framing in R1-Searcher and Search-R1 literature. **Implication:** interleaved SFT-RL may strictly dominate at 2B.

### 13.C Pulled ablation numbers — actual deltas, not headlines

#### C1. DAPO ablation — Qwen2.5-32B base, AIME 2024 pass@1

| Stage | Score | Δ |
|---|---:|---:|
| Vanilla GRPO | 30 | — |
| + Overlong filtering | 36 | +6 |
| + Clip-Higher | 38 | +2 |
| + Soft-overlong-penalty | 41 | +3 |
| + Token-level loss | 42 | +1 |
| **+ Dynamic sampling** | **50** | **+8** ← largest single contributor |

**Total +20 over GRPO; dynamic sampling alone explains nearly half.** At **0.5B** (Lavaee replication), all four DAPO components reportedly **hurt** — recipe is scale-conditioned.

#### C2. Dr. GRPO

Quality delta vs vanilla GRPO across math benches: **~1-3 points absolute**, *not* recipe-defining. Bigger impact: **token efficiency** (correct responses don't inflate to 6-8k tokens of repeated reasoning). λ-GRPO ([2510.06870](https://arxiv.org/abs/2510.06870)) reports Dr. GRPO is **strictly dominated** by learnable token preferences in heterogeneous settings.

#### C3. R1-Searcher headline

HotpotQA +48.2%, 2WikiMultiHopQA +21.7%, Bamboogle +4.0% (LLM-as-Judge). Gain mostly **format-discovery via stage-1 retrieve-reward**, not absolute reasoning improvement. Absolute EM numbers are modest.

#### C4. R1-Searcher++ ablation

vs Search-R1 baseline: **+4.3% absolute and 42.9% fewer retrieval calls.** The retrieval-call reduction is the more interesting metric — internal-knowledge gating has compute upside, not just accuracy.

#### C5. Search-R1 (incl. v0.3 empirical study, [2505.15117](https://arxiv.org/abs/2505.15117))

- **Retrieved-token loss masking (Table 4): significant.** Without it, RL is unstable because policy gets gradient through retrieved tokens it didn't generate. **Non-negotiable.**
- **top-k retrieval (Table 7):** top-3 best at step 500; top-1 underperforms, top-5 plateaus.
- **Base vs Instruct (Figure 4):** Instruct converges faster from a better init; final scores similar; instruct preferred for compute-bound runs.
- Headline: +41% on Qwen2.5-7B, +20% on 3B over RAG baselines — but these are vs naïve RAG.

#### C6. LUFFY

+6.4 avg on six math benches; +6.2 OOD. **Ablation:** "Shaping" and "NoClip" both contribute; **on-policy alone with shaping/NoClip yields no improvement** — the off-policy data is necessary, not the IS trick alone.

#### C7. Tina

DeepSeek-R1-Distill-Qwen-1.5B + LoRA + GRPO: **+20% reasoning on AIME24 at $9 USD compute.** 43.33% Pass@1 AIME24. **260× cost reduction vs full-FT 1.5B RL baselines.** Strongest single $/AIME-point datapoint in the literature.

#### C8. RePO

Qwen2.5-Math-1.5B: **+18.4 abs avg** over GRPO. Qwen3-1.7B: **+4.1 abs avg**. Replay helps more when base policy is weaker.

#### C9. LLDS

**+37.8% on Qwen2.5-3B, +32.0% on Qwen2.5-7B** across 7 OD/multi-hop QA benches over Search-R1+GRPO. Three-phase collapse trajectory documented: early stagnation → steady decay → accelerated collapse.

#### C10. StepSearch

**+11.2 abs at 3B, +4.2 at 7B** over global-reward search-RL baselines using **only 19k training data**.

#### C11. ZeroSearch

Real-search RL: **$586.70** per 64k queries. ZeroSearch (LM simulator): **$70.80** — **8.3× cheaper**. Performance comparable with 7B simulator; **better with 14B simulator** (a strong simulator can produce more useful retrievals than the real index).

#### C12. s3

**2.4k training samples** to outperform Search-R1 trained on 70× more (170k+). First strong evidence you don't need to RL-train the answerer to get RAG gains.

#### C13. BRIDGE

Qwen2.5-3B: **44% faster training + 13% perf gain.** Qwen3-8B: 14% faster + 10% gain.

### 13.D Adjacent literatures

- **RL-trained retrievers / co-training**: [Query-Doc Co-Augmentation ([2506.18670](https://arxiv.org/abs/2506.18670))], **R3-RAG** (EMNLP 2025), **GRAIL** ([2508.05498](https://arxiv.org/abs/2508.05498)).
- **DPO/IPO/KTO for tool-use**: **DiaTool-DPO** (SIGDIAL 2025) — multi-turn DPO/KTO with token-level masking. General consensus: DPO variants competitive for *preference* tasks, lag GRPO for *verifiable-reward* reasoning.
- **Cost-Pareto frontier**: [Algorithmic Efficiency / Cost of Inference ([2511.23455](https://arxiv.org/abs/2511.23455))] — **24.5×/year cost-of-pass decline on MATH 500, 3.23× on AIME 2024**.
- **Frameworks**: **OpenRLHF** (PPO/GRPO/Dr.GRPO/RLOO/REINFORCE++ via single flag, async vLLM, Ray scaling), **AReaL** (2.57× speedup over best sync systems, multi-node), **ASearcher** (long-horizon search agent RL, >100 turns/trajectory, 400k tokens).
- **Optimizers**: **Muon** ([2502.16982](https://arxiv.org/abs/2502.16982)) — ~2× compute efficiency for *pretraining*. **Important caveat:** Muon-finetuned-after-AdamW-pretrained does NOT clearly beat AdamW-finetune; optimizer-pretraining-match matters. Qwen2.5/3.5 is AdamW-pretrained → **Muon unlikely to win for fine-tuning your model.** **bnb 8-bit Adam**: standard ~75% optimizer-state reduction; safe pick for 2B on a single A100.
- **Self-play**: **Agent0** ([2511.16043](https://arxiv.org/abs/2511.16043)) — Curriculum + Executor co-evolve via shared tool integration; **+18% math, +24% general reasoning on Qwen3-8B-Base from zero data**. **Tool-R0** ([2602.21320](https://arxiv.org/abs/2602.21320)) — generator/solver co-evolution. **SeRL** ([2505.20347](https://arxiv.org/abs/2505.20347)) — self-instruction + self-rewarding via majority voting; useful when seed dataset is tiny.

### 13.E Hugging Face trending — quick notes

The HF papers landing page is title-only without abstracts on a single GET; a more thorough scrape would need the API. From the trending titles in late April 2026 that map to this review's scope: Co-Evolving Policy Distillation (2604.27083), Step-level Optimization for Tool-use Agents (2604.27151), Length-Value Model token-level pretraining (2604.27039) — all RL-flavored small-model work; abstracts pending. **Most of the genuinely high-impact 2025-26 papers in this review surfaced via arxiv listings + awesome-lists rather than HF trending** — HF trending skews toward last-week visibility, not "best papers of the field over 12 months". Treat HF as a discovery feed for what's *new*, not as a curated list of what's *important*.

---

## 14. (v2 SUPERSEDES §11) Revised experiment plan — Variant C

> **What changed from §11.** Two failure modes emerged in v2 that the v1 plan didn't account for: (1) GRPO+Search-R1+Qwen2.5-3B has a documented LLD death spiral at β=0 (LLDS paper, [2512.04220](https://arxiv.org/abs/2512.04220)); (2) dynamic sampling can starve the model of signal at 2B-with-low-pass-rate, where every group is nearly all-wrong. The revised plan below either replaces or adds insurance against both.

**Variant A (control)**: current GRPO recipe — β=0.001, group-norm advantages, no filter, no spec-dec.

**Variant C (revised stack — what to actually run)**:

1. **Keep**: Dr. GRPO normalizations, EAGLE-3 spec-dec for rollout, **Search-R1 retrieved-token loss masking** (this is non-negotiable per Table 4 of empirical study).
2. **Change β=0 → β=1e-3 to 5e-3 with anneal-to-0**. Compute cost trivial; insurance against LLD death spiral and the documented Qwen-1.5B random-number-stuffing failure.
3. **Add LLDS regularizer ([2512.04220](https://arxiv.org/abs/2512.04220))** — the only paper specifically targeting GRPO+Search-R1 collapse on Qwen2.5-3B; +37.8 abs reported. **Single highest-EV addition.**
4. **Replace dynamic sampling with BAPO-style adaptive clipping ([2510.18927](https://arxiv.org/abs/2510.18927))**. At 2B with hard task, all-zero groups are common early; BAPO handles them via clip adaptation rather than dropping.
5. **Add a small replay buffer (RePO/ExGRPO style, ~5-20% of batch from buffer)** with **freshness-decay weighting** ([2604.16918](https://arxiv.org/abs/2604.16918) — without decay, replay *hurts*).
6. **Off-policy seed of ~1-5k high-quality search-trajectory traces (LUFFY-style)** from R1-Searcher-7B or DeepResearcher-7B. Dominates any cold-start SFT alternative because they're already structurally consistent with your reward.
7. **Curriculum on retrieval-difficulty (RAG-RL style)**: 0-1 distractor passages first, ramp to 5+ over 30% of training.
8. **Use ZeroSearch-style simulated retrieval for the first ~30% of training** if your real corpus is slow. Switch to real corpus once format and basic search behavior are learned.
9. **Optional: GSPO sequence-level IS** if long-trajectory variance issues appear (likely with multi-turn search). One flag in NeMo-RL.
10. **Skip**: Muon optimizer (Qwen2.5/3.5 is AdamW-pretrained); LoRA (Tina works at 1.5B-distilled-on-math; multi-turn search at 2B+LoRA is unproven and you have 80GB).

**By EV**:
- **Highest**: LLDS regularizer (low effort, +37.8 abs on Qwen2.5-3B in your exact setting).
- **Second**: off-policy seed traces (LUFFY-style; medium effort, large reported gain in your setting).
- **Third**: small-β + small replay buffer with freshness decay (medium effort, well-documented).

**Failure mode to instrument explicitly:** LLD three-phase trajectory (early stagnation → steady decay → accelerated collapse). Log mean log-likelihood of *correct* rollouts every step; if it monotonically decreases for >100 steps, you're in the death spiral and need either LLDS or higher β immediately.

---

## 15. Updated reference list (v2 additions)

Off-policy / replay: [LUFFY 2504.14945](https://arxiv.org/abs/2504.14945), [ExGRPO 2510.02245](https://arxiv.org/abs/2510.02245), [RePO 2506.09340](https://arxiv.org/abs/2506.09340), [BAPO 2510.18927](https://arxiv.org/abs/2510.18927), [Buffer Matters 2602.20722](https://arxiv.org/abs/2602.20722), [Freshness-Aware PER 2604.16918](https://arxiv.org/abs/2604.16918), [Prioritized Replay 2601.02648](https://arxiv.org/abs/2601.02648), [RLEP 2507.07451](https://arxiv.org/abs/2507.07451).

Intrinsic motivation: [CDE 2509.09675](https://arxiv.org/abs/2509.09675), [IMAGINE 2505.17621](https://arxiv.org/abs/2505.17621), [MERCI 2510.16614](https://arxiv.org/abs/2510.16614), [PREPO 2511.00794](https://arxiv.org/abs/2511.00794).

Self-play / self-improvement: [Co-rewarding 2508.00410](https://arxiv.org/abs/2508.00410), [SeRL 2505.20347](https://arxiv.org/abs/2505.20347), [Agent0 2511.16043](https://arxiv.org/abs/2511.16043), [Tool-R0 2602.21320](https://arxiv.org/abs/2602.21320), [SDPO 2601.20802](https://arxiv.org/abs/2601.20802), [Self-Distillation Zero 2604.12002](https://arxiv.org/abs/2604.12002).

Hybrid SFT-RL: [BRIDGE / Beyond Two-Stage 2509.06948](https://arxiv.org/abs/2509.06948), [Interleaved Online FT 2506.07527](https://arxiv.org/abs/2506.07527), [RL Heals OOD Forgetting 2509.12235](https://arxiv.org/abs/2509.12235).

GRPO variants and counter-evidence: [λ-GRPO 2510.06870](https://arxiv.org/abs/2510.06870), [VAPO 2504.05118](https://arxiv.org/abs/2504.05118), [GTPO 2508.03772](https://arxiv.org/abs/2508.03772), [EDGE-GRPO 2507.21848](https://arxiv.org/abs/2507.21848), [AEPO 2510.08141](https://arxiv.org/abs/2510.08141), [MO-GRPO 2509.22047](https://arxiv.org/abs/2509.22047), [LLDS / GRPO Collapse in Search-R1 2512.04220](https://arxiv.org/abs/2512.04220), [Hidden Objective Biases 2601.05002](https://arxiv.org/abs/2601.05002).

Search agents: [ZeroSearch 2505.04588](https://arxiv.org/abs/2505.04588), [s3 2505.14146](https://arxiv.org/abs/2505.14146), [StepSearch 2505.15107](https://arxiv.org/abs/2505.15107), [MaskSearch 2505.20285](https://arxiv.org/abs/2505.20285), [DeepResearcher 2504.03160](https://arxiv.org/abs/2504.03160), [WebDancer 2505.22648](https://arxiv.org/abs/2505.22648), [LiteResearcher 2604.17931](https://arxiv.org/abs/2604.17931), [PaperSearchQA 2601.18207](https://arxiv.org/abs/2601.18207), [Agentic Search Survey 2510.16724](https://arxiv.org/abs/2510.16724), [Agentic RL Landscape 2509.02547](https://arxiv.org/abs/2509.02547).

Optimizers / memory: [Muon 2502.16982](https://arxiv.org/abs/2502.16982), [GUM 2510.17802](https://arxiv.org/abs/2510.17802), [Sophia 2305.14342](https://arxiv.org/abs/2305.14342).

Cost-Pareto: [Tina 2504.15777](https://arxiv.org/abs/2504.15777) (ICLR 2026), [Magistral 2506.10910](https://arxiv.org/abs/2506.10910), [Algorithmic Efficiency 2511.23455](https://arxiv.org/abs/2511.23455).

Curriculum: [Curriculum E2H 2506.06632](https://arxiv.org/abs/2506.06632), [CLPO 2509.25004](https://arxiv.org/abs/2509.25004), [Rethinking Easy-to-Hard 2603.27226](https://arxiv.org/abs/2603.27226), [LLMs Compose Skills 2509.25123](https://arxiv.org/abs/2509.25123), [RAG-RL 2503.12759](https://arxiv.org/abs/2503.12759).

Reward shaping / hacking: [Composite Rewards 2509.15557](https://arxiv.org/abs/2509.15557), [Reward Shaping vs Hacking 2502.18770](https://arxiv.org/abs/2502.18770), [Beyond Correctness 2509.03403](https://arxiv.org/abs/2509.03403), [Detecting/Mitigating Reward Hacking 2507.05619](https://arxiv.org/abs/2507.05619).

Frameworks: [OpenRLHF 2501.03262](https://arxiv.org/abs/2501.03262), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [ASearcher](https://github.com/inclusionAI/ASearcher), [Cameron Wolfe — GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks), [Sebastian Raschka — State of LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training), [Awesome-LLM-RLVR](https://github.com/smiles724/Awesome-LLM-RLVR), [opendilab/awesome-RLVR](https://github.com/opendilab/awesome-RLVR), [Awesome RL-Agentic-Search](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers).

---

## 16. (v3) Hugging Face trending + alphaxiv + latest papers (May 2026)

> **Note**: alphaxiv.org and connectedpapers.com WebFetch were blocked by the sandbox, so §16.A is partial (search-result snippets only); §16.B is empty. §16.C covers 16 HF-trending papers + 5 late-breaking bonus papers retrieved via direct HF scrape and arxiv listings.

### 16.A alphaxiv insights (partial — search-snippet based)

Three papers surfaced via alphaxiv search results (not from alphaxiv's UI itself) are genuinely new and highly relevant:

1. **M-GRPO: Momentum-Anchored Policy Optimization** ([2512.13070](https://arxiv.org/abs/2512.13070), Dec 2025). Identifies long-horizon "policy collapse" in self-supervised RL where scaling rollouts only delays collapse. Uses a slowly-evolving EMA target model as anchor + addresses rapid policy-entropy collapse. **Independent confirmation of the LLDS-style failure mode** (v2 §13.B1). Momentum-anchor is orthogonal to LLDS and could be combined with it. **Applicability: HIGH.** **Effort: low** (EMA-target is a standard technique).

2. **JustRL: Scaling a 1.5B LLM with a Simple RL Recipe** ([2512.16649](https://arxiv.org/abs/2512.16649), Dec 2025; ICLR 2026 Blogpost track). Single-stage GRPO with FIXED hyperparameters, NO curriculum, NO dynamic sampling, NO length penalty, NO multi-stage scheduling — reaches 54.9% / 64.3% average across nine math benchmarks. **Ablation bombshell: adding "standard tricks" (explicit length penalties, robust verifiers) actively DEGRADES performance by collapsing exploration.** Counter-evidence to several v2 §14 recommendations. Same hyperparameters transfer between two different 1.5B models. **Applicability: VERY HIGH.** Run "JustRL minimal" as a control variant alongside Variant C to validate whether the v2 stack is helping or hurting.

3. **Agentic RL for Search is Unsafe** ([2510.17431](https://arxiv.org/abs/2510.17431), Oct 2025). Search-R1-style RL inherits refusal from instruction tuning but is fragile: "Search attack" (force response to begin with a search) and "Multi-search attack" drop refusal up to 60.0%, answer-safety by 82.5%, query-safety by 82.4% on Qwen and Llama bases. **Applicability: HIGH for production phase; LOW for capability development.** Instrument this as a final-eval gate.

### 16.B connectedpapers derivative-works graphs

> **Status:** connectedpapers.com is blocked by the sandbox (requires interactive JS-rendered SPA; no search-engine index of derivative works). **No data retrievable in this session.** To fix, use a manual browser session against https://www.connectedpapers.com/main/<arxiv_id> for Search-R1 (2503.09516), LLDS (2512.04220), LUFFY (2504.14945).

### 16.C Hugging Face trending papers — RL/reasoning (last 30 days)

Sixteen papers NEW to v1+v2:

#### **C1. Apriel-Reasoner** ([2604.02007](https://arxiv.org/abs/2604.02007), ServiceNow, Apr 3)

**ELI5.** Multi-domain RL usually fails because math needs long reasoning traces but instruction-following needs short answers. Apriel adds two knobs: "keep each domain's data ratio fixed even when harder domains need longer rollouts" and "spend more tokens on hard problems, fewer on easy." Enables a single model accurate across all domains while staying short-answered.

**Applicability: medium.** Single-task setting (multi-hop QA) doesn't need multi-domain balancing. But **difficulty-aware length penalty** is directly transferable and could replace flat length caps in reward shaping. **Effort: low** (one adaptive term).

#### **C2. Rethinking On-Policy Distillation of LLMs** ([2604.13016](https://arxiv.org/abs/2604.13016), Tsinghua, Apr 15, 202 upvotes)

**ELI5.** Teaching a small model to copy a big model only works if (a) they "think alike" (share compatible token-production patterns) and (b) the big model knows things the small one doesn't. Otherwise you're wasting compute on patterns the student already had. Successful distillation concentrates on a small shared token set carrying 97-99% of probability mass.

**Applicability: HIGH.** Directly informs the LUFFY-style cold-start (v2 §14 item 6). **Pick the off-policy seed from a teacher in the SAME family as your student** (R1-Searcher-7B with Qwen2.5 base for Qwen3.5-2B student is good; cross-family seed is wasted). **Effort: none** (config-level). **Impact: high** — wrong family of seed can null out LUFFY gains.

#### **C3. KnowRL** ([2604.12627](https://arxiv.org/abs/2604.12627), Tianjin + Baidu, Apr 15, 100 upvotes)

**ELI5.** Hard RL problems give zero reward forever, so you need hints. Most hint methods throw text at the model; KnowRL breaks hints into tiny independent facts ("knowledge points") and finds the smallest combination that actually helps. A weird side effect: removing one fact can make training *easier*, but removing two can make it *harder* (interactions matter).

**Applicability: HIGH.** Reward sparsity at 2B + multi-hop search is your central problem. The knowledge-point decomposition could map onto search-trajectory structure (each retrieved doc as a knowledge atom). **Effort: medium** (decompose gold passages into atomic facts; train a selector network).

#### **C4. DiPO — Disentangled Perplexity Policy Optimization** ([2604.13902](https://arxiv.org/abs/2604.13902), Apr 16, 62 upvotes)

**ELI5.** Hard problems and easy problems need different RL strategies. DiPO splits your training data by how surprised the model is (perplexity) and applies different reward bonuses to each pile — avoids one pile's rewards canceling the other's.

**Applicability: medium.** Perplexity-conditioning is similar to PREPO (v2 A4) but at per-sample vs per-prompt level. **Effort: low** (one additional tracking pass). Not a swap target; more of a refinement.

#### **C5. Where Does Output Diversity Collapse in Post-Training?** ([2604.16027](https://arxiv.org/abs/2604.16027), USTC, Apr 20, 22 upvotes)

**ELI5.** People blame RL for making model outputs boring. This paper says no — boredom comes from your *training data*, not the algorithm. Once it's baked into weights by SFT, no inference-time trick recovers it. Decomposing into quality-control vs residual narrowing: Think (CoT distillation) loses more diversity at SFT than RL does later.

**Applicability: HIGH for diagnosis.** If you SFT-cold-start with R1-Searcher traces, you may collapse search-strategy diversity *before* RL even starts. **Counter-evidence to v2 §14 item 6**: SFT seed quality matters more than seed quantity. **Effort: none** (diagnostic insight).

#### **C6. HiExp — Hierarchical Experience for Agentic Search** ([2604.08124](https://arxiv.org/abs/2604.08124), Alibaba Cloud, Apr 10, 5 upvotes)

**ELI5.** Instead of letting the agent randomly try search queries, mine successful past trajectories for "lessons" (e.g., "if you see a date, search the date first") and use those as soft training signals. Converts random exploration into strategic exploration.

**Applicability: HIGH.** Direct fit for multi-hop search. Comparable in spirit to StepSearch (v2 A9) but acts at trajectory-cluster level. **Worth a head-to-head test**: StepSearch vs HiExp on your multi-hop validation set. **Effort: medium** (contrastive clustering + regularizer).

#### **C7. Step-Level Advantage Selection / SAS** ([2604.24003](https://arxiv.org/abs/2604.24003), AMD + UNC, Apr 28, 7 upvotes)

**ELI5.** When the model's wrong answer was confidently-but-wrongly-confirmed by your verifier, don't punish it for that step (the verifier might just be broken). Zero-out reward on verifier-failed-steps-in-correct-reasoning and low-confidence-steps-in-correct-rollouts. Cleaner training, shorter reasoning.

**Applicability: HIGH.** Your EM verifier is brittle (one of v2's known failure modes, B1). SAS is a low-effort fix to verifier-induced gradient noise. **Effort: low** (zero out selected steps in reward computation).

#### **C8. Lightning OPD** ([2604.13010](https://arxiv.org/abs/2604.13010), NVIDIA, Apr 15, 13 upvotes)

**ELI5.** Distillation usually needs the teacher running live during student training (expensive). Lightning-OPD precomputes teacher predictions once and reuses — BUT ONLY if the same teacher was used for the SFT step. Mix teachers and training breaks.

**Applicability: medium.** If you SFT-cold-start with R1-Searcher-7B, **keep R1-Searcher-7B as the teacher for downstream distillation** — don't swap to a different model. **Effort: none** (constraint-level, not algorithmic).

#### **C9. UDM-GRPO** ([2604.18518](https://arxiv.org/abs/2604.18518), Apr 21, 19 upvotes)

**ELI5.** GRPO for diffusion image generation. Not your setting.

**Applicability: low.** T2I, not text. Mention only as "GRPO continues to ship across modalities."

#### **C10. Faithful GRPO / FGRPO** ([2604.08476](https://arxiv.org/abs/2604.08476), Apr 10, 8 upvotes)

**ELI5.** Multiple goals (be right AND be faithful to sources) can cancel in GRPO's group-normalization step. FGRPO enforces each goal via Lagrangian multipliers without signal cancellation.

**Applicability: medium.** If you ever add a faithfulness reward (e.g., "your answer must come from a retrieved doc"), use Lagrangian-dual-ascent to avoid breaking the EM signal. **Effort: medium** (constraint-solver integration).

#### **C11. Beyond Accuracy: Inefficiency Patterns in Tool-Integrated Reasoning / PTE** ([2604.05404](https://arxiv.org/abs/2604.05404), USTC, Apr 8, 42 upvotes)

**ELI5.** Counting tool calls or tokens is a bad measure of how slow your agent really is — every tool call kicks tokens out of GPU cache, forcing recomputation. PTE (Prefill Token Equivalents) is the proper hardware-aware metric.

**Applicability: medium.** You measure rollout latency for the spec-dec ablation; PTE is a better metric to log. **Effort: low** (one metric addition).

#### **C12. Visual Reasoning through Tool-Supervised RL / ToolsRL** ([2604.19945](https://arxiv.org/abs/2604.19945), Amazon, Apr 23, 4 upvotes)

**ELI5.** Don't train a model to use tools AND solve tasks simultaneously — first teach it tools, then tasks. Decoupling avoids optimization conflict between tool-mastery and task-mastery.

**Applicability: HIGH.** Clean 2-phase recipe for tool-use RL. For your setting: phase 1 = format reward only (learn `<search>...</search>` syntax); phase 2 = EM reward only (learn what to search). Already implicit in Search-R1 but here it's explicit with ablation. **Effort: none** (scheduling-level; no algo change). **Impact: medium** (known good practice, not novel).

#### **C13. Near-Future Policy Optimization / NPO** ([2604.20733](https://arxiv.org/abs/2604.20733), Apr 23, 73 upvotes)

**ELI5.** Replay buffers reuse old versions of the model. Teacher distillation uses foreign models. NPO uses *future* versions of the same model — train ahead on a fast-lane policy, then teach the main policy with those future rollouts. Sounds paradoxical but it's bootstrapping with an EMA-ahead checkpoint.

**Applicability: HIGH.** Direct alternative to LUFFY-style external-teacher seeding (v2 §14 item 6). If R1-Searcher traces aren't available, NPO uses your own training run. Could pair with replay buffer (v2 item 5). **Effort: low** (EMA checkpoint + async training). **Impact: high** if external seed unavailable.

#### **C14. GFT — Group Fine-Tuning** ([2604.14258](https://arxiv.org/abs/2604.14258), Zhejiang, Apr 21, 29 upvotes)

**ELI5.** Standard SFT silently behaves like a buggy RL algorithm (sparse reward + unstable inverse-probability weighting). GFT replaces SFT with group-relative learning + adaptive clipping — doesn't break the model before RL even starts.

**Applicability: HIGH.** Direct replacement for the SFT cold-start step (v2 §14 item 6). If you SFT R1-Searcher traces conventionally, you may already be hurting downstream GRPO. GFT is the fix. **Effort: medium** (group-learning + dynamic clipping). **Impact: high** on SFT-quality.

#### **C15. The Illusion of Certainty / CaOPD** ([2604.16830](https://arxiv.org/abs/2604.16830), Apr 21, 14 upvotes)

**ELI5.** When a small model copies a big model, it ends up sure it's right even when it's wrong. Overconfidence breaks downstream RL because exploration dies. CaOPD makes the small model honest about what it knows.

**Applicability: medium-high.** Confidence collapse is a precursor to LLDS-style death spirals (v2 §13.B1). If you use OPD anywhere in your stack, calibration matters. **Effort: low** (confidence decoupling; no RL changes).

#### **C16. Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding** ([2604.26779](https://arxiv.org/abs/2604.26779), NVIDIA, Apr 30, 6 upvotes)

**ELI5.** Spec-decoding in NeMo-RL makes RL rollouts 1.8× faster, but only if the draft model was trained on data matching your RL task. Use draft length k=3. If you grab a generic draft, longer drafts can actually slow you down.

**Applicability: VERY HIGH.** This is the **canonical reference for EAGLE-3 spec-dec** (v2 §14 item 1) — lives directly in NeMo-RL. Operational config: draft length k=3, initialize on Qwen3.5-2B's rollout distribution, don't waste effort on online adaptation if init is good. **Key finding: draft length k≥5 is SLOWER than autoregressive on hard reasoning** despite being faster on easy gen. N-gram drafting is slower despite longer draft length.

**Updated action:** Update v2 §14 item 1 with these operational parameters.

### 16.D Bonus papers (late-breaking, last 7 days)

Five papers from late-April / early-May 2026 arxiv that escape the v2 list:

1. **M-GRPO** ([2512.13070](https://arxiv.org/abs/2512.13070)) — already listed in §16.A.

2. **JustRL** ([2512.16649](https://arxiv.org/abs/2512.16649)) — already listed in §16.A.

3. **TCOD — Temporal Curriculum in OPD for Multi-turn Agents** ([2604.24005](https://arxiv.org/abs/2604.24005), Tongyi Lab, Apr 28). First systematic study of OPD stability in **multi-turn agent settings**. Vanilla OPD's stability claims don't generalize to long-horizon agents. Temporal curriculum recovers stability. **Applicability: HIGH.** Multi-hop search QA is exactly the multi-turn setting where v2-cited OPD tricks may break. **Effort: low** (curriculum scheduling).

4. **TESSY — Teacher-Student Cooperation Data Synthesis** ([2604.14164](https://arxiv.org/abs/2604.14164), Shanghai AI Lab, Apr 17). Identifies "stylistic divergence" as the failure mode of synthetic-SFT-data pipelines. Cooperative token interleaving between teacher and student. On Qwen3-8B: vanilla SFT *drops* code performance −3.25% on LiveCodeBench, +10.02% on OJBench; TESSY fixes to +11.25% and +6.68%. **Applicability: HIGH.** Direct fix for LUFFY-style cold start if R1-Searcher traces are stylistically far from Qwen3.5-2B's distribution. **Effort: medium** (trace fusion during SFT).

5. **Agentic RL for Search is Unsafe** ([2510.17431](https://arxiv.org/abs/2510.17431)) — already listed in §16.A.

---

## 17. FINAL revised experiment plan (all three rounds consolidated)

> **What changed from v2 §14.** The v3 round surfaced three high-EV updates and confirmed that JustRL's counter-evidence (adding tricks degrades) deserves a head-to-head test.

**Variant A (control)**: your current GRPO recipe — β=0.001, group-norm advantages, no filter, no spec-dec.

**Variant C (recommended production stack)**:

*Core algorithmic stack:*
1. **β anneal from 1e-3 to 0 over the run** (not β=0 from the start) — insurance against LLDS + documented Qwen-1.5B random-number-stuffing.
2. **Dr. GRPO normalizations** — remove length-norm + std-norm (v2 §3).
3. **LLDS regularizer** ([2512.04220](https://arxiv.org/abs/2512.04220)) — +37.8 abs on Qwen2.5-3B multi-hop QA. **Highest single EV.**
4. **M-GRPO momentum-anchor** ([2512.13070](https://arxiv.org/abs/2512.13070)) — independent insurance against long-horizon policy collapse. Cheap (EMA target). Run alongside LLDS.
5. **BAPO adaptive clipping** ([2510.18927](https://arxiv.org/abs/2510.18927)) instead of DAPO dynamic sampling — handles zero-advantage groups at low pass-rate via adaptation, not dropping.
6. **Small replay buffer (5-20% of batch)** with **freshness-decay weighting** ([2604.16918](https://arxiv.org/abs/2604.16918)) — large gains; without decay, replay hurts.
7. **Off-policy seed of 1-5k search traces** from R1-Searcher-7B (or DeepResearcher-7B) using **SAME-family teacher** (v2 §14 item 6, refined per C2 above). Via LUFFY-style IS clipping.
8. **GFT (Group Fine-Tuning)** ([2604.14258](https://arxiv.org/abs/2604.14258)) for the SFT cold-start (not vanilla SFT) — prevents diversity collapse before RL.
9. **Curriculum on retrieval-difficulty** (0-1 distractors → 5+ distractors over 30% of training, v2 §14 item 7).
10. **ZeroSearch-style simulated retrieval** for first 30% of training (v2 §14 item 8).
11. **EAGLE-3 spec-dec for rollout** with **draft length k=3, in-domain draft init, skip online adaptation** (v2 §14 item 1, refined per C16 above). 1.8× rollout speedup confirmed.
12. **Optional: GSPO sequence-level IS** if long-trajectory variance appears (v2 §14 item 9, one flag flip).
13. **Optional: KnowRL knowledge-point hints** (C3) if early-phase EM plateaus below 10% — decompose gold passages into atomic facts, use as soft signals.
14. **Instrument LLD-death-spiral explicitly:** log mean log-likelihood of correct rollouts every step; if monotonic decrease for >100 steps, activate LLDS regularizer boost.
15. **Instrument verifier brittleness (SAS):** zero out reward on steps where verifier fails in otherwise-reasonable trajectories (C7).

*Skip:*
- Muon optimizer (Qwen2.5/3.5 is AdamW-pretrained).
- LoRA (you have 80GB; full FT is cheaper per-quality than low-rank).
- PRMs / step-level outcome shaping (null evidence for multi-hop QA).
- Curriculum (just-train-hard confirmed for math; retrieval-difficulty curriculum is the exception).

**Variant C-minimal (control for JustRL counter-evidence)**:

Run **one seed** of ultra-minimal GRPO to validate that v2's stack of tricks actually helps:
- Single-stage GRPO, **fixed hyperparameters** (same as Variant C but no per-step tuning).
- **No length penalty, no robust verifier adjustments, no curriculum.**
- **No dynamic sampling, no BAPO, no replay buffer.**
- **No SFT cold-start, no off-policy seed.**
- Pure GRPO with EM reward only.

If C-minimal beats Variant C on your eval suite, the stack is hurting. If Variant C beats C-minimal by 2-5 EM points, the stack is justified.

**By EV** (what to invest sweat in):
1. **Highest:** LLDS + M-GRPO regularizers (low effort, large confirmed upside).
2. **Second:** Off-policy seed from same-family teacher (v2 item 6, refined per C2).
3. **Third:** Replay buffer + freshness decay (v2 item 5).
4. **Fourth:** Spec-dec with proper config (v2 item 1, refined per C16).
5. **Fifth:** GFT for SFT (C14) or TESSY for traces (16.D item 4) — one or the other.

**Expected outcome**: Variant C should hit 8-12 EM points above C-minimal and 4-7 above your current baseline (if Search-R1 is your baseline). If JustRL's counter-evidence holds and C-minimal beats Variant C, pivot to a "maximal tricks stripping" experiment to identify which pieces are hurting.

**Failure-mode instrumentation:**
- Log mean/max log-likelihood of *correct* rollouts every 10 steps; alert if monotonic decrease for >100 steps.
- Log "all-1" and "all-0" group frequencies per step; alert if >30% of groups are flat (starvation).
- Log top-3 most-common `<search>` queries per 100 steps; flag if stuck on 1-2 queries (mode collapse).
- Log EM@top-k for k ∈ {1, 3, 5} group members to diagnose pass@k trends.
- Log verifier disagreements: reward=1 but manual QA disagrees, or vice versa; aim for <2% discrepancy.

---

## 18. Final reference list (v3 additions)

Hugging Face + alphaxiv + late-breaking (May 2026):

- [Apriel-Reasoner 2604.02007](https://arxiv.org/abs/2604.02007)
- [Rethinking OPD 2604.13016](https://arxiv.org/abs/2604.13016)
- [KnowRL 2604.12627](https://arxiv.org/abs/2604.12627)
- [DiPO 2604.13902](https://arxiv.org/abs/2604.13902)
- [Output Diversity Collapse 2604.16027](https://arxiv.org/abs/2604.16027)
- [HiExp 2604.08124](https://arxiv.org/abs/2604.08124)
- [SAS 2604.24003](https://arxiv.org/abs/2604.24003)
- [Lightning OPD 2604.13010](https://arxiv.org/abs/2604.13010)
- [UDM-GRPO 2604.18518](https://arxiv.org/abs/2604.18518)
- [FGRPO 2604.08476](https://arxiv.org/abs/2604.08476)
- [PTE 2604.05404](https://arxiv.org/abs/2604.05404)
- [ToolsRL 2604.19945](https://arxiv.org/abs/2604.19945)
- [NPO 2604.20733](https://arxiv.org/abs/2604.20733)
- [GFT 2604.14258](https://arxiv.org/abs/2604.14258)
- [CaOPD 2604.16830](https://arxiv.org/abs/2604.16830)
- [NVIDIA Spec-Dec 2604.26779](https://arxiv.org/abs/2604.26779)
- [M-GRPO 2512.13070](https://arxiv.org/abs/2512.13070)
- [JustRL 2512.16649](https://arxiv.org/abs/2512.16649)
- [Agentic RL Safety 2510.17431](https://arxiv.org/abs/2510.17431)
- [TCOD 2604.24005](https://arxiv.org/abs/2604.24005)
- [TESSY 2604.14164](https://arxiv.org/abs/2604.14164)
