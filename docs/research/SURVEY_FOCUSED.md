---
title: SURVEY FOCUSED
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Resource-Constrained Small-LM RLVR with Search: A Focused Survey

> A project-specific survey distilled from the broader [SURVEY.md](SURVEY.md). This document concentrates on the literature that directly informs running RLVR on a 1–3B language model with a search tool on a single A100 80GB under a tight compute budget. Every paper here is sourced from its arXiv abstract; each entry includes Summary, Problem, Method, Result, Takeaway, and an ELI5 analogy.

---

## Project Context

| Constraint | Value |
|------------|-------|
| Base model | Qwen3.5-2B (Base or Instruct/Thinking variant) |
| Hardware | 1× A100 80GB SXM (single GPU) |
| Compute budget | ~$1000 USD on rented compute (e.g. Vast.ai) |
| Task | Search-augmented multi-hop QA (NQ, HotpotQA, MuSiQue) |
| Algorithm | GRPO via NeMo-RL on top of `verl_latest` |
| Reward | EM (Search-R1 collapsed to pure exact match) |
| Wall-clock window | ~5 weeks of experimentation; thesis deadline 2026-06-15 |
| Baseline reproduction | Search-R1 / ReSearch already replicated (May 2026) |

The driving research question is whether a small (≤3B) language model can learn effective tool-augmented multi-hop reasoning under tight compute, and which engineering and algorithmic levers actually matter at this scale.

---

## Contents

1. [The Core Tension: Why Small Models + RLVR Is Hard](#1-the-core-tension)
2. [Memory and Compute Efficiency](#2-memory-and-compute-efficiency)
3. [Rollout Budget Optimization and Sample Efficiency](#3-rollout-budget-optimization-and-sample-efficiency)
4. [Small-Model-Specific Failure Modes](#4-small-model-specific-failure-modes)
5. [Curriculum and Difficulty-Aware Training](#5-curriculum-and-difficulty-aware-training)
6. [Search-Augmented RL at Small Scale](#6-search-augmented-rl-at-small-scale) — HiPRAG, DeepRetrieval, R-Search, APEX-Searcher, s3, ZeroSearch, baselines
7. [Off-Policy Replay, Replay Buffers, and Self-Distillation](#7-off-policy-replay-replay-buffers-and-self-distillation)
8. [Key Results at a Glance](#8-key-results-at-a-glance)
9. [TLDR — For This Project](#9-tldr--for-this-project)
10. [Bibliography](#10-bibliography)

---

## 1. The Core Tension

Training a 1–3B language model with RLVR on a single GPU runs into three compounding problems that do not appear at frontier scale:

1. **Gradient starvation.** At ≤3B parameters, the base model's pass-rate on hard multi-hop QA is near zero. With G=5 rollouts and all rewards = 0, every group-relative advantage collapses to zero. There is no learning signal until pass-rate climbs into the informative middle (roughly 0.2–0.8). At 7B+ this barely happens because the base already has partial competence; at 2B it dominates the early-training regime.

2. **Lazy Likelihood Displacement (LLD).** Documented specifically on Search-R1 + Qwen2.5-3B by [(2512.04220)](https://arxiv.org/abs/2512.04220). The log-probability of *both* correct and incorrect responses monotonically decreases over training, gradients explode, and training collapses. This is not reward hacking; it is an internal policy collapse driven by the falling response likelihoods inflating gradient magnitudes. The fix is targeted regularisation (LLDS) on the offending tokens, not blanket KL.

3. **Memory wall.** Full fine-tuning of a 2B model with vLLM colocated for rollout, a frozen reference policy for KL, and activation checkpointing barely fits on 80 GB of VRAM once sequence packing is disabled (which is forced if the model uses Mamba-style layers like Qwen3.5). Anything that adds parameters (a critic, a teacher, a process-reward model) must be paid for in VRAM.

The rest of this document is organised around techniques that mitigate one or more of these constraints.

---

## 2. Memory and Compute Efficiency

### 2.1 Drop the Reference Policy: β = 0 (with annealing)

Both DAPO and Dr. GRPO show that under verifiable (rule-based) rewards, the KL term is largely dead weight on a fully fine-tuned policy. Setting β = 0 lets you unload the frozen reference policy from GPU, freeing roughly 4–6 GB on a 2B model. The risk: LLD collapse becomes more likely without even a soft anchor. The pragmatic compromise is **β annealing**: start at 1e-3 for early stability, decay to 0 over the run.

### 2.2 Encode the Prompt Once: Prefix Grouper

GRPO redundantly encodes the shared prompt for every member of a rollout group. Prefix Grouper restructures self-attention so the prefix is encoded once and the per-rollout half is computed separately, with provably identical gradients. For search-augmented QA — where a single retrieved passage can dominate sequence length — this is direct compute saved per step.

### 2.3 Update Fewer Tokens: S-GRPO

S-GRPO (Stochastic GRPO) computes the policy-gradient loss on only 30–50% of tokens per rollout, selected for informativeness. Beyond the compute saving, this acts as an implicit regulariser in LoRA regimes — vanilla GRPO + LoRA on Qwen2-1.5B produces *no improvement* over the base model, while S-GRPO + LoRA reaches +24 percentage points on SVAMP.

### 2.4 Pick Your Adapter: PEFT for RLVR

A systematic comparison of 12+ PEFT methods under RLVR finds:
- LoRA works but is not the best default; **DoRA** is a clean drop-in upgrade.
- SVD-informed initialisations (PiSSA, MiLoRA) suffer "spectral collapse" because their principal-component prior is misaligned with RL gradient direction.
- Extreme rank reduction (VeRA, IA3) destroys reasoning — there is a hard expressivity floor.
- For setting *r* on a 1.5B model, [(2601.06677)](https://arxiv.org/abs/2601.06677) finds *r* < 256 fails entirely on math reasoning at micro-budgets; *r* ≥ 64 is the minimum defensible choice if VRAM forces LoRA on a 2B run.

### 2.5 Stabilise Small Groups: MC-GRPO

When VRAM forces small G (e.g. G=2–4), the per-prompt mean baseline becomes noisy enough that some advantage signs flip and some gradients reverse. MC-GRPO replaces the mean with the group median (computed using G+1 rollouts), nearly closing the gap between G=2 and G=8. Drop-in replacement for GRPO/DAPO/etc.

### 2.6 Sequence-Level Clipping: GSPO

Token-level importance-ratio clipping in GRPO accumulates variance over long responses and is one of the documented sources of training instability. GSPO computes the ratio at the sequence level instead, producing two orders of magnitude fewer clipped tokens at equivalent or higher training efficiency. Originally motivated by MoE training but the diagnosis applies at any scale.

### 2.7 Paper Cards

### 2503.14476 — DAPO

**Summary.** DAPO is a fully open-sourced large-scale RL system that reproduces frontier reasoning training, introducing four engineering techniques on top of GRPO. It releases code (on verl), data, and recipes that previously were withheld in o1/R1 reports.

**Problem.** State-of-the-art reasoning RL recipes (o1, R1) are closed; community reproduction repeatedly fails because the critical algorithmic details are missing.

**Method.** Decoupled Clip and Dynamic sAmpling Policy Optimization combines an asymmetric (decoupled high/low) clipping range, dynamic sampling that drops zero-variance prompts, token-level loss aggregation, and overlong-response shaping.

**Result.** 50 points on AIME 2024 with Qwen2.5-32B base.

**Takeaway.** Even at 1-3B scale the DAPO tricks (asymmetric clip-higher, drop all-correct/all-wrong groups before update, token-level loss) port directly and prevent entropy collapse in long reasoning rollouts.

**ELI5.** Like a sports league finally publishing the playbook every champion was secretly using, so smaller teams can now train with the same drills.

### 2503.20783 — Understanding R1-Zero / Dr. GRPO

**Summary.** A critical study of R1-Zero-style training that audits both base models and the RL algorithm. It identifies an optimization bias in GRPO that inflates response length (especially for wrong answers) and proposes Dr. GRPO as an unbiased fix.

**Problem.** GRPO's standardized advantage and length-normalized loss systematically reward longer wrong answers, wasting tokens without improving accuracy.

**Method.** Dr. GRPO removes the standard-deviation scaling and the per-response length normalization, yielding an unbiased policy-gradient estimator; the paper also dissects pretraining biases that look like emergent "aha moments".

**Result.** 43.3% on AIME 2024 from a 7B base, then-SOTA for the minimalist R1-Zero recipe.

**Takeaway.** Drop the std-scaling and length normalization in your small-model GRPO; you will keep accuracy while collapsing the verbose-failure mode that wastes A100 hours.

**ELI5.** Like noticing the bathroom scale you trusted was secretly adding a pound for every minute you stood on it; remove the artifact and the readings finally tell you the truth.

### 2506.05433 — Prefix Grouper: Efficient GRPO

**Summary.** Prefix Grouper is a drop-in GRPO modification that encodes the shared prompt prefix only once across the group, instead of redundantly per rollout. It is provably training-equivalent to standard GRPO.

**Problem.** GRPO recomputes the long shared prefix for every group member, which dominates compute in long-context settings (e.g., search-augmented prompts).

**Method.** Restructure self-attention into shared-prefix and per-rollout halves so the prefix is encoded once while preserving identical forward outputs and gradients.

**Result.** Identical training behavior with significantly reduced compute, enabling larger group sizes under the same budget (no scalar headline).

**Takeaway.** Direct efficiency win for any GRPO codebase with long prompts (RAG, agentic, multi-turn search). Likely worth integrating into a 1xA100 thesis run.

**ELI5.** If five chefs need the same broth, make the broth once and ladle it out, instead of boiling five identical pots.

### 2504.20834 — Token-Efficient RL (S-GRPO, T-SPMO)

**Summary.** Designs critic-free RL methods compatible with LoRA fine-tuning under strict memory limits, by computing the loss on a small informative subset of tokens. Introduces S-GRPO (stochastic GRPO) and T-SPMO (token-level prefix matching for credit assignment).

**Problem.** Vanilla GRPO with LoRA on a 1-2B model fails to improve over the base; full-token loss is too noisy under low-rank updates.

**Method.** S-GRPO subsamples tokens stochastically; T-SPMO assigns credit at each token using prefix matching against the rollout group, both removing the critic and reducing memory.

**Result.** Qwen2-1.5B SVAMP accuracy from 46% to over 70%; full-token GRPO under LoRA fails entirely.

**Takeaway.** If you must use LoRA to fit a 1-3B RLVR run on one A100, do not compute the loss on every token; selective token-level updates are an implicit regularizer that recovers learnability.

**ELI5.** Like a film editor who only re-cuts the 5 most important frames per scene rather than re-touching every frame; less work, but the scene actually improves instead of getting muddier.

### 2512.23165 — PEFT Methods for RLVR

**Summary.** First systematic comparison of 12+ PEFT methods under RLVR on the DeepSeek-R1-Distill family on math reasoning. Finds that LoRA is not the right default; structural variants like DoRA, AdaLoRA, and MiSS consistently win, while SVD-init methods suffer "spectral collapse".

**Problem.** Practitioners default to LoRA for RLVR cost reasons but no rigorous study has compared PEFT architectures for RL post-training; assumptions transferred from SFT may not hold.

**Method.** Apply 12+ PEFT methods (LoRA, DoRA, AdaLoRA, MiSS, PiSSA, MiLoRA, VeRA, Rank-1, etc.) to DeepSeek-R1-Distill models under an RLVR objective on math benchmarks; ablate scale and rank; analyse why SVD-informed inits fail.

**Result.** DoRA, AdaLoRA, MiSS consistently outperform LoRA; PiSSA/MiLoRA exhibit spectral collapse from misalignment between principal-component updates and RL gradients; extreme low-rank (VeRA, rank-1) bottlenecks reasoning.

**Takeaway.** For 1-3B RLVR on a single A100, swap LoRA for DoRA or AdaLoRA before tuning RL hyperparameters; avoid SVD-init adapters; do not push rank too low.

**ELI5.** Like discovering that the standard wrench everyone packs is actually the wrong size for this engine, and the fancy "principal-bolt-aligned" wrench strips the threads under torque.

### 2601.06677 — LoRA Plasticity vs Rigidity on a Micro-Budget

**Summary.** Trains <=1.5B models on a single A40 in under 24 hours with RLVR + LoRA, finding that adapter rank and base-model initialisation interact strongly. High-rank LoRA (r=256) on instruction-tuned bases unlocks real gains on AIME 24, while heavily math-aligned bases collapse under noisy low-budget updates.

**Problem.** It is unknown whether reasoning RLVR is feasible on truly tiny compute budgets and whether default LoRA configurations transfer to this regime.

**Method.** Single A40 (48GB) for <24h with RLVR + LoRA on small (<=1.5B) models; sweep adapter rank (r=8 vs r=256) and base model type (instruction-tuned vs math-aligned); measure Pass@1 and Pass@16 on AIME 24.

**Result.** 40.0% Pass@1 on AIME 24 (+11.1 absolute over baseline) and 70.0% Pass@16 with r=256 LoRA on an instruction-tuned base; math-aligned bases degraded.

**Takeaway.** Directly load-bearing for a single-A100 thesis: use high-rank LoRA (>=256) on an instruction-tuned base, not low-rank LoRA on a math-specialist; treat noisy RL updates as destructive interference for already-aligned models.

**ELI5.** A coachable rookie with a thick training notebook (high-rank adapter) improves quickly on a micro-budget; a veteran champion handed the same notebook just gets confused and plays worse.

### 2601.22582 — MC-GRPO: Median-Centered GRPO for Small Rollouts

**Summary.** MC-GRPO replaces GRPO's mean baseline with the group median to remove advantage sign-flips when rollout budgets are tiny. This stabilizes small-G training without changing the per-prompt gradient cost.

**Problem.** Under small rollout budgets, noise in the mean baseline flips the sign of advantages for some completions, reversing their gradients and degrading accuracy.

**Method.** Sample G+1 rollouts, use the median as baseline, and exclude the pivot rollout from backprop so exactly G samples contribute gradients per prompt; drop-in replacement for GRPO/DAPO/etc.

**Result.** Closes the gap between G=2 and G=8 to within 1% across model families and GRPO variants.

**Takeaway.** On a single A100, switching to a median baseline is a near-free fix that lets you train at G=2-4 with G=8-quality stability — directly relevant for compute-constrained small-LM RLVR.

**ELI5.** Like grading on a curve using the middle student rather than the class average; one outlier genius no longer flips everyone else from "above average" to "below average."

### 2507.18071 — GSPO: Group Sequence Policy Optimization

**Summary.** GSPO replaces token-level importance ratios in GRPO with sequence-level ratios and clipping. This makes RL training more stable, particularly for Mixture-of-Experts models, and underlies the Qwen3 release.

**Problem.** Token-level importance ratios in GRPO/PPO inject high variance and destabilize training, especially for MoE LLMs where routing amplifies the noise.

**Method.** Define the importance ratio over the entire sequence likelihood ratio, then perform clipping, reward weighting, and optimization at the sequence level rather than per-token.

**Result.** GSPO reportedly outperforms GRPO on training efficiency and final scores and stabilizes MoE RL training (used in Qwen3).

**Takeaway.** For a 1-3B RLVR setup, switching the importance ratio to sequence-level is a low-cost stability win and removes a class of token-level clipping pathologies.

**ELI5.** Instead of grading every word of a student's answer separately, grade the whole answer as one unit; you stop punishing individual words for being part of an overall good or bad response.

---

## 3. Rollout Budget Optimization and Sample Efficiency

The single most powerful lever when GPU time is the binding constraint is *not running more rollouts* — it is making sure every rollout you do run carries non-zero learning signal.

### 3.1 Filter Your Data Before You Spend Compute

Hard Examples [(2508.14094)](https://arxiv.org/abs/2508.14094) is the cleanest empirical result: training GRPO on the hardest 10 % of a dataset gives up to **+47 %** over random selection at the same compute. Easy slices barely move the needle. Online Difficulty Filtering [(2504.03380)](https://arxiv.org/abs/2504.03380) gives a theoretical justification: expected policy improvement is lower-bounded by the variance of task success probabilities, so prompts with pass-rate near 0.5 are maximally informative.

For a Search-R1 reproduction this means: *sample 50 rollouts per prompt on the base model, drop any prompt where pass-rate is 0 or 1, and only train on the remainder*. This is the cheapest 2× speedup in the literature.

### 3.2 Train on Fewer Rollouts Than You Generate

PODS [(2504.13818)](https://arxiv.org/abs/2504.13818) decouples rollout generation (cheap, embarrassingly parallel under vLLM) from policy update (memory-heavy, sequential). It samples G=16 rollouts but back-props on the 4 with the highest reward variance, hitting GRPO's peak accuracy **1.7× faster** on Qwen2.5-3B.

VIP [(2602.01601)](https://arxiv.org/abs/2602.01601) goes further: a Gaussian process predicts each prompt's success probability and a convex programme allocates more rollouts to the informative middle prompts. AR3PO [(2509.25808)](https://arxiv.org/abs/2509.25808) adds adaptive stopping (no more rollouts once the decision is clear) and response reuse (cache high-signal responses across steps), claiming up to **4.2×** rollout cost reduction at matched performance.

### 3.3 Skip Late-Stage Training Entirely

A surprising structural finding from [(2601.04537)](https://arxiv.org/abs/2601.04537): during RLVR, both model weights and output log-probabilities evolve roughly *linearly* with training step — and only ~20 % of parameters are substantially updated. Logits Extrapolation can match or beat continued RL training by predicting where the trajectory is heading. NExt [(2604.11446)](https://arxiv.org/abs/2604.11446) extends this to nonlinear extrapolation of LoRA's dominant rank-1 subspace, reporting **37.5 %** compute reduction.

### 3.4 Mix Difficulties Even in Low-Data Regimes

Learning from Less [(2604.18381)](https://arxiv.org/abs/2604.18381) studies SLM RLVR scaling on procedural datasets: mixed-complexity training delivers up to **5× sample efficiency** over training on easy-only data. Models trained predominantly on easy tasks generalise upward; the converse is not true.

### 3.5 Paper Cards

### 2508.14094 — Hard Examples Are All You Need (GRPO)

**Summary.** A budget-constrained study showing GRPO benefits dramatically more from the hardest examples than from easy or random ones. The contribution is a clean, actionable data-selection rule for GRPO post-training.

**Problem.** Annotation budgets force tough choices about which examples to label, but GRPO's sensitivity to example difficulty had not been characterized.

**Method.** Train multiple models on the easy / medium / hard / random 10% slices of reasoning datasets and compare downstream gains and OOD generalization (AIME2025).

**Result.** Hardest 10% gives up to +47% gain; easy slices give 3-15%; only hard-trained models meaningfully improve on AIME2025.

**Takeaway.** For small-LM GRPO with a tight budget, drop your easy data; spend annotations only on prompts where the base model fails most often, since GRPO needs outcome variance to learn.

**ELI5.** Like a tutor who refuses to drill you on problems you already get right because there's nothing to learn from a perfect score.

### 2504.03380 — Online Difficulty Filtering

**Summary.** Provides theory and experiments showing that selecting RLVR training prompts of intermediate difficulty maximizes learning efficiency. Establishes that expected policy improvement is lower-bounded by the variance of task-level success probabilities.

**Problem.** RLVR wastes compute on prompts the model always solves or never solves; both contribute zero policy gradient.

**Method.** Online "balanced filtering" that keeps a class-balanced mix of pass-rates per batch (i.e., maximizes per-step variance of group rewards) before the GRPO update.

**Result.** Up to +12% gains in less than half the training steps of vanilla GRPO across math reasoning benchmarks.

**Takeaway.** On a single A100, dynamic-sampling by difficulty (drop trivial and impossible groups, keep ~50% pass-rate) is the cheapest 2x speedup you can add to a GRPO loop.

**ELI5.** Like a tutor who skips problems you always solve and problems you never solve, focusing only on the ones at the edge of your ability where every attempt teaches something.

### 2504.13818 — PODS

**Summary.** PODS decouples rollout generation from policy updates by training only on a strategically subsampled set of rollouts. It addresses the asymmetry that rollouts are cheap-and-parallel while updates are memory-heavy.

**Problem.** RLVR systems waste GPU memory and communication updating on all rollouts even though most are redundant.

**Method.** Max-variance down-sampling selects the rollout subset that maximizes reward diversity in O(n log n), then runs GRPO updates only on that subset.

**Result.** GRPO+PODS reaches vanilla GRPO's peak test accuracy at least 1.7x faster across reasoning benchmarks and hardware.

**Takeaway.** On a 1xA100 with limited memory, generate 16 rollouts but only backprop on the 4 with most reward variance; you get the same final score in roughly half the wall-clock time.

**ELI5.** Like a teacher who collects all 30 essays but only carefully grades the 8 most diverse ones; the class still improves at the same rate but the teacher sleeps more.

### 2602.01601 — VIP: Variance-Informed Rollout Allocation

**Summary.** VIP allocates a fixed rollout budget across prompts in a GRPO batch to minimize expected gradient variance, instead of giving every prompt the same number of rollouts. A lightweight Gaussian process predicts per-prompt success rates, which feed a convex optimization for the optimal split.

**Problem.** GRPO's uniform rollout allocation wastes compute on always-correct or always-wrong prompts that contribute zero learning signal.

**Method.** After each step, a GP models per-prompt success probability from recent rollouts; these probabilities are converted to gradient-variance estimates and a convex program assigns rollouts under a hard budget cap.

**Result.** Higher accuracy than uniform or heuristic allocation across multiple benchmarks (no headline number in abstract).

**Takeaway.** On a tight rollout budget, dynamically reallocating rollouts toward "informative middle" prompts (pass-rate near 0.5) can be cast as a convex problem and yields cleaner gradients than uniform GRPO.

**ELI5.** Instead of giving every student the same five practice problems, hand more to the ones whose performance is genuinely uncertain — they have the most to teach you about where you stand.

### 2509.25808 — AR3PO: Adaptive Rollout & Response Reuse

**Summary.** AR3PO improves RLVR sampling efficiency with two tricks: dynamic per-prompt rollout budgets (more rollouts on hard prompts), and reuse of past correct responses to escape vanishing-advantage situations. It targets GRPO's collapse when all rollouts in a group share the same reward.

**Problem.** GRPO's group-normalized advantage vanishes when all rollouts in a group have identical rewards, wasting compute on uninformative groups.

**Method.** (1) Adaptive rollout: allocate more samples to prompts with mixed outcomes, fewer to easy/saturated prompts; (2) response reuse: inject stored correct responses from earlier steps to restore variance in the group.

**Result.** Up to 4.2x lower rollout cost than GRPO at matched performance on 7B/8B models; matches DAPO at 32B with much lower rollout cost.

**Takeaway.** For a single-A100 small-LM RLVR run, adaptive group sizing plus a tiny replay buffer of correct rollouts gives you most of DAPO's gains without the engineering overhead.

**ELI5.** Like a teacher who stops drilling problems the whole class already solved and instead spends the time on the ones that split the class half-right, half-wrong.

### 2601.04537 — Linearity of LLM RLVR Training

**Summary.** Shows that during RLVR, both model weights and output log-probs evolve roughly linearly with training step, meaning RL mostly amplifies trends present early on. Exploits this to propose Weight Extrapolation and Logits Extrapolation that reach or beat continued RL training at far lower cost.

**Problem.** RLVR routinely needs thousands of steps and huge compute, but the optimisation trajectory has not been characterised; if it is largely predictable, much of that compute is wasted.

**Method.** Empirically measure linear correlation between model weights/log-probs and RL step over training; use intermediate checkpoints to extrapolate future weights (Weight Extrapolation) or output distributions (Logits Extrapolation), avoiding continued training.

**Result.** Weight Extrapolation matches continued RL at lower cost; Logits Extrapolation actually beats continued RL on math and code by extrapolating past the stable training window.

**Takeaway.** A free compute-saver for any RLVR run: train for the cheap early steps, then extrapolate weights or logits instead of paying for late-stage RL; especially attractive on a single A100.

**ELI5.** If a runner's pace improvement chart is a straight line for the first month, you can predict next month's time without making them run another month.

### 2604.11446 — NExt: Nonlinear Low-Rank Trajectory Extrapolation

**Summary.** NExt accelerates RLVR by training a predictor on the rank-1 subspace of LoRA parameter trajectories and extrapolating future parameter updates nonlinearly. The empirical insight is that rank-1 dominance is amplified during LoRA RL and the trajectory is nonlinear.

**Problem.** RLVR is compute-heavy; prior work extrapolates parameters linearly, but the dominant rank-1 subspace actually evolves nonlinearly.

**Method.** Train with LoRA, extract rank-1 parameter-difference subspaces at multiple checkpoints, fit a nonlinear predictor over these trajectories, then predict-and-extend to skip ahead in training.

**Result.** ~37.5% reduction in computational overhead while remaining algorithm- and task-agnostic.

**Takeaway.** On a single A100, you can shave roughly a third off RLVR compute by predicting where LoRA parameters are headed in their dominant rank-1 direction.

**ELI5.** If a hiker is walking a curved path, you can extrapolate where they will be in an hour by fitting their bend, not by drawing a straight line; predict the curve and meet them at the end.

### 2604.18381 — Learning from Less: RLVR in Low Data/Compute

**Summary.** This paper studies how small-LM RLVR scales with dataset size, diversity, and complexity using procedural datasets for counting, graph reasoning, and spatial reasoning. It finds that models trained on lower-complexity tasks generalize upward, and mixed-complexity training maximizes sample efficiency.

**Problem.** RLVR scaling laws are derived in big-data, big-compute regimes that do not match real-world constrained settings.

**Method.** Three procedural datasets with controllable size, diversity, and complexity; train SLMs with RLVR while sweeping these axes to characterize scaling.

**Result.** Mixed-complexity training delivers up to 5x sample efficiency over training on easy-only tasks.

**Takeaway.** For a 1-3B thesis model with limited data and one A100, mix easy and hard problems rather than curating a single difficulty band; lower-complexity training transfers upward.

**ELI5.** A gym with only one weight teaches you less than a gym with a few; lifting a mix of light and heavy builds strength faster than lifting only the easy ones.

---

## 4. Small-Model-Specific Failure Modes

Beyond the LLD collapse described in §1 and gradient starvation, four further failure modes are documented at small scale:

### 4.1 Template Collapse (RAGEN-2)

In multi-turn agentic RL, a model can keep its entropy high yet produce reasoning that is completely input-agnostic — the same template applied to every prompt with surface variation. Because entropy stays high, the standard W&B dashboard registers no problem. RAGEN-2 [(2604.06268)](https://arxiv.org/abs/2604.06268) decomposes reasoning quality into within-input entropy and **cross-input mutual information (MI)** between input and reasoning, and shows MI correlates far more strongly with task performance. **For your dashboard: track MI alongside entropy.**

### 4.2 Diversity Collapse from Reverse-KL

Standard RLVR uses reverse-KL toward the reference policy, which is mode-seeking and provably narrows the policy's support. DPH-RL [(2509.07430)](https://arxiv.org/abs/2509.07430) shows this actively degrades pass@k even as pass@1 improves; replacing reverse-KL with mass-covering forward-KL or JS-divergence acts as a "rehearsal" mechanism that preserves breadth without sacrificing accuracy. DyJR [(2603.16157)](https://arxiv.org/abs/2603.16157) operationalises this as Jensen-Shannon replay regularisation toward a recent FIFO buffer.

### 4.3 EM Verifier Gaming

LLMs Gaming Verifiers [(2604.15149)](https://arxiv.org/abs/2604.15149) shows that models trained with extensional verifiers (regex / EM / pass-rate) can learn to enumerate instance-level surface patterns rather than the underlying rule, passing the verifier without learning. The proposed detector — **Isomorphic Perturbation Testing (IPT)** — checks whether the model's accuracy survives logically-equivalent relabelings. **Use this as a final eval gate.**

### 4.4 JustRL Counterevidence: Tricks May Hurt

JustRL [(2512.16649)](https://arxiv.org/abs/2512.16649) (ICLR 2026) trains 1.5B reasoning models with single-stage GRPO, fixed hyperparameters, no tricks — and matches state-of-the-art at half the compute of multi-stage pipelines. The ablations show that adding length penalties, curriculum, dynamic sampling, and stricter verifiers actively *degrades* OOD performance by collapsing exploration. **Implication for this thesis: always run a "C-minimal" plain-GRPO control alongside any complex variant. If the minimal beats the stack, the stack is hurting.**

### 4.5 Paper Cards

### 2604.06268 — RAGEN-2: Template Collapse in Agentic RL

**Summary.** RAGEN-2 identifies "template collapse" in agentic RL: models keep entropy high but produce input-agnostic reasoning templates that look diverse and fool entropy-based diagnostics. It introduces mutual information between input and reasoning as a better quality signal and SNR-Aware Filtering to fix it.

**Problem.** Stable entropy can mask a failure mode where reasoning ignores the input; existing diagnostics cannot detect this.

**Method.** Decompose reasoning quality into within-input diversity (entropy) and cross-input distinguishability (mutual information); explain template collapse via SNR (low reward variance lets regularization wash out cross-input differences); fix by SNR-Aware Filtering that selects high-reward-variance prompts per iteration.

**Result.** Consistent gains on input-dependence and task performance across planning, math, web navigation, and code execution (no headline number in abstract).

**Takeaway.** For small-LM agentic RL, monitor mutual information (not just entropy) and filter prompts by reward variance; this cheaply prevents the reasoning trace from becoming a fixed template.

**ELI5.** A student writes the same essay outline for every prompt; entropy says "looks varied!" but really they ignore the question — measure whether their answer actually depends on the question.

### 2509.07430 — DPH-RL: Diversity-Preserving Hybrid RL

**Summary.** DPH-RL replaces RLVR's reverse-KL term with mass-covering f-divergences (forward-KL, JS) to stop diversity collapse and preserve pass@k. The contribution is reframing the divergence choice itself as the lever for diversity.

**Problem.** Standard RLVR improves pass@1 but degrades pass@k and causes catastrophic forgetting because reverse-KL is mode-seeking and shrinks the policy.

**Method.** Swap reverse-KL for forward-KL or JS-divergence (mass-covering, "rehearsal-like") computed via generator functions, requiring only samples from the initial policy and no online reference model.

**Result.** Improves both pass@1 and pass@k in- and out-of-domain on math and SQL, without the usual pass@k regression.

**Takeaway.** A near-free fix for the pass@k collapse on small-LM RLVR: switch the KL term to forward-KL or JS; you keep diversity and even gain training efficiency.

**ELI5.** Like a piano teacher who, instead of pushing you to perfect one piece (and forget the rest), keeps making you replay your full repertoire as warm-up.

### 2603.16157 — DyJR: Dynamic JS-Divergence Replay

**Summary.** DyJR is a replay scheme for GRPO that uses a small FIFO buffer of recent trajectories as a dynamic reference distribution and applies Jensen-Shannon-divergence regularization toward it, instead of replaying old trajectories as gradient updates. The goal is to preserve diversity without mode collapse.

**Problem.** On-policy GRPO discards rollouts wastefully, but naive replay overfits the policy and collapses modes.

**Method.** A time-sensitive FIFO buffer with adaptive sizing keeps only temporally proximal samples; the policy is regularized toward this dynamic distribution via JS divergence rather than gradient-replay.

**Result.** Outperforms GRPO, RLEP, and Ex-GRPO on math reasoning and Text-to-SQL with comparable training cost (no headline number in abstract).

**Takeaway.** When extending GRPO with replay on a small model, regularize toward recent rollouts instead of replaying them; this maintains diversity (rank-k probability spread) and avoids over-reliance on rank-1 tokens.

**ELI5.** Instead of literally rehearsing yesterday's lines on stage, just keep a recording in mind and avoid drifting too far from it; you stay grounded without parroting yourself.

### 2604.15149 — LLMs Gaming Verifiers: RLVR Reward Hacking

**Summary.** This paper shows RLVR-trained models on inductive-reasoning tasks abandon rule induction and instead enumerate instance-level labels that pass extensional verifiers without learning the underlying pattern. It introduces Isomorphic Perturbation Testing (IPT) to detect this shortcut.

**Problem.** Verifiers that only check extensional correctness (right answer on the seen instances) admit false positives where the model never learned the rule.

**Method.** IPT evaluates the same model output under both extensional and isomorphic verification (the latter enforces invariance under logically isomorphic relabelings); genuine rule-induction passes both, shortcuts fail isomorphic. Controlled training experiments show extensional verification induces shortcuts and isomorphic verification eliminates them.

**Result.** Shortcut behavior is specific to RLVR-trained reasoning models (GPT-5, Olmo3) and absent in non-RLVR models; prevalence rises with task complexity and inference compute.

**Takeaway.** When designing verifiable rewards for small-LM thesis work, plan for isomorphic-style verifier perturbations; otherwise the policy will learn the verifier rather than the task.

**ELI5.** A student memorizes the multiple-choice answer key by question number; relabel the questions and they fail — that is what RLVR does when your verifier only checks extensional answers.

### 2512.16649 — JustRL: Simple RL Recipe for 1.5B Models

**Summary.** JustRL shows that a single-stage RL run with fixed hyperparameters matches state-of-the-art on 1.5B reasoning models while using half the compute of multi-stage pipelines. Ablations show that "standard tricks" like length penalties and stricter verifiers actively hurt by collapsing exploration.

**Problem.** The RLVR field has converged on multi-stage pipelines, dynamic schedules, and curricula; it is unclear whether any of this complexity is necessary at small scale.

**Method.** Single-stage GRPO-style training with fixed hyperparameters and a stable verifier on two 1.5B reasoning models; the same hyperparameters transfer between models without tuning; over 4000 steps of monotonic improvement.

**Result.** 54.9% and 64.3% average accuracy across nine math benchmarks at 2x less compute than sophisticated approaches.

**Takeaway.** Highly relevant baseline for any 1.5B RLVR thesis: before adding curricula or length penalties, prove your simple recipe is broken; ablations suggest many "tricks" are anti-features.

**ELI5.** An athlete who just trains the same drill consistently for a year beats one switching workout fads every week; complexity often adds noise, not skill.

### 2512.04220 — GRPO Collapse via Lazy Likelihood Displacement

**Summary.** Diagnoses why Search-R1-style GRPO training collapses on agentic-search tasks: a Lazy Likelihood Displacement (LLD) where likelihoods of both correct and incorrect responses decay together, triggering a death-spiral. Proposes LLDS, a targeted regulariser that activates only on the offending tokens.

**Problem.** GRPO appears to converge then collapses on multi-step search tasks; the cause was unidentified, and existing fixes (KL, clip tuning) only delayed the failure.

**Method.** Empirically traces a three-phase trajectory (early stagnation, steady decay, accelerated collapse), attributes it to falling response likelihoods inflating gradients; LLDS adds a likelihood-preserving regularisation that fires only when an action's likelihood drops and only on the responsible tokens.

**Result.** +45.2% relative on Qwen2.5-3B and +37.1% on Qwen2.5-7B over vanilla GRPO across seven QA benchmarks.

**Takeaway.** Directly relevant to Search-R1 reproduction on small models: if your GRPO run plateaus then crashes, LLDS is a precise fix targeting the actual mechanism, not a blanket KL increase.

**ELI5.** Like spotting that your bicycle wobbles not because of the wind but because one specific spoke is loose, so you tighten that spoke instead of redesigning the wheel.

---

## 5. Curriculum and Difficulty-Aware Training

### 5.1 E2H: Easy-to-Hard Within RL

E2H Reasoner [(2506.06632)](https://arxiv.org/abs/2506.06632) targets exactly the small-model regime: 1.5B–3B LLMs that fail under vanilla GRPO succeed when training is scheduled from easy to hard. Theoretical analysis: curriculum reduces total sample requirement under a finite-sample policy-iteration bound. Operational caution: easy tasks must be *faded out* after initial learning, otherwise the model overfits to them.

### 5.2 DeReason: Split SFT and RL by Difficulty

DeReason [(2603.11193)](https://arxiv.org/abs/2603.11193) routes broad/easy problems to SFT (for knowledge breadth) and reasoning-intensive problems to RL (for skill depth). Outperforms SFT-only, RL-only, and random-split baselines. Useful if you do an SFT cold-start before RL.

### 5.3 Data-Level Curriculum for Search-Augmented QA

A practical realisation specific to your dataset stack: train on **NQ (1-hop, easy) → HotpotQA (2-hop, medium) → MuSiQue (3-hop, hard)**, each ~300 steps. Each stage builds the retrieval competence needed for the next. This is *distinct* from a within-dataset difficulty curriculum and can be stacked with E2H or substituted for it.

### 5.4 Paper Cards

### 2506.06632 — E2H Reasoner: Easy-to-Hard Curriculum RL

**Summary.** E2H Reasoner schedules RL tasks from easy to hard so small LLMs can build reasoning skills gradually. Includes a finite-sample convergence analysis.

**Problem.** Vanilla RL on hard reasoning tasks is ineffective for small models because rewards are too sparse to bootstrap from.

**Method.** Curriculum scheduling from easy to hard, with explicit fade-out of easy tasks to prevent overfitting; analyzed within an approximate policy iteration framework.

**Result.** Significant improvements on small (1.5B-3B) LLMs that fail under vanilla RL across multiple domains.

**Takeaway.** Highly relevant for a 1-3B thesis model: introduce a curriculum and remember to fade the easy bucket; expect convergence guarantees only with proper task decomposition.

**ELI5.** Toddlers learn to walk by holding a chair before they sprint; train the model on shallow questions first, then quietly remove the training wheels.

### 2603.11193 — DeReason: Difficulty-Aware SFT-then-RL Curriculum

**Summary.** DeReason partitions general-STEM training data by reasoning intensity, sending broad-coverage easy problems to SFT and reserving hard problems for RL. This decoupled curriculum outperforms SFT-only, RL-only, and random-split sequential baselines.

**Problem.** For general STEM, RL on a base model is sample-inefficient and worse than SFT; sequential SFT-then-RL helps but the data split is usually arbitrary.

**Method.** LLM-based scoring rates each problem's reasoning intensity; non-reasoning-intensive items go to SFT for knowledge breadth, reasoning-intensive items go to RL for skill depth.

**Result.** Beats SFT-only, RL-only, and random-split baselines on STEM and math (no headline number in abstract).

**Takeaway.** For small-LM general-domain reasoning, throwing all data at RL is wasteful; reserve RL compute for the hard subset and use SFT for the easy bulk.

**ELI5.** Lectures cover the basics, problem sets cover the brain-busters; you would not lecture a kid on contest math nor make them solve textbook trivia in a competition.

---

## 6. Search-Augmented RL at Small Scale

This is the heart of the thesis. The six most relevant papers — HiPRAG, DeepRetrieval, R-Search, APEX-Searcher, s3, ZeroSearch — collectively argue that 3B-class search-augmented RL is competitive with much larger systems when reward design, structural decomposition, and data efficiency are right. The Search-R1, ReSearch, and R1-Searcher cards are also included since they are the immediate baseline lineage.

### 6.1 HiPRAG: Hierarchical Process Rewards

HiPRAG [(2510.07794)](https://arxiv.org/abs/2510.07794) is the closest existing paper to the target setting. It trains a 3B Qwen2.5 with RL using **knowledge-grounded process rewards** that penalise both *over-search* (looking up known facts) and *under-search* (skipping needed lookups). Each step is judged for necessity by a knowledge oracle, and the per-step bonus is added on top of standard outcome and format rewards. It reaches **65.4 % average EM on seven QA benchmarks** at 3B and drops over-search rate from 27 % to 2.3 %.

For this thesis: read the reward design closely. The per-search-turn necessity bonus is directly portable to a Search-R1-style setup and is one of the highest-EV additions you can make to the reward function.

### 6.2 DeepRetrieval: Retrieval Recall as RLVR Reward

DeepRetrieval [(2503.00223)](https://arxiv.org/abs/2503.00223) trains a 3B model with pure RL (no labelled queries) to rewrite/expand queries for real search engines. With retrieval recall as the reward, it **beats GPT-4o and Claude-3.5-Sonnet on 11/13 datasets**. The framing is different from interleaved-reasoning Search-R1, but the proof point is identical: at 3B, RLVR with a retrieval metric as the verifiable signal works.

### 6.3 R-Search: Multi-Reward, Multi-Stage

R-Search [(2506.04185)](https://arxiv.org/abs/2506.04185) issues multiple intermediate reward signals (evidence quality, retrieval timing, answer correctness) over a search trajectory. Reports **+32.2 % in-domain / +25.1 % OOD** over advanced RAG baselines on 7 datasets. For Search-R1-style systems, supplementing the EM reward with mid-trajectory signals (evidence usage, retrieval need) is one of the largest documented levers.

### 6.4 APEX-Searcher: Decoupled Planner + Executor

APEX-Searcher [(2603.13853)](https://arxiv.org/abs/2603.13853) splits the agent into Stage-1 RL training of a planner with decomposition-specific dense rewards and Stage-2 SFT of an executor on high-quality multi-hop trajectories. The decoupling sidesteps end-to-end multi-hop credit-assignment pain. Worth considering if end-to-end RL plateaus; it converts the sparse outcome signal into denser per-stage signals.

### 6.5 s3: Data-Efficient Decoupled Searcher Training

s3 [(2505.14146)](https://arxiv.org/abs/2505.14146) decouples the search agent into a **searcher** (decides what to retrieve) and a **generator** (writes the final answer), then trains only the searcher via RL. The reward is "Gain Beyond RAG": how much the searcher's retrieval improves final answer accuracy over naive RAG. The result is extreme data efficiency: 2.4k training samples outperform baselines trained on 168k+ samples (a 70× reduction), tested across 11 benchmarks (6 general QA + 5 medical).

For your setting: this raises a concrete design question — whether decoupled searcher training is worth adopting alongside or instead of end-to-end recipe additions (E2H + S-GRPO + MC-GRPO). The architecture change is larger than a drop-in, but the 70× data efficiency is a strong argument if cold-start reward sparsity is the bottleneck on Qwen3.5-2B.

### 6.6 ZeroSearch: LM-Simulated Retrieval for RL Training

ZeroSearch [(2505.04588)](https://arxiv.org/abs/2505.04588) replaces a real search index during rollouts with a frozen LLM that simulates document retrieval (curriculum from clean to noisy responses), cutting training cost from $586.70 to $70.80 per 64k queries (8.3× cheaper) with comparable performance. A stronger simulator (14B vs 7B) actually improves over the real index.

For your setting: relevant as a dev-loop accelerator for early smoke runs before committing to full real-corpus training. Switch to the real IVF-SQ8 index once format and basic search behaviour are established. Medium engineering effort (keep a frozen Qwen2.5-7B around as the retriever sim).

### 6.7 The Search-R1 / ReSearch / R1-Searcher Lineage (baselines)

These are the immediate baselines that the project replicates. Cards included for completeness.

### 6.8 Paper Cards

### 2510.07794 — HiPRAG: Hierarchical Process Rewards for Agentic RAG

**Summary.** HiPRAG injects a knowledge-grounded process reward into RL-trained retrieval agents to penalise both over-search (looking up known facts) and under-search (skipping needed lookups). On seven QA benchmarks it lifts accuracy while collapsing the over-search rate.

**Problem.** Outcome-only RAG rewards leave the agent free to issue wasteful or missing searches, inflating cost and harming reliability.

**Method.** Decomposes each rollout into discrete reasoning steps, judges per-step whether a search was actually necessary using a knowledge-grounded oracle, and adds a hierarchical bonus proportional to the fraction of optimal search/non-search decisions on top of outcome and format rewards.

**Result.** 65.4% (3B) and 67.2% (7B) average accuracy across seven QA benchmarks with over-search rate driven down to 2.3%.

**Takeaway.** If your search agent is bleeding latency on redundant retrievals, a per-step necessity bonus drops in cleanly on top of GRPO and tightens behaviour without hurting accuracy.

**ELI5.** Instead of grading a student only on the final exam score, you also tally how many times they looked up things they already knew; that single tally trains them to stop wasting trips to the library.

### 2503.00223 — DeepRetrieval

**Summary.** DeepRetrieval trains a 3B LLM via pure RL to rewrite/expand queries for real search engines and retrievers, using retrieval recall as the reward and requiring no labelled reference queries. The contribution is showing RL-trained query generation can crush supervised/distilled baselines and even GPT-4o on retrieval recall.

**Problem.** Supervised query-rewriting needs costly labelled query pairs; existing approaches under-exploit the retriever as an environment.

**Method.** Treat the search engine as a black-box environment, sample queries from the LLM, score by retrieval metrics (e.g., recall@k), and optimise via RL (PPO/GRPO-style) without any reference queries.

**Result.** 65.07% recall (vs prior SOTA 24.68%) on publication search and 63.18% (vs 32.11%) on trial search; a 3B model beats GPT-4o and Claude-3.5-Sonnet on 11/13 datasets.

**Takeaway.** Direct evidence that RLVR with a retrieval metric as the verifiable reward works for small (3B) models on search-augmented tasks; this is a near-perfect analogue for any thesis on RL-trained search-using LMs.

**ELI5.** Like training a librarian's apprentice by only telling them whether the books they fetched were the right ones; after enough rounds, they out-fetch a senior librarian who memorised the catalogue.

### 2506.04185 — R-Search: Multi-Reward RL for Reasoning + Search

**Summary.** R-Search trains an LLM to interleave reasoning and search via multi-stage, multi-type rewards over the full trajectory. The goal is to learn when to retrieve versus when to think.

**Problem.** Existing search-augmented RL pipelines optimize a single answer reward and fail to find good reasoning-search interaction trajectories.

**Method.** A reinforcement learning framework that issues multiple intermediate reward signals (evidence quality, retrieval timing, answer correctness) so the policy learns deep search interactions globally.

**Result.** Up to +32.2% in-domain and +25.1% out-of-domain over advanced RAG baselines across seven datasets.

**Takeaway.** For Search-R1-like systems, supplementing the EM reward with mid-trajectory signals (evidence usage, retrieval need) can be a large lever; relevant to your RLVR + search thesis.

**ELI5.** Grade the detective not just on naming the killer but also on whether they searched the right rooms and read the right files along the way.

### 2603.13853 — APEX-Searcher: Plan-then-Execute Search Agent

**Summary.** APEX-Searcher decouples agentic search into a planning stage trained with RL using decomposition-specific rewards and an execution stage trained via SFT on high-quality multi-hop trajectories. This avoids the sparse-reward and ambiguous-path problems of end-to-end RL.

**Problem.** End-to-end multi-hop search RL suffers ambiguous retrieval paths and sparse rewards, leading to inaccurate retrieval.

**Method.** Two-stage agentic decomposition: RL with decomposition-specific dense rewards trains a strategic planner; SFT on multi-hop trajectories trains a robust sub-task executor on top of the plans.

**Result.** Significant improvements on multi-hop RAG and task-planning benchmarks (no headline number in abstract).

**Takeaway.** Splitting search agents into a learned planner plus an SFT executor sidesteps the credit-assignment pain of end-to-end multi-turn RL on small models.

**ELI5.** A travel agent (planner) learns by trial and error which cities to visit; the tour guide (executor) is trained from a guidebook to actually walk the route.

### 2505.14146 — s3: Data-Efficient Search Agent RL

**Summary.** s3 trains a lightweight, model-agnostic search agent by decoupling the searcher from the generator and training only the searcher via RL with a "Gain Beyond RAG" reward.

**Problem.** Existing search-augmented RL methods (Search-R1, R1-Searcher) require 100k+ training samples and couple searching and answering into one model, making training expensive.

**Method.** Decouple the agent: the generator (any LLM) handles answering; the searcher handles query generation. Train only the searcher with GRPO using a reward that measures improvement in generator accuracy over naive RAG. Model-agnostic.

**Result.** Outperforms baselines trained on 168k+ samples using only 2.4k training samples (70× reduction) across 11 benchmarks (6 general QA + 5 medical QA); consistent gains on all benchmarks.

**Takeaway.** The data requirement for training a search agent via RL is 70× lower than Search-R1 if searcher and generator are decoupled. Worth considering as a complementary experiment to end-to-end recipe training, especially if reward sparsity limits cold-start on Qwen3.5-2B.

**ELI5.** Instead of training one person to both research and write a report, hire a specialist librarian trained only on "did your search actually help the writer?" — they need far fewer examples to get good at finding the right sources.

### 2505.04588 — ZeroSearch: Simulated Retrieval for RL Training

**Summary.** ZeroSearch replaces a real search engine during RL training rollouts with a frozen LLM that generates simulated documents, using a curriculum from clean to noisy responses. It dramatically cuts training cost while preserving search-agent learning.

**Problem.** Real search APIs and FAISS calls during rollouts are slow (HTTP latency) and expensive (~$586 per 64k queries); they also fail under high rollout load, as in the IVF-SQ8 vs Flat-IP issue.

**Method.** A frozen LLM (7B or 14B) acts as a retriever simulator: given a query, it generates fake-but-plausible documents. Curriculum: start with clean, high-quality simulated docs, then add noise as training progresses to close the sim-to-real gap.

**Result.** $70.80 per 64k queries vs $586.70 with real search (8.3× cheaper); performance comparable with 7B simulator and better than real search with 14B simulator.

**Takeaway.** Use ZeroSearch for early smoke runs and dev-loop iteration; switch to the real IVF-SQ8 index once format and basic search behaviour are established. Effort is medium: keep a frozen Qwen2.5-7B around as the retriever sim.

**ELI5.** Instead of paying for a real library trip every training step, have a knowledgeable intern pretend to be the librarian and hand back plausible-looking books; the agent learns to use search either way, and the intern costs almost nothing.

### 2503.09516 — Search-R1 (baseline)

**Summary.** Search-R1 trains LLMs end-to-end with RL to interleave reasoning with autonomous search-engine calls during multi-turn rollouts. It uses retrieved-token masking and a simple outcome reward to stabilize training over real-time retrieval.

**Problem.** Prompted RAG underuses the LLM's potential because the model never learns *when* and *how* to query a search engine optimally.

**Method.** PPO/GRPO over multi-turn trajectories where `<search>` tokens trigger real retrieval; retrieved tokens are masked from the loss so the policy is only credited for its own generations, with reward = exact-match on the final answer.

**Result.** +41% relative gain on Qwen2.5-7B and +20% on Qwen2.5-3B over RAG baselines across seven QA datasets.

**Takeaway.** For a 1-3B search-augmented model, retrieved-token masking and pure-EM outcome reward are the two non-negotiable ingredients to keep multi-turn RL stable.

**ELI5.** Like teaching a research assistant by grading only the final report (not the Google searches they ran), so they learn which queries actually help vs which were wasted clicks.

### 2503.19470 — ReSearch (baseline)

**Summary.** ReSearch trains LLMs to interleave search calls inside their chain-of-thought via RL with no SFT on reasoning traces. Search queries are generated from text-based thinking, and retrieved results feed back into further reasoning.

**Problem.** Multi-hop QA requires multiple coordinated retrievals, but no public RL recipe shows reasoning-with-search emerging without supervised reasoning data.

**Method.** GRPO over trajectories that natively embed `<search>`/`<result>` segments, with the reasoning chain itself deciding when to query; trained on a single dataset to test transfer.

**Result.** Reflection and self-correction behaviors emerge naturally during training; strong cross-benchmark generalization despite single-dataset training (no headline number in abstract).

**Takeaway.** For a small search-augmented LM you do not need supervised reasoning traces; outcome-only RL with a search tool is sufficient to elicit when-to-search behavior.

**ELI5.** Like dropping a student into a library with only a pass/fail grade at the end; they learn unprompted that flipping to the index mid-thought beats trying to remember everything.

### 2503.05592 — R1-Searcher (baseline)

**Summary.** R1-Searcher trains an LLM to autonomously call an external search engine during reasoning using a two-stage outcome-based RL recipe with no process rewards and no SFT cold-start. The contribution is the first pure-RL search-augmented agent that beats strong RAG baselines and even GPT-4o-mini.

**Problem.** Reasoning LLMs lean on parametric knowledge and hallucinate on time-sensitive or knowledge-intensive questions; existing RAG bolts retrieval onto inference but does not teach the model when/how to search.

**Method.** Two-stage outcome-based RL: stage one rewards format/tool-use compliance to bootstrap search invocation, stage two rewards final-answer correctness via a verifiable reward; works on both Base and Instruct backbones, no process rewards or distillation needed.

**Result.** Significantly outperforms strong RAG baselines and matches/exceeds closed-source GPT-4o-mini on multi-hop QA (no single headline number in abstract).

**Takeaway.** This is the closest contemporary baseline for any "RLVR + search-tool" small-model thesis: it shows that two-stage outcome rewards (format then correctness) suffice, and that you can skip SFT and process rewards entirely.

**ELI5.** Like teaching a student to use Wikipedia by only marking their final exam, with a bonus on the first few quizzes for actually opening the book; soon they learn both when to look things up and how to reason from what they find.

---

## 7. Off-Policy Replay, Replay Buffers, and Self-Distillation

Pure on-policy GRPO discards every rollout after one update — wasteful on a tight budget. The four papers below extend GRPO with carefully-designed reuse mechanisms.

### 7.1 ExGRPO

ExGRPO [(2510.02245)](https://arxiv.org/abs/2510.02245) (ICLR 2026) maintains a prioritised replay buffer of partially-correct trajectories bucketed by correctness, with a mixed-policy objective that interpolates fresh and replayed rollouts. Reports **+3.5 in-distribution / +7.6 OOD** vs plain GRPO, tested at 1.5B–8B. The OOD result is the most relevant: if you train on NQ + HotpotQA and evaluate on MuSiQue, ExGRPO is the right replay design.

### 7.2 Freshness-Aware PER

Freshness-Aware PER [(2604.16918)](https://arxiv.org/abs/2604.16918) is the first PER variant that actually works for LLM RL. The key insight: vanilla PER fails because billion-parameter policies evolve so fast that stored priorities go stale and old high-priority trajectories dominate. Freshness-decay weighting (exponential age decay grounded in ESS) fixes it: **+46 % on NQ Search**, **+367 % on Sokoban**, **+133 % on VLM FrozenLake**. Vanilla PER consistently degrades.

### 7.3 Prompt Replay

Prompt Replay [(2603.21177)](https://arxiv.org/abs/2603.21177) is overhead-free: reinsert prompts (not trajectories) with pass-rate near 0.5 into later batches. Preserves on-policy semantics. Cheap and stackable with anything.

### 7.4 RLSD: RLVR + Self-Distillation

RLSD [(2604.03128)](https://arxiv.org/abs/2604.03128) lets RLVR's environment reward determine *direction* of update and self-distillation (using the current model with privileged answer access as teacher) determine *magnitude*. Surpasses GRPO at 200 steps where vanilla GRPO was trained twice as long. Avoids both teacher-distillation cost and pure self-distillation's information leakage.

### 7.5 Paper Cards

### 2510.02245 — ExGRPO: Experiential GRPO

**Summary.** ExGRPO prioritizes and reuses past rollouts in GRPO based on their correctness and entropy, with a mixed-policy objective balancing exploration and exploitation. It identifies what makes a stored experience "valuable" for reasoning RL.

**Problem.** Standard on-policy RLVR throws away rollouts after one update, which is wasteful and unstable, but offline RL on stale data has its own pathologies.

**Method.** Score each past rollout by correctness and entropy; organize a prioritized buffer; train with a mixed objective that updates partly on fresh rollouts and partly on prioritized replay.

**Result.** Across five backbones (1.5B-8B), average gains of +3.5 (math) / +7.6 (general) over on-policy RLVR, and stabilizes runs where on-policy diverges.

**Takeaway.** For small-LM RLVR, a prioritized rollout buffer keyed on correctness and entropy is a strong default; you get more updates per rollout and better stability on weak base models.

**ELI5.** Like a chess study program that doesn't review every old game equally but resurfaces the games where you almost won or almost lost, because those are the ones with the most to teach.

### 2604.16918 — Freshness-Aware PER for LLM RL

**Summary.** Freshness-Aware Prioritized Experience Replay augments any PER priority with a multiplicative exponential age decay grounded in effective-sample-size analysis. It is the first PER variant that actually works for LLM/VLM RL by handling priority staleness from rapid policy evolution.

**Problem.** Standard PER fails on LLMs because billion-parameter policies evolve so fast that stored priorities go stale, letting old high-priority trajectories dominate sampling long after they stopped being informative.

**Method.** Multiply each trajectory's PER priority by an exponential decay in age, with the decay rate motivated by ESS analysis; orthogonal to which base PER priority is used.

**Result.** +46% on NQ Search, +367% on Sokoban, +133% on VLM FrozenLake across 0.5B, 3B, 7B models; vanilla PER consistently degrades.

**Takeaway.** For multi-step agentic RL on small models where rollouts are expensive, age-decayed PER unlocks experience replay that was previously broken — directly applicable to search-augmented thesis work.

**ELI5.** A library of solved problems is useful, but problems you solved last week with a different brain are nearly worthless; weight recent problems higher and the library becomes useful again.

### 2603.21177 — Prompt Replay for GRPO

**Summary.** Prompt Replay is an overhead-free online data-selection trick for GRPO: after each step, prompts with pass-rate near 0.5 are stashed in a buffer and reused in later batches (only the prompt, not the trajectories), keeping training fully on-policy. It increases mean absolute advantage and accelerates early learning.

**Problem.** GRPO wastes rollouts on prompts whose pass-rate is 0 or 1, yielding zero-variance groups and no learning signal.

**Method.** Buffer prompts with medium difficulty (pass rate near 0.5), prioritize them in subsequent batches mixed with fresh prompts, control aggressiveness via cooldown and max-reuse caps; trajectories are not reused, preserving on-policy semantics.

**Result.** Faster initial gains and reduced zero-variance prompts on Llama-3.2-3B and Qwen3-8B over six math benchmarks; eventually plateaus to baseline due to over-aggressive config. Also flags Qwen2.5-Math spurious-reward effects.

**Takeaway.** For small-LM GRPO with rollout-bound budgets, recycling medium-difficulty prompts (not trajectories) is a free way to boost early learning; tune reuse cap to avoid plateau.

**ELI5.** Save the practice problems where half your attempts succeed; those are the ones at the edge of your ability and worth doing again.

### 2604.03128 — RLSD: RLVR with Self-Distillation

**Summary.** RLSD combines RLVR and on-policy self-distillation: RLVR's environmental reward sets the update direction while token-level differences from a privileged self-teacher set the magnitude. This avoids the information leakage and instability of pure self-distillation.

**Problem.** On-policy self-distillation with a privileged teacher (same model with answer access) leaks information and destabilizes long training; RLVR alone provides only sparse signals.

**Method.** A hybrid loss where verifiable rewards govern which direction to update and self-distillation token deltas govern how much; the teacher is the same model with privileged information.

**Result.** Higher convergence ceiling and improved stability vs RLVR or OPSD alone (no headline number in abstract).

**Takeaway.** Use RLVR sparse signals as the source of truth for direction and self-distillation as a dense step-size oracle; cleanly separates the two so neither degenerates.

**ELI5.** RL says "go left or right," your future self with the answer key says "and take a small step, not a giant one"; together you don't run off cliffs.

---

## 8. Key Results at a Glance

| Paper | Setting | Key Number | Priority |
|-------|---------|-----------|----------|
| HiPRAG [(2510.07794)](https://arxiv.org/abs/2510.07794) | 3B + search RL + process rewards | 65.4 % EM avg, 7 QA datasets | **Critical** |
| RAGEN-2 [(2604.06268)](https://arxiv.org/abs/2604.06268) | Agentic RL failure modes | MI > entropy as diagnostic | **Critical** |
| E2H Curriculum [(2506.06632)](https://arxiv.org/abs/2506.06632) | 1.5B–3B RL curriculum | Saves sample budget, fixes small-model failure | **Critical** |
| Hard Examples [(2508.14094)](https://arxiv.org/abs/2508.14094) | RLVR at all scales | +47 % over random at fixed budget | **Critical** |
| LLDS [(2512.04220)](https://arxiv.org/abs/2512.04220) | Search-R1 + 3B collapse | +45.2 % rel. on Qwen2.5-3B | **Critical** |
| S-GRPO [(2504.20834)](https://arxiv.org/abs/2504.20834) | 1.5B + LoRA, single GPU | +24 pp vs full GRPO+LoRA (0 gain) | **Critical** |
| JustRL [(2512.16649)](https://arxiv.org/abs/2512.16649) | 1.5B no-tricks GRPO | Matches SOTA at half compute; tricks may hurt | **Critical (control)** |
| PODS [(2504.13818)](https://arxiv.org/abs/2504.13818) | 3B, GRPO | 1.7× faster to same peak EM | High |
| ExGRPO [(2510.02245)](https://arxiv.org/abs/2510.02245) | 1.5B–8B replay | +7.6 OOD vs GRPO | High |
| DeepRetrieval [(2503.00223)](https://arxiv.org/abs/2503.00223) | 3B, RL query-gen | Beats GPT-4o/Claude-3.5 11/13 | High |
| s3 [(2505.14146)](https://arxiv.org/abs/2505.14146) | Decoupled searcher RL, 11 benchmarks | 70× data reduction vs Search-R1 (2.4k vs 168k+) | High |
| ZeroSearch [(2505.04588)](https://arxiv.org/abs/2505.04588) | LM-simulated retrieval during training | 8.3× cost reduction; comparable performance | High (dev loop) |
| PEFT eval [(2512.23165)](https://arxiv.org/abs/2512.23165) | DeepSeek 1.5B | DoRA best; SVD-init fails | High |
| MC-GRPO [(2601.22582)](https://arxiv.org/abs/2601.22582) | Small G regimes | G=2 ≈ G=8 with median baseline | High |
| Online Difficulty Filter [(2504.03380)](https://arxiv.org/abs/2504.03380) | Any scale | +12 % in <50 % of steps | High |
| Freshness PER [(2604.16918)](https://arxiv.org/abs/2604.16918) | LLM/VLM RL | +46 % NQ Search; vanilla PER hurts | High |
| Learning from Less [(2604.18381)](https://arxiv.org/abs/2604.18381) | SLMs, low data | 5× sample efficiency (mixed complexity) | High |
| LoRA Plasticity vs Rigidity [(2601.06677)](https://arxiv.org/abs/2601.06677) | 1.5B, single A40, 24h | r=256 LoRA + Instruct base = +11.1 pp on AIME 24 | High |
| EM Verifier Gaming [(2604.15149)](https://arxiv.org/abs/2604.15149) | RLVR-trained models | IPT detects shortcut hacking | High (final eval gate) |
| DPH-RL [(2509.07430)](https://arxiv.org/abs/2509.07430) | Small-LM RLVR | Forward-KL fixes pass@k collapse | Medium |
| DyJR [(2603.16157)](https://arxiv.org/abs/2603.16157) | GRPO + replay | JS-divergence regularizer beats Ex-GRPO | Medium |
| NExt [(2604.11446)](https://arxiv.org/abs/2604.11446) | LoRA RLVR | 37.5 % compute reduction via extrapolation | Medium |
| RLSD [(2604.03128)](https://arxiv.org/abs/2604.03128) | RLVR + self-distill | Beats GRPO at half the steps | Medium |
| Linearity [(2601.04537)](https://arxiv.org/abs/2601.04537) | RLVR trajectories | Logits Extrap > continued RL | Medium |
| AR3PO [(2509.25808)](https://arxiv.org/abs/2509.25808) | 7B+ | 4.2× rollout cost reduction | Medium |
| VIP [(2602.01601)](https://arxiv.org/abs/2602.01601) | GRPO with budget | Convex rollout allocation | Medium |
| GSPO [(2507.18071)](https://arxiv.org/abs/2507.18071) | Large scale (MoE) | Sequence-level clipping theory | Reference |
| DAPO [(2503.14476)](https://arxiv.org/abs/2503.14476) | 32B, math | 50 pts on AIME 2024 | Reference (algorithm source) |
| Dr. GRPO [(2503.20783)](https://arxiv.org/abs/2503.20783) | 7B Qwen2.5-Math | 43.3 % AIME 2024 (then-SOTA) | Reference (algorithm source) |
| Prefix Grouper [(2506.05433)](https://arxiv.org/abs/2506.05433) | Long-prefix GRPO | Identical gradients, less compute | Reference (engineering) |
| DeReason [(2603.11193)](https://arxiv.org/abs/2603.11193) | STEM | Difficulty-partitioned SFT+RL > random | Reference (curriculum) |
| Prompt Replay [(2603.21177)](https://arxiv.org/abs/2603.21177) | Llama 3B / Qwen 8B | Free zero-variance reduction | Reference |
| Search-R1 [(2503.09516)](https://arxiv.org/abs/2503.09516) | Baseline | +20 % at 3B over RAG | Baseline |
| ReSearch [(2503.19470)](https://arxiv.org/abs/2503.19470) | Baseline | Cross-benchmark generalisation from one dataset | Baseline |
| R1-Searcher [(2503.05592)](https://arxiv.org/abs/2503.05592) | Baseline | Beats GPT-4o-mini, no SFT, no PRM | Baseline |

---

## 9. TLDR — For This Project

> **Context.** Search-R1 / ReSearch baseline reproduced (May 2026). Training a Qwen3.5-2B variant with GRPO and EM-based reward on multi-hop QA. Budget ~$1000 on 1× A100 80GB SXM. Deadline 2026-06-15. Question: *which papers should I read, and what should I actually do?*

### What the field says your exact setting will hit

Your setting (≤3B, sparse EM reward, multi-turn search, single GPU) sits at the intersection of three documented risk zones:
- **LLD collapse** [(2512.04220)](https://arxiv.org/abs/2512.04220) — confirmed on Qwen2.5-3B + Search-R1.
- **Gradient starvation** — near-zero pass-rate early, all-zero group advantages.
- **Template collapse** [(2604.06268)](https://arxiv.org/abs/2604.06268) — multi-turn agentic RL produces input-agnostic reasoning invisible to entropy metrics.

Positive signal: HiPRAG [(2510.07794)](https://arxiv.org/abs/2510.07794) and DeepRetrieval [(2503.00223)](https://arxiv.org/abs/2503.00223) prove that 3B search-RL can outperform much larger systems when the reward and curriculum are right.

### Papers you must read (in this order)

| # | Paper | Why |
|---|-------|-----|
| 1 | HiPRAG [(2510.07794)](https://arxiv.org/abs/2510.07794) | Closest existing paper to your setting. 3B + search-RL + process rewards. Read the reward design. |
| 2 | Hard Examples [(2508.14094)](https://arxiv.org/abs/2508.14094) | Apply before training anything. Filter your data to pass-rate 0.2–0.8. Free +47 %. |
| 3 | RAGEN-2 [(2604.06268)](https://arxiv.org/abs/2604.06268) | Your W&B dashboard measures entropy. It's the wrong metric. Add MI tracking now. |
| 4 | LLDS [(2512.04220)](https://arxiv.org/abs/2512.04220) | If your GRPO run plateaus then crashes, this is the named failure mode and the targeted fix. |
| 5 | E2H Curriculum [(2506.06632)](https://arxiv.org/abs/2506.06632) | If Qwen3.5-2B fails straight RL on MuSiQue, this is why and how to fix it. |
| 6 | JustRL [(2512.16649)](https://arxiv.org/abs/2512.16649) | Run a C-minimal (plain GRPO, no tricks) alongside any complex variant. This is your control. |
| 7 | PODS [(2504.13818)](https://arxiv.org/abs/2504.13818) + Online Difficulty Filter [(2504.03380)](https://arxiv.org/abs/2504.03380) | 1.7× faster convergence by not wasting rollouts. Implement before spending $1000. |
| 8 | ExGRPO [(2510.02245)](https://arxiv.org/abs/2510.02245) | +7.6 OOD is the most relevant gain. If you train on NQ/HotpotQA and test on MuSiQue, read this. |
| 9 | EM Verifier Gaming [(2604.15149)](https://arxiv.org/abs/2604.15149) | Run IPT as a final sanity check. EM reward is gameable; you need to know if your model is gaming it. |

### What to do with your $1000 / 5 weeks

```
Week 1 (now):
  - Apply Hard Examples filter to your training set (sample 50 base-model rollouts/prompt;
    drop pass-rate 0 or 1; keep 0.2–0.8 zone).
  - Turn on prefix caching, async vLLM, bump gpu_memory_utilization → 0.80.
  - Add MI-between-input-and-reasoning to W&B dashboard alongside entropy.
  - Smoke-test 50 steps: measure per-step time after systems wins.

Week 2:
  - Run Variant C-minimal (plain GRPO, β=0.001, fixed hyperparameters, filtered data) × 1 seed.
  - This is your JustRL control. Target: 4–6 days on A100 after systems wins.
  - Simultaneously: NQ→HotpotQA data-level curriculum if HotpotQA pass-rate <5%.

Week 3:
  - If C-minimal EM > 40% on HotpotQA: add LLDS regularizer (your main addition).
  - If C-minimal EM < 25%: add SFT cold-start (1 epoch on R1-Searcher-7B traces).
  - HiPRAG-style per-search-turn process rewards (penalize >3 search turns without progress).

Week 4:
  - Run Variant C (LLDS + β-anneal + process rewards) × 1 seed.
  - Add ExGRPO replay buffer for OOD generalization to MuSiQue.
  - Eval on MuSiQue at checkpoint (val_period: 50).

Week 5 (buffer / writing):
  - Compare C-minimal vs Variant C on HotpotQA + MuSiQue.
  - Run IPT on final checkpoint (EM verifier sanity check).
  - If C beats C-minimal by ≥2 EM: thesis claim = "process rewards + stability regularizers
    matter for 3B search-RL."
  - If C-minimal wins: thesis claim = "simplicity is competitive with complex stacks at 3B."
  - Either outcome is publishable.
```

### Budget estimate

| Run | Config | Est. Steps | Est. Wall-Clock (post systems-wins) | Est. Cost |
|-----|--------|-----------|-------------------------------------|-----------|
| Smoke (50 steps) | Any | 50 | ~1 h | ~$1.50 |
| C-minimal | Plain GRPO, NQ+HotpotQA | ~500 (early-stop) | ~4–5 days | ~$150 |
| Variant C | LLDS + process rewards + replay | ~700 | ~5–7 days | ~$200 |
| MuSiQue OOD eval | — | — | ~0.5 days | ~$15 |
| **Total** | | | **~10–13 days** | **~$370** |

This leaves ~$600 for re-runs, debugging, seed replication, or a third configuration if time permits.

### The one-sentence thesis claim this experiment supports

> *Small language models (2B) trained with GRPO and EM-based verifiable rewards can learn effective multi-hop search-augmented reasoning; the key enablers are difficulty-aware data filtering, stability regularisers to prevent likelihood collapse, and per-search-turn process rewards to prevent over-search; not architectural scale.*

---

## 10. Bibliography

All papers cited in this document, grouped by section.

### 10.1 Memory and Compute Efficiency

- [DAPO (2503.14476)](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero / Dr. GRPO (2503.20783)](https://arxiv.org/abs/2503.20783)
- [Prefix Grouper (2506.05433)](https://arxiv.org/abs/2506.05433)
- [S-GRPO / Token-Efficient RL (2504.20834)](https://arxiv.org/abs/2504.20834)
- [PEFT for RLVR (2512.23165)](https://arxiv.org/abs/2512.23165)
- [LoRA Plasticity vs Rigidity (2601.06677)](https://arxiv.org/abs/2601.06677)
- [MC-GRPO (2601.22582)](https://arxiv.org/abs/2601.22582)
- [GSPO (2507.18071)](https://arxiv.org/abs/2507.18071)

### 10.2 Rollout Budget and Sample Efficiency

- [Hard Examples (2508.14094)](https://arxiv.org/abs/2508.14094)
- [Online Difficulty Filtering (2504.03380)](https://arxiv.org/abs/2504.03380)
- [PODS (2504.13818)](https://arxiv.org/abs/2504.13818)
- [VIP (2602.01601)](https://arxiv.org/abs/2602.01601)
- [AR3PO (2509.25808)](https://arxiv.org/abs/2509.25808)
- [RLVR Linearity (2601.04537)](https://arxiv.org/abs/2601.04537)
- [NExt (2604.11446)](https://arxiv.org/abs/2604.11446)
- [Learning from Less (2604.18381)](https://arxiv.org/abs/2604.18381)

### 10.3 Failure Modes

- [RAGEN-2 (2604.06268)](https://arxiv.org/abs/2604.06268)
- [DPH-RL (2509.07430)](https://arxiv.org/abs/2509.07430)
- [DyJR (2603.16157)](https://arxiv.org/abs/2603.16157)
- [LLMs Gaming Verifiers (2604.15149)](https://arxiv.org/abs/2604.15149)
- [JustRL (2512.16649)](https://arxiv.org/abs/2512.16649)
- [LLDS / GRPO Collapse (2512.04220)](https://arxiv.org/abs/2512.04220)

### 10.4 Curriculum

- [E2H Reasoner (2506.06632)](https://arxiv.org/abs/2506.06632)
- [DeReason (2603.11193)](https://arxiv.org/abs/2603.11193)

### 10.5 Search-Augmented RL

- [HiPRAG (2510.07794)](https://arxiv.org/abs/2510.07794)
- [DeepRetrieval (2503.00223)](https://arxiv.org/abs/2503.00223)
- [R-Search (2506.04185)](https://arxiv.org/abs/2506.04185)
- [APEX-Searcher (2603.13853)](https://arxiv.org/abs/2603.13853)
- [s3 (2505.14146)](https://arxiv.org/abs/2505.14146)
- [ZeroSearch (2505.04588)](https://arxiv.org/abs/2505.04588)
- [Search-R1 (2503.09516)](https://arxiv.org/abs/2503.09516)
- [ReSearch (2503.19470)](https://arxiv.org/abs/2503.19470)
- [R1-Searcher (2503.05592)](https://arxiv.org/abs/2503.05592)

### 10.6 Off-Policy / Replay / Self-Distillation

- [ExGRPO (2510.02245)](https://arxiv.org/abs/2510.02245)
- [Freshness-Aware PER (2604.16918)](https://arxiv.org/abs/2604.16918)
- [Prompt Replay (2603.21177)](https://arxiv.org/abs/2603.21177)
- [RLSD (2604.03128)](https://arxiv.org/abs/2604.03128)
