---
title: INTEGRATION GUIDE
tags: []
source: internal
created: 2026-05-04
updated: 2026-05-06
---

# Integration Guide: Efficiency + Algorithm Evolution

**Purpose**: Navigate the three rounds of research (v1 → v2 → v3) + two parallel paths (algorithms in PARADIGM_REVIEW, systems in RUNTIME_EFFICIENCY) to make concrete decisions about your training run.

**Why this doc exists**: PARADIGM_REVIEW.md has evolved significantly as new papers and counter-evidence emerged. The v1 recommendations (clean, simple) were revised in v2 (failure modes discovered) and then partially challenged in v3 (JustRL's "tricks hurt" finding). RUNTIME_EFFICIENCY.md adds orthogonal systems wins. This doc is a map through all of that, with decision points, caveats, and a recommended path.

**Audience**: You (implementing the run), collaborators, and future readers trying to understand why certain choices were made.

---

## A. The evolution: v1 → v2 → v3

### v1 (original, PARADIGM_REVIEW §2-12)

**Thesis:** GRPO is solid; clean up the known failure modes and optimize for small models on single GPU.

**Stack (Variant B)**:
1. Drop KL entirely (β=0)
2. Dr. GRPO normalizations (remove length-norm + std-norm)
3. DAPO dynamic sampling (drop all-1/all-0 groups)
4. EAGLE-3 spec-dec for rollout

**Why these were chosen**: Each had published ablations showing 10-50% improvements. β=0 saved VRAM. Dr. GRPO matched research precedent. Dynamic sampling fixed the "useless gradient" problem. Spec-dec was proven at 8B.

**Measured claim**: "Variant B should hit 4-7 EM above your current baseline."

---

### v2 (PARADIGM_REVIEW §13-15: Failure modes + new techniques)

**Critical new finding**: **β=0 + Search-R1 + Qwen2.5-3B has a documented death spiral** (LLDS paper, [2512.04220](https://arxiv.org/abs/2512.04220)).

> The model's likelihood on *both* correct and incorrect responses drops systematically. Gradients explode. Training collapses. Not reward hacking — internal collapse.

**Other counter-evidence**:
- **Dynamic sampling starves the model early** (when pass-rate is low, all groups are nearly all-wrong; dropping them removes learning signal exactly when it's needed most).
- **Dr. GRPO's normalizations aren't unambiguously best** — λ-GRPO and plain GRPO with length monitoring can outperform Dr. GRPO in heterogeneous settings.
- **Format reward can reward-hack** if structured additively (model inserts answer early to claim credit).

**Revised Stack (Variant C, §14)**:
1. **β annealing**: 1e-3 → 0 over the run (not β=0 from start). Insurance against LLD.
2. Keep Dr. GRPO but monitor length bias.
3. **Add LLDS regularizer** (the direct fix for Search-R1 collapse; +37.8 abs on Qwen2.5-3B).
4. **Replace dynamic sampling with BAPO adaptive clipping** (handles zero-advantage groups via adaptation, not dropping).
5. **Add small replay buffer** with freshness-decay weighting (5-20% of batch; -46% gain without decay weighting).
6. **Off-policy seed of 1-5k traces from same-family teacher** (R1-Searcher-7B or DeepResearcher-7B). LUFFY-style with IS clipping.
7. **Curriculum on retrieval difficulty** (0-1 distractors → 5+ over first 30% of training).
8. **ZeroSearch simulated retrieval** for first 30% (accelerates cold-start while real corpus learns format).
9. Optional: GSPO sequence-level IS if variance issues.

**Why these changed**:
- v1 recommended "pure techniques based on math papers"; v2 found Search-R1 + small models have unique failure modes not covered by math RLVR literature.
- LLDS is specifically about Search-R1 collapse — highest EV single addition.
- BAPO > DAPO when you expect zero-advantage groups early (which you do at 2B).
- Replay buffers require careful setup (freshness decay!) but high upside for sparse-reward settings.

---

### v3 (PARADIGM_REVIEW §16-17: Latest papers + JustRL counter-evidence)

**Bombshell**: [JustRL ([2512.16649](https://arxiv.org/abs/2512.16649), Dec 2025; ICLR 2026)] shows that **adding tricks actively degrades performance**.

> Single-stage GRPO, fixed hyperparameters, no curriculum, no dynamic sampling, no length penalty, no multi-stage scheduling: 54.9% / 64.3% average across nine math benches. **Ablation: adding "standard tricks" (length penalties, robust verifiers, curriculum) actively DEGRADES performance by collapsing exploration.**

**Three new high-EV techniques**:
1. **M-GRPO ([2512.13070](https://arxiv.org/abs/2512.13070))**: Momentum-anchored target model to prevent long-horizon policy collapse. Independent of LLDS (can combine). Low effort (EMA target).
2. **KnowRL ([2604.12627](https://arxiv.org/abs/2604.12627))**: Decompose rewards into atomic "knowledge points" and find minimal sufficient set. Directly targets reward sparsity.
3. **Agentic RL Safety ([2510.17431](https://arxiv.org/abs/2510.17431))**: Search-R1 inherits refusal from base but is fragile to attack. **Instrument as a final eval gate.**

**Final Consolidated Plan (Variant C, §17)**:
Integrates all three rounds with JustRL as a control:

*Core stack (Variant C):*
1. β annealing from 1e-3 to 0 over the run.
2. Dr. GRPO normalizations.
3. LLDS + M-GRPO regularizers (both are cheap; both target collapse).
4. BAPO adaptive clipping (not DAPO).
5. Replay buffer (5-20% of batch) with freshness-decay weighting.
6. Off-policy seed from same-family teacher (pick the right family per C2).
7. GFT for SFT cold-start (prevents diversity collapse before RL).
8. Curriculum on retrieval-difficulty (0-1 → 5+ distractors over 30% of training).
9. ZeroSearch simulated retrieval for first 30%.
10. EAGLE-3 spec-dec with draft-length k=3, in-domain init, skip online adaptation.
11. Optional: GSPO, KnowRL if early plateau, instrumentation for LLD spirals.

*Control (Variant C-minimal):*
Pure GRPO with no tricks, fixed hyperparameters. **If C-minimal beats Variant C on your eval suite, the stack is hurting.** This is the JustRL replication.

*By EV* (what to invest sweat in first):
1. LLDS + M-GRPO (confirmed upside on your exact setting).
2. Off-policy seed from same-family teacher (high reported gain).
3. Replay buffer + freshness decay (well-documented, cheap).
4. Spec-dec with proper config.
5. GFT or TESSY for traces.

---

## B. Two parallel paths: Algorithms + Systems

PARADIGM_REVIEW and RUNTIME_EFFICIENCY are **orthogonal**. They compose multiplicatively.

### Algorithms (PARADIGM_REVIEW)

**Scope**: Which RL techniques, regularizers, sampling strategies, and architecture choices.

**Key tensions**:
- **v1 vs v2**: β=0 works for math but fails for Search-R1. Insurance (small β, LLDS) is cheap.
- **Dr. GRPO vs λ-GRPO**: No clear winner in heterogeneous settings. Monitor and swap if needed.
- **Dynamic sampling vs BAPO**: DAPO can starve early; BAPO adapts to regime.
- **Tricks vs minimalism (JustRL)**: Variant C is complex; C-minimal tests if complexity helps or hurts.

**Validation lever**: If C-minimal beats C by 2+ EM points, **something in the stack is hurting**. Then run a "tricks stripping" experiment to identify which piece.

### Systems (RUNTIME_EFFICIENCY)

**Scope**: vLLM config (prefix caching, async engine, gpu_memory_utilization, generation_batch_size), optimizer kernels (fused AdamW), colocation overhead, dynamic batching, torch.compile.

**Measured baseline** (1× A100 80GB, from [`SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config)):
- **Per-step**: 15–24 minutes (linear vs sub-linear scaling estimate).
- **1005 steps**: 11–17 days wall-clock.

**Best initial wins**:
1. **Prefix caching** (R1): 1.5–2.0× on rollout phase.
2. **Async vLLM** (R2): 1.3–1.5× on rollout.
3. **GPU memory utilization bump** 0.6 → 0.85 (R4): 1.2–1.4× on rollout.
4. **Raise gen batch size** (R5): 1.1–1.3× on rollout.
5. **Fused AdamW** (O1): 1.03–1.08× on full step.

**Stacked together**: ~2–3× on rollout phase alone, which is 50–65% of step time → ~1.5–2× overall step time → **11–17 days → ~6–12 days**.

**Caveats**: Colocation (vLLM ⇄ DTensor swap, ~35 GiB per step) is the #2 bottleneck ([SMOKE_RESULTS_2026-05-06.md "Bottlenecks identified"](../training/SMOKE_RESULTS_2026-05-06.md#bottlenecks-identified)). Systems wins are nearly free; algorithmic wins require careful validation (see JustRL).

---

## C. Decision tree: What to actually do

### Start here

**Q1: Do you want to validate whether the v2/v3 stack helps or hurts?**

**YES** → Run **Variant C (full stack) + Variant C-minimal (JustRL replica) + systems wins**.
- Variant C: full v3 stack.
- C-minimal: pure GRPO, fixed hyperparameters, no tricks.
- Systems: R1 + R2 + R4 + R5 + O1 (should give ~2× step time reduction).
- Outcome: if C beats C-minimal by 2-5 EM, the stack justified; if C-minimal wins, investigate which pieces hurt.

**NO, just give me one clear plan** → Run **Variant C + systems stack**. You're trusting the research rounds. Expected outcome: 8-12 EM above C-minimal, 4-7 above baseline (per PARADIGM_REVIEW §17).

---

### Architecture & data prep (pre-training)

**Q2: SFT cold-start, or jump straight to RL?**

**Straight RL** (faster, proven at 1.5B on math): Only if your base Qwen3.5-2B already handles tool use reasonably (tests < 10% pass-rate). If pass-rate is <1%, do SFT.

**SFT cold-start** (safer, Variant C default): 1-2 epochs on R1-Searcher-7B traces, using GFT (group fine-tuning) to prevent diversity collapse (§16.C5 finding). 1-2 days upfront, buys stability.

**Q3: Off-policy seed — real or simulated?**

**Real (LUFFY-style)**: 1-5k high-quality traces from R1-Searcher-7B or DeepResearcher-7B, same family as Qwen3.5-2B. **Critical: match families** (C2 finding — wrong family nulls the gains). Medium effort, highest EV after LLDS.

**Simulated (ZeroSearch)**: Use a frozen 7B model as the retriever sim for first 30% of training, then switch to real corpus. Accelerates cold-start. Effort: you keep the 7B around. Cost-benefit: saves wall-clock if corpus is slow; adds complexity.

---

### During training: The fork

**Variant C (full stack — if validating the research)**:
1. β annealing (1e-3 → 0).
2. LLDS + M-GRPO regularizers.
3. BAPO adaptive clipping.
4. Replay buffer with freshness decay.
5. Off-policy seed from same-family teacher.
6. Curriculum on retrieval difficulty (0-1 → 5+).
7. ZeroSearch for first 30%.
8. EAGLE-3 spec-dec.
9. Instrumentation: LLD spirals, all-1/all-0 frequency, search-query mode collapse, EM@top-k, verifier disagreement.

**Variant C-minimal (control, if testing whether tricks help)**:
- Pure GRPO.
- Fixed hyperparameters (no annealing, no curriculum, no dynamic sampling, no explicit length penalty).
- EM reward only.
- Same base config otherwise.

**Metric**: Run both to convergence. If C beats C-minimal by 2+, the stack justified. If C-minimal wins, investigate which pieces hurt via ablation.

---

### Systems wins (applied to both variants)

**Safe stack (implement first, ~2 days)**:
1. **R1**: Prefix caching (`vllm_kwargs.enable_prefix_caching: true`).
2. **R2**: Async vLLM (`vllm_cfg.async_engine: true`).
3. **R4**: Bump gpu_memory_utilization 0.6 → 0.80 (watch W&B `gpu_monitoring`).
4. **R5**: Raise generation_batch_size 32 → 64.
5. **O1**: Fused AdamW (`fused: true`, `foreach: false` → `true` if stable).

**Expected speedup**: ~2× on rollout phase → ~1.5–2× on full step → 11–17 days → 6–12 days.

**Next tier (if measuring R1-R5 on a few steps first)**:
6. **R3**: Async GRPO (overlap rollout/train). Adds small staleness; bounded by ε=0.2.
7. **R6**: Retrieval LRU cache (query-keyed, process-local).

**Skip on 1× A100**:
- Decolocation (needs 2 GPUs; breaks simplicity).
- O3 LoRA (capacity ceiling at <2B risky; you have 80GB VRAM).
- O6 sequence packing (blocked by Qwen3.5 GDN kernel; only viable if you swap model).

---

## D. Caveats & failure modes

### Failure mode: LLD Death Spiral

**What**: Model's likelihood on correct and incorrect responses both drop, gradients explode, training collapses.

**Why**: Pure GRPO at scale on sparse-reward tasks (Search-R1, 2B, low pass-rate) can have unstable dynamics.

**Mitigation in Variant C**:
1. **LLDS regularizer** (+37.8 abs on Qwen2.5-3B, your exact setting).
2. **M-GRPO momentum-anchor** (independent, orthogonal).
3. **β annealing** (small β early for stability, anneals to 0).
4. **Instrumentation**: Log mean log-likelihood of *correct* rollouts every step. If monotonic decrease for >100 steps, you're in the spiral — activate LLDS boost immediately.

**In Variant C-minimal**: No protection. If LLD appears, Variant C's LLDS + M-GRPO should prevent it.

---

### Failure mode: Gradient starvation (all-zero advantage groups)

**What**: In early training, all 5 rollouts in a group get EM=0. Advantage is zero for all. No learning signal.

**Why**: 2B model starts with very low pass-rate on hard multi-hop tasks.

**Mitigation in Variant C**:
1. **BAPO adaptive clipping** (adapts clip bounds when off-policy drift is large; handles zero-advantage groups via adaptation, not dropping).
2. **Replay buffer** (5-20% of batch are historical EM=1 trajectories; provides learning signal).
3. **Curriculum on retrieval difficulty** (start with easy, ramp to hard over first 30%; gives time to see non-zero rewards before hitting hard regime).

**Why NOT dynamic sampling here**: DAPO drops all-zero groups. At 2B with low early pass-rate, all-zero groups are 80%+ of batches. Dropping them = starving the model of training signal.

---

### Failure mode: Tricks causing worse generalization (JustRL finding)

**What**: Adding curriculum, length penalties, robust verifiers, dynamic sampling might collapse exploration and hurt OOD performance.

**Why**: Not fully understood yet. JustRL hypothesis: tricks prematurely narrow the model's behavior; pure GRPO keeps diversity longer.

**Mitigation**:
1. **Run Variant C-minimal as a control**. If it beats Variant C, something in the stack is hurting.
2. **Measure on both in-distribution (e.g., NQ+HotpotQA) and OOD (e.g., MuSiQue if trained on NQ)** to catch this.
3. **If C-minimal wins**: Run a "tricks stripping" experiment. Disable one piece at a time (e.g., no curriculum, no replay buffer, no LLDS). Identify which one hurts.

---

### Failure mode: SFT diversity collapse (§16.C5 finding)

**What**: If you SFT cold-start on R1-Searcher traces, the model loses search-strategy diversity *before* RL even starts. No inference trick recovers it.

**Why**: SFT memorization is different from RL exploration. Once collapsed by SFT, it's baked in.

**Mitigation in Variant C**:
1. **GFT (Group Fine-Tuning)** instead of vanilla SFT. Keeps diversity by training on *groups* of trajectories instead of individual traces.
2. **TESSY** alternative (task-level early-stop SFT on traces). Another way to avoid over-fitting to a single trajectory type.
3. **Monitor**: Log the top-3 most-common search queries per 100 steps. If collapsing to 1-2 queries, you're losing strategy diversity.

---

### Failure mode: Off-policy seed family mismatch (§16.C2 finding)

**What**: If you seed RL with traces from a *different* model family than your student, you waste compute. Distillation only works if "they think alike."

**Why**: Successful distillation concentrates on a small shared token set (97-99% of probability mass). Cross-family students don't share that set.

**Mitigation**: **Pick the off-policy seed from the SAME family as your student**. R1-Searcher-7B with Qwen2.5 base → good seed for Qwen3.5-2B student. LLama-7B base → bad seed.

---

## E. Data: MuSiQue + NQ/HotpotQA mix

Your current config trains on NQ+HotpotQA. MuSiQue is harder (3-hop, lower base EM, sparser rewards).

### Q: Train on pure MuSiQue, or mix?

**Option 1: Pure MuSiQue**
- More direct to your goal.
- But: lower base pass-rate → higher risk of LLD spiral + gradient starvation.
- **Mitigation**: Use Variant C's full stack (LLDS, M-GRPO, BAPO, replay, curriculum). Expect 15-20 days instead of 11-17.

**Option 2: NQ (easy) → HotpotQA (medium) → MuSiQue (hard) — curriculum at the data level**
- Each 300-step stage builds capacity for the next.
- Proven in curriculum literature (§16.B4).
- But: Variant C already has curriculum on retrieval difficulty (within MuSiQue); stacking *both* might over-constrain learning.
- **Mitigation**: If doing stage-wise, disable the retrieval-difficulty curriculum and let data difficulty be the curriculum signal.

**Recommendation**: Start with Option 2 (easier ramping). If JustRL's "tricks hurt" holds for your setting, you'll be grateful for the stability. Once validation is done, try pure MuSiQue for speed.

---

## F. Validation: What to measure

### During training (every step)

1. **Loss & reward signals**:
   - `mean_logprob_correct`: Mean log-likelihood of correct rollouts. **Alert if monotonic decrease for >100 steps** (LLD spiral signal).
   - `mean_logprob_incorrect`: Should also drop, but slower than correct.
   - `all_1_freq`, `all_0_freq`: % of groups where all 5 rollouts have same reward. **Alert if >30%** (starvation).

2. **Optimization signals**:
   - `grad_norm`: Should not explode (cap ~10 with your max_grad_norm=1.0).
   - `clip_ratio`: Fraction of updates where PPO clip activates. ~20-40% is healthy; >60% = too conservative; <5% = not clipping enough.
   - `kl_divergence`: Even with anneal β, should stay <0.5 bits in early phase.

3. **Rollout diversity**:
   - Top-3 most-common `<search>` queries per 100 steps. **Alert if stuck on 1-2** (mode collapse).
   - `generation_diversity`: Entropy of first-token choices across group.

4. **Retrieval signals** (if available):
   - `retrieval_success_rate`: % of search queries that get >0 docs. If <95%, your local retriever is failing.

### Validation data (if enabled, per VALIDATION.md)

- **EM@top-k for k ∈ {1, 3, 5}**: Diagnose whether the model is learning to search strategically (top-1 EM low, top-3 or top-5 much higher → good exploration) or stuck (all are low → collapse; top-1 high but others low → mode collapse).
- **Verifier disagreement**: Manual QA on 50 samples per 200 steps. Aim for <2% (reward=1 but manual QA disagrees, or vice versa).

### Final eval (after training)

- **In-distribution** (e.g., HotpotQA test set): Expected 50–65% EM (baseline search-R1 numbers).
- **OOD** (e.g., MuSiQue test if trained on NQ/HotpotQA): Expected <40% EM; Variant C should bridge some of that gap.
- **Safety gate** ([2510.17431](https://arxiv.org/abs/2510.17431)): Test refusal robustness (vs "Search attack" injection).

---

## G. Timeline & resource plan

### Pre-run (1–2 weeks)

- [ ] Implement systems wins (R1, R2, R4, R5, O1) on current config. Smoke-test 10 steps to confirm speedup.
- [ ] Prepare off-policy seed: decide same-family teacher (R1-Searcher-7B or DeepResearcher-7B) and export 1-5k high-quality traces.
- [ ] If doing SFT cold-start: prepare traces, implement GFT or TESSY (1–2 days).
- [ ] Set up instrumentation dashboards in W&B (LLD spiral detection, all-1/all-0 frequency, search-query mode).

### Run phase A (Variant C, 11–17 days on 1× A100)

- Implement full Variant C stack.
- Run 3 seeds if possible; 1 seed minimum.
- Monitor instrumentation every 100 steps. If LLD spiral detected, activate LLDS boost.
- Log top-3 search queries, EM@top-k, grad norms.

### Run phase B (Variant C-minimal, 11–17 days)

- Same setup but pure GRPO, fixed hyperparameters, no tricks.
- Run same 3 seeds.
- Compare final EM vs Variant C. If C-minimal beats C, investigate which pieces hurt (ablation sweep).

### Post-run (1–2 weeks)

- Analyze Variant C vs C-minimal on in-distribution and OOD.
- If C wins: ship Variant C config to production notes.
- If C-minimal wins: run mini-ablation (one trick off at a time) to identify which piece to drop.
- Run safety eval on final checkpoint.

---

## H. References

### Key papers by category

**Core failure-mode papers (v1 correction)**:
- LLDS ([2512.04220](https://arxiv.org/abs/2512.04220)) — Search-R1 collapse.
- M-GRPO ([2512.13070](https://arxiv.org/abs/2512.13070)) — Long-horizon collapse, momentum-anchor fix.
- JustRL ([2512.16649](https://arxiv.org/abs/2512.16649)) — Tricks may hurt, ablation counter-evidence.

**Algorithmic techniques (v2/v3 stack)**:
- LLDS, M-GRPO, BAPO (above).
- LUFFY ([2504.14945](https://arxiv.org/abs/2504.14945)) — Off-policy guidance.
- Freshness-aware PER ([2604.16918](https://arxiv.org/abs/2604.16918)) — Replay buffer decay.
- GFT ([2604.14258](https://arxiv.org/abs/2604.14258)) — SFT without diversity collapse.
- KnowRL ([2604.12627](https://arxiv.org/abs/2604.12627)) — Knowledge-point hints for sparse reward.
- ZeroSearch ([2505.04588](https://arxiv.org/abs/2505.04588)) — Simulated retrieval for dev-loop.

**Safety & robustness**:
- Agentic RL Safety ([2510.17431](https://arxiv.org/abs/2510.17431)) — Refusal under attack.
- Composite Rewards ([2509.15557](https://arxiv.org/abs/2509.15557)) — Format reward hacking mitigation.

**Systems & speedup**:
- RUNTIME_EFFICIENCY.md (this project) — vLLM config, async, colocation.
- [SMOKE_RESULTS_2026-05-06.md](../training/SMOKE_RESULTS_2026-05-06.md) (this project) — Measured per-step baseline.

---

## I. FAQ

**Q: Should I run Variant C or C-minimal first?**

A: Run both in parallel (if budget allows; 6 total seeds = 3 runs × 11–17 days). If budget is tight, run C-minimal first. It's simpler, matches JustRL setup, and validates whether the tricks are even worth it before you invest in the full stack.

**Q: What if LLDS regularizer isn't in NeMo-RL?**

A: LLDS paper has a reference implementation on GitHub ([2512.04220](https://arxiv.org/abs/2512.04220)). It's a lightweight regularizer term; ~50 lines of code to add to your loss function. Effort: 1–2 days.

**Q: Can I skip M-GRPO if I have LLDS?**

A: LLDS targets likelihood collapse; M-GRPO targets policy collapse (entropy death) via momentum. Orthogonal fixes. Together: very safe. Separately: LLDS is higher-EV; M-GRPO is cheap insurance. If time-constrained, do LLDS. If both fit: do both.

**Q: My base Qwen3.5-2B already does tool-use OK. Can I skip SFT?**

A: Yes, jump to Variant C RL directly (Magistral [2506.10910](https://arxiv.org/abs/2506.10910) validates this). If pass-rate is <1%, do 1 epoch SFT with GFT to avoid diversity collapse.

**Q: What if my retriever is slow?**

A: Use ZeroSearch for first 30%, then switch to real retriever. Frozen 7B sim only needs to be loaded once, adds <1 min to setup. Recovers ~3-5 days if corpus retrieval is >1s per query.

**Q: Should I tune hyperparameters (lr, ε, β schedule) or keep them fixed?**

A: **Keep them fixed.** JustRL's counter-evidence is precisely that per-step tuning can hurt generalization. Run the stack as-is. If EM plateaus early (<100 steps), investigate *why* (instrumentation will show). Then address root cause (LLD? All-zero groups? Diversity collapse?) rather than tuning λ.

**Q: How many seeds?**

A: 3 is standard for RL (captures variance). 1 seed minimum if budget is tight. Run both Variant C and C-minimal at same seed count for fair comparison.

---

## J. Checklist: Before you run

- [ ] **Systems**: R1, R2, R4, R5, O1 implemented and smoke-tested on 10 steps.
- [ ] **Off-policy seed**: 1-5k traces from same-family teacher (if doing Variant C).
- [ ] **SFT (if needed)**: GFT or TESSY implemented.
- [ ] **Instrumentation**: W&B dashboards set up for LLD, all-zero groups, search-query diversity, EM@top-k.
- [ ] **Control variant**: Variant C-minimal config ready (pure GRPO, fixed hyperparameters).
- [ ] **Validation**: VALIDATION.md re-enabled if you want intermediate EM measurements.
- [ ] **Safety**: Final eval gate for refusal robustness (Agentic RL Safety test).
- [ ] **Documentation**: Config version in comments; link to this guide.

---

## K. How this guide will evolve

- **v2**: After your Variant C and C-minimal runs complete, update this guide with actual results (final EM, which instrumentation signals mattered most, which pieces of the stack were worth it).
- **v3**: If other projects train Search-R1 variants, add their findings to the Caveats section.
