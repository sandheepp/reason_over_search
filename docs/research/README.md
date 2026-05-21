---
title: README
tags: []
source: internal
created: 2026-05-04
updated: 2026-05-07
---

# Research: Literature, Algorithms, Systems

Two strands live in this folder:

1. **Literature** (`LITERATURE_REVIEW.md` + the three `SURVEY*.md` files): the RLVR paper landscape, project-personal annotations, and the project-specific recipe selection.
2. **Algorithm + systems engineering** (`INTEGRATION_GUIDE.md`, `PARADIGM_REVIEW.md`, `RUNTIME_EFFICIENCY.md`): how to make GRPO training efficient and effective for a 2B model on multi-hop search-augmented QA on a single A100.

Both strands feed the M2 ablation plan (E2H curriculum + S-GRPO + MC-GRPO with a JustRL plain-GRPO control); see [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) for the strand-level snapshot and [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 6 for the supervisor-facing version.

Alongside the two strands, [`QUESTIONS.md`](QUESTIONS.md) is the running register of open research questions and uncertainties to chase later. When the user "registers a question", entries are appended there.

---

## Quick Navigation

**Strand status**: [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) — what's done, decisions, what feeds where.

**Algorithm + systems entry point**: [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md)

A complete decision guide covering:
- The evolution from v1 (original recommendations) → v2 (failure modes discovered) → v3 (latest papers & JustRL counter-evidence)
- How algorithms (PARADIGM_REVIEW) and systems (RUNTIME_EFFICIENCY) compose
- Decision trees for your constraints (1× A100, 2B model, MuSiQue)
- Failure modes and how to detect/mitigate them
- Complete pre-run checklist

**Read this if**: You want a clear, actionable path forward and don't have time to read three long papers.

**Literature entry point**: [`SURVEY_FOCUSED.md`](SURVEY_FOCUSED.md) — project-specific subset with full paper cards and the source of the candidate recipe.

---

## Literature documents

| File | What |
|---|---|
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | Project-personal working notebook. Thesis framing, introduction outline, per-paper annotations with `==notes==` and `⭐` markers, and five appendices preserving thinking-as-it-happened. Drove the v0 / v1 decisions. |
| [`SURVEY.md`](SURVEY.md) | Reference-style survey of the RLVR field. 14 thematic sections (Foundations, Verifier Design, Policy Optimization, Exploration, Reward Hacking, Self-Verification, Multi-Task, Open-Ended, Tool-Use, Mechanism Studies, Theoretical Foundations, Measurement, Open Challenges, Bibliography). Each section ends with paper cards in standardised Summary / Problem / Method / Result / Takeaway / ELI5 format. 96 cards total. |
| [`SURVEY_FOCUSED.md`](SURVEY_FOCUSED.md) | Project-specific subset of `SURVEY.md`: only the papers relevant to Qwen3.5-2B / single-A100 / search-augmented. Source of the next-steps recipe (E2H curriculum + S-GRPO + MC-GRPO) used in § 6 of `../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`. |
| [`SURVEY_OVERFLOW.md`](SURVEY_OVERFLOW.md) | 15 adjacent papers (foundational priors, surveys, alternative directions) kept for completeness. |
| [`QUESTIONS.md`](QUESTIONS.md) | Running register of open research questions and uncertainties; one heading per question, resolved entries kept in place with a `Resolution` block as a learning trail. |

---

## The three algorithm/systems documents

### 1. [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) — Decision framework

**What it covers**:
- Three research rounds (v1 → v2 → v3) and why each recommendation changed
- Decision tree: Variant C (full stack) vs Variant C-minimal (JustRL control)
- How PARADIGM_REVIEW (algorithms) and RUNTIME_EFFICIENCY (systems) fit together
- Failure modes (LLD spiral, gradient starvation, tricks hurting, SFT diversity collapse) and mitigations
- Pre-run checklist, validation plan, resource timeline

**Length**: ~900 lines, structured as sections A–K

**Best for**: Actually making decisions, understanding trade-offs, implementing the run

---

### 2. [`PARADIGM_REVIEW.md`](PARADIGM_REVIEW.md) — Algorithmic literature review

**What it covers**:
- **v1 (§2-12)**: Original literature survey on RL post-training techniques. Clean recommendations (β=0, Dr. GRPO, DAPO, spec-dec) with evidence from math RLVR papers.
- **v2 (§13-15)**: 13 new techniques discovered (LUFFY, BAPO, LLDS, etc.), counter-evidence to v1 (β=0 fails on Search-R1, DAPO starves early), revised Variant C stack.
- **v3 (§16-17)**: Latest papers (JustRL bombshell, M-GRPO, KnowRL) and final consolidated plan integrating all three rounds.

**Key findings**:
- β=0 works for math but has LLD Death Spiral on Search-R1 + Qwen2.5-3B. Insurance (small β, LLDS regularizer) is cheap.
- DAPO dynamic sampling can starve gradients in early training when all groups are zero-reward. BAPO (adaptive clipping) is more robust.
- JustRL found that adding tricks (curriculum, length penalties, dynamic sampling) can *degrade* performance. Variant C needs validation via C-minimal control.

**Length**: ~800 lines (v1) + ~550 lines (v2) + ~200 lines (v3) = ~1550 total

**Best for**: Understanding *why* the recommendations evolved, reading the literature, diving into specific techniques

---

### 3. [`RUNTIME_EFFICIENCY.md`](RUNTIME_EFFICIENCY.md) — Systems & engineering

**What it covers**:
- Measured baseline from [SMOKE_RESULTS_2026-05-06.md](../training/SMOKE_RESULTS_2026-05-06.md): 15–24 min/step on 1× A100 → 11–17 days for 1005 steps
- 21 engineering levers (R1-R7 rollout, C1 colocation, G1-G3 algorithm, O1-O6 optimizer, M1-M3 misc) with speedups and risks
- Detailed explanations for rollout optimizations (prefix caching 1.5–2.0×, async vLLM 1.3–1.5×, GPU memory util 1.2–1.4×)
- Suggested ordering: safe stack (~2 days) gives ~2× speedup on rollout phase → 11–17 days → 6–12 days overall

**Key findings**:
- Colocation swap (vLLM ⇄ DTensor, ~35 GiB freed per step) is bottleneck #2 after rollout. Async GRPO or 2-GPU split helps.
- Prefix caching is free and high-impact — all 5 generations in a group share system prompt + first-turn retrieval.
- LoRA is a capacity trap at <2B on multi-turn tasks — Plasticity vs Rigidity shows r<256 fails on reasoning at ≤1.5B. Skip unless desperate for VRAM.

**Length**: ~600 lines, densely packed

**Best for**: Understanding the systems constraints, picking which engineering wins matter most, estimating wall-clock impact

---

## How to use these docs

### If you have 30 minutes
Read **INTEGRATION_GUIDE.md § A–C** (evolution + decision tree). Gives you the "what changed and why" and the key fork (Variant C vs C-minimal).

### If you have 2 hours
Read **INTEGRATION_GUIDE.md** all the way through. It's self-contained; it cites PARADIGM_REVIEW and RUNTIME_EFFICIENCY without requiring you to read them first.

### If you want deep context
1. Read **INTEGRATION_GUIDE.md** (decision framework).
2. Skim **PARADIGM_REVIEW.md** §2-12 (v1 original) to understand the baseline.
3. Read **PARADIGM_REVIEW.md** §13-15 (v2 failure modes + Variant C) for the key corrections.
4. Skim **PARADIGM_REVIEW.md** §16-17 (v3 latest + JustRL) for what's new.
5. Read **RUNTIME_EFFICIENCY.md** § 1-2 (measured baseline + lever map) to understand the systems side.
6. Cross-reference specific techniques as needed.

### If you want to implement the run
1. Start with **INTEGRATION_GUIDE.md** § G-J (timeline, checklist).
2. Use **RUNTIME_EFFICIENCY.md** § 2-7 to understand which systems levers to implement first.
3. Use **PARADIGM_REVIEW.md** § 17 (final plan) to understand the algorithmic stack (or Variant C-minimal for the JustRL control).
4. Set up instrumentation per **INTEGRATION_GUIDE.md** § F.
5. Run both Variant C and C-minimal (if budget allows) to validate the stack.

---

## Key tensions & unknowns

### 1. **Tricks help or hurt?** (Variant C vs C-minimal)

- **v1**: Stack of clean techniques should boost perf.
- **v2**: Added more techniques to fix failure modes.
- **v3**: JustRL found adding tricks can *degrade* generalization.

**Resolution**: Run both variants. If C beats C-minimal by 2+, the stack is justified. If C-minimal wins, investigate which pieces hurt (ablation sweep).

### 2. **β=0 or small β?**

- **v1**: β=0 (no KL, saves VRAM & compute).
- **v2**: β=0 causes LLD Death Spiral on Search-R1. Use β anneal from 1e-3 to 0 (insurance).
- **v3**: Confirmed; LLDS regularizer is the surgical fix, but β > 0 is still safer.

**Current recommendation**: β anneal 1e-3 → 0 + LLDS regularizer. Belt and suspenders, but Search-R1 is specific enough to warrant both.

### 3. **Dr. GRPO + static vs adaptive normalization?**

- **v1**: Dr. GRPO (no length-norm, no std-norm) is clean.
- **v2**: λ-GRPO and plain GRPO + monitoring can outperform Dr. GRPO in heterogeneous settings.
- **v3**: No update; still context-dependent.

**Current recommendation**: Use Dr. GRPO by default. If length collapse appears, swap to λ-GRPO or plain GRPO + monitoring.

### 4. **Curriculum or just train hard?**

- **v1**: "Just train hard" (Hard Examples paper).
- **v2**: Curriculum on retrieval difficulty (e.g., 0-1 → 5+ distractors) has good evidence.
- **v3**: No strong update; both can work.

**Current recommendation**: Retrieval-difficulty curriculum is cheap and well-motivated for multi-hop QA. Include it.

### 5. **How many tricks is too many?**

- **v1**: A few clean ones.
- **v2**: Lots (LLDS, M-GRPO, BAPO, replay, curriculum, ZeroSearch, etc.).
- **v3**: Maybe none? JustRL minimal stack outperformed tricks.

**Current recommendation**: Run Variant C (full) and C-minimal (pure) to validate. Let the data answer this.

---

## Measured baseline (from SMOKE_RESULTS_2026-05-06.md)

| Hardware | Per-step | 1005 steps |
|---|---:|---:|
| 1× A100 80GB SXM (measured) | 15–24 min | **11–17 days** |
| 1× H100 80GB SXM | 7–12 min | 5–8.5 days |
| 2× A100 80GB SXM | 9–14 min | 6.5–9.5 days |

With systems wins (R1+R2+R4+R5+O1): **→ 6–12 days on 1× A100**.

With algorithmic wins (if they help): **→ 3–8 days** depending on how much DAPO + dynamic sampling + curriculum + LLDS each contribute.

---

## Links to related docs

- **Training**: [`docs/training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md) — Measured per-step numbers and bottleneck analysis (co-location swap is #2 cost).
- **Training**: [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) — How the training recipe compares to Search-R1 paper.
- **Training**: [`docs/training/VALIDATION.md`](../training/VALIDATION.md) — How to re-enable validation for early stopping.
- **Config**: [`training/configs/grpo_qwen3.5_2b_1xa100.yaml`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml) — The current config (base for both Variant C and C-minimal).

---

## Version history

- **v1** (PARADIGM_REVIEW §2-12, drafted 2026-05-03): Original literature survey.
- **v2** (PARADIGM_REVIEW §13-15, drafted 2026-05-03): Failure modes + new techniques discovered.
- **v3** (PARADIGM_REVIEW §16-17, drafted 2026-05-03): Latest papers + JustRL counter-evidence.
- **v1** (RUNTIME_EFFICIENCY, INTEGRATION_GUIDE, drafted 2026-05-03): Systems + synthesis.

The three research docs were written in the same session; INTEGRATION_GUIDE and RUNTIME_EFFICIENCY are new to tie them together and provide actionable guidance.

---

## FAQ

**Q: Where do I start?**

A: [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md). It's designed as a standalone decision guide. It references PARADIGM_REVIEW and RUNTIME_EFFICIENCY without requiring you to read them.

**Q: What's the recommended stack?**

A: **Variant C** (from PARADIGM_REVIEW §17):
- Algorithms: β anneal, LLDS, M-GRPO, BAPO, replay buffer, off-policy seed, curriculum, ZeroSearch, EAGLE-3 spec-dec.
- Systems: R1 prefix caching, R2 async vLLM, R4 GPU mem util bump, R5 gen batch size bump, O1 fused AdamW.

**But validate with Variant C-minimal** (pure GRPO, fixed hyperparameters). If C-minimal beats C, something is hurting.

**Q: Do I need to read all three docs?**

A: No. Read INTEGRATION_GUIDE.md. That's the entry point. Reference PARADIGM_REVIEW or RUNTIME_EFFICIENCY only if you want to understand a specific decision deeper.

**Q: What's new compared to v1?**

A: v2 adds LLDS, M-GRPO, BAPO, replay, curriculum, ZeroSearch. v3 surfaces JustRL ("tricks may hurt") and KnowRL (atomic hints). INTEGRATION_GUIDE and RUNTIME_EFFICIENCY are new synthesis docs.

**Q: What's likely to change in the next update?**

A: After you run Variant C and C-minimal, INTEGRATION_GUIDE will be updated with actual results (which EM improvements were real, which instrumentation signals mattered most). If other teams train Search-R1 variants, their findings will be added to the Caveats section.
