---
title: SURVEY
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Reinforcement Learning with Verifiable Rewards: A Consolidated Survey

> A unified treatment of the RLVR literature spanning foundations, verifier design, optimization, self-verification, multi-task hybrid rewards, domain extensions, tool-use integration, theoretical foundations, and open problems. Coverage current through Q1 2026.

> **Companion documents:**
> - [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md) — project-specific synthesis for resource-constrained small-LM RLVR with search (Qwen3.5-2B, single A100, $1000 budget). Contains the full per-paper cards for the project-critical literature plus a week-by-week experiment plan.
> - [SURVEY_OVERFLOW.md](SURVEY_OVERFLOW.md) — 15 adjacent papers (foundational priors, surveys, alternative directions) that are in the bibliography but not part of the main thematic narrative.

---

## Contents

1. [Foundations and Objectives](#1-foundations-and-objectives)
2. [Verifier Design and Task Coverage](#2-verifier-design-and-task-coverage)
3. [Policy Optimization Algorithms](#3-policy-optimization-algorithms)
4. [Exploration, Optimization, and Credit Assignment](#4-exploration-optimization-and-credit-assignment)
5. [Reward Hacking, Safety, and Robustness](#5-reward-hacking-safety-and-robustness)
6. [Online Self-Verification (RISE)](#6-online-self-verification-rise)
7. [Multi-Task Learning with Verifiable and Non-Verifiable Rewards](#7-multi-task-learning-with-verifiable-and-non-verifiable-rewards)
8. [Extensions to Open-Ended, Multimodal, and Sparse-Data Tasks](#8-extensions-to-open-ended-multimodal-and-sparse-data-tasks)
9. [Tool-Use and Search Integration](#9-tool-use-and-search-integration)
10. [Mechanism Studies: What Does RLVR Actually Do?](#10-mechanism-studies-what-does-rlvr-actually-do)
11. [Theoretical Foundations: Symbolic, Compositional, and Formal-Methods RL](#11-theoretical-foundations-symbolic-compositional-and-formal-methods-rl)
12. [Measurement, Evaluation, and System Design](#12-measurement-evaluation-and-system-design)
13. [Open Challenges and Future Directions](#13-open-challenges-and-future-directions)
14. [Consolidated Bibliography](#14-consolidated-bibliography)

---

## 1. Foundations and Objectives

Reinforcement Learning with Verifiable Rewards (RLVR) is a post-training paradigm that replaces learned reward models with **deterministic, programmatically computable verifiers**. The agent — typically a large language model — generates structured outputs (often including reasoning traces) which are scored by rule-based checks, automated evaluators, or hybrid procedures, and the policy is optimized via standard policy-gradient methods.

### 1.1 Primary Optimization Objective

The canonical RLVR objective augments the expected verifiable reward with a Kullback–Leibler regularizer toward a frozen reference policy:

$$
\mathcal{J}_{\text{RLVR}}(\theta) \;=\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\bigl[r(x, y)\bigr] \;-\; \beta \cdot D_{\text{KL}}\bigl(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr)
$$

where:

| Symbol | Meaning |
|--------|---------|
| $\pi_\theta$ | Parametric policy (the LLM under training) |
| $\pi_{\text{ref}}$ | Frozen reference policy (typically the SFT initialization) |
| $r(x, y)$ | Deterministic verifier-based reward, binary or scalar |
| $\beta$ | KL regularization coefficient |
| $\mathcal{D}$ | Prompt distribution |
| $D_{\text{KL}}$ | Kullback–Leibler divergence |

The KL term mitigates two failure modes simultaneously: (i) catastrophic forgetting of pre-trained capabilities, and (ii) reward hacking via degenerate policies that exploit verifier blind spots.

### 1.2 Reward Function Taxonomy

A general verifier admits the decomposition:

$$
r(x, y) \;=\; \sum_{i=1}^{k} w_i \cdot r_i(x, y), \qquad r_i \in \{0, 1\} \text{ or } r_i \in [0, 1]
$$

with components $r_i$ drawn from one or more of:

- **Rule-based** ($r_{\text{rule}}$): Programmatic checks — string match, regex, unit-test pass/fail, math equation-solver verdict.
- **Model-based** ($r_{\text{judge}}$): An auxiliary LLM scores the response.
- **Format** ($r_{\text{fmt}}$): Schema or tag compliance (e.g., `<think>...</think>` blocks).
- **Composite/Soft** ($r_{\text{soft}}$): Continuous similarity scores (BLEU, ROUGE, encoder cosine, F1).
- **Process/Step-wise** ($r_{\text{proc}}$): Per-step rewards from a Process Reward Model (PRM).
- **Intrinsic** ($r_{\text{int}}$): Model-internal signals — self-certainty, entropy, token probabilities.

### 1.3 Three-Stage Loop

RLVR systems follow a three-stage operating loop:

1. **Generation.** Sample a structured output $y \sim \pi_\theta(\cdot \mid x)$ with reasoning trace.
2. **Verification.** Compute reward $r(x, y)$ using rule-based, model-based, or hybrid verifiers.
3. **Policy Update.** Apply policy-gradient step (PPO, GRPO, REINFORCE) with KL regularization toward $\pi_{\text{ref}}$.

### 1.4 Why It Matters

RLVR is the operational successor to RLHF in domains where ground-truth checking is cheap and cheap-to-scale. It removes the dependence on human preference annotation and the cost of training a learned reward model. Recent work demonstrates that verifiable rewards implicitly incentivize the *correctness of intermediate reasoning chains* — not merely the correctness of final answers — producing emergent self-verification and reflection behaviors absent in the SFT initialization [(2506.14245)](https://arxiv.org/abs/2506.14245).

### Paper Cards

### 2506.14245 — RLVR Implicitly Incentivizes Correct Reasoning

**Summary.** Argues and shows that RLVR does more than boost sampling efficiency: it expands the reasoning boundary, even with answer-only rewards. Introduces CoT-Pass@K, a metric that scores the chain as well as the final answer.

**Problem.** It was unclear whether RLVR's pass@1 gains came from real new capability or just from sharpening pass@k toward a single mode.

**Method.** Revisit Pass@K experiments under GRPO, propose CoT-Pass@K to credit correct intermediate reasoning, and build a theoretical account of how outcome-only rewards encourage correct intermediate steps.

**Result.** Both math and code reasoning boundaries extend under RLVR; intermediate reasoning quality improves early in training (no scalar headline).

**Takeaway.** Useful framing for your thesis: report CoT-Pass@K alongside EM to argue your RLVR run is gaining real reasoning, not just sharpening.

**ELI5.** RLVR is not just teaching the chess player to pick their best existing move faster; it actually expands the set of moves they can find at all.

---

## 2. Verifier Design and Task Coverage

Verifier design is the central engineering decision in any RLVR system. The literature now spans six broad families:

### 2.1 Rule-Based Verifiers

The original RLVR formulation, used in DeepSeek-R1 and TinyZero. Examples:

- Math: deterministic answer comparison after parsing.
- Code: unit tests, compiler verdict, LeetCode-style judge.
- Format: regex over `<think>` and `<answer>` tags.

Strengths: zero-cost at training time, fully reproducible, no reward model drift.
Weaknesses: brittle on natural-language responses, false negatives common ([2505.22203](https://arxiv.org/abs/2505.22203)).

### 2.2 Model-Based (Judge) Verifiers

A separate LLM scores responses against a reference. Used when rule-based checks are too brittle.

- [Crossing the Reward Bridge (2503.23829)](https://arxiv.org/abs/2503.23829) — Soft generative judges for medicine, psychology, economics; uses z-score normalization over reference answers.
- Failure mode: false positives (model halucinates correctness), high inference cost.

### 2.3 Hybrid Verifiers

The current dominant paradigm. Combines rule-based hard constraints with model-based soft constraints.

- [VerIF (2506.09942)](https://arxiv.org/abs/2506.09942) — Code-checks for hard constraints, LLM judge for content semantics; SOTA on instruction-following.
- [Reward Hacking Mitigation (2509.15557)](https://arxiv.org/abs/2509.15557) — Composite penalties + Sentence-BERT semantic leak detection.

Composite reward form:

$$
r_{\text{hybrid}}(x, y) \;=\; \alpha \cdot r_{\text{rule}}(x, y) \;+\; (1 - \alpha) \cdot r_{\text{judge}}(x, y) \;-\; \lambda \cdot \mathbb{1}[\text{hack}(y)]
$$

### 2.4 Rubric-Based Verifiers

Multi-criterion rubrics serve as explicit reward functions for subjective tasks.

- [Rubrics as Rewards (2507.17746)](https://arxiv.org/abs/2507.17746) — Decomposes ill-defined tasks into criterion-wise verifiable signals.

### 2.5 Generative Reward Models (GenRM)

Pairwise or pointwise generative judges trained from RL-collected exploration data.

- [Writing-Zero (2506.00103)](https://arxiv.org/abs/2506.00103) — Self-principled critique pairs and Bootstrapped Relative Policy Optimization (BRPO) for creative writing.
- [Agentic Reward Modeling (2502.19328)](https://arxiv.org/abs/2502.19328) — Modular routers combining base reward models with aspect-specific verifiable signals.

### 2.6 Self-Verification (RISE Family)

Solution generation and critique are co-trained within a single online policy. Treated as a first-class objective; see §6.

### 2.7 Verifier Reliability

- [From Accuracy to Robustness (2505.22203)](https://arxiv.org/abs/2505.22203) — Quantifies false negatives in rule-based verifiers and false positives in model-based verifiers.
- [VerifyBench (2505.15801)](https://arxiv.org/abs/2505.15801) — First systematic benchmark for reference-based reward systems; highlights gaps in difficult/ambiguous cases.

### Paper Cards

### 2505.22203 — Rule vs Model Verifiers in Math RL

**Summary.** A systematic study of rule-based and model-based verifiers used as reward signals in RLVR for math reasoning. The paper quantifies both static accuracy and downstream RL effects.

**Problem.** Rule-based verifiers under-credit equivalent answers (false negatives) and model-based verifiers can be hacked, but neither failure mode was well measured.

**Method.** Compare open-source rule-based verifiers and model-based verifiers in static evaluation and within RL training across multiple math datasets and policy strengths.

**Result.** Rule-based verifiers exhibit non-negligible false-negative rates that worsen as the policy strengthens; model-based verifiers achieve higher static accuracy but are exploited during RL, producing inflated rewards.

**Takeaway.** For an EM-only RLVR pipeline (e.g., Search-R1 style), expect false negatives from a strict rule verifier and reward hacking if you swap in a learned judge; combine carefully.

**ELI5.** A strict spell-checker marks "color" wrong because the answer key has "colour"; a lenient AI grader gets bribed by flowery phrasing. Both fail in opposite directions.

### 2503.23829 — Crossing the Reward Bridge

**Summary.** Extends RLVR beyond math/code into medicine, chemistry, psychology, economics, and education where structured reference answers are unavailable. It shows binary verification is consistent across LLM judges when expert references exist, and uses a small generative reward model for soft signals on free-form answers.

**Problem.** RLVR works only where you can write a regex or run a unit test; most real-world domains have free-form answers that defy simple verifiers.

**Method.** Train a 7B generative reward model that scores free-form answers against expert references with a soft signal, then use it in standard RLVR.

**Result.** Outperforms Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B on cross-domain free-form benchmarks (no single headline number).

**Takeaway.** A small generative judge can replace exact-match for unstructured answers; this is the path to RLVR on QA tasks where EM/F1 is too brittle.

**ELI5.** Like switching from a multiple-choice grader to a TA who reads the essay; the TA only needs the answer key (not the rubric for every possible essay) to give a useful score.

### 2506.09942 — VerIF: Verification Engineering for Instruction Following

**Summary.** VerIF combines rule-based code verifiers with a large reasoning model (QwQ-32B) judge to produce reliable RLVR rewards for instruction following. Released with VerInstruct, a 22k-instance dataset with verification signals.

**Problem.** Instruction-following constraints are too varied for pure rules and too objective for pure preference models, leaving RLVR underexplored here.

**Method.** Hybrid verification: rule-based code checks for hard constraints plus an LLM judge for soft constraints, used as the reward in RL training on VerInstruct.

**Result.** State-of-the-art instruction-following among comparable-size models, with general capabilities preserved.

**Takeaway.** For multi-constraint RLVR (your search-augmented QA likely has format + answer constraints), a hybrid rule-plus-LLM verifier is a sturdier reward than either alone.

**ELI5.** Use a calculator to check the math part and an English teacher to check the essay part of the same homework, then combine their grades.

### 2509.15557 — Composite Rewards Against Reward Hacking (Medical QA)

**Summary.** Adds explicit penalty terms to RLVR's reward for two specific medical-QA reward-hacking patterns: skipping reasoning, and using non-standard reasoning formats. The composite reward yields cleaner traces with comparable accuracy.

**Problem.** RLVR on medical QA tends to game the reward by emitting answers without reasoning or by using degenerate reasoning formats that satisfy the verifier.

**Method.** Construct a composite reward = correctness + format penalty (no/short reasoning) + structure penalty (non-standard reasoning format), applied within standard RLVR.

**Result.** Better-formatted reasoning with reduced reward hacking and comparable accuracy to baselines (no specific numbers).

**Takeaway.** When you observe a small-LM model collapsing into a degenerate reasoning shortcut, encode an explicit penalty for that shortcut into the reward; it is a more direct fix than tuning KL or temperature.

**ELI5.** Like docking points on a math test for handing in just the answer with no working, even if the answer is right; the format itself becomes part of the grade.

### 2507.17746 — Rubrics as Rewards (RaR)

**Summary.** RaR extends RLVR to non-verifiable domains by using instance-specific rubrics as the reward signal during on-policy training. The key idea is structured multi-criteria rubrics outperform Likert-based LLM-as-judge rewards, especially for smaller judges.

**Problem.** RLVR works only where correctness is binary; medical and scientific reasoning need multi-criteria judgment that current judge-based rewards capture poorly.

**Method.** For each prompt, attach a rubric of weighted criteria; aggregate rubric checks into a scalar reward (several aggregation strategies tested) and run on-policy RL.

**Result.** Up to +31% relative on HealthBench and +7% on GPQA-Diamond over Likert LLM-judge baselines.

**Takeaway.** When you cannot get clean string-match rewards for a small-LM task, prefer a rubric checklist over a single judge score; it is more stable and scales down to small judge models.

**ELI5.** Like grading an essay with a 10-item rubric ("thesis present? evidence cited? counter-argument?") instead of asking a single grader for an overall vibe-score.

### 2506.00103 — Writing-Zero: RLVR for Non-Verifiable Tasks

**Summary.** Writing-Zero brings RLVR-style training to creative writing by replacing scalar reward models with a pairwise generative reward model and a new policy update rule. The result is a writing-tuned model trained without SFT.

**Problem.** Subjective tasks like creative writing have no ground truth, and scalar reward models hack length and over-explain.

**Method.** A writing-principle-based pairwise GenRM produces verifiable preferences; Bootstrapped Relative Policy Optimization (BRPO) compares each rollout to a bootstrapped sibling from within the group as a reference-free anchor.

**Result.** Consistent improvement and strong reward-hacking resistance versus scalar baselines, competitive on in-house and open-source writing benchmarks (no headline number).

**Takeaway.** If you must train on a fuzzy task, build a pairwise judge and use within-group anchoring instead of an absolute scalar reward.

**ELI5.** Two essays are easier to compare than to score absolutely, like asking "which dish is better?" instead of "rate this 7.4/10".

### 2502.19328 — Agentic Reward Modeling (RewardAgent)

**Summary.** RewardAgent fuses a standard preference reward model with verifiable correctness signals (factuality and instruction-following) into a single agentic reward. The contribution is showing that mixing learned preference rewards with rule-based verifiers produces better RMs both at inference-time best-of-n and as DPO training-pair generators.

**Problem.** Pure preference RMs ignore verifiable correctness signals, so they reward fluent-but-wrong outputs; pure rule-based rewards miss style and helpfulness.

**Method.** A reward agent routes each candidate through a preference RM plus two verifier modules (factuality check, instruction-following check) and combines their scores; the combined reward is used both for best-of-n at inference and to construct DPO preference pairs for training.

**Result.** Outperforms vanilla RMs on RM benchmarks and downstream best-of-n; DPO training on RewardAgent-built pairs beats DPO from conventional RMs.

**Takeaway.** For RLVR pipelines on QA, the right move is rarely "EM only" or "RM only"; a small agent that combines a verifier (EM/F1) with a learned preference signal gives a stronger, less hackable reward.

**ELI5.** Like grading a history essay with both a teacher's red pen and an automatic fact-checker; the final mark blends taste with truth so students cannot bluff their way to an A.

### 2505.15801 — VerifyBench

**Summary.** Introduces VerifyBench and VerifyBench-Hard, two benchmarks specifically for evaluating reference-based reward systems used in RLVR (not preference rewards). Constructs cases via meticulous data curation and human annotation.

**Problem.** Existing reward benchmarks measure preference comparisons; nobody had benchmarked the verifiers that decide RLVR's reward signal against ground truth references.

**Method.** Curate reference-based reasoning tasks, hand-annotate correctness, and evaluate verifier systems (rule-based and LLM-judge) on standard and hard cases.

**Result.** Larger LLM-judge verifiers do well on standard cases but all systems show substantial room on the hard split (no single headline number).

**Takeaway.** Pick your RLVR verifier deliberately; the wrong judge silently bottlenecks your whole RL run. Use VerifyBench-Hard to stress-test it before training.

**ELI5.** Like grading the graders before the exam; if the answer key is wrong, no amount of student practice helps.

---

## 3. Policy Optimization Algorithms

### 3.1 PPO (Proximal Policy Optimization)

The classical actor-critic baseline. Surrogate objective:

$$
\mathcal{L}^{\text{PPO}}(\theta) \;=\; \mathbb{E}_t\!\left[\min\!\left(\rho_t(\theta)\,\hat{A}_t,\; \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon)\,\hat{A}_t\right)\right]
$$

where $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ and $\hat{A}_t$ is an advantage estimate. PPO requires a value head, doubling memory cost.

### 3.2 GRPO (Group Relative Policy Optimization)

Introduced by DeepSeek to remove the value head. Given $G$ rollouts $\{y_i\}_{i=1}^{G}$ from prompt $x$, GRPO normalizes within the group:

$$
\hat{A}_i \;=\; \frac{r(x, y_i) - \mu_g}{\sigma_g}, \qquad \mu_g = \frac{1}{G}\sum_{j=1}^{G} r(x, y_j), \quad \sigma_g = \text{std}\bigl(\{r(x, y_j)\}\bigr)
$$

The policy update uses the same clipped surrogate as PPO with $\hat{A}_i$ broadcast across all tokens of $y_i$:

$$
\mathcal{L}^{\text{GRPO}}(\theta) \;=\; \mathbb{E}_{x, \{y_i\}}\!\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\bigl(\rho_i(\theta)\,\hat{A}_i,\; \text{clip}(\rho_i(\theta), 1 - \epsilon, 1 + \epsilon)\,\hat{A}_i\bigr)\right] \;-\; \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

Theoretical analysis: [(2503.06639)](https://arxiv.org/abs/2503.06639) derives GRPO's effective loss, dynamics, and success-amplification properties.

### 3.3 GRPO Variants

| Variant | Reference | Modification |
|---------|-----------|--------------|
| BRPO | [2506.00103](https://arxiv.org/abs/2506.00103) | Bootstrapped reference for pairwise comparison |
| RC-GRPO | [2602.03025](https://arxiv.org/abs/2602.03025) | Reward-conditioned rollouts for multi-turn tool agents |
| Fission-GRPO | [2601.15625](https://arxiv.org/abs/2601.15625) | Error-recovery via execution-failure supervision |
| MC-GRPO | [2601.22582](https://arxiv.org/abs/2601.22582) | Median-centered baseline for small-rollout regimes |
| F-GRPO | [2602.06717](https://arxiv.org/abs/2602.06717) | Difficulty-aware advantage scaling |
| Prefix Grouper | [2506.05433](https://arxiv.org/abs/2506.05433) | Efficient training via shared-prefix grouping |
| Guide-GRPO | [2506.13923](https://arxiv.org/abs/2506.13923) | Adaptive guidance hints when all rollouts fail |
| Group Turn PO | [2511.14846](https://arxiv.org/abs/2511.14846) | Per-turn grouping for multi-turn tool reasoning |

### 3.4 REINFORCE-Family

- [Crossing the Reward Bridge (2503.23829)](https://arxiv.org/abs/2503.23829) employs REINFORCE, REINFORCE++, and RLOO for soft-reward settings.
- Useful when groups are small or batch heterogeneity is high.

### 3.5 Algorithmic Failure Modes

- [GRPO Collapse in Agent Search (2512.04220)](https://arxiv.org/abs/2512.04220) — Lazy likelihood-displacement; group-baseline collapse under sparse rewards.
- Mitigation strategies: variance reduction via shrinkage [(2511.03710)](https://arxiv.org/abs/2511.03710), median baselines [(2601.22582)](https://arxiv.org/abs/2601.22582), entropy bonuses.

### Paper Cards

### 2503.06639 — GRPO's Effective Loss & Success Amplification

**Summary.** This paper provides a theoretical analysis of GRPO under verifiable binary rewards, showing the algorithm reduces to a weighted contrastive loss against synthetic samples from the previous policy. It derives closed-form optimal policies for variants differing in reward normalization (mean-only vs mean+variance) and KL regularization (mirror, reference, or both).

**Problem.** GRPO is widely used but lacks a principled understanding of how its reward calibration and KL terms shape the optimal policy and convergence behavior.

**Method.** The authors derive an explicit form for the per-iteration optimal policy in terms of the binary reward and first/second-order reward statistics, then characterize the resulting probability-of-success recurrence and its fixed point.

**Result.** Proves the GRPO fixed point's probability of success strictly exceeds the reference policy's, formally establishing success amplification.

**Takeaway.** When tuning small-model GRPO, choose between mean-only vs mean+variance normalization knowing each induces a different effective contrastive loss; the reference-KL strength directly controls the asymptotic success rate.

**ELI5.** Like discovering that a coach who pits a student against slightly-worse versions of themselves will, with the right handicap, always push the student above their starting skill ceiling.

### 2602.03025 — RC-GRPO: Reward-Conditioned GRPO

**Summary.** RC-GRPO injects discrete reward tokens (e.g., <|high_reward|>) into prompts during a preliminary SFT stage so the model can produce trajectories of controllable quality, then samples a mix of these tokens within each GRPO group during RL to keep within-group diversity high.

**Problem.** In multi-turn tool-calling, GRPO groups often collapse to all-0 or all-1 rewards, killing the group-normalized advantage and stalling learning.

**Method.** First, SFT a Reward-Conditioned Trajectory Policy on mixed-quality data with reward-goal special tokens; during RL, condition different rollouts on different reward tokens within each group to manufacture diversity.

**Result.** Qwen2.5-7B-Instruct trained with RC-GRPO surpasses all closed-source API models on BFCLv4 multi-turn.

**Takeaway.** For small tool-calling agents stuck with all-correct or all-wrong groups, deliberately conditioning some rollouts to "fail" is a cheap way to recover gradient signal without bigger groups.

**ELI5.** Tell some students "try your best" and others "deliberately attempt this poorly"; now the class has a real spread of grades to learn from instead of everyone scoring identically.

### 2601.15625 — Fission-GRPO: Recover from Tool-Call Errors

**Summary.** Fission-GRPO turns failed tool-call trajectories into on-policy training data by splitting each failure, injecting diagnostic feedback from a fine-tuned Error Simulator, and resampling recovery rollouts. This teaches multi-turn agents to interpret error messages instead of looping on invalid calls.

**Problem.** Standard RL collapses tool-call failures into sparse negative rewards, and pre-collected error datasets quickly drift away from the policy's actual failure modes.

**Method.** Each failed rollout is "fissioned" into a new training instance with simulator-injected diagnostic feedback, then multiple recovery rollouts are sampled on-policy and used as corrective supervision inside GRPO.

**Result.** +5.7% absolute error-recovery and +4.0% overall accuracy (42.75 to 46.75) on BFCLv4 Multi-Turn for Qwen3-8B; up to +17.4% on TAU-Bench variants.

**Takeaway.** For small tool-using models, harvesting the agent's own errors as on-policy curriculum is more effective than static error-correction SFT data.

**ELI5.** When an apprentice fumbles a recipe, an instructor whispers what went wrong and lets them try again on the same dish; the lesson sticks because it is about their own mistake, not someone else's.

### 2601.22582 — MC-GRPO: Median-Centered GRPO for Small Rollouts

**Summary.** MC-GRPO replaces GRPO's mean baseline with the group median to remove advantage sign-flips when rollout budgets are tiny. This stabilizes small-G training without changing the per-prompt gradient cost.

**Problem.** Under small rollout budgets, noise in the mean baseline flips the sign of advantages for some completions, reversing their gradients and degrading accuracy.

**Method.** Sample G+1 rollouts, use the median as baseline, and exclude the pivot rollout from backprop so exactly G samples contribute gradients per prompt; drop-in replacement for GRPO/DAPO/etc.

**Result.** Closes the gap between G=2 and G=8 to within 1% across model families and GRPO variants.

**Takeaway.** On a single A100, switching to a median baseline is a near-free fix that lets you train at G=2-4 with G=8-quality stability — directly relevant for compute-constrained small-LM RLVR.

**ELI5.** Like grading on a curve using the middle student rather than the class average; one outlier genius no longer flips everyone else from "above average" to "below average."

### 2602.06717 — F-GRPO: Don't Forget the Rare Correct

**Summary.** F-GRPO adds a difficulty-aware, focal-loss-inspired advantage scaling that down-weights updates on already-easy prompts, preventing GRPO from concentrating mass on common solutions and forgetting rare-correct trajectories. It is a drop-in modifier for GRPO/DAPO/CISPO.

**Problem.** Small group sizes bias GRPO toward already-likely trajectories; even when total correct mass grows, mass on rare-correct modes can shrink.

**Method.** A focal-style coefficient scales the advantage based on a prompt's success rate, deemphasizing high-success prompts so rare-correct modes survive; no extra rollouts required.

**Result.** Qwen2.5-7B pass@256 improves 64.1 to 70.3 (GRPO), 69.3 to 72.5 (DAPO), 73.2 to 76.8 (CISPO), with pass@1 preserved.

**Takeaway.** If you care about pass@k and exploration on a small-LM run, a one-line focal reweighting on advantages preserves rare solutions for free.

**ELI5.** Stop drilling the multiplication tables you already know; spend the practice time on the tricky problems you only occasionally get right.

### 2506.05433 — Prefix Grouper: Efficient GRPO

**Summary.** Prefix Grouper is a drop-in GRPO modification that encodes the shared prompt prefix only once across the group, instead of redundantly per rollout. It is provably training-equivalent to standard GRPO.

**Problem.** GRPO recomputes the long shared prefix for every group member, which dominates compute in long-context settings (e.g., search-augmented prompts).

**Method.** Restructure self-attention into shared-prefix and per-rollout halves so the prefix is encoded once while preserving identical forward outputs and gradients.

**Result.** Identical training behavior with significantly reduced compute, enabling larger group sizes under the same budget (no scalar headline).

**Takeaway.** Direct efficiency win for any GRPO codebase with long prompts (RAG, agentic, multi-turn search). Likely worth integrating into a 1xA100 thesis run.

**ELI5.** If five chefs need the same broth, make the broth once and ladle it out, instead of boiling five identical pots.

### 2506.13923 — Guide: Adaptive Hints in RL

**Summary.** Guide is an online RL algorithm that injects natural-language hints into the prompt only when all rollouts fail, then off-policy-corrects so the final policy still works without hints. Decomposes RLVR gains into "compression of pass@k into pass@1" versus "true capability gain".

**Problem.** Vanilla RLVR cannot learn problems where every rollout fails; pass@k stays zero so there is no signal.

**Method.** When all rollouts are wrong, add a hint to the context, generate guided trajectories, then adjust importance sampling so the policy is optimized for the hint-free deployment context. Variants for GRPO and PPO.

**Result.** Up to +4% macro-average across math benchmarks for Guide-GRPO on 7B and 32B models.

**Takeaway.** Closely related to Agent-RLVR's guidance idea but with explicit off-policy correction. Useful pattern for hard problems where pass@k = 0; integrates cleanly with GRPO.

**ELI5.** When the student cannot solve a problem at all, the tutor whispers a hint, then teaches them to solve future problems without the whisper.

### 2511.14846 — GTPO: Group Turn Policy Optimization

**Summary.** GTPO targets multi-turn tool-integrated reasoning by replacing GRPO's trajectory-level reward with turn-level rewards plus return-based advantages and a self-supervised code-execution shaping signal. It outperforms GRPO across math and non-math benchmarks at negligible overhead.

**Problem.** GRPO's single trajectory reward is too coarse for multi-turn TIR (think-code-execute loops), causing training to stagnate.

**Method.** Three additions on top of GRPO: (1) per-turn reward assignment, (2) advantages computed from normalised discounted returns instead of group means, (3) self-supervised reward shaping that uses code-execution signals to densify the binary outcome reward.

**Result.** +3.0% over GRPO on math reasoning and +3.9% on commonsense and program-synthesis benchmarks.

**Takeaway.** For any Search-R1-style multi-turn RLVR loop on small models, adopting turn-level credit assignment is a higher-ROI change than tweaking clip ratios.

**ELI5.** Instead of paying a basketball team only when they win the whole game, pay them after each successful possession; players learn what to do mid-game, not just what to celebrate.

### 2512.04220 — GRPO Collapse via Lazy Likelihood Displacement

**Summary.** Diagnoses why Search-R1-style GRPO training collapses on agentic-search tasks: a Lazy Likelihood Displacement (LLD) where likelihoods of both correct and incorrect responses decay together, triggering a death-spiral. Proposes LLDS, a targeted regulariser that activates only on the offending tokens.

**Problem.** GRPO appears to converge then collapses on multi-step search tasks; the cause was unidentified, and existing fixes (KL, clip tuning) only delayed the failure.

**Method.** Empirically traces a three-phase trajectory (early stagnation, steady decay, accelerated collapse), attributes it to falling response likelihoods inflating gradients; LLDS adds a likelihood-preserving regularisation that fires only when an action's likelihood drops and only on the responsible tokens.

**Result.** +45.2% relative on Qwen2.5-3B and +37.1% on Qwen2.5-7B over vanilla GRPO across seven QA benchmarks.

**Takeaway.** Directly relevant to Search-R1 reproduction on small models: if your GRPO run plateaus then crashes, LLDS is a precise fix targeting the actual mechanism, not a blanket KL increase.

**ELI5.** Like spotting that your bicycle wobbles not because of the wind but because one specific spoke is loose, so you tighten that spoke instead of redesigning the wheel.

### 2511.03710 — Shrinkage Baselines for RLVR

**Summary.** This paper applies Stein-style shrinkage to GRPO's per-prompt mean baseline, blending each prompt's empirical reward mean with the across-prompt mean to lower variance. The result is a drop-in baseline with provably lower-variance gradients and zero extra compute.

**Problem.** GRPO uses the empirical mean of a prompt's small group as its baseline, which is noisy in the few-rollout regime and inflates gradient variance.

**Method.** Replace the per-prompt mean with a shrinkage estimator that pulls each prompt's mean toward the global mean, motivated by Stein's paradox; theoretical bounds show strict variance reduction with no new hyperparameters.

**Result.** Consistent variance reduction and improved training stability over the empirical-mean baseline (no specific accuracy headline in abstract).

**Takeaway.** For low-rollout-count GRPO (e.g. 4-8 generations), swapping in a shrinkage baseline is a free stability win that survives any model scale.

**ELI5.** When a teacher has only three test scores per student to estimate ability, blending each student's average with the class average gives a less jumpy estimate than trusting three scores alone.

---

## 4. Exploration, Optimization, and Credit Assignment

Sparse reward and entropy collapse are the dominant pathologies.

### 4.1 Entropy Collapse

Empirically, GRPO drives policy entropy toward zero before reward saturates:

$$
\mathcal{H}(\pi_\theta(\cdot \mid x)) \;\xrightarrow{t \to \infty}\; 0
$$

producing premature commitment to suboptimal patterns.

- [Trial-and-Error Analysis (2508.07534)](https://arxiv.org/abs/2508.07534) — Diagnoses entropy collapse; proposes perplexity-based advantage shaping.

### 4.2 Exploration Mechanisms

- [Forward-KL Exploration (2510.03865)](https://arxiv.org/abs/2510.03865) — Replaces reverse-KL with forward-KL to encourage out-of-distribution search.
- [Beyond 80/20 (2506.01939)](https://arxiv.org/abs/2506.01939) — High-entropy "forking tokens" carry disproportionate learning signal; RLVR efficiency comes from optimizing these.
- [First Return, Entropy-Eliciting Explore (2507.07017)](https://arxiv.org/abs/2507.07017) — Entropy-aware exploration heuristics.

### 4.3 Credit Assignment & Advantage Shaping

- [UCAS — Uncertainty-aware Advantage Shaping (2510.10649)](https://arxiv.org/abs/2510.10649) — Refines per-token credit using model confidence.
- [PACR — Progressively Ascending Confidence Reward (2510.22255)](https://arxiv.org/abs/2510.22255) — Dense stepwise reward signals for complex multi-step reasoning.
- [StepHint (2507.02841)](https://arxiv.org/abs/2507.02841) — Multi-level stepwise hints with rubric-shaped rewards under sparse-reward regimes.

### 4.4 Variance Reduction

Within-group variance can dominate gradient noise. James–Stein-style shrinkage:

$$
\hat{A}_i^{\text{shrink}} \;=\; \left(1 - \frac{(G - 3)\sigma^2}{\|r - \mu_g\|^2}\right)_+ \cdot (r_i - \mu_g)
$$

- [Shrinkage Baselines (2511.03710)](https://arxiv.org/abs/2511.03710) — Reduces variance under low-rollout budgets.

### 4.5 Adaptive Guidance

- [Adaptive Guidance / Guide-GRPO (2506.13923)](https://arxiv.org/abs/2506.13923) — When all rollouts in a group fail, inject in-context hints; off-policy importance-ratio correction restores stability.

### Paper Cards

### 2508.07534 — Systematic Analysis of RLVR Exploration

**Summary.** A technical report dissecting how exploration works inside RLVR across exploration-space shaping, the entropy-performance trade-off, and translating exploration into measurable performance. The contribution is a unifying analytical framework rather than a new algorithm.

**Problem.** RLVR's exploration mechanisms are empirically successful but poorly understood, blocking principled improvements.

**Method.** Define quantitative metrics for capability boundaries; track entropy across training stages, instances, and tokens; analyze how exploration gains translate (or fail to translate) into pass@1.

**Result.** No single headline number; the report consolidates findings on entropy collapse, sample-level entropy dynamics, and exploration ROI.

**Takeaway.** Use this paper as a reference checklist when diagnosing your small-LM RLVR run: track entropy at multiple granularities and watch the exploration-to-performance translation.

**ELI5.** Like a coaching manual that doesn't introduce a new drill but maps out which existing drills actually move the needle and which just look productive.

### 2510.03865 — RAPO: Rewards-Aware Policy Optimization

**Summary.** RAPO addresses the pass@k collapse in RLVR by replacing the reverse-KL regularizer with a forward-KL term and reweighting the reference policy for adaptive in-distribution exploration. Applied to Qwen2.5-3B/7B without any SFT.

**Problem.** As sampling budget grows, RLVR-trained models lose their edge over the base because reverse-KL's mode-seeking traps the policy inside the base model's support.

**Method.** Two changes inside the RLVR objective: (i) swap reverse-KL for forward-KL (mass-covering, encourages OOD exploration); (ii) reweight the reference policy adaptively for finer in-distribution exploration. No SFT, trained on 8K SimpleRL-Zero.

**Result.** On AIME2024 and AIME2025, RAPO surpasses the base model's pass@k ceiling and solves problems previously intractable for the base.

**Takeaway.** For a 3B model with no SFT budget, RAPO is a drop-in replacement for vanilla GRPO that lets the policy escape the base-model support; concretely, just changing the divergence direction is the bulk of the gain.

**ELI5.** Like loosening the leash from "stay within arm's reach" to "stay roughly in the park"; the dog can now sniff places it never could before, but won't run into traffic.

### 2506.01939 — Forking Tokens (80/20 in RLVR)

**Summary.** Identifies that only the high-entropy "forking" tokens in a CoT actually steer reasoning, and shows RLVR mostly nudges those tokens. Restricting policy gradients to the top-entropy 20% of tokens improves results.

**Problem.** Standard RLVR updates every token equally even though most tokens are deterministic continuations that contribute no decision-making.

**Method.** Mask the policy gradient to the top 20% highest-entropy tokens (the forks), leaving low-entropy continuations untouched.

**Result.** +11.04 on AIME'25 and +7.71 on AIME'24 for Qwen3-32B; matches full updates on Qwen3-8B; gains scale with model size.

**Takeaway.** Cheap and powerful trick for any GRPO recipe: filter the loss to forking tokens. Probably more impactful than tuning hyperparameters.

**ELI5.** Coach the quarterback only on the moments when he chose the play, not on every routine handoff; those are the choices that actually decide games.

### 2507.07017 — FR3E: First Return, Entropy-Eliciting Explore

**Summary.** FR3E is a structured exploration framework for RLVR that locates high-uncertainty decision points inside a reasoning trajectory and performs targeted re-rollouts from those points. The contribution is to convert sparse outcome rewards into semantically grounded intermediate feedback without needing dense process supervision.

**Problem.** Vanilla RLVR exploration is unstable, with rollouts collapsing onto a small set of trajectories and providing little learning signal on hard prompts.

**Method.** Detect tokens with high policy entropy as "decision points," roll out multiple continuations from each, and use the diverging outcomes as intermediate (process-style) signals fed back into the policy update.

**Result.** On AIME24, FR3E produces longer, more coherent responses and a higher fraction of fully correct trajectories than standard RLVR baselines.

**Takeaway.** For a small-LM RLVR pipeline, branching rollouts at entropy spikes is a cheap way to get process-level credit without training a separate reward model.

**ELI5.** Like a chess coach who, instead of grading whole games, pauses at the moments you hesitated and makes you replay just that move five different ways.

### 2510.10649 — UCAS: Uncertainty-Aware Advantage Shaping

**Summary.** UCAS replaces GRPO's uniform per-token advantage with a model-internal uncertainty-shaped credit signal so high-stakes uncertain tokens dominate learning. This unlocks deeper exploration and prevents entropy collapse on math RLVR.

**Problem.** Standard GRPO broadcasts the same advantage to every token, drowning the rare pivotal decisions and triggering entropy collapse.

**Method.** Two-stage shaping: response-level advantage is modulated by a logit-space self-confidence proxy, then an asymmetric token-level penalty based on raw logit certainty rewards correct uncertain tokens and punishes confident wrong ones.

**Result.** Outperforms strong RLVR baselines on five math benchmarks at both 1.5B and 7B scales while preserving reasoning diversity.

**Takeaway.** A drop-in advantage reweighting (no extra model, no reward model) buys you exploration depth on small-model RLVR; very cheap to try in a GRPO loop.

**ELI5.** Instead of paying every musician in the orchestra the same flat fee per note, you pay extra at the moments where one wrong note would ruin the symphony, so they practise those bars hardest.

### 2510.22255 — PACR: Progressively Ascending Confidence Reward

**Summary.** PACR adds a dense intrinsic shaping reward that fires whenever the model's belief in the gold answer rises step by step along its reasoning trajectory. This densifies the sparse outcome signal of RLVR and accelerates convergence.

**Problem.** Outcome-only RLVR gives no intermediate guidance, so exploration is slow and many trajectories teach nothing.

**Method.** Compute the model's posterior probability of the ground-truth answer at each reasoning step; reward trajectories whose probability monotonically ascends, encoding the inductive bias that good reasoning should make the right answer increasingly likely.

**Result.** Reaches reward saturation with fewer trajectories and improves accuracy across multiple reasoning benchmarks (no single headline number in abstract).

**Takeaway.** For sparse-reward RLVR you can synthesise a process reward from the policy itself (no separate verifier) by tracking ascending belief in the gold; cheap and architecture-agnostic.

**ELI5.** Like grading a detective not only on naming the killer but on whether their suspicion meter climbs steadily as the clues come in, instead of zig-zagging.

### 2507.02841 — StepHint: Multi-Level Stepwise Hints

**Summary.** StepHint provides the model with the first k steps of a stronger model's solution as a hint at multiple granularities, tackling near-miss reward and exploration stagnation. Uses adaptive partitioning to find natural step boundaries.

**Problem.** RLVR penalizes a near-correct chain entirely (near-miss problem) and traps models in a comfort zone of familiar solutions.

**Method.** Generate solutions from a stronger teacher, partition them into reasoning steps adaptively, and feed the model multiple hint depths simultaneously during RL rollouts.

**Result.** Outperforms competitive RLVR methods on six math benchmarks with strong out-of-domain generalization (no scalar headline).

**Takeaway.** Multi-depth prefix hints are a richer training signal than single-shot hints; combined with Guide/Agent-RLVR ideas, this is a converging design pattern for hard-task RLVR.

**ELI5.** Some students need just the first sentence of a worked example, others need three sentences; show all depths and let each part of the model pick what it needs.

---

## 5. Reward Hacking, Safety, and Robustness

### 5.1 Hacking Patterns

Common exploits:
- Verbatim copying from prompt or context.
- Format gaming (filling tags without semantic content).
- Tripwire avoidance under instruction-following.
- Reward-correlated but task-orthogonal artifacts.

### 5.2 Mitigation

- **Composite penalties**: $r_{\text{final}} = r_{\text{task}} - \lambda_1 r_{\text{leak}} - \lambda_2 r_{\text{format-violation}}$ — [(2509.15557)](https://arxiv.org/abs/2509.15557)
- **Trap instructions and intent alignment**: [IFDECORATOR (2508.04632)](https://arxiv.org/abs/2508.04632) wraps the RL loop with adversarial probes.
- **Semantic leak detection**: Sentence-BERT comparison between response and forbidden-content references.

### 5.3 Safety Under KL Constraint

- [Breaking the Safety-Capability Tradeoff (2511.21050)](https://arxiv.org/abs/2511.21050) — Demonstrates that KL-constrained RLVR preserves safety guardrails even as reasoning capacity scales; the $\beta D_{\text{KL}}$ term provably bounds policy drift in the safety-relevant subspace.

### 5.4 Spurious Reward Sensitivity

- [Spurious Rewards (2506.10947)](https://arxiv.org/abs/2506.10947) — RLVR gains persist under *uncorrelated* reward signals on models with strong pre-instilled reasoning patterns (code preferences rise from 65% to 90% even with random rewards). The reward signal is partly a *trigger* for latent capability, not solely an *information* source.

### Paper Cards

### 2508.04632 — IFDecorator: RLVR for Instruction Following

**Summary.** IFDecorator wraps RLVR for instruction-following with a co-evolving instruction-verification flywheel, an intent-alignment bypass, and "trip wire" trap instructions to detect reward hacking. It addresses sample inefficiency and verifier shortcut exploitation.

**Problem.** RLVR for instruction following is sample-inefficient and prone to verifier-shortcut hacking, where the model satisfies the verifier without satisfying the user's real intent.

**Method.** Three components: (1) cooperative-adversarial pipeline that jointly hardens instructions and verifications; (2) IntentCheck module enforcing intent alignment; (3) trip-wire trap prompts that catch reward-hacking behaviors during training.

**Result.** Qwen2.5-32B-Instruct-IFDecorator hits 87.43% on IFEval, beating GPT-4o.

**Takeaway.** For RLVR on open-ended instructions, plant trap prompts in your training stream; they are a cheap reward-hacking smoke detector and let you catch shortcuts before the policy bakes them in.

**ELI5.** Like seeding an exam with a few questions that have obviously wrong "easy" tricks; if the student keeps falling for them, you know they're gaming, not learning.

### 2511.21050 — RLVR Maintains Safety Guardrails

**Summary.** This paper shows that, unlike SFT and RLHF, RLVR can boost reasoning without eroding safety alignment. It backs the claim with KL-bounded theory and experiments on five adversarial safety benchmarks.

**Problem.** Conventional fine-tuning trades capability for safety; whether RLVR inherits that trade-off was an open question.

**Method.** Derive upper bounds on safety drift under KL-constrained RLVR optimisation, prove conditions for zero degradation, then empirically sweep optimisation algorithms, model scales, and task domains across five safety benchmarks.

**Result.** RLVR can simultaneously enhance reasoning and maintain or improve safety guardrails across all five benchmarks (no single headline percentage in abstract).

**Takeaway.** Reassuring for thesis framing: KL-constrained verifiable-reward training does not have to be paired with extra safety post-training to be deployment-safe.

**ELI5.** Tutoring a student in math with strict honour-code rules turns out not to make them ruder, unlike letting them ramble freely with a personality coach.

### 2506.10947 — Spurious Rewards in RLVR

**Summary.** Demonstrates that RLVR with random or spurious rewards can still produce large gains on Qwen2.5-Math because GRPO's clipping bias amplifies high-prior pretrained behaviors. The effect is highly model-family dependent.

**Problem.** RLVR results are usually attributed to the reward signal, but that attribution may be wrong if the algorithm itself amplifies pretraining priors.

**Method.** Train Qwen2.5-Math-7B with random rewards under GRPO, dissect the clip term, and replicate across Qwen, Llama3, and OLMo2 families.

**Result.** Random rewards yield +21.4 points on MATH-500 for Qwen2.5-Math-7B (vs. +29.1 with true rewards); code-reasoning frequency grows from 65% to 90%; gains do not transfer to Llama3 or OLMo2.

**Takeaway.** Critical caveat for your thesis: validate RLVR effects on more than one model family, otherwise your gains may be a Qwen-specific GRPO clip artifact, not real learning.

**ELI5.** Praising a parrot at random still makes it talk more, because it already knew the words; a different bird that never spoke will not learn just from random claps.

---

## 6. Online Self-Verification (RISE)

RISE (Reasoning + Self-Evaluation) frameworks intertwine solution generation and critique within a single online RL process. Departs from classical post-hoc verification by using verifiable signals to **co-train both roles simultaneously**.

### 6.1 RISE Mechanism

A unified policy $\pi_\theta$ produces both a solution $y$ and a self-critique $c$:

$$
(y, c) \sim \pi_\theta(\cdot \mid x)
$$

The integrated reward aggregates solution and verification correctness:

$$
r_{\text{RISE}}(x, y, c) \;=\; r_{\text{ans}}(x, y) \;+\; \gamma \cdot r_{\text{ver}}(c, y, x)
$$

with $r_{\text{ver}}$ rewarding self-critique for agreeing with the deterministic outcome verifier on $y$. Optimization is on-policy via PPO or GRPO.

### 6.2 Key Papers

- [Trust, But Verify (2505.13445)](https://arxiv.org/abs/2505.13445) — Core RISE framework with PPO; mathematics domain.
- [Incentivizing LLMs to Self-Verify (2506.01369)](https://arxiv.org/abs/2506.01369) — Unified gen-verify with GRPO; achieves 87.2% verification accuracy on MATH500.
- [RISE-CoT for VLMs (2508.13229)](https://arxiv.org/abs/2508.13229) — Vision-LLM extension for image annotation with self-supervised composite rewards.
- [LMs Better Reasoners with Self-Verification (2212.09561)](https://arxiv.org/abs/2212.09561) — Foundational backward-verification baseline.

### 6.3 Reported Results

| Model | Task | Reasoning Acc. | Verification Acc. |
|-------|------|----------------|-------------------|
| RISE-7B | MATH500 | 42.9% | 69.2% |
| Baseline SFT | MATH500 | 11.3% | — |
| Qwen-7B Self-Verify | MATH500 | 83.6% | 87.2% (F1 92.8%) |
| Qwen2-VL-2B (RISE-CoT) | Vision annot. | mAP 0.404 | JSD 0.071 |

### 6.4 Limitations

- Restricted to deterministic rule-based verifiable outcomes.
- Open-ended tasks (free-form code generation, commonsense) remain unresolved.
- Risk of *overconfident incorrect self-verification* persists; verifier head can collude with solution head.

### Paper Cards

### 2505.13445 — RISE: Trust But Verify

**Summary.** Online RL framework that simultaneously trains a model to solve problems and to verify its own solutions, both via verifiable rewards. Targets the "superficial self-reflection" failure where models claim to verify but don't.

**Problem.** RLVR-trained models often produce reflection tokens that don't actually catch errors, because nothing in standard RL rewards real verification.

**Method.** Each iteration, the model generates a solution and then critiques its own on-policy solution; the outcome verifier scores both the solution and the critique, and both trajectories contribute to the policy update.

**Result.** Consistent gains on math benchmarks plus genuinely more frequent and accurate self-verification (no headline number in abstract).

**Takeaway.** For small reasoning models, train solver and verifier as the same policy with two reward heads; you avoid distilling a separate critic and the verification skill emerges in-band.

**ELI5.** Like a student who, after every problem, has to grade their own work and is graded on both the solution and the grading; soon they actually catch their own mistakes instead of nodding at them.

### 2506.01369 — Self-Verifying RL

**Summary.** This paper unifies answer generation and answer verification within a single RL run so the policy learns to grade its own outputs. The trained model can then self-scale at test time without external verifiers.

**Problem.** Test-time scaling with external reward models gives only marginal gains for specialized post-trained generators because the generator and judge live in different distributions.

**Method.** Joint RL objective rewards both correct generations and correct self-verifications, trained on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B.

**Result.** Improves post-training accuracy and yields effective self-verification-driven test-time scaling on multiple math benchmarks.

**Takeaway.** A single policy that generates and verifies eliminates the generator/judge distribution gap; attractive for small models where you cannot afford a separate critic.

**ELI5.** Train the chess player to also call "blunder" on their own moves; at game time they can prune their own bad ideas without an extra coach.

### 2508.13229 — RISE: Self-Supervised CoT Rewards for VLMs

**Summary.** RISE generates verified chains-of-thought for VLM image annotation tasks via a closed-loop "annotation-reasoning-annotation" RL, then fine-tunes on the filtered CoTs. The novelty is producing high-quality CoTs without manual annotation by checking that the CoT can reconstruct the label.

**Problem.** Visual RFT produces inconsistent CoTs because no verified rationales exist for VLM pre-training, limiting reasoning quality.

**Method.** Stage 1 (RISE-CoT): RL loop where a CoT is rewarded if it lets the model reconstruct the original label without leakage. Stage 2 (RISE-R1): SFT on the highest-reward CoTs followed by RFT.

**Result.** RISE-trained Qwen2-VL-2B beats SFT and Visual-RFT on complex and simple image-annotation tasks (no specific numbers in abstract).

**Takeaway.** For small VLMs, you can bootstrap a process-reward signal by checking whether a generated rationale is sufficient to recover the answer, no human CoTs needed.

**ELI5.** Like accepting a witness's story only if, when you replay it from scratch, it lands on the same verdict; the story is its own polygraph.

### 2212.09561 — LLMs are Better Reasoners with Self-Verification

**Summary.** The paper shows that LLMs can re-verify their own chain-of-thought conclusions by treating the answer as a condition and checking whether the original problem can be reconstructed. The contribution is a backward self-verification step that scores candidate answers and selects the highest, improving reasoning accuracy.

**Problem.** CoT reasoning is brittle: any single-step mistake propagates and corrupts the final answer.

**Method.** Generate multiple CoT answers; for each, plug the answer back into the question and ask the LM to re-derive the masked premise; score candidates by reconstruction consistency and pick the best.

**Result.** Improves accuracy across arithmetic, commonsense, and logical reasoning datasets (no single headline number in abstract).

**Takeaway.** A cheap inference-time "answer-then-verify" wrapper is complementary to RLVR; the same backward-check signal could even serve as a verifiable reward when ground truth is unavailable.

**ELI5.** Like a detective who writes a suspect's name on the board and then re-reads the case file asking "if this were true, do all the clues fit?"

---

## 7. Multi-Task Learning with Verifiable and Non-Verifiable Rewards

The hardest open problem: jointly optimizing objectives where some are verifiable (math correctness, code tests) and some are intrinsically subjective (preferences, style, aesthetic).

### 7.1 Formal Setup

For $K$ tasks with reward functions $\{r_1, \ldots, r_K\}$, where $r_1, \ldots, r_m$ are verifiable and $r_{m+1}, \ldots, r_K$ are preference-based:

$$
\mathcal{J}_{\text{multi}}(\theta) \;=\; \sum_{i=1}^{m} w_i \cdot \mathbb{E}[r_i^{\text{ver}}] \;+\; \sum_{j=m+1}^{K} w_j \cdot \mathbb{E}[r_j^{\text{pref}}] \;-\; \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

The challenge: $w_i$ are usually treated as hyperparameters, but verifiable and preference rewards live on different scales and exhibit different noise properties.

### 7.2 Approaches

- [Multi-task RL Cross-learning in RKHS (2008.11895)](https://arxiv.org/abs/2008.11895) — Constrains task-specific policies within neighborhood of shared central policy.
- [Multi-Task IRL for Common Sense (2402.11367)](https://arxiv.org/abs/2402.11367) — Decomposes task-specific verifiable + shared common-sense rewards.
- [Combining Reward Sources (2103.12142)](https://arxiv.org/abs/2103.12142) — Multitask Inverse Reward Design with uncertainty propagation.
- [Variational IRL (2206.09498)](https://arxiv.org/abs/2206.09498) — Empowerment regularization for transferable, robust rewards.
- [Multi-Task Reward Learning from Human Ratings (2506.09183)](https://arxiv.org/abs/2506.09183) — Adaptive weighting reflecting per-task uncertainty.
- [Agentic Reward Modeling (2502.19328)](https://arxiv.org/abs/2502.19328) — Modular routers combining base preference models with aspect-specific verifiable signals.
- [Imperfect also Deserves Reward (2104.04748)](https://arxiv.org/abs/2104.04748) — Hierarchical decomposition for dialog (domain/act/slot levels).

### 7.3 Hybrid and Bridge Approaches

- [Writing-Zero (2506.00103)](https://arxiv.org/abs/2506.00103) — Pairwise GenRM transforms subjective preferences into quasi-verifiable signals.
- [Auditable-choice Reframing (2511.02463)](https://arxiv.org/abs/2511.02463) — Verifiable Multiple-Choice Reformulation: convert open-ended tasks into MCQ form.
- [Crossing the Reward Bridge (2503.23829)](https://arxiv.org/abs/2503.23829) — Soft model-based rewards for noisy/ambiguous domains.

### 7.4 Specialized Multi-Objective Settings

- [RAIDEN-R1 (2505.10218)](https://arxiv.org/abs/2505.10218) — Role-aware GRPO with verifiable rewards.
- [SATORI-R1 (2505.19094)](https://arxiv.org/abs/2505.19094) — Decomposes multimodal reasoning into explicit verifiable subtasks.

### Paper Cards

### 2008.11895 — Cross-learning for Multi-task RL in RKHS

**Summary.** This paper introduces "cross-learning," a multi-task RL scheme in which agents solving related tasks are constrained to keep their policies close to a shared central policy in a reproducing kernel Hilbert space. The contribution is a projected policy gradient algorithm with provable near-optimal convergence and a multi-task policy that adapts quickly to unseen related tasks.

**Problem.** Standard single-task RL is sample-hungry and generalises poorly when the task is even mildly perturbed.

**Method.** Each agent optimises its own task reward subject to a constraint that its RKHS policy stays within a bounded distance of a learned central policy; optimisation is via projected policy gradient with high-probability convergence guarantees.

**Result.** On a navigation benchmark, the cross-learned central policy avoids obstacle shapes never seen during training (no scalar metric in abstract).

**Takeaway.** A KL- or distance-regularised "anchor" policy across related tasks is an old idea that prefigures the per-prompt reference policy used in GRPO; it suggests RLVR pipelines benefit from cross-task regularisation as much as from per-task reward shaping.

**ELI5.** Like a music school where every student must stay close to a shared "house style" while learning their own instrument, so any of them can sit in for another at short notice.

### 2402.11367 — Multi-task IRL for Common-Sense Reward

**Summary.** This paper splits the agent's reward into a known task-specific term and an unknown "common-sense" term that captures general expected behaviour, and learns the latter via inverse RL across many tasks. The contribution is showing that single-task IRL fails to recover transferable rewards but multi-task IRL succeeds.

**Problem.** Single-task IRL can train a competent agent yet learn a reward that does not actually transfer to a new agent, indicating it never recovered the "true" common-sense component.

**Method.** Run IRL jointly over multiple tasks, sharing a common-sense reward term while each task adds its own task reward; the shared term is forced to explain expert behaviour across tasks and so generalises.

**Result.** Multi-task IRL produces a common-sense reward that transfers to new agents, unlike single-task IRL which does not (qualitative result, no scalar in abstract).

**Takeaway.** If you want a reward model that captures "good behaviour" beyond the verifiable answer (e.g., good search hygiene), train it across many task families rather than fitting it on a single one.

**ELI5.** Like learning manners by watching dinners in many cultures: you separate "use a fork in Paris" from the universal "do not chew with your mouth open."

### 2103.12142 — MIRD: Combining Conflicting Reward Sources

**Summary.** This paper studies how to combine two reward functions learned from different (possibly misspecified) sources without overcommitting to either. It proposes Multitask Inverse Reward Design (MIRD), which retreats to a broader posterior over rewards when sources conflict.

**Problem.** Multiplying likelihoods of two reward signals fails badly when one or both observation models are misspecified.

**Method.** MIRD treats each reward source as evidence about a latent task, maintains a distribution over compatible reward functions, and acts to maximise expected reward under that broadened distribution; a variant MIRD-IF refines the trade-off.

**Result.** Toy-environment experiments show MIRD/MIRD-IF balance conservatism and informativeness better than naive baselines (no scalar metric in abstract).

**Takeaway.** When mixing a verifiable reward (EM, exec) with a learned preference reward, do not naively sum or multiply; explicitly model that one of them might be wrong and act under the broader posterior.

**ELI5.** Like a juror who hears two witnesses contradicting each other; instead of believing the louder one, they widen their suspicion to cover both stories until more evidence arrives.

### 2206.09498 — Variational Multi-task IRL with Empowerment

**Summary.** This paper learns transferable rewards and policies across multiple robotic tasks from unlabeled expert demos using a variational, empowerment-regularised inverse RL framework. The contribution is "situational empowerment," a mutual-information regulariser that disentangles task intent from environment dynamics.

**Problem.** Standard IRL on single tasks overfits to environment dynamics and fails to transfer when dynamics change.

**Method.** Building on GAN-style adversarial IRL, the method maximises a variational lower bound on mutual information between actions and futures conditioned on state and sub-task, jointly learning a multi-task reward and policy.

**Result.** Outperforms prior imitation baselines in performance and data efficiency on several multi-task transfer benchmarks (no scalar in abstract).

**Takeaway.** When designing reward shaping for multi-skill agents (search vs reason vs answer), an information-theoretic regulariser that ties actions to future task progress can stabilise transfer.

**ELI5.** Like teaching a chef to cook in any kitchen by rewarding moves that visibly change the dish, not moves that just rearrange the same kitchen.

### 2506.09183 — Multi-Task Reward Learning from Human Ratings

**Summary.** Proposes a reward-learning method that jointly models human ratings as both a classification and a regression signal with learnable weights. The aim is to capture the multi-strategy nature of human judgment.

**Problem.** Standard RLHF reduces human reasoning to a single isolated task (just classify or just regress), ignoring uncertainty in human decision-making.

**Method.** Infer a reward function from human ratings in reward-free environments, with a learned weighting between classification and regression heads.

**Result.** Outperforms existing rating-based RL methods and sometimes traditional RL on synthetic-rating experiments (no scalar headline).

**Takeaway.** When using human ratings as the reward source, consider modeling multiple judgment modes jointly rather than picking one head.

**ELI5.** Asking a film critic both "thumbs up?" and "stars 1-10?" together captures more than either alone; the model learns to weight which question matters when.

### 2104.04748 — Multi-level Reward Modeling for Dialog RL

**Summary.** This paper attacks reward sparsity in task-oriented dialog RL by factoring the reward into a three-level hierarchy of domain, dialog act, and slot. The contribution is an inverse adversarial RL recipe that learns interpretable, fine-grained rewards from real dialog data.

**Problem.** Sparse end-of-dialog rewards make dialog-management RL slow and unstable, while flat learned rewards are uninterpretable.

**Method.** An adversarial inverse-RL discriminator is trained at each of the three semantic levels (domain, act, slot) to score state-action pairs, producing a multi-level dense reward used by the dialog policy.

**Result.** Across multiple RL dialog managers, the multi-level reward improves both task success and convergence speed (no scalar metric in abstract).

**Takeaway.** Decomposing a sparse outcome reward along the natural compositional structure of the task (here: domain/act/slot, in QA: retrieval/reasoning/answer) gives denser signal without abandoning verifiability.

**ELI5.** Like grading an essay on thesis, structure, and grammar separately instead of just one final letter grade, so the writer knows which lever to pull.

### 2511.02463 — VMR-RLVR: Verifiable Multiple-Choice Reformulation

**Summary.** VMR-RLVR sidesteps the missing-ground-truth problem of open-ended RLVR by recasting open-ended prompts as verifiable multiple-choice questions. This lets RLVR train on creative writing and subjective QA without a learned reward model.

**Problem.** RLVR has been stuck in math/code because open-ended tasks have no automatic verifier; reward-model RL is the usual fallback but it is noisy.

**Method.** Restructure each open-ended training example into a multiple-choice format with a known correct answer, then run standard verifiable-reward RL on the MC version while transferring gains to the open-ended task.

**Result.** +3.29 average points across seven open-ended benchmarks over RL with a learned reward model.

**Takeaway.** If you want to extend a small-model RLVR pipeline beyond math without building a reward model, MC-reformulation is a near-free trick to keep the verifier deterministic.

**ELI5.** Translating a free-response essay into a multiple-choice version lets the autograder still grade it, and what the student learns transfers back to writing essays.

### 2505.10218 — RAIDEN-R1

**Summary.** Applies GRPO with a Verifiable Role-Awareness Reward (VRAR) to train role-playing conversational agents that maintain character consistency. Builds a role-aware CoT dataset via multi-LLM collaboration to seed reasoning.

**Problem.** Role-playing agents drift out of character because consistency is hard to quantify as a reward.

**Method.** Mine "role-specific keys" via singular and multi-term strategies, use their presence/absence as a binary verifiable reward, then GRPO on a 14B base.

**Result.** 88.04% / 88.65% on RAIDEN's Script-Based Knowledge and Conversation Memory metrics.

**Takeaway.** Even fuzzy properties like persona consistency can be made into RLVR rewards by extracting role-specific keyword/key-phrase checks; do not assume RLVR is only for math.

**ELI5.** Like training an actor by checking, after each scene, whether they used their character's signature phrases; a simple rubric turns out to be enough to keep them in character.

### 2505.19094 — SATORI-R1: Anchored Multimodal RL

**Summary.** SATORI-R1 applies RLVR to visual question answering by decomposing the task into three verifiable sub-stages (caption, region localization, answer) instead of free-form chains. Each stage gets its own explicit reward, keeping the model's attention on the image rather than drifting through long unverifiable text.

**Problem.** R1-style free-form CoT for VQA diffuses visual focus and amplifies policy-gradient variance because the intermediate steps are unverifiable.

**Method.** Decompose VQA into global captioning, region localization, and answer prediction; train with stage-wise verifiable rewards on a 12k caption+bbox dataset (VQA-Verify).

**Result.** Up to +15.7% accuracy over the R1-like baseline across seven VQA benchmarks.

**Takeaway.** For multimodal or tool-using small models, breaking a long trajectory into stages with their own dense verifiable signals beats one terminal reward over a long chain.

**ELI5.** Instead of grading a student only on the final answer of a long word problem, grade them on each step (read the picture, point at the relevant area, then answer); they stop daydreaming.

---

## 8. Extensions to Open-Ended, Multimodal, and Sparse-Data Tasks

### 8.1 Mathematics and General Reasoning

- [DeepSeek-R1 (2501.12948)](https://arxiv.org/abs/2501.12948) — Foundational; accuracy reward via deterministic math/code checkers, format reward via LLM judge.
- [DeepSeekMath-V2 (2511.22570)](https://arxiv.org/abs/2511.22570) — Self-verifiable mathematical reasoning with second-LLM explanation scoring.
- [One Training Example (2504.20571)](https://arxiv.org/abs/2504.20571) — Near-doubling of MATH500 with a single curated example.

### 8.2 Medical

- [Med-RLVR (2502.19655)](https://arxiv.org/abs/2502.19655) — 3B base model, MCQ verifiable labels, +8pp OOD gain; first demonstration of RLVR domain transfer.
- [Open-Medical-R1 (2504.13950)](https://arxiv.org/abs/2504.13950) — Difficulty-based sample filtering for robustness.
- [EHR-Based Reasoning (2505.24105)](https://arxiv.org/abs/2505.24105) — Application to electronic health records.

### 8.3 Multimodal and Embodied

- [R1-Omni (2503.05379)](https://arxiv.org/abs/2503.05379) — Multimodal emotion recognition.
- [SATORI-R1 (2505.19094)](https://arxiv.org/abs/2505.19094) — Spatial grounding with verifiable subtask rewards.
- [MoDoMoDo (2505.24871)](https://arxiv.org/abs/2505.24871) — Multi-domain data mixtures for multimodal RL.
- [ManipLVM-R1 (2505.16517)](https://arxiv.org/abs/2505.16517) — Embodied robotic manipulation.
- [Few-Shot Vision-Language for Satellite (2507.21745)](https://arxiv.org/abs/2507.21745) — Lightweight rule-based verification for vision-LLM adaptation.

### 8.4 World Models and Forecasting

- [RLVR-World (2505.13934)](https://arxiv.org/abs/2505.13934) — World models trained with F1, LPIPS metrics as verifiable rewards.
- [Outcome-based Forecasting (2505.17989)](https://arxiv.org/abs/2505.17989) — Binary/noisy reward adaptations for prediction tasks.

### 8.5 Open-Ended and Creative

- [Writing-Zero (2506.00103)](https://arxiv.org/abs/2506.00103) — BRPO over self-principled critique pairs.
- [Auditable-choice Reframing (2511.02463)](https://arxiv.org/abs/2511.02463) — MCQ reformulation for verifiability.
- [Rubrics as Rewards (2507.17746)](https://arxiv.org/abs/2507.17746) — Multi-criterion rubrics as explicit reward functions.

### 8.6 Verifier-Free Extensions

- [RLPR (2506.18254)](https://arxiv.org/abs/2506.18254) — Token probabilities as implicit correctness signals; scalable to general domains.
- [Learning to Reason without External Rewards (2505.19590)](https://arxiv.org/abs/2505.19590) — Self-certainty (KL from uniform) as intrinsic reward; mode-seeking, length-bias-resistant.

### 8.7 Specialized

- [Korean Word-Chain Game (2510.03394)](https://arxiv.org/abs/2510.03394) — Curriculum RLVR for phonological rule enforcement.

### Paper Cards

### 2501.12948 — DeepSeek-R1

**Summary.** DeepSeek-R1 shows that pure RL with verifiable rewards (no SFT cold-start in R1-Zero) can incentivise emergent self-reflection, verification, and strategy switching in LLMs. The contribution is the first large-scale demonstration that RLVR alone, on math/code/STEM tasks, surpasses SFT-on-CoT and that the resulting reasoning patterns can be distilled into smaller models.

**Problem.** Strong reasoning previously required heavy human-annotated CoT supervision, which is expensive and caps capability at the annotators' level.

**Method.** GRPO-style group-relative policy optimisation with rule-based verifiable rewards (exact-match for math, unit-test pass for code) on a base LLM, with no human reasoning traces; reasoning behaviours like reflection emerge during training and are then distilled into smaller dense models.

**Result.** Achieves frontier-level reasoning on math/coding/STEM benchmarks via pure RL; published in Nature 2025 (no single scalar in abstract).

**Takeaway.** This is the canonical reference for "RLVR + GRPO works on small base models too"; the distillation finding directly justifies a 1-3B thesis target as a tractable but credible setting.

**ELI5.** Like teaching a student maths by only ever marking the final answer right or wrong; eventually they invent their own habit of checking their work, and you can pass that habit on to younger students.

### 2511.22570 — DeepSeekMath-V2: Self-Verifiable Math Reasoning

**Summary.** DeepSeekMath-V2 trains an LLM verifier for theorem proving and uses it as the reward model for a proof generator, with the generator incentivised to fix its own proof issues before submitting. Verification compute scales to keep pace as the generator improves.

**Problem.** Final-answer rewards do not check whether reasoning is rigorous, and many tasks (theorems, open problems) lack a checkable numeric answer.

**Method.** Two-loop self-verifiable training: train a faithful proof verifier, use it as the RL reward for the generator, scale verifier compute to label new hard-to-verify proofs, and update the verifier to maintain the generation/verification gap.

**Result.** Gold-medal scores on IMO 2025 and CMO 2024 and 118/120 on Putnam 2024 with scaled test-time compute.

**Takeaway.** When pure outcome rewards saturate, training a verifier-as-reward is the next ladder; even at small scale, a coupled verifier loop may extend RLVR beyond exact-match domains.

**ELI5.** A student writes proofs while a co-trained TA grades each line; both keep getting smarter, like a debater and their critic sparring nightly.

### 2504.20571 — 1-shot RLVR

**Summary.** Demonstrates that RLVR with literally one training example can dramatically boost LLM math reasoning. Shows the gain comes from the policy-gradient loss itself, not from "grokking", and relies on entropy-based exploration.

**Problem.** Conventional wisdom holds RL needs thousands of diverse prompts; nobody had probed the floor.

**Method.** Run GRPO/PPO on Qwen2.5-Math-1.5B with a single carefully-selected math problem, with entropy-loss coefficient tuned for exploration; analyze post-saturation generalization and self-reflection emergence.

**Result.** One example raises MATH500 from 36.0% to 73.6%; matches the 1.2k DeepScaleR subset.

**Takeaway.** Before scaling your RLVR dataset, validate the recipe with 1-2 exemplary prompts; if you do not see post-saturation generalization, the problem is your loss/exploration, not your data.

**ELI5.** Like learning to play guitar from one well-chosen song; the right example unlocks general technique, but only if you keep an exploratory ear open.

### 2502.19655 — Med-RLVR

**Summary.** Med-RLVR applies pure RLVR (no SFT) to a 3B base model on medical multiple-choice QA, using answer-correctness as the verifiable reward. The contribution is the first evidence that RLVR's emergent-reasoning effect generalises beyond math/code into a knowledge-intensive medical domain.

**Problem.** RLVR's success was largely confined to math and code; whether it transfers to knowledge-intensive domains was unknown.

**Method.** Take a 3B base model, run RLVR with MCQA correctness as the binary reward signal, and analyse training dynamics for emergent reasoning behaviours.

**Result.** Matches SFT in-distribution and beats it by +8 accuracy points out-of-distribution on medical QA benchmarks.

**Takeaway.** A 3B base + pure RLVR can produce emergent reasoning in domains other than math/code, validating small-model RLVR for thesis-scale projects with verifiable answers (e.g., open-domain QA with EM).

**ELI5.** Like giving a junior medic only a pass/fail buzzer at the end of each case; with enough cases they teach themselves to think through differential diagnoses, and they generalise better than peers who just memorised textbook answers.

### 2504.13950 — Open-Medical-R1

**Summary.** Studies four data-selection strategies for RLVR training on MedQA-USMLE using Gemma-3-12b-it as base. Compares random vs filtering by Phi-4, Gemma-3-27b-it, and self-filtering by Gemma-3-12b-it.

**Problem.** RLVR data selection is well-studied for math but unexplored for specialized medical domains.

**Method.** GRPO on Gemma-3-12b-it with four prompt-filtering strategies, evaluated on MMLU, GSM8K, MMLU-Pro, CMMLU.

**Result.** Filtered-data training beats random; self-filtering wins in-domain (medical) but loses robustness; larger same-family filter gives best overall robustness (no single headline number).

**Takeaway.** When picking RLVR data for a niche domain on a small model, filter with a *larger* model from the same family for cross-benchmark robustness rather than self-filtering.

**ELI5.** Like having a senior doctor pre-select practice cases for a resident; the senior's judgment beats the resident grading themselves and beats picking cases at random.

### 2505.24105 — EHRMIND: RLVR for Clinical Reasoning

**Summary.** EHRMIND adapts RLVR to electronic health record reasoning tasks via a two-stage SFT-then-RLVR recipe, addressing healthcare-specific failure modes. It is evaluated on medical calculations, trial matching, and diagnosis.

**Problem.** Off-the-shelf RLVR struggles in healthcare because models suffer from misapplied or missing domain knowledge that pure outcome rewards cannot fix.

**Method.** A lightweight SFT warm-up injects missing domain knowledge and stabilizes outputs, followed by RLVR that refines decision-making against verifiable correctness.

**Result.** Consistent gains in accuracy and cross-task generalization on MEDCALC, TREC Clinical Trials, and EHRSHOT (no headline number in abstract).

**Takeaway.** When the base model lacks the prerequisite knowledge for your domain, RLVR alone fails; a small SFT warm-up unlocks it, a useful pattern for any specialized 1-3B agent.

**ELI5.** You cannot RL-tune a chef who has never seen an oven; first show them ten recipes, then let them iterate via taste-test.

### 2503.05379 — R1-Omni

**Summary.** R1-Omni applies RLVR to an omni-modal (vision + audio + text) LLM for emotion recognition, optimising both classification accuracy and explicit reasoning chains. The contribution is the first RLVR application to multimodal emotion recognition, with measurable gains in accuracy, OOD robustness, and modality-attribution interpretability.

**Problem.** Multimodal emotion recognition models are accuracy-focused and opaque; standard SFT does not yield reasoning over which modality drives a prediction.

**Method.** Apply RLVR (DeepSeek-R1-style verifiable reward) to an Omni multimodal LLM, with verifiable labels on emotion classes plus structured reasoning outputs that expose visual vs audio contributions.

**Result.** Improves in-distribution accuracy and OOD robustness over SFT baselines and yields interpretable per-modality contributions (no scalar in abstract).

**Takeaway.** RLVR generalises beyond text-only reasoning into multimodal classification with explainability as a free side-effect; useful citation if you argue RLVR's reasoning emergence is domain-general.

**ELI5.** Like training a band judge to not only say "happy or sad" but also to point at whether the singer's face or voice is doing the heavy lifting, and they get better at both jobs at once.

### 2505.24871 — MoDoMoDo: Multi-Domain Mixtures for MLLM RLVR

**Summary.** MoDoMoDo is a multimodal RLVR framework that learns the optimal mixture proportions across heterogeneous vision-language datasets. It treats data mixing as a predictable optimization problem.

**Problem.** Combining diverse verifiable VL tasks under one RL run causes objective conflicts, but uniform mixing leaves performance on the table.

**Method.** Curate verifiable multimodal tasks, train multi-domain online RL, and learn a model that predicts post-RL outcomes from mixture distributions to pick the best mix.

**Result.** +5.24% average OOD accuracy over uniform mixing and +20.74% over the pre-finetuning baseline.

**Takeaway.** For multi-task RLVR (e.g., search + math + instruction), mixture weights matter as much as the algorithm; consider learning them rather than guessing.

**ELI5.** Instead of feeding a horse equal hay, oats, and apples, you A/B test diets and let a small model predict which mix wins the race.

### 2505.16517 — ManipLVM-R1

**Summary.** Applies RLVR to large vision-language models for embodied robot manipulation, replacing costly human annotation with rule-based verifiable rewards. Targets out-of-domain generalization in physical interaction tasks.

**Problem.** Manipulation LVLMs depend on expensive annotated trajectories and break down on OOD scenarios.

**Method.** Two rule-based rewards: an Affordance Perception Reward (correct localization of interaction regions) and a Trajectory Match Reward (physically plausible action paths), used in an RLVR loop on an LVLM.

**Result.** No headline number stated in abstract; emphasis is on improved OOD generalization.

**Takeaway.** Designing two complementary spatial-logical rule rewards (where to act + how to move) is enough to bootstrap RLVR for embodied tasks without manual labels.

**ELI5.** Like teaching a robot via two checklists ("did you grab the right spot?" and "did your arm take a sensible path?") instead of having a human grade every video frame; cheap but informative.

### 2507.21745 — Few-Shot RLVR for Satellite Imagery

**Summary.** First application of few-shot RLVR to vision-language models on remote-sensing tasks, using only rule-based binary or IoU rewards. Shows even one labeled example can meaningfully steer a VLM.

**Problem.** Annotated satellite-imagery data is scarce and expensive, blocking RLVR-style post-training in specialized vision domains.

**Method.** Adapt the "1-shot RLVR" recipe from text models to VLMs: policy gradients on as few as 1-128 curated examples with IoU/binary correctness rewards, no caption supervision.

**Result.** 128 examples match or exceed models trained on thousands of annotated samples across classification, VQA, and grounding.

**Takeaway.** You can extend a small-VLM RLVR pipeline to a niche domain with a handful of programmatically checkable examples; data scarcity is no longer a blocker.

**ELI5.** Like teaching a junior analyst a new map style by handing them one well-marked example and a checking key, instead of a thousand annotated atlases.

### 2505.13934 — RLVR-World

**Summary.** Generalizes RLVR beyond reasoning to training world models (state-transition predictors) across language, web, video, and robotics domains. Treats decoded predictions' task-metric (accuracy, perceptual quality) as the verifiable reward.

**Problem.** World models trained with MLE are mis-aligned with the metrics that actually matter (transition accuracy, perceptual fidelity), since likelihood and quality diverge.

**Method.** Standard autoregressive token prediction for world states, but the post-training loss is an RLVR loss where reward = the task-specific metric on the decoded output.

**Result.** Substantial gains on text games, web navigation, and robot manipulation world models (no single headline number).

**Takeaway.** RLVR is a general post-training paradigm, not a reasoning-only trick; any time your eval metric differs from cross-entropy, RLVR can directly close the gap.

**ELI5.** Like training a weather forecaster on "did it actually rain?" rather than on "did your sentence look like the meteorologist's"; the forecast itself becomes the thing you optimize.

### 2505.17989 — Outcome-based RL for Forecasting

**Summary.** Applies RLVR to the noisy, delayed-reward task of forecasting real-world events, training a 14B reasoning model on prediction-market questions with associated news. Achieves frontier-level forecasting from a compact model.

**Problem.** Real-world forecasting has noisy, delayed binary outcomes (the event resolves months later), making RL credit assignment notoriously hard.

**Method.** RLVR on a curated dataset of recent prediction-market questions + news headlines; uses synthetic-question augmentation, learning-stability guardrails, and median prediction sampling at inference.

**Result.** Matches/surpasses o1's accuracy and calibration; simulated Polymarket trading yields >10% ROI on the test set.

**Takeaway.** RLVR can work even with noisy, sparse, delayed rewards if you augment with synthetic questions and add stability guardrails; do not assume your reward needs to be clean for RLVR to help.

**ELI5.** Like training a sports gambler who only finds out the score weeks later; with enough past games and a steady hand on bet size, they still learn to pick winners.

### 2506.18254 — RLPR: Verifier-Free RLVR via Token Probabilities

**Summary.** RLPR replaces explicit verifiers with the LLM's own token-probability score on a reference answer, making RLVR work in general (non-math, non-code) domains. Adds variance-reduction tricks to stabilize the noisy probability reward.

**Problem.** RLVR is gated on having a domain-specific verifier, which does not exist for most natural-language tasks.

**Method.** Use the policy's probability of generating the reference answer as the reward, with prob-to-reward and stabilizing transforms to control high variance.

**Result.** +7.6 on TheoremQA and +7.5 on Minerva over concurrent VeriFree; +1.6 average across seven benchmarks over General-Reasoner (which uses a verifier model).

**Takeaway.** A practical bridge from RLVR to general-domain QA without building verifiers; competitive with verifier-based methods if you tame the variance.

**ELI5.** Instead of asking a judge "is this answer right?", measure how shocked the model would be at the reference answer; less shock means it implicitly agrees.

### 2505.19590 — Intuitor: RL from Internal Feedback

**Summary.** Intuitor replaces the verifiable reward in GRPO with the model's own self-certainty score, enabling RL fine-tuning with no labels, no gold answers, and no test cases. The framework is called RLIF (Reinforcement Learning from Internal Feedback).

**Problem.** RLVR is bottlenecked by the cost and domain-specificity of producing verifiable rewards, which limits its reach to math and code.

**Method.** Compute per-rollout self-certainty (a function of the model's own token probabilities) and use it as the GRPO reward, removing any external supervision.

**Result.** Matches GRPO on math benchmarks while generalizing better to out-of-domain code generation, with no labeled data.

**Takeaway.** A surprisingly strong baseline: if you cannot build a verifier for your domain, the policy's own confidence may carry you most of the way.

**ELI5.** Instead of a teacher correcting the student's homework, the student learns by trusting answers they wrote with conviction; for many subjects, that is almost as good.

### 2510.03394 — Korean Word-Chain Game with RLVR

**Summary.** Studies the Korean word-chain game (kkeunmal-itgi) under RLVR and finds rule-derived rewards naturally conflict (e.g., validity vs novelty). Shows curriculum learning resolves these conflicts.

**Problem.** Even seemingly simple rule-based RLVR rewards can conflict internally, blocking learning on language-puzzle tasks.

**Method.** Define multiple rule-based reward components for the Korean word-chain task, identify their pairwise conflicts experimentally, then mitigate by ordering rewards via a curriculum (easier rule subsets first).

**Result.** Curriculum learning empirically mitigates the reward conflicts (no specific numbers in abstract).

**Takeaway.** For a small-LM RLVR run with multiple rule rewards, check pairwise reward conflicts before tuning weights; a simple curriculum that introduces rewards in stages can avoid intractable optimization landscapes.

**ELI5.** Like teaching a kid Scrabble: don't enforce "must be a real word AND must be high-scoring AND must use a triple-letter" all at once; introduce each rule one round at a time.

---

## 9. Tool-Use and Search Integration

The most direct bridge from RLVR to non-verifiable domains: train the model to *invoke tools* whose outputs are verifiable, even when the model's free-form reasoning is not.

### 9.1 Search-Augmented Reasoning

- [ReSearch (2503.19470)](https://arxiv.org/abs/2503.19470) — GRPO with rule-based answer reward (F1 / exact match) + format reward; uses instruction-tuned model with `<search>...</search>` tags.
- [R1-Searcher (2503.05592)](https://arxiv.org/abs/2503.05592) — Two-stage: Stage-1 retrieve-rewards teach search-tool format; Stage-2 answer-rewards train end-to-end QA.
- [Search-R1 (2503.09516)](https://arxiv.org/abs/2503.09516) — Multi-turn search queries, real-time retrieval, retrieved-token masking for stable optimization.
- [Dr. Zero (2601.07055)](https://arxiv.org/abs/2601.07055) — Self-evolving search agents without supervised training data.

### 9.2 Tool-Use as Reward Surrogate

- [ToolRL (2504.13958)](https://arxiv.org/abs/2504.13958) — Reward-only signal sufficient for tool learning; no SFT.
- [Replacing Thinking with Tool Usage (2507.05065)](https://arxiv.org/abs/2507.05065) — Substitute internal reasoning with tool calls in small LMs.
- [Advancing SLM Tool-Use via RL (2509.04518)](https://arxiv.org/abs/2509.04518) — Small-model adaptations.
- [Acting Less is Reasoning More (2504.14870)](https://arxiv.org/abs/2504.14870) — Efficiency reward to teach when *not* to call a tool.
- [ToolOrchestra (2511.21689)](https://arxiv.org/abs/2511.21689) — Efficient model and tool orchestration framework.
- [Training LMs to Use Prolog as a Tool (2512.07407)](https://arxiv.org/abs/2512.07407) — Symbolic reasoning via Prolog tool calls.
- [ToolExpander (2510.07737)](https://arxiv.org/abs/2510.07737) — Extending tool-use frontier to weak LLMs.

### 9.3 Tool-Augmented Verification

- [CoSineVerifier (2512.01224)](https://arxiv.org/abs/2512.01224) — Tool-augmented answer verification for computation-oriented scientific questions.

### 9.4 Multi-Turn Tool RL

The dominant 2026 frontier — handling stateful, error-prone, sequential tool use:

- [RC-GRPO (2602.03025)](https://arxiv.org/abs/2602.03025) — Reward-conditioned rollouts for multi-turn tool calling; restores diversity in sparse-reward scenarios.
- [Fission-GRPO (2601.15625)](https://arxiv.org/abs/2601.15625) — Converts execution failures into corrective supervision.
- [Group Turn Policy Optimization (2511.14846)](https://arxiv.org/abs/2511.14846) — Per-turn grouping for multi-turn tool reasoning.
- [On GRPO Collapse in Agent Search (2512.04220)](https://arxiv.org/abs/2512.04220) — Lazy likelihood-displacement in multi-turn agentic search.
- [MC-GRPO (2601.22582)](https://arxiv.org/abs/2601.22582) — Median-centered baseline for small-rollout regimes.
- [F-GRPO (2602.06717)](https://arxiv.org/abs/2602.06717) — Difficulty-aware advantage scaling.
- [Prefix Grouper (2506.05433)](https://arxiv.org/abs/2506.05433) — Computational efficiency via shared-prefix grouping.

### 9.5 Software Engineering Agents

- [Agent-RLVR (2506.11425)](https://arxiv.org/abs/2506.11425) — Pedagogically-inspired guidance with environment rewards for SWE; reports ×2.4 pass@1 improvement on sparse-reward agentic tasks.
- [Agent-R1 (2511.14460)](https://arxiv.org/abs/2511.14460) — End-to-end agentic RL.
- [SWE-Universe (2602.02361)](https://arxiv.org/abs/2602.02361) — Autonomous MoE building agent constructs million-scale verifiable SWE environments from GitHub PRs.

### 9.6 Meta-Reasoning Rewards

- [RLVMR (2507.22844)](https://arxiv.org/abs/2507.22844) — Reward signals on meta-reasoning steps for long-horizon agents.
- [DeepSearch (2509.25454)](https://arxiv.org/abs/2509.25454) — MCTS integration with RLVR to overcome the rollout bottleneck.

### Paper Cards

### 2503.19470 — ReSearch

**Summary.** ReSearch trains LLMs to interleave search calls inside their chain-of-thought via RL with no SFT on reasoning traces. Search queries are generated from text-based thinking, and retrieved results feed back into further reasoning.

**Problem.** Multi-hop QA requires multiple coordinated retrievals, but no public RL recipe shows reasoning-with-search emerging without supervised reasoning data.

**Method.** GRPO over trajectories that natively embed `<search>`/`<result>` segments, with the reasoning chain itself deciding when to query; trained on a single dataset to test transfer.

**Result.** Reflection and self-correction behaviors emerge naturally during training; strong cross-benchmark generalization despite single-dataset training (no headline number in abstract).

**Takeaway.** For a small search-augmented LM you do not need supervised reasoning traces; outcome-only RL with a search tool is sufficient to elicit when-to-search behavior.

**ELI5.** Like dropping a student into a library with only a pass/fail grade at the end; they learn unprompted that flipping to the index mid-thought beats trying to remember everything.

### 2503.05592 — R1-Searcher

**Summary.** R1-Searcher trains an LLM to autonomously call an external search engine during reasoning using a two-stage outcome-based RL recipe with no process rewards and no SFT cold-start. The contribution is the first pure-RL search-augmented agent that beats strong RAG baselines and even GPT-4o-mini.

**Problem.** Reasoning LLMs lean on parametric knowledge and hallucinate on time-sensitive or knowledge-intensive questions; existing RAG bolts retrieval onto inference but does not teach the model when/how to search.

**Method.** Two-stage outcome-based RL: stage one rewards format/tool-use compliance to bootstrap search invocation, stage two rewards final-answer correctness via a verifiable reward; works on both Base and Instruct backbones, no process rewards or distillation needed.

**Result.** Significantly outperforms strong RAG baselines and matches/exceeds closed-source GPT-4o-mini on multi-hop QA (no single headline number in abstract).

**Takeaway.** This is the closest contemporary baseline for any "RLVR + search-tool" small-model thesis: it shows that two-stage outcome rewards (format then correctness) suffice, and that you can skip SFT and process rewards entirely.

**ELI5.** Like teaching a student to use Wikipedia by only marking their final exam, with a bonus on the first few quizzes for actually opening the book; soon they learn both when to look things up and how to reason from what they find.

### 2503.09516 — Search-R1

**Summary.** Search-R1 trains LLMs end-to-end with RL to interleave reasoning with autonomous search-engine calls during multi-turn rollouts. It uses retrieved-token masking and a simple outcome reward to stabilize training over real-time retrieval.

**Problem.** Prompted RAG underuses the LLM's potential because the model never learns *when* and *how* to query a search engine optimally.

**Method.** PPO/GRPO over multi-turn trajectories where `<search>` tokens trigger real retrieval; retrieved tokens are masked from the loss so the policy is only credited for its own generations, with reward = exact-match on the final answer.

**Result.** +41% relative gain on Qwen2.5-7B and +20% on Qwen2.5-3B over RAG baselines across seven QA datasets.

**Takeaway.** For a 1-3B search-augmented model, retrieved-token masking and pure-EM outcome reward are the two non-negotiable ingredients to keep multi-turn RL stable.

**ELI5.** Like teaching a research assistant by grading only the final report (not the Google searches they ran), so they learn which queries actually help vs which were wasted clicks.

### 2601.07055 — Dr. Zero: Self-Evolving Search Agents

**Summary.** Dr. Zero is a data-free self-evolution framework where a proposer LLM generates increasingly hard search questions for a solver LLM (initialized from the same base) to learn from, removing the need for any training data. The two roles co-evolve through an automated curriculum and a new GRPO variant tailored to multi-hop search.

**Problem.** Multi-turn search agents need annotated multi-hop QA data that is scarce, and naive self-evolution wastes compute on per-query difficulty estimation.

**Method.** A proposer-solver self-play loop generates questions and solves them with verifiable rewards; Hop-grouped Relative Policy Optimization (HRPO) clusters structurally similar questions to share group baselines, cutting the rollout overhead of estimating per-prompt difficulty.

**Result.** Data-free Dr. Zero matches or surpasses fully supervised search agents (no specific number reported in abstract).

**Takeaway.** For small-LM search agents, you can dispense with curated multi-hop datasets entirely if you pair self-play question generation with grouped baselines that amortize rollouts across structurally similar prompts.

**ELI5.** Instead of buying textbooks, two clones of yourself take turns writing pop quizzes for each other; HRPO is grading the quizzes by topic so you don't have to guess the difficulty of every single question.

### 2504.13958 — ToolRL

**Summary.** First systematic study of reward design for tool-use RL, varying type, scale, granularity, and temporal dynamics. Builds a principled tool-use reward and trains via GRPO.

**Problem.** Coarse answer-matching rewards fail for tool use because invocations have many parameters that need finer-grained credit.

**Method.** GRPO with a composite reward decomposing tool-name match, parameter match, and final-answer correctness, with calibrated scales and timing.

**Result.** +17% over base models and +15% over SFT models on tool-use benchmarks.

**Takeaway.** For a search-augmented small model, decompose your reward into (correct tool selection) + (correct query/parameters) + (correct final answer); pure outcome reward leaves accuracy on the table.

**ELI5.** Like grading a cooking exam not just on the final dish but also on whether the student picked the right knife and held it properly; partial credit at each step teaches faster than only tasting the result.

### 2507.05065 — Tools Replace Thinking in Small LMs

**Summary.** Argues that for small LMs (up to 3B), expending inference compute via tool interactions is more effective than emitting natural-language "thoughts". Demonstrates on Python code repair.

**Problem.** SFT+RLVR on natural-language CoT struggles to make small models scale inference-time compute usefully; thoughts are slow to sample and weakly rewarded.

**Method.** Replace CoT tokens with a multi-turn DSL trace controlling a stateful tool; the tool's new state is appended to context each turn, providing dense per-turn signal.

**Result.** Even up-to-3B models learn to productively spend more inference compute on the malfunctioning-Python-repair task (no scalar headline).

**Takeaway.** For your 1-3B search-augmented thesis, this validates the bet: tool calls (search) are a better inference-compute medium than long internal CoT; expect denser rewards and cheaper rollouts.

**ELI5.** A small model trying to "think" out loud is like a toddler reasoning aloud about a math problem; give them an abacus and they get further than whispering equations.

### 2509.04518 — Advancing SLM Tool-Use with GRPO

**Summary.** Applies GRPO to small language models on function-calling/JSON tool-use tasks using a structured reward over format, tool selection, and parameter correctness. The contribution is a recipe and reward design tailored for SLM tool use.

**Problem.** SLMs lag LLMs at structured tool use (JSON output, correct tool, correct args), limiting their deployment as agents.

**Method.** GRPO with a composite rule-based reward: well-formed JSON, correct tool selection, and correct parameter values, no human preference data.

**Result.** No specific numbers in the abstract; reports significant tool-use accuracy gains on SLMs.

**Takeaway.** For a 1-3B agent, decompose your tool-use reward into format / tool / args sub-rewards rather than one monolithic correctness signal; SLMs need the denser structure.

**ELI5.** Like grading a waiter on three things separately (took the order in writing, picked the right dish, got the side correct) instead of just "did the customer leave happy."

### 2504.14870 — OTC-PO: Acting Less is Reasoning More

**Summary.** Adds an efficiency term to tool-integrated reasoning RL so models produce correct answers with minimal tool calls. Targets the "cognitive offloading" pathology where RL-trained agents over-call tools because nothing penalizes excess.

**Problem.** Outcome-only RL drives agents to call search/code-interp every chance they get, inflating cost and atrophying internal reasoning.

**Method.** Optimal Tool Call-controlled Policy Optimization (OTC-PPO and OTC-GRPO) adds a tool-integrated reward jointly weighing correctness and number of tool calls; introduces "tool productivity" (correct/total-calls) as the diagnostic metric.

**Result.** Up to 68.3% fewer tool calls and up to 215.4% higher tool productivity at comparable accuracy on Qwen-2.5 / Qwen-Math QA benchmarks.

**Takeaway.** Add a small negative reward per tool call (or a soft cap) to your search-RL recipe; you cut search-engine API spend by 2-3x with little accuracy cost.

**ELI5.** Like paying an intern per question they answer but charging them for each Google search; they quickly learn to think first and only google when needed.

### 2511.21689 — ToolOrchestra: 8B Orchestrator over Tools and Models

**Summary.** ToolOrchestra trains a small (8B) orchestrator that routes problems across many tools and bigger models using RL with outcome, efficiency, and user-preference rewards. The resulting Orchestrator beats GPT-5 on Humanity's Last Exam at a fraction of the cost.

**Problem.** Solving frontier agentic benchmarks like HLE with a single huge model is conceptually and computationally expensive; existing tool-use agents do not balance accuracy against cost.

**Method.** RL-train a small orchestrator policy that selects among intelligent tools and larger model calls; reward combines task outcome, compute efficiency, and alignment with user tool preferences.

**Result.** 37.1% on Humanity's Last Exam vs. GPT-5's 35.1%, at 2.5x lower cost; on tau2-Bench and FRAMES, beats GPT-5 at ~30% of cost.

**Takeaway.** Strong evidence that the right place for a small RLVR-trained model is as a router, not a soloist; relevant if your thesis discusses cost-aware tool selection.

**ELI5.** A clever receptionist who knows exactly which specialist to dial for each visitor outperforms one overworked super-doctor trying to do every consultation alone.

### 2512.07407 — Training LMs to Use Prolog as a Tool

**Summary.** Fine-tunes Qwen2.5-3B-Instruct with GRPO to call Prolog as an external symbolic tool on GSM8K, sweeping prompt formats, reward shapes, and inference protocols. Reveals an accuracy-vs-auditability trade-off where reward hacking lets the model push reasoning back into natural language.

**Problem.** LLMs produce plausible-but-wrong reasoning; symbolic tools could verify steps but it is unclear how to train a 3B model to use them faithfully.

**Method.** GRPO on a cleaned GSM8K (released as gsm8k-prolog-prover); ablate prompt structure, reward composition (execution, syntax, semantics, structure), and inference protocols (single-try, multiple-try, agentic); analyse what behaviours each reward shape elicits.

**Result.** 3B model reaches zero-shot performance competitive with 7B few-shot baselines on MMLU-STEM and MMLU-Pro; reward-hacking trade-off identified between correctness and auditable Prolog use.

**Takeaway.** Direct evidence that on a 3B model, RLVR with symbolic-tool calls can match much larger few-shot baselines, but reward design must explicitly demand symbolic structure or the model will cheat.

**ELI5.** A student told only "get the right answer" learns to scribble in the margin and only use the calculator at the last step; if you also grade them on showing calculator work, they actually use it.

### 2510.07737 — ToolExpander: Tool-Use RL for Weak LLMs

**Summary.** ToolExpander adapts GRPO for small models on tool-use tasks where standard training collapses because rollouts rarely produce a correct answer. It adds dynamic hard-sample substitution and a self-exemplifying thinking objective to keep gradient signal alive in resource-constrained settings.

**Problem.** GRPO frequently mid-training-collapses on small models because zero-reward groups give no learning signal, especially in tool-using regimes.

**Method.** Two changes: (1) Dynamic Multi-Round Hard Sampling swaps in high-quality few-shot demos for prompts that fail all 10 rollouts, paired with exponential LR decay; (2) Self-Exemplifying Thinking removes KL, adjusts clip coefficients, and adds a tiny (0.01) reward for autonomously generating and analysing few-shot examples.

**Result.** Reports significant gains in tool-use capability and training stability on small-scale models (no single headline number; the +0.01 reward is the key tuning constant).

**Takeaway.** For 1-3B GRPO with sparse-reward tools, do not just tune clipping; recycle failing prompts with demonstrations and let the model self-exemplify to keep group variance non-zero.

**ELI5.** Like a coach who, when a beginner keeps missing every shot, briefly hands them a video of a pro making it; they imitate, score once, and the lesson loop stays alive instead of dying from zero feedback.

### 2512.01224 — CoSineVerifier: Tool-Augmented Answer Verification

**Summary.** CoSineVerifier is a verifier that calls external symbolic tools (CAS, executors) to handle algebraic equivalence and physical-constant substitution, replacing brittle string matching. Trained via cold-start SFT then multi-turn tool-integrated RL, it serves as a reward model for RLVR with consistent gains on AIME.

**Problem.** Standard RLVR verifiers cannot tell if "(x+1)^2" equals "x^2+2x+1" or recognise unit substitutions, especially in STEM, leaving training rewards noisy.

**Method.** Two-stage pipeline: cold-start SFT teaches the model to invoke CAS-style tools, then multi-turn RL with tool integration tunes verification behaviour; the verifier is then deployed as the reward source for downstream RLVR runs.

**Result.** SOTA on VerifyBench-Hard and SCI-Bench and consistently beats rubric- and model-based verifiers on AIME'24/'25 when used as the RLVR reward.

**Takeaway.** If your RLVR reward is exact-match string compare, you are leaving accuracy on the table; a tool-augmented verifier is now a viable, reusable component.

**ELI5.** Instead of a teacher who only marks answers right when the writing matches the answer key letter-for-letter, give them a calculator and a textbook so they recognise the same answer in disguise.

### 2506.11425 — Agent-RLVR: SWE Agents with Guidance

**Summary.** Agent-RLVR makes RLVR tractable in sparse-reward agentic settings (software engineering) by injecting teacher-style guidance into failed trajectories. Trains the policy to recover with hints, then to solve without them.

**Problem.** RLVR collapses in agentic environments because most rollouts fail and provide no learning signal.

**Method.** When initial trajectories fail unit tests, augment with guidance (plans, error feedback, environment hints), let the agent retry, then update the policy on the guided rewards.

**Result.** Lifts Qwen-2.5-72B-Instruct from 9.4% to 22.4% pass@1 on SWE-Bench Verified, and to 27.8% with a guidance-trained reward model.

**Takeaway.** For sparse-reward search/agent tasks, mid-rollout teacher hints turn dead trajectories into useful gradient; analogous to your search-rollout failure modes.

**ELI5.** A driving instructor who only watches silently sees a lot of crashes; one who shouts "brake!" mid-skid creates teachable moments out of every near-miss.

### 2511.14460 — Agent-R1: End-to-End RL Framework for LLM Agents

**Summary.** Agent-R1 is a modular training framework that recasts LLM-agent tool-use as an extended MDP and trains agents end-to-end with RL. It is positioned as a flexible substrate for the nascent area of agentic RL, validated on multi-hop QA.

**Problem.** RL for LLM agents lacks both a clean MDP formalism and a flexible, extensible training framework; existing code is brittle and task-specific.

**Method.** Extend the standard MDP to define agent state, action, environment, and reward components for tool-using LLMs; provide a modular open-source framework supporting plug-in tools, environments, and RL algorithms; demonstrate on multi-hop QA.

**Result.** Initial validation on multi-hop QA benchmarks (no headline numerical result in abstract).

**Takeaway.** A potential reference codebase if you are building an agentic-RL stack from scratch; the MDP formalisation is useful for clarifying state/action boundaries in tool-use papers.

**ELI5.** Less a new recipe and more a well-equipped kitchen: drawers labelled state, action, tool, reward, so you can swap ingredients without rewiring the stove.

### 2602.02361 — SWE-Universe: Million-Scale SWE Environments

**Summary.** SWE-Universe automatically constructs verifiable software-engineering environments from GitHub PRs at million scale, using a custom-trained building agent with iterative self-verification. The result is 807,693 multilingual SWE tasks usable for agentic mid-training and RL.

**Problem.** Real-world SWE RL environments are expensive to build, with low yield, weak verifiers, and easy reward hacking.

**Method.** A specialized agent (small custom model) iteratively builds, tests, and self-verifies PR-derived environments while running in-loop hacking detection to discard tasks with exploitable verifiers.

**Result.** 75.3% on SWE-Bench Verified after applying the data to Qwen3-Max-Thinking; ~800k usable environments produced.

**Takeaway.** Even for small models, the bottleneck for agentic RL is verifiable environment supply; lightweight builder-agents with self-verification can scale environment generation by orders of magnitude.

**ELI5.** Instead of hiring contractors to set up training labs one at a time, you train a junior engineer to spin up labs by the million — and to throw out any lab whose tests can be cheated.

### 2507.22844 — RLVMR: Verifiable Meta-Reasoning Rewards

**Summary.** RLVMR augments outcome-only agent RL with dense, programmatic rewards for meta-reasoning tags (plan / explore / reflect). It targets long-horizon agent tasks where outcome-only RL reinforces flawed reasoning paths.

**Problem.** Outcome-only RL on long-horizon agents reinforces brittle "lucky" trajectories and fails to teach coherent reasoning, hurting generalization.

**Method.** The agent emits explicit cognitive tags (`<plan>`, `<explore>`, `<reflect>`); rule-based verifiers reward well-formed and effective use of these tags, combined with the final outcome reward via critic-free policy gradient.

**Result.** 7B model reaches 83.6% success on the hardest unseen split of ALFWorld and ScienceWorld, a new SOTA.

**Takeaway.** For a small-LM agent on long-horizon tool tasks, mark up the trace with structural tags and reward them programmatically; you get most of the benefit of process supervision without a learned PRM.

**ELI5.** Like grading a math student not just on the final answer but for legibly writing "Plan:" and "Check:" headings at the right spots, judged by a simple rubric stamp.

### 2509.25454 — DeepSearch: MCTS Inside RLVR Training

**Summary.** DeepSearch embeds Monte Carlo Tree Search into the RLVR training loop (not just inference) to overcome RLVR's training plateau. It uses global frontier selection, entropy-guided supervision, and a replay buffer with solution caching.

**Problem.** RLVR's sparse rollout exploration causes training plateaus after thousands of steps; more compute stops yielding gains.

**Method.** Run MCTS during training to systematically expand reasoning trees; (1) select promising frontier nodes globally, (2) pick paths via entropy-based confidence for supervision, (3) cache solutions in an adaptive replay buffer.

**Result.** 62.95% average accuracy on math reasoning benchmarks with 5.7x fewer GPU hours than extended-training baselines.

**Takeaway.** When your small-LM RLVR run plateaus, the fix is structural exploration (training-time tree search) rather than longer training; even a lightweight MCTS frontier helps.

**ELI5.** Like a chess engine that, during practice games, doesn't just play out one line but expands the whole opening tree and learns from the sharpest branch.

---

## 10. Mechanism Studies: What Does RLVR Actually Do?

A consistent empirical finding across 2025: **RLVR optimizes the selection and frequency of pre-existing reasoning patterns; it does not invent new ones.**

### 10.1 Pattern Selection vs. Pattern Discovery

- [Reasoning Pattern Selection (2506.04695)](https://arxiv.org/abs/2506.04695) — Formalizes the claim: RLVR shifts probability mass over patterns already in the base distribution. Distillation can introduce novel patterns; RLVR cannot.
- [Does RL Really Incentivize Reasoning Beyond the Base Model? (2504.13837)](https://arxiv.org/abs/2504.13837) — RLVR boosts pass@1 but base models cover more diverse solutions at high $k$ in pass@$k$.
- [The Invisible Leash (2507.14843)](https://arxiv.org/abs/2507.14843) — RLVR preserves base model support, reducing entropy but limiting novel discovery; argues for explicit exploration strategies.

### 10.2 Implicit Reasoning Incentive

- [RLVR Implicitly Incentivizes Correct Reasoning (2506.14245)](https://arxiv.org/abs/2506.14245) — Introduces **CoT-Pass@K**: a sample is credited only if both the answer *and* its reasoning chain are correct. Shows RLVR meaningfully improves CoT-Pass@K, not merely Pass@K.

### 10.3 Token-Level Mechanics

- [Beyond 80/20 (2506.01939)](https://arxiv.org/abs/2506.01939) — High-entropy "forking tokens" carry the bulk of the learning signal. ~20% of tokens drive ~80% of the gradient; restricting updates to high-entropy positions matches full RLVR.

### 10.4 Self-Distillation as Learning Mechanism

- [Adaptive Guidance (2506.13923)](https://arxiv.org/abs/2506.13923) — Empirically, self-distillation (the model imitating its own correct rollouts) dominates capability gain; RLVR's reward signal acts as a filter for which trajectories to imitate.

### 10.5 Spurious Reward Phenomenon

- [Spurious Rewards (2506.10947)](https://arxiv.org/abs/2506.10947) — Even uncorrelated reward signals trigger gains in models with strong code-reasoning priors. Implications: pre-training quality is more load-bearing than reward signal quality.

### 10.6 Small-Model Boundary

- [Generalization of RLVR via Causal Reasoning (2512.20760)](https://arxiv.org/abs/2512.20760) — Argues RLVR fails on smaller models; strong reasoning foundations through pre-training or initial SFT may be prerequisite.
- TinyZero counterexample: 3B base model develops self-verification and search abilities through pure RL at <\$30 cost. The resolution likely lies in *strong SFT initialization + targeted verifiers + appropriate domain*, not in RLVR alone.

### Paper Cards

### 2506.04695 — Theory of RL Training Dynamics via Pattern Selection

**Summary.** A combined empirical and theoretical analysis showing that RL on LLMs reshapes reasoning by selecting among existing patterns through a sparse subset of critical tokens. Covers both RLVR and RLIF.

**Problem.** RL training dynamics on LLMs were poorly understood, particularly why some runs converge fast and others stall.

**Method.** Reasoning-pattern-level and token-level empirical analysis combined with a formal model of two reward regimes (verifiable rewards and internal-feedback rewards).

**Result.** Base-model reasoning quality determines RLVR convergence; RLIF can initially help but eventually degrade with continued training (no scalar headline).

**Takeaway.** For a 1-3B base model, expect RLVR convergence to depend strongly on the base's existing reasoning patterns; RLIF (e.g., Intuitor-style) needs early stopping.

**ELI5.** RL does not teach new dance moves; it picks which moves the model already knows to use more. If your dancer never learned the spin, RL will not invent it.

### 2504.13837 — Does RL Actually Incentivize New Reasoning?

**Summary.** Critically probes whether RLVR truly expands a base model's reasoning capacity using pass@k at large k as the metric. Finds that RLVR-trained models win at small k but base models surpass them at large k, meaning RLVR sharpens existing trajectories rather than discovering new ones.

**Problem.** The community assumes RLVR teaches genuinely new reasoning patterns; nobody had stress-tested this against base-model coverage at high sampling budgets.

**Method.** Systematic pass@k evaluation across model families, RL algorithms (six popular ones), and math/code/visual benchmarks; coverage and perplexity analyses to localize where reasoning originates.

**Result.** Base models outperform RLVR-trained models at large k across all six RL algorithms; distillation, by contrast, does introduce new patterns.

**Takeaway.** Treat RLVR on a small model as a sampling-efficiency improver, not a capability extender; if you need new abilities, distill from a stronger teacher rather than RL forever.

**ELI5.** Like teaching a chef to consistently cook their best 3 dishes vs teaching them new recipes; RLVR does the former, distillation does the latter.

### 2507.14843 — The Invisible Leash: Limits of RLVR

**Summary.** An empirical study arguing RLVR mostly sharpens what the base model can already do rather than discovering new reasoning. It frames RLVR as support-constrained optimization with an entropy-reward trade-off.

**Problem.** It is unclear whether RLVR genuinely expands a model's reasoning frontier or just amplifies high-reward modes already in the base distribution.

**Method.** Compare base vs RLVR policies under varying sampling budgets (pass@k), measuring empirical-support shrinkage, token-level entropy, and answer-level entropy across training.

**Result.** RLVR improves pass@1 reliably but the shrinkage of empirical support outweighs its expansion at large k, so RLVR models lose correct answers the base model could find.

**Takeaway.** If you care about pass@k or diverse solutions for a 1-3B model, plain RLVR will hurt you; you need explicit diversity-preserving objectives or seeding mechanisms.

**ELI5.** Like sharpening a pencil so much that the lead becomes a single perfect point but breaks the moment you try to draw outside that one spot.

### 2512.20760 — RLVR Generalisation via Causal Reasoning

**Summary.** Studies when RLVR generalises by training on probabilistic queries over causal graphs, varying both query level (associational, interventional, counterfactual) and graph complexity. Finds RLVR generalises better than SFT only when the base model already has sufficient reasoning competence.

**Problem.** RLVR is widely used but the conditions under which it actually transfers across reasoning levels and complexities are poorly characterised.

**Method.** Build a causal-graph QA dataset spanning Pearl's ladder and graph-size axes; fine-tune Qwen-2.5-Instruct (3B-32B) with RLVR vs SFT; measure within-level and across-level generalisation as a function of model scale and training query level.

**Result.** RLVR beats SFT for specific (model size, query level) combinations and meaningfully improves marginalisation strategy and intermediate probability calculation; benefits emerge only when the base model has sufficient initial competence.

**Takeaway.** Validates a thesis-friendly point: at 1-3B scale, RLVR generalisation is conditional on initial competence; if your base lacks the subskill, RLVR will not invent it; consider SFT cold-start or curriculum.

**ELI5.** RL coaching only sharpens a pianist who already knows scales; if the pianist cannot find middle C, more practice sessions will not teach it from scratch.

---

## 11. Theoretical Foundations: Symbolic, Compositional, and Formal-Methods RL

The theory branch of the verifiable-rewards lineage, predating modern LLM RLVR but sharing its conceptual core.

### 11.1 Symbolic and Example-Based Rewards

- [Learning Intrinsic Symbolic Rewards (2010.03694)](https://arxiv.org/abs/2010.03694) — Interpretable, low-dimensional symbolic trees as verifiable reward functions.
- [Replacing Rewards with Examples (2103.12656)](https://arxiv.org/abs/2103.12656) — Recursive classification from success examples; replaces explicit reward engineering.

### 11.2 Compositional and Modular Systems

- [Verifiable and Compositional RL Systems (2106.05864)](https://arxiv.org/abs/2106.05864) — Decomposes global RL objective into verifiable subsystems with formal interfaces.
- [Verifiable RL via Compositionality (2309.06420)](https://arxiv.org/abs/2309.06420) — Parametric MDPs with iterative adaptation mechanisms.

### 11.3 Formal Methods and Temporal Logic

- [Omega-Regular Reward Machines (2308.07469)](https://arxiv.org/abs/2308.07469) — Unifies reward machines with omega-regular languages for infinite-horizon safety, liveness, and fairness specifications.

### 11.4 Robustness Verification

- [Reward Martingales (2312.09695)](https://arxiv.org/abs/2312.09695) — Neural function certificates with provable bounds on cumulative rewards under perturbations.
- [Set-Based RL (2408.09112)](https://arxiv.org/abs/2408.09112) — Zonotope-based reachability analysis for certifiable safety guarantees post-training.

### 11.5 GRPO Theoretical Properties

- [GRPO's Effective Loss, Dynamics, and Success Amplification (2503.06639)](https://arxiv.org/abs/2503.06639) — Closed-form policy update analysis for binary verifiable rewards.

### Paper Cards

### 2010.03694 — Symbolic Trees for Intrinsic Reward Discovery

**Summary.** The paper learns dense surrogate reward functions represented as small symbolic expression trees over observations, used to supervise a neural policy. The contribution is interpretable, low-dimensional discovered rewards that beat black-box neural reward learners.

**Problem.** Hand-engineered dense rewards need domain expertise, while neural reward networks are opaque and hard to debug.

**Method.** A symbolic regression search produces shallow trees of arithmetic operators that map state to a scalar bonus reward; this is then used as the reward signal for standard policy gradient training.

**Result.** Outperforms a contemporary neural reward-discovery baseline on every Mujoco, Atari, and Pygame environment tested (no single headline number in abstract).

**Takeaway.** When designing process or shaping rewards for RLVR, prefer the simplest interpretable form (a short formula) over a learned reward net; it is auditable and avoids reward-hacking surprises.

**ELI5.** Instead of hiring a black-box critic to grade your dish, the chef writes down a one-line recipe like "salt + 2*umami - bitterness" and grades against that.

### 2103.12656 — Example-Based RL via Recursive Classification

**Summary.** This paper replaces hand-coded reward functions with a small set of success-state examples, deriving an algorithm that maximises the future probability of reaching such states. The contribution is a single-stage classifier-based value learner that skips the usual "learn reward, then optimise it" pipeline.

**Problem.** Writing reward functions by hand is laborious and brittle; two-stage IRL approaches add hyperparameters and instability.

**Method.** A recursive classifier learns a value function directly from off-policy transitions plus success examples by satisfying a new "data-driven Bellman equation" in which examples replace the reward term.

**Result.** Outperforms prior reward-learning baselines across continuous-control benchmarks (no single metric in abstract).

**Takeaway.** For RLVR-style outcome rewards, you can sometimes skip building an explicit reward model and treat success demonstrations as the reward signal directly via a classifier-shaped value function.

**ELI5.** Instead of writing a recipe for "tasty," show the apprentice a tray of finished dishes and let them learn to steer toward those plates one bite at a time.

### 2106.05864 — Verifiable & Compositional RL

**Summary.** The paper proposes a framework where multiple RL subsystems are trained against subtask specifications, then composed under a high-level parametric MDP that gives end-to-end correctness guarantees. The contribution is automatic decomposition of a probabilistic task spec (e.g. reach goal w.p. >= 0.95) into per-subsystem specs, plus an iterative refinement loop when specs are infeasible.

**Problem.** Monolithic RL agents give no formal guarantees on whether a complex task is actually solved with the required probability.

**Method.** A pMDP abstracts subsystem interfaces; subtask specs are derived by parameter optimisation in the pMDP; subsystems train independently and either meet their specs (giving compositional guarantees) or trigger a spec update.

**Result.** Compositional policies provably reach goals with probability >= 0.95 in the demonstrated environments.

**Takeaway.** For multi-step search-augmented agents, decomposing the verifiable reward into per-step "subtask" specifications can give both denser learning signal and stronger end-to-end guarantees.

**ELI5.** Like building a Lego rocket where each module is stress-tested alone against a safety budget, then snapped together with a guarantee that the whole rocket meets the mission spec.

### 2309.06420 — Verifiable RL via Compositionality

**Summary.** The journal-level extension of the compositional verifiable-RL framework: multiple deep RL subsystems trained under partial observability are composed via a parametric MDP that yields end-to-end probabilistic guarantees. The contribution is theoretical guarantees plus an iterative subtask-spec refinement loop covering discrete/continuous, deterministic/stochastic settings.

**Problem.** Deep RL gives no formal end-to-end guarantees, especially under partial observability and stochastic dynamics.

**Method.** A high-level pMDP plans over subsystem interfaces; subtask specs (entry/exit conditions with probability bounds) are derived automatically; subsystems train independently as deep POMDP agents, with infeasible specs triggering pMDP-level re-optimisation.

**Result.** Compositional policies achieve target reachability probabilities (e.g., >= 0.95) across discrete, continuous, deterministic, and stochastic benchmarks.

**Takeaway.** For agentic RLVR with long tool-use trajectories, treating each tool call as a subsystem with its own verifiable spec lets you prove (and debug) composite behaviour rather than only the final answer.

**ELI5.** Like an aviation safety audit: each subsystem (engine, autopilot, landing gear) gets its own probabilistic spec, and the plane is certified only when the specs compose to "safe flight."

### 2308.07469 — Omega-Regular Reward Machines

**Summary.** This paper unifies reward machines (quantitative non-Markovian rewards) with omega-regular languages (qualitative temporal specs) into a single formalism called omega-regular reward machines. The contribution is a model-free RL algorithm that computes epsilon-optimal policies against this richer reward class.

**Problem.** Markovian rewards cannot express many real objectives that depend on temporal history (e.g., "eventually reach A while avoiding B").

**Method.** Omega-regular reward machines extend reward machines with an omega-automaton tracking infinite-trace acceptance; a model-free RL algorithm with theoretical convergence guarantees optimises epsilon-optimal policies against the combined automaton.

**Result.** Empirically computes epsilon-optimal strategies on tasks requiring both quantitative and qualitative objectives (no scalar in abstract).

**Takeaway.** For multi-step tool-using agents whose "success" is a temporal pattern (e.g., "search then answer," not just "right answer"), formalising the reward as an automaton gives a principled way to enforce trajectory-level constraints.

**ELI5.** Like grading a movie not just on the ending but on whether key plot beats happen in the right order, expressed as a checklist that the script must walk through.

### 2312.09695 — Reward Martingales for Robustness Verification

**Summary.** This paper introduces reward martingales, a martingale-theoretic tool that gives provably-sound bounds on expected and tail cumulative reward of a deep RL policy under state perturbations. The contribution is the first robustness verification method for DRL controllers that produces certified reward bounds.

**Problem.** Deployed DRL controllers are vulnerable to state perturbations, but no prior method gave formal performance bounds under such perturbations.

**Method.** A neural network is trained to satisfy reward-martingale conditions over the perturbed state space, yielding mathematically certified upper/lower and tail bounds on cumulative reward against any perturbation in a given set.

**Result.** Certified bounds tightly enclose simulation outcomes across multiple DRL control benchmarks (no scalar in abstract).

**Takeaway.** Robustness verification ideas from control RL hint at how to certify worst-case reward of an LLM agent under prompt perturbations; relevant if your thesis discusses safety/robustness of RLVR-trained search agents.

**ELI5.** Like a stress test that proves your bridge design will still carry trucks even if every beam shifts a little, written as a single inequality the engineers can sign off on.

### 2408.09112 — Set-Based RL for Verifiable Robustness

**Summary.** The paper trains RL agents on entire reachable sets of perturbed states (not just sampled adversarial points) to maximise worst-case reward, using reachability analysis lifted from neural-network verification. The contribution is verifiably robust continuous-control agents that beat adversarial-training baselines on robustness.

**Problem.** Adversarial training only patches sampled attacks, leaving formal robustness gaps in safety-critical RL deployments.

**Method.** During training, propagate input perturbation sets through the policy via reachability analysis and back-propagate against the worst-case reward over the entire reachable set, producing a policy with formally tighter robustness certificates.

**Result.** Verifiably more robust than adversarial-training baselines across four continuous control benchmarks (no scalar in abstract).

**Takeaway.** Even if your RLVR pipeline never invokes formal verification, the principle of training against worst-case perturbations of the input prompt (not random samples) is a cheap robustness lever worth piloting.

**ELI5.** Like training a tightrope walker by tilting the wire to every angle within a safe range simultaneously, instead of just trying a few random tilts.

---

## 12. Measurement, Evaluation, and System Design

### 12.1 Beyond Headline Accuracy

- [Hidden Costs and Measurement Gaps (2509.21882)](https://arxiv.org/abs/2509.21882) — Position paper advocating "tax-aware" evaluation: calibration, reliability, contamination, robustness must accompany accuracy metrics.

### 12.2 Verifier Benchmarking

- [VerifyBench (2505.15801)](https://arxiv.org/abs/2505.15801) — First systematic benchmark for reference-based reward systems; quantifies gaps in difficult/ambiguous cases.

### 12.3 Workload and Systems

- [RL in the Wild (2509.25279)](https://arxiv.org/abs/2509.25279) — PolyTrace benchmark for workload-aware system optimization and scaling of deployed RLVR pipelines.

### 12.4 Training Infrastructure

| Resource | Reference |
|----------|-----------|
| DAPO scalable RL framework | [2503.14476](https://arxiv.org/abs/2503.14476) |
| Understanding R1-Zero training | [2503.20783](https://arxiv.org/abs/2503.20783) |
| DeepSeek-V3.2 RL framework | [2512.02556](https://arxiv.org/abs/2512.02556) |
| Olmo 3 full-lifecycle release | [2512.13961](https://arxiv.org/abs/2512.13961) |

### 12.5 Reasoning Environments and Synthetic Data

| Resource | Reference | Description |
|----------|-----------|-------------|
| REASONING GYM | [2505.24760](https://arxiv.org/abs/2505.24760) | 100+ procedural environments with deterministic checkers |
| Enigmata | [2505.19914](https://arxiv.org/abs/2505.19914) | Synthetic verifiable puzzles |
| SHARP | [2505.14147](https://arxiv.org/abs/2505.14147) | Pipeline for high-quality verifiable reasoning problems |
| Golden Goose | [2601.22975](https://arxiv.org/abs/2601.22975) | Fill-in-the-middle MCQ synthesis from unverifiable text |
| SWE-Universe | [2602.02361](https://arxiv.org/abs/2602.02361) | Million-scale SWE environments from GitHub PRs |

### Paper Cards

### 2509.21882 — Hidden Costs and Measurement Gaps of RLVR

**Summary.** A position paper arguing many headline RLVR gains shrink or vanish once you control for budget mismatch, attempt inflation, calibration drift, and benchmark contamination. Proposes a minimum measurement standard for RLVR claims.

**Problem.** RLVR papers conflate true policy improvement with confounds (mismatched evaluation budgets, abstentions converted into confident wrong answers, contaminated benchmarks).

**Method.** Run budget-matched reproductions and partial-prompt contamination probes on widely cited RLVR results; codify a "tax-aware minimum standard" (saturation curves, calibration, abstention, judge robustness, contamination screen).

**Result.** Several widely cited RLVR gaps shrink substantially or disappear under proper controls.

**Takeaway.** When reporting your small-LM RLVR results, match the baseline's compute budget, track abstention/calibration, and explicitly screen contamination; otherwise reviewers will (and should) discount the gains.

**ELI5.** Like a fitness study where the "wonder pill" group also happened to sleep more and eat better; once you match the controls, the pill's effect is much smaller than the headline.

### 2509.25279 — RL in the Wild: Characterizing RLVR Training

**Summary.** A systems characterization of real RLVR training workloads in production LLM deployment, with the PolyTrace benchmark suite. Identifies GPU idling, parallelism mismatches, and load imbalance as dominant inefficiencies.

**Problem.** RLVR's data flow (rollout, reward, update) is heterogeneous across tasks and steps, but no system-level study quantifies where compute is actually wasted.

**Method.** Trace RLVR workloads across training steps and tasks in a production LLM deployment; analyze sequence-length skew, parallelism strategy mismatches, data management, and load imbalance; package into the PolyTrace benchmark suite.

**Result.** PolyTrace replays workloads at 94.7% accuracy versus the production traces.

**Takeaway.** For a single-A100 small-LM RLVR setup, expect long-tail rollout sequences to dominate idle time; static parallelism is mis-tuned and dynamic batching pays off more than algorithmic tweaks at this scale.

**ELI5.** Like timing a restaurant kitchen and finding chefs spend most of the dinner rush waiting for one slow-cooking dish, not actually cooking; a real fix has to target the bottleneck dish.

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

### 2512.02556 — DeepSeek-V3.2: Sparse Attention + Scalable RL

**Summary.** DeepSeek-V3.2 combines a sparse-attention architecture (DSA) with a scaled RL post-training pipeline and a large-scale agentic-task synthesis pipeline. The high-compute variant matches Gemini-3.0-Pro and wins gold at IMO and IOI 2025.

**Problem.** Pushing reasoning-and-agent frontier capability while keeping inference cost tractable in long-context settings.

**Method.** Three pillars: (1) DeepSeek Sparse Attention reduces long-context compute, (2) a robust RL framework scales post-training compute, (3) an agentic-task synthesis pipeline auto-generates tool-use training data at scale.

**Result.** DeepSeek-V3.2-Speciale gold-medals at both IMO 2025 and IOI 2025, on par with Gemini-3.0-Pro and surpassing GPT-5.

**Takeaway.** Confirms that scaling RL post-training data via synthetic agentic trajectories is now table-stakes for frontier reasoning; useful citation for justifying synthetic-rollout pipelines in small-model work.

**ELI5.** A new flagship laptop that runs cooler (sparse attention), trains harder in evening school (scaled RL), and gets quizzed on a mountain of synthetic homework (agentic synthesis).

### 2512.13961 — Olmo 3: Fully-Open 7B/32B Reasoning Models

**Summary.** Olmo 3 is a fully-open family of 7B and 32B language models targeting long-context reasoning, function calling, coding, and chat, released with the entire model flow (every checkpoint, dataset, dependency). The flagship Olmo 3 Think 32B is positioned as the strongest fully-open thinking model to date.

**Problem.** Open reasoning research is hampered by partial releases (weights without data or training recipes), making rigorous comparison and post-training research difficult.

**Method.** Releases the complete model lifecycle (data, every checkpoint, every dependency) at 7B and 32B scales, with explicit attention to long-context reasoning, function calling, coding, instruction following, and knowledge recall.

**Result.** Olmo 3 Think 32B is reported as the strongest fully-open thinking model released to date (no abstract-level numerical metric).

**Takeaway.** For RLVR ablations needing a clean, fully-open base, Olmo 3 (especially the Think variant) is the new reference; the released intermediate checkpoints enable mid-training experiments not possible on closed models.

**ELI5.** Most open models hand you a finished cake; Olmo 3 hands you the cake plus the full recipe, ingredient list, and every batter sample from each step of mixing.

### 2505.24760 — Reasoning Gym

**Summary.** Reasoning Gym (RG) is an open library of 100+ procedural data generators and verifiers spanning algebra, logic, geometry, graphs, games and more. It targets RLVR training and evaluation.

**Problem.** Existing reasoning datasets are fixed-size, so they get memorized and cannot probe difficulty curves.

**Method.** Procedural generation with adjustable complexity yields effectively infinite training and evaluation data, each paired with a deterministic verifier.

**Result.** Demonstrates effective use for both evaluation and RL training of reasoning models (no single headline number in abstract).

**Takeaway.** A drop-in source of verifiable tasks for any RLVR project; especially useful for curriculum work where you want a difficulty knob.

**ELI5.** Like a level editor for math problems: you can crank the difficulty slider and the game ships with an automatic referee.

### 2505.19914 — Enigmata: Synthetic Verifiable Puzzles

**Summary.** Enigmata is a suite of 36 puzzle tasks, each with an unbounded generator and a rule-based verifier, designed for scalable multi-task RLVR training. It also ships Enigmata-Eval as a held-out benchmark.

**Problem.** LLMs trained with RLVR for math and code still flunk human-easy puzzles, and there is no large, controllable, verifier-equipped puzzle dataset.

**Method.** Pair every puzzle with a difficulty-controllable generator and a deterministic verifier, then train multi-task GRPO/RLVR with optimized mixture strategies on Qwen2.5-32B.

**Result.** Qwen2.5-32B-Enigmata reaches 32.8% on ARC-AGI and beats o3-mini-high / o1 on the puzzle benchmarks; further boosts AIME and GPQA when added to a 200B Seed1.5-Thinking model.

**Takeaway.** Synthetic generator+verifier task suites are a cheap way to scale RLVR data and can transfer to math/STEM if mixed properly.

**ELI5.** Instead of buying a few crossword books, the model gets a printer that produces unlimited custom crosswords plus an instant answer key; practice never runs out.

### 2505.14147 — SHARP

**Summary.** A pipeline for synthesizing graduate/Olympiad-level STEM problems with verifiable answers, designed to feed RLVR training of large reasoning models. Combines self-alignment principles with a three-phase Alignment-Instantiation-Inference framework.

**Problem.** RLVR for reasoning is bottlenecked by the scarcity of difficult, diverse, *verifiable* training problems; CoT-prompted synthesis produces too-easy or unverifiable items.

**Method.** Use a strong LRM to infer and self-verify hard STEM problems following alignment principles (difficulty, logical consistency, unambiguous answers); RLVR-loop refines the model on these synthetic problems.

**Result.** Substantial GPQA gains, "pushing LRM performance closer to expert-level" (no headline number).

**Takeaway.** If your verifiable-reward training plateaus on standard datasets, generate harder problems with an LRM-as-author + verifier, rather than scraping more data of the same difficulty.

**ELI5.** Like a tutor who not only writes harder problems for the student but also solves them first to make sure the answer key is right; the student is then guaranteed honest feedback on every attempt.

### 2601.22975 — Golden Goose: Synthesizing Unlimited RLVR Tasks

**Summary.** Golden Goose generates verifiable RLVR tasks from arbitrary internet text by masking key reasoning steps and adding plausible distractors, turning unverifiable corpora into multiple-choice fill-in-the-middle problems. It produces GooseReason-0.7M and revives RLVR runs that have saturated on existing math/code data.

**Problem.** RLVR scaling is bottlenecked by the limited supply of naturally verifiable problems; gains saturate fast on existing corpora.

**Method.** An LLM identifies and masks reasoning-critical spans in source text, then synthesizes distractors to form a verifiable MCQ; this lets reasoning-rich but unverifiable corpora (textbooks, FineWeb scrapes) feed RLVR.

**Result.** New SOTA for 1.5B and 4B-Instruct models across 15 benchmarks; a 4B cyber-domain model trained this way beats a 7B specialist.

**Takeaway.** If your small-LM RLVR run is plateauing on verifiable math/QA, you can extend it indefinitely by synthesizing verifiable MCQs from any reasoning-rich text, including domain corpora with no native ground truth.

**ELI5.** Take any well-written paragraph, blank out the punchline sentence, and offer four wrong endings; suddenly the entire internet is a verifiable exam and you'll never run out of questions.

---

## 13. Open Challenges and Future Directions

### 13.1 Verifier Frontier
- Unified theory of verifier reliability (false negatives × false positives).
- Cost-aware verifier selection (rule vs. judge vs. hybrid trade-off curves).
- Cross-domain transferable verifiers.

### 13.2 Exploration Frontier
- Principled (rather than heuristic) escape from base-model support.
- Multi-turn agent-environment exploration without entropy collapse.
- MCTS integration with policy-gradient RL.

### 13.3 Bridging to Non-Verifiable Domains
- Tool use as a verifier surrogate (under-explored).
- Process reward models without per-step ground truth.
- Generative judges that themselves benefit from RLVR.

### 13.4 Small-Model Frontier
- Resolving the contested boundary: when does RLVR help small (≤3B) models?
- SFT-prerequisite quantification.
- Negative sampling and cold-start strategies.

### 13.5 Multi-Objective Learning
- Principled aggregation of verifiable + preference rewards.
- Pareto-frontier characterization across reasoning vs. style vs. safety.
- Domain-mix curriculum (transfer interference vs. transfer benefit).

### 13.6 Long-Horizon and Infinite-Context
- [InftyThink+ (2602.06960)](https://arxiv.org/abs/2602.06960) — Infinite-horizon iterative reasoning with autonomous summarization.
- State compression while preserving verifiability.

### 13.7 Theoretical
- Connection between RLVR and classical compositional/formal-methods RL.
- Sample complexity bounds for verifier-based learning.
- When does RLVR provably exceed the base model support?

### Paper Cards

### 2602.06960 — InftyThink+: RL for Iterative Reasoning

**Summary.** InftyThink+ is an end-to-end RL framework that learns when to summarize, what to keep, and how to resume during long-horizon iterative reasoning, replacing fixed heuristics or SFT-only training. It uses an SFT cold-start followed by trajectory-level RL.

**Problem.** Long chain-of-thought has quadratic cost, hits context limits, and suffers lost-in-the-middle; existing iterative-summarization methods rely on fixed schedules.

**Method.** The model controls iteration boundaries and emits explicit summaries; a two-stage recipe (SFT cold-start then trajectory-level RL) optimizes the entire iterative trajectory rather than per-step tokens.

**Result.** +21% accuracy on AIME24 with DeepSeek-R1-Distill-Qwen-1.5B over conventional long-CoT RL, plus reduced inference latency.

**Takeaway.** For 1.5B reasoners with limited context, learning summarize-and-resume policies via RL beats fixed CoT and is a strong fit for compute-constrained thesis setups.

**ELI5.** Instead of writing a single 50-page essay, the model learns to take notes, file them away, and start a fresh page; RL teaches it which notes are worth keeping.

---

## 14. Consolidated Bibliography

Deduplicated. Grouped by primary research focus. arXiv links only. Full per-paper cards (Summary / Problem / Method / Result / Takeaway / ELI5) for Tier-1 references are now inline under the relevant thematic sections (§1-13). For the project-specific resource-constrained RLVR literature with cards see [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md); for adjacent / overflow papers see [SURVEY_OVERFLOW.md](SURVEY_OVERFLOW.md).

### 14.1 Foundational LLM RL

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (2501.12948)](https://arxiv.org/abs/2501.12948)
- [Training language models to follow instructions with human feedback — InstructGPT (2203.02155)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (2305.18290)](https://arxiv.org/abs/2305.18290)
- [LoRA: Low-Rank Adaptation (2106.09685)](https://arxiv.org/abs/2106.09685)

### 14.2 Self-Verification (RISE Family)

- [Trust, But Verify (2505.13445)](https://arxiv.org/abs/2505.13445)
- [Incentivizing LLMs to Self-Verify (2506.01369)](https://arxiv.org/abs/2506.01369)
- [RISE: VLM Image Annotation (2508.13229)](https://arxiv.org/abs/2508.13229)
- [LMs Better Reasoners with Self-Verification (2212.09561)](https://arxiv.org/abs/2212.09561)
- [DeepSeekMath-V2 (2511.22570)](https://arxiv.org/abs/2511.22570)
- [Learning to Reason without External Rewards (2505.19590)](https://arxiv.org/abs/2505.19590)
- [RLVMR (2507.22844)](https://arxiv.org/abs/2507.22844)

### 14.3 Reward Modeling

- [Crossing the Reward Bridge (2503.23829)](https://arxiv.org/abs/2503.23829)
- [VerIF (2506.09942)](https://arxiv.org/abs/2506.09942)
- [Rubrics as Rewards (2507.17746)](https://arxiv.org/abs/2507.17746)
- [Agentic Reward Modeling (2502.19328)](https://arxiv.org/abs/2502.19328)
- [Reward Hacking Mitigation Composite (2509.15557)](https://arxiv.org/abs/2509.15557)
- [Accuracy to Robustness Verifiers (2505.22203)](https://arxiv.org/abs/2505.22203)
- [VerifyBench (2505.15801)](https://arxiv.org/abs/2505.15801)
- [Shaping Explanations (2509.13081)](https://arxiv.org/abs/2509.13081)
- [Survey of Reward Models (2504.12328)](https://arxiv.org/abs/2504.12328)
- [Writing-Zero (2506.00103)](https://arxiv.org/abs/2506.00103)

### 14.4 Optimization Algorithms

- [GRPO Effective Loss (2503.06639)](https://arxiv.org/abs/2503.06639)
- [DAPO (2503.14476)](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero (2503.20783)](https://arxiv.org/abs/2503.20783)
- [RC-GRPO (2602.03025)](https://arxiv.org/abs/2602.03025)
- [Fission-GRPO (2601.15625)](https://arxiv.org/abs/2601.15625)
- [MC-GRPO (2601.22582)](https://arxiv.org/abs/2601.22582)
- [F-GRPO (2602.06717)](https://arxiv.org/abs/2602.06717)
- [Group Turn PO (2511.14846)](https://arxiv.org/abs/2511.14846)
- [Prefix Grouper (2506.05433)](https://arxiv.org/abs/2506.05433)
- [GRPO Collapse in Agent Search (2512.04220)](https://arxiv.org/abs/2512.04220)

### 14.5 Exploration and Credit Assignment

- [Trial-and-Error LLM Exploration (2508.07534)](https://arxiv.org/abs/2508.07534)
- [Forward-KL Exploration (2510.03865)](https://arxiv.org/abs/2510.03865)
- [UCAS — Uncertainty-aware Advantage Shaping (2510.10649)](https://arxiv.org/abs/2510.10649)
- [PACR — Confidence Reward (2510.22255)](https://arxiv.org/abs/2510.22255)
- [Shrinkage Baselines (2511.03710)](https://arxiv.org/abs/2511.03710)
- [Adaptive Guidance / Guide-GRPO (2506.13923)](https://arxiv.org/abs/2506.13923)
- [Beyond 80/20 — High-Entropy Tokens (2506.01939)](https://arxiv.org/abs/2506.01939)
- [First Return, Entropy-Eliciting Explore (2507.07017)](https://arxiv.org/abs/2507.07017)
- [StepHint (2507.02841)](https://arxiv.org/abs/2507.02841)

### 14.6 Safety, Robustness, Reward Hacking

- [Breaking Safety-Capability Tradeoff (2511.21050)](https://arxiv.org/abs/2511.21050)
- [IFDECORATOR (2508.04632)](https://arxiv.org/abs/2508.04632)
- [Spurious Rewards (2506.10947)](https://arxiv.org/abs/2506.10947)
- [Verifiable yet Noisy Rewards (2510.00915)](https://arxiv.org/abs/2510.00915)

### 14.7 Mechanism Studies

- [Reasoning Pattern Selection (2506.04695)](https://arxiv.org/abs/2506.04695)
- [Does RL Really Incentivize Reasoning? (2504.13837)](https://arxiv.org/abs/2504.13837)
- [The Invisible Leash (2507.14843)](https://arxiv.org/abs/2507.14843)
- [RLVR Implicitly Incentivizes Reasoning (2506.14245)](https://arxiv.org/abs/2506.14245)
- [Generalization of RLVR via Causal Reasoning (2512.20760)](https://arxiv.org/abs/2512.20760)

### 14.8 Multi-Task and Hybrid Rewards

- [Multi-task RL Cross-learning RKHS (2008.11895)](https://arxiv.org/abs/2008.11895)
- [Multi-Task IRL Common Sense (2402.11367)](https://arxiv.org/abs/2402.11367)
- [Combining Reward from Multiple Sources (2103.12142)](https://arxiv.org/abs/2103.12142)
- [Variational IRL Transferable Rewards (2206.09498)](https://arxiv.org/abs/2206.09498)
- [Multi-Task Reward Learning Human Ratings (2506.09183)](https://arxiv.org/abs/2506.09183)
- [Imperfect also Deserves Reward (Dialog) (2104.04748)](https://arxiv.org/abs/2104.04748)
- [Auditable-choice Reframing (2511.02463)](https://arxiv.org/abs/2511.02463)

### 14.9 Domain Extensions

**Mathematics:**
- [One Training Example (2504.20571)](https://arxiv.org/abs/2504.20571)

**Medical:**
- [Med-RLVR (2502.19655)](https://arxiv.org/abs/2502.19655)
- [Open-Medical-R1 (2504.13950)](https://arxiv.org/abs/2504.13950)
- [EHR-Based Reasoning RL (2505.24105)](https://arxiv.org/abs/2505.24105)

**Multimodal & Embodied:**
- [R1-Omni (2503.05379)](https://arxiv.org/abs/2503.05379)
- [SATORI-R1 (2505.19094)](https://arxiv.org/abs/2505.19094)
- [MoDoMoDo (2505.24871)](https://arxiv.org/abs/2505.24871)
- [ManipLVM-R1 (2505.16517)](https://arxiv.org/abs/2505.16517)
- [Few-Shot Vision-Language Satellite (2507.21745)](https://arxiv.org/abs/2507.21745)

**World Models & Forecasting:**
- [RLVR-World (2505.13934)](https://arxiv.org/abs/2505.13934)
- [Outcome-based Forecasting (2505.17989)](https://arxiv.org/abs/2505.17989)

**Specialized:**
- [RAIDEN-R1 (2505.10218)](https://arxiv.org/abs/2505.10218)
- [Korean Word-Chain Game RLVR (2510.03394)](https://arxiv.org/abs/2510.03394)

**Multi-Domain:**
- [Multi-Domain Reasoning RL (2507.17512)](https://arxiv.org/abs/2507.17512)
- [Synthetic Data Generation Multi-Step RL (2504.04736)](https://arxiv.org/abs/2504.04736)

### 14.10 Tool-Use and Search Integration

**Search-Augmented:**
- [ReSearch (2503.19470)](https://arxiv.org/abs/2503.19470)
- [R1-Searcher (2503.05592)](https://arxiv.org/abs/2503.05592)
- [Search-R1 (2503.09516)](https://arxiv.org/abs/2503.09516)
- [Dr. Zero (2601.07055)](https://arxiv.org/abs/2601.07055)

**Tool-Use Reward:**
- [ToolRL (2504.13958)](https://arxiv.org/abs/2504.13958)
- [Replacing Thinking with Tool Usage (2507.05065)](https://arxiv.org/abs/2507.05065)
- [Advancing SLM Tool-Use (2509.04518)](https://arxiv.org/abs/2509.04518)
- [Acting Less is Reasoning More (2504.14870)](https://arxiv.org/abs/2504.14870)
- [ToolOrchestra (2511.21689)](https://arxiv.org/abs/2511.21689)
- [LMs Use Prolog as Tool (2512.07407)](https://arxiv.org/abs/2512.07407)
- [ToolExpander (2510.07737)](https://arxiv.org/abs/2510.07737)

**Tool-Augmented Verification:**
- [CoSineVerifier (2512.01224)](https://arxiv.org/abs/2512.01224)

**Agentic / Software Engineering:**
- [Agent-RLVR (2506.11425)](https://arxiv.org/abs/2506.11425)
- [Agent-R1 (2511.14460)](https://arxiv.org/abs/2511.14460)
- [SWE-Universe (2602.02361)](https://arxiv.org/abs/2602.02361)
- [DeepSearch via MCTS (2509.25454)](https://arxiv.org/abs/2509.25454)

### 14.11 Verifier-Free / Intrinsic Rewards

- [RLPR — Extrapolating RLVR without Verifiers (2506.18254)](https://arxiv.org/abs/2506.18254)
- [Self-Certainty Reward (2505.19590)](https://arxiv.org/abs/2505.19590)

### 14.12 Theoretical Foundations (Symbolic / Compositional / Formal Methods)

- [Intrinsic Symbolic Rewards (2010.03694)](https://arxiv.org/abs/2010.03694)
- [Replacing Rewards with Examples (2103.12656)](https://arxiv.org/abs/2103.12656)
- [Verifiable Compositional RL Systems (2106.05864)](https://arxiv.org/abs/2106.05864)
- [Verifiable RL via Compositionality (2309.06420)](https://arxiv.org/abs/2309.06420)
- [Omega-Regular Reward Machines (2308.07469)](https://arxiv.org/abs/2308.07469)
- [Reward Martingales (2312.09695)](https://arxiv.org/abs/2312.09695)
- [Set-Based RL (2408.09112)](https://arxiv.org/abs/2408.09112)
- [LLM Multi-Agent RL Survey (2405.11106)](https://arxiv.org/abs/2405.11106)

### 14.13 Environments, Data Generation, and Benchmarks

- [REASONING GYM (2505.24760)](https://arxiv.org/abs/2505.24760)
- [Enigmata (2505.19914)](https://arxiv.org/abs/2505.19914)
- [SHARP (2505.14147)](https://arxiv.org/abs/2505.14147)
- [Golden Goose (2601.22975)](https://arxiv.org/abs/2601.22975)
- [VerifyBench (2505.15801)](https://arxiv.org/abs/2505.15801)

### 14.14 Measurement and Systems

- [Hidden Costs and Measurement Gaps (2509.21882)](https://arxiv.org/abs/2509.21882)
- [RL in the Wild — PolyTrace (2509.25279)](https://arxiv.org/abs/2509.25279)
- [DAPO (2503.14476)](https://arxiv.org/abs/2503.14476)
- [DeepSeek-V3.2 (2512.02556)](https://arxiv.org/abs/2512.02556)
- [Olmo 3 (2512.13961)](https://arxiv.org/abs/2512.13961)

### 14.15 Long-Horizon and Frontier

- [InftyThink+ (2602.06960)](https://arxiv.org/abs/2602.06960)
- [Learning to Reason in 13 Parameters (2602.04118)](https://arxiv.org/abs/2602.04118)
- [Reinforcement World Model Learning (2602.05842)](https://arxiv.org/abs/2602.05842)
- [Era of Agentic Organization (2510.26658)](https://arxiv.org/abs/2510.26658)
- [Autonomous Agents for Scientific Discovery (2510.09901)](https://arxiv.org/abs/2510.09901)

### 14.16 Open-Review Submissions and Pre-prints

- [Enabling Tool Use without Verifiable Reward via SFT-RL Loop](https://openreview.net/pdf/1ca67ceed76207273bb57d9dc64f0ce06c209123.pdf)
- [Extending RLVR to Open-Ended Tasks via Verifiable Multiple-Choice Reformulation](https://openreview.net/pdf?id=uZxyvmN72d)
- [Knowledge-to-Verification: Unlocking RLVR for Knowledge-Intensive Domains](https://openreview.net/forum?id=EVS7SeKBqI)
- [Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards](https://openreview.net/forum?id=ujLgLz6QQa)

### 14.17 Resource-Constrained and Small-Model RLVR

**Memory and compute efficiency:**
- [S-GRPO / Token-Efficient RL (2504.20834)](https://arxiv.org/abs/2504.20834)
- [Prefix Grouper (2506.05433)](https://arxiv.org/abs/2506.05433) — also in §14.4
- [MC-GRPO (2601.22582)](https://arxiv.org/abs/2601.22582) — also in §14.4
- [GSPO (2507.18071)](https://arxiv.org/abs/2507.18071)
- [Evaluating PEFT for RLVR (2512.23165)](https://arxiv.org/abs/2512.23165)
- [Plasticity vs Rigidity — LoRA RL (2601.06677)](https://arxiv.org/abs/2601.06677)

**Rollout budget and sample efficiency:**
- [Hard Examples (2508.14094)](https://arxiv.org/abs/2508.14094)
- [PODS — Down-Sampling Rollouts (2504.13818)](https://arxiv.org/abs/2504.13818)
- [Online Difficulty Filtering (2504.03380)](https://arxiv.org/abs/2504.03380)
- [AR3PO — Rollout Reuse (2509.25808)](https://arxiv.org/abs/2509.25808)
- [VIP — Adaptive Rollout Allocation (2602.01601)](https://arxiv.org/abs/2602.01601)
- [RLVR Training Linearity (2601.04537)](https://arxiv.org/abs/2601.04537)
- [NExt — Nonlinear Trajectory Extrapolation (2604.11446)](https://arxiv.org/abs/2604.11446)
- [Learning from Less — Low-Data RLVR (2604.18381)](https://arxiv.org/abs/2604.18381)

**Curriculum and difficulty-aware training:**
- [E2H Curriculum Reasoner (2506.06632)](https://arxiv.org/abs/2506.06632)
- [DeReason — Difficulty-Partitioned SFT+RL (2603.11193)](https://arxiv.org/abs/2603.11193)

**Replay buffers and off-policy:**
- [ExGRPO — Experience Replay (2510.02245)](https://arxiv.org/abs/2510.02245)
- [Freshness-Aware PER (2604.16918)](https://arxiv.org/abs/2604.16918)
- [Prompt Replay (2603.21177)](https://arxiv.org/abs/2603.21177)
- [Self-Distilled RLVR / RLSD (2604.03128)](https://arxiv.org/abs/2604.03128)
- [BAPO-Buffer — Off-Policy Replay (2602.20722)](https://arxiv.org/abs/2602.20722)
- [DyJR — Dynamic JSD Replay (2603.16157)](https://arxiv.org/abs/2603.16157)

**Small-model failure modes:**
- [RAGEN-2 — Reasoning Collapse / Template Collapse (2604.06268)](https://arxiv.org/abs/2604.06268)
- [Divergence Choice / DPH-RL (2509.07430)](https://arxiv.org/abs/2509.07430)
- [JustRL — Tricks May Hurt (2512.16649)](https://arxiv.org/abs/2512.16649)
- [LLMs Gaming Verifiers (2604.15149)](https://arxiv.org/abs/2604.15149)

**Search-augmented RL at small scale:**
- [HiPRAG — Hierarchical Process Rewards Search 3B (2510.07794)](https://arxiv.org/abs/2510.07794)
- [DeepRetrieval — 3B RL Query Gen (2503.00223)](https://arxiv.org/abs/2503.00223)
- [R-Search — Multi-Reward Search RL (2506.04185)](https://arxiv.org/abs/2506.04185)
- [APEX-Searcher — Decoupled Planning+Execution RL (2603.13853)](https://arxiv.org/abs/2603.13853)
- [BAPO — Boundary-Aware Search RL (2601.11037)](https://arxiv.org/abs/2601.11037)

---


## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\pi_\theta$ | Policy parameterized by $\theta$ |
| $\pi_{\text{ref}}$ | Frozen reference policy |
| $r(x, y)$ | Verifier-based reward |
| $\beta$ | KL regularization coefficient |
| $G$ | Group size in GRPO |
| $\hat{A}_i$ | Advantage estimate for rollout $i$ |
| $\rho_t(\theta)$ | Likelihood ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$ |
| $\epsilon$ | PPO clip range |
| $\mathcal{H}$ | Policy entropy |
| $D_{\text{KL}}$ | Kullback–Leibler divergence |

---

*Coverage current through Q2 2026. Total deduplicated arXiv references: 140+. Section 15 and bibliography §14.17 added 2026-05-03, covering resource-constrained small-model RLVR.*
