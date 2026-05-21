---
title: LITERATURE REVIEW
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# RLVR: Reinforcement Learning with Verifiable Rewards

A comprehensive survey of recent advances in reinforcement learning for language models, focusing on verifiable rewards, reasoning emergence, and domain extension.

> **Companion documents:**
> - [SURVEY.md](SURVEY.md) — equation-driven consolidated survey of the full RLVR landscape with inline per-paper cards (Summary / Problem / Method / Result / Takeaway / ELI5) for ~96 Tier-1 references.
> - [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md) — project-specific synthesis for the Qwen3.5-2B / single-A100 / search-augmented thesis setting, with cards for the project-critical literature and a week-by-week experiment plan.
> - [SURVEY_OVERFLOW.md](SURVEY_OVERFLOW.md) — 15 adjacent papers (foundational priors, surveys, alternative directions) kept for completeness.
>
> Use this literature review for the project-specific framing and personal annotations; use the SURVEY family for the unified reference treatment with cards.

---

## 1. Core Paradigm

### What is RLVR?

Reinforcement Learning with Verifiable Rewards (RLVR) is a training paradigm that uses deterministic, model-based reward functions to optimize LLM behavior without requiring expensive human preference annotations. It represents a shift from RLHF (Reinforcement Learning from Human Feedback) to systems that can automatically verify correctness.

**Key Properties:**
- Deterministic and model-based reward functions
- Requires verifiable outcomes (answers that can be objectively checked)
- Binary or soft reward signals
- Works in domains with clear ground truth: mathematics, code, multiple choice questions
- Policy optimization via PPO, GRPO, or other policy gradient methods
- Reward acts as the advantage signal with KL-divergence penalty for regularization

### Historical Context

- **InstructGPT** (2022): Showed small models can outperform larger ones with RLHF
- **RLHF era**: Required expensive human preference labels
- **DeepSeek-R1** (Jan 2025): Foundational proof that reasoning-like behavior emerges from pure RL without supervised reasoning trajectories
- **Current frontier**: Extending RLVR beyond verifiable domains

### Why RLVR Works

1. **Emergent Reasoning**: R1 demonstrated that reasoning patterns emerge through RL optimization, not just SFT distillation
2. **Pattern Selection & Optimization**: RL selects and optimizes existing reasoning patterns already present in the base model (distillation can introduce new patterns)
3. **Strong SFT Coupling**: Efficacy is tightly coupled with initialization—rapid convergence with a strong SFT base
4. **Self-Distillation**: Reasoning emerges through the model refining its own outputs

### Known Limitations & Challenges

- **Base Model Dependence**: Heavily dependent on pre-training quality; doesn't work well on weak base models
- **Reward Hacking**: Models learn to exploit reward signals in unintended ways
- **Model Dependence**: Strong correlation between upstream SFT quality and downstream RL gains
- **Upper Bound by Base Model**: Reasoning abilities are bounded by what's already in the base model (without distillation)
- **Metric Limitations**: Pass@k may be insufficient; need metrics like CoT-pass@k where both answer AND reasoning are correct
- **Spurious Reward Dependency**: Gains depend heavily on the base model—even spurious rewards work with some models but not others

### Domain Adaptation Perspective

> **Key insight**: Domain adaptation means learning how to _interact with_ the domain (via tool use), not internalizing the domain knowledge.

Domain adaptation through RL should focus on:
- **Policy adaptation over tools**: Can we get domain-based reasoning by enabling tool use?
- **Tool use + MCTS**: Combining tool use with Monte Carlo Tree Search to learn domain reasoning
- **Tool use as surrogate objective**: Model non-verifiable domains as environments where tool use generates verifiable signals

---

## 1.7 Consolidated Survey Reference

All thematic synthesis, mathematical formulations, per-survey breakdowns, and cross-cutting analysis across the six major RLVR topic reviews have been consolidated into three companion documents:

- [SURVEY.md](SURVEY.md) — full RLVR landscape with equation-driven treatment, deduplicated arXiv bibliography, and inline per-paper cards (Summary / Problem / Method / Result / Takeaway / ELI5) for ~96 Tier-1 references.
- [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md) — project-specific synthesis for resource-constrained small-LM RLVR with search (Qwen3.5-2B, single A100, ~$1000 budget) including a week-by-week experiment plan.
- [SURVEY_OVERFLOW.md](SURVEY_OVERFLOW.md) — 15 adjacent papers (foundational priors, surveys, alternative directions) kept for completeness.

---

## 2. Approaches & Techniques

### 2.1 Reward Modeling

The design of reward functions is central to RLVR. Different approaches have emerged for different domains:

**Rule-Based Rewards:**
- Unit tests, math checkers, code verifiers
- Format rewards (e.g., LLM judge for ensuring `<think>` tag compliance)
- Binary correctness signals
- Efficient and interpretable

**Model-Based / Semantic Rewards:**
- Encoder-based similarity matching (e.g., sentence embedding cosine similarity)
- Soft rewards using reference solutions
- Learned reward models from domain data
- Semantic Reward Modeling with Encoder-Only Transformers

**Hybrid Reward Setups:**
- Combining rule-based + semantic rewards
- Answer reward + format reward (DeepSeek-R1 pattern)
- Evidence-grounded with RAG
- Constraints (predefined rules or on-the-fly generation)

**Soft Reward Mechanisms:**
- Z-score normalization for soft scores
- Model-based rewards from domain data
- Rubrics-based scoring
- Self-certainty / confidence-based rewards (KL-divergence from uniform distribution)

**Non-Verifiable Domain Extensions:**
- Process Reward Models (PRMs) for step-wise evaluation
- Semantic matching for similarity assessment
- Latent state comparison
- Keyword matching
- Schema compliance (regex, format validation)
- Pair-wise ranking (Bradley-Terry style)
- Generative soft model-based rewards with z-score normalization

### 2.2 Meta-Reasoning & Tool Use

Research shows meta-reasoning tags improve model behavior:

**Structured Reasoning Patterns:**
- `<thinking>...</thinking>` for internal reasoning
- `<tool_use>...</tool_use>` for external tool invocation
- Combined reasoning and tool use for complex problems
- Self-verification and reflection patterns

**Tool-Use as Domain Bridge:**
- Tools provide verifiable signals in non-verifiable domains
- Tool invocation can be rewarded (meta-level optimization)
- Tool use + reasoning improves QA performance
- Multi-turn search and reasoning with tools

**Process Reward Models (PRM):**
- Score each reasoning step, not just final answer
- Learn to judge LLM explanations during training
- Enables fine-grained feedback
- Trade-off: computational overhead vs. improved guidance

### 2.3 Training Algorithms

**GRPO (Group Relative Policy Optimization):**
- Used in DeepSeek-R1, ReSearch, and recent RLVR work
- Compares relative performance within groups
- More stable than PPO for small batch sizes
- Variants: RC-GRPO (reward-conditioned), BRPO (bootstrapped relative)

**PPO (Proximal Policy Optimization):**
- Classic policy gradient algorithm
- Works well with strong SFT initialization
- Requires careful hyperparameter tuning

**Emerging Variants:**
- Off-policy importance-ratio correction for stable training
- Guide-GRPO: adaptive guidance acceleration
- REINFORCE, REINFORCE++, RLOO for different problem settings

**Curriculum Learning:**
- Pedagogically structured progression improves convergence
- Step-wise difficulty increase
- Shown to accelerate reasoning development
- Effectiveness depends on task domain

### 2.4 Domain Extension: Bridging Verifiable & Non-Verifiable

**From Pure Math/Code to Broader Domains:**

The frontier of RLVR is extending beyond deterministic, verifiable domains:

- **Medical QA** (MedRLVR): Multiple choice answers as verifiable labels, format compliance, correctness rewards
- **Scientific Reasoning** (ReSearch): Search tool integration with rule-based answer rewards
- **Open-Ended Generation** (Writing Zero): Self-principled critique pairs, phase-dependent rewards
- **Instruction Following** (VerIF): Combining rule-based and LLM-based semantic verification
- **General Domains** (RLPR): Using intrinsic model properties as reward signals

**Key Insight for Non-Verifiable Domains:**
Tool use acts as a surrogate for verifiable rewards—the model learns when and how to use tools to find verifiable answers, making the overall trajectory verifiable even if the individual generation step is not.

---

## 3. Open Questions & Research Gaps

### Fundamental Questions

**Reward Design for Non-Deterministic Domains:**
- How can we automatically create reward signals in domains without objective ground truth?
- Can semantic similarity metrics (encoder-based) serve as reliable rewards?
- How do we combine rule-based and learned rewards effectively?
- What role can rubrics play in structured reward modeling?

**Process vs. Outcome Rewards:**
- How effective are process reward models compared to outcome-only rewards?
- Can step-wise rewards improve convergence without excessive computational overhead?
- How do we evaluate individual reasoning steps when there's no ground truth?

**Domain Bridging:**
- How can we extend RLVR into non-verifiable domains?
- Can tool use serve as a reliable bridge between verifiable and non-verifiable domains?
- What makes a good surrogate objective for non-verifiable tasks?

**Small Model Capabilities:**
- What is the actual reasoning capacity ceiling for 1B-3B models?
- Can negative sampling with SFT for cold-start improve small model performance?
- How much does base model quality limit downstream RLVR gains?

**Reasoning Patterns:**
- What kind of reasoning emerges in small models through pure RL?
- Can self-verification and reflection develop without explicit training?
- What's the difference between pattern selection (optimization) and pattern emergence (novelty)?

### Emerging Research Directions

- Monte Carlo Tree Search integration in RLVR training
- Transfer learning across domains (can skills from one domain help others?)
- Multi-turn agent-environment interaction
- Continual scaling of RL paradigms
- Personalized guidance strategies
- Infinite-horizon reasoning with state compression
- Combining internal and external reward signals

---

## 4. Key Examples & Reproductions

### Foundational Work

**TinyZero**
- Replicates DeepSeek-R1-Zero approach with minimal resources
- 3B base LM develops self-verification and search abilities through pure RL
- Emerges reasoning-like behavior and emergent self-verification
- Training cost: <$30
- **Significance**: Proves reasoning emergence is achievable in small models

**DeepSeek-R1** (Jan 2025)
- Foundational architecture demonstrating RLVR at scale
- Accuracy reward (verifiable via LeetCode compiler for code, deterministic math solver)
- Format reward (LLM judge for compliance with `<think>` tags)
- Shows reasoning-like behavior emerges from pure RL without supervised reasoning trajectories
- Extensive comparison with PRM (limited advantages despite computational overhead)

### Domain-Specific Reproductions

**ReSearch** (Sep 2025) ⭐ EXACTLY YOUR PATTERN
- Combines search tool use with GRPO and RLVR
- Rule-based rewards: answer reward (F1/exact match) + format reward
- Uses instruction-tuned models
- Directly applicable to QA domains with search

**MedRLVR** (Feb 2025)
- Extends RLVR to medical domain with 3B base model
- Uses MCQ answers as verifiable labels
- Reward signals: correctness + format compliance
- Future direction: penalize short chain-of-thought

**Writing Zero** (Jun 2025)
- Self-principled critique pairs with phase-dependent rewards
- GenRM learned from SFT + rule-based RL to serve as judge
- Bootstrapped Relative Policy Optimization (BRPO)
- Shows RLVR applicability to creative domains

**VerIF** (Jun 2025)
- Combines rule-based and LLM-based semantic verification
- Instruction-following dataset with verifiable signals
- Demonstrates hybrid reward approach

### Supporting Infrastructure

- REASONING GYM (May 2025): Over 100 data generators and verifiers with adjustable difficulty

---

## 5. Comprehensive Paper Bibliography

### 5.1 Meta-Reasoning & Self-Verification

- [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844)
- [Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2505.13445) (May 2025) — Core RISE framework
- [Incentivizing LLMs to Self-Verify Their Answers](https://arxiv.org/abs/2506.01369) (Jun 2025) — GRPO-based unified gen-verify; 87.2% verification on MATH500
- [RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning](https://arxiv.org/abs/2508.13229) (Aug 2025) — RISE-CoT vision-LLM extension
- [Large Language Models are Better Reasoners with Self-Verification](https://arxiv.org/abs/2212.09561) (Weng et al., 2022) — Foundational backward verification
- [RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245) (Jun 2025) — Introduces CoT-Pass@K metric
- [DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://www.alphaxiv.org/abs/2511.22570) (Nov 2025)
- [Learning to Reason without External Rewards](https://www.alphaxiv.org/abs/2505.19590)

### 5.2 Tool Use & Search Integration

- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://www.alphaxiv.org/abs/2503.05592) ⭐
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://www.alphaxiv.org/abs/2503.09516) ⭐
- [Dr. Zero: Self-Evolving Search Agents without Training Data](https://www.alphaxiv.org/abs/2601.07055) ⭐
- [Replacing thinking with tool usage enables reasoning in small language models](https://www.alphaxiv.org/abs/2507.05065)
- [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://www.alphaxiv.org/abs/2511.21689)
- [CoSineVerifier: Tool-Augmented Answer Verification for Computation-Oriented Scientific Questions](https://www.alphaxiv.org/abs/2512.01224)

**Tool Use in Multi-Agent Settings:**
- [RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents](https://alphaxiv.org/abs/2602.03025)
- [Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](https://alphaxiv.org/abs/2601.15625)
- [ToolRL: Reward is All Tool Learning Needs](https://alphaxiv.org/abs/2504.13958)
- [Advancing SLM Tool-Use Capability using Reinforcement Learning](https://alphaxiv.org/abs/2509.04518)
- [Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization](https://alphaxiv.org/abs/2511.14846)
- [Acting Less is Reasoning More! Teaching Model to Act Efficiently](https://alphaxiv.org/abs/2504.14870)

### 5.3 Reward Modeling (Rule-Based, Soft, Hybrid)

- [Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO](https://www.alphaxiv.org/overview/2509.13081)
- [A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future](https://www.alphaxiv.org/abs/2504.12328)
- [Using Semantic Similarity as Reward for Reinforcement Learning in Sentence Generation](https://aclanthology.org/P19-2056.pdf)
- [Crossing The Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains](https://arxiv.org/abs/2503.23829) (Mar 2025)
- [Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals](https://arxiv.org/abs/2502.19328) (Feb 2025) — Modular routers combining preferences + verifiable signals
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746) (Jul 2025) — Multi-criterion rubrics as explicit reward functions
- [Reward Hacking Mitigation using Verifiable Composite Rewards](https://arxiv.org/abs/2509.15557) (Sep 2025) — Sentence-BERT semantic detection for medical QA
- [From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning](https://arxiv.org/abs/2505.22203) (May 2025) — False negatives in rule, false positives in model verifiers
- [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](https://arxiv.org/abs/2505.15801) (May 2025) — First systematic benchmark for verifier reliability

### 5.4 Training Algorithms & Optimization

- [STEPHINT: MULTI-LEVEL STEPWISE HINTS ENHANCE REINFORCEMENT LEARNING TO REASON](https://www.alphaxiv.org/abs/2507.02841)
- [Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models](https://arxiv.org/abs/2506.13923) (Jun 2025)
- [First Return, Entropy-Eliciting Explore](https://www.alphaxiv.org/abs/2507.07017)
- [On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement](https://alphaxiv.org/abs/2512.04220)
- [GRPO's Effective Loss, Dynamics, and Success Amplification](https://arxiv.org/abs/2503.06639) (Mar 2025) — Theoretical analysis of GRPO under verifiable rewards
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning](https://arxiv.org/abs/2506.01939) (Jun 2025) — High-entropy "forking tokens" steer reasoning
- [Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration](https://arxiv.org/abs/2510.03865) (Oct 2025) — Forward-KL penalties for OOD search
- [From Trial-and-Error to Improvement: A Systematic Analysis of LLM Exploration in RLVR](https://arxiv.org/abs/2508.07534) (Aug 2025) — Identifies entropy collapse, PPL-based advantage shaping
- [Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping (UCAS)](https://arxiv.org/abs/2510.10649) (Oct 2025) — Confidence-based credit assignment
- [PACR: Progressively Ascending Confidence Reward for LLM Reasoning](https://arxiv.org/abs/2510.22255) (Oct 2025) — Dense stepwise signals
- [Shrinking the Variance: Shrinkage Baselines for RLVR](https://arxiv.org/abs/2511.03710) (Nov 2025) — James-Stein estimators for low-rollout variance reduction

### 5.5 Understanding RLVR Mechanisms

- [On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models](https://arxiv.org/abs/2506.04695) (Jun 2025)
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (Apr 2025)
- [The Invisible Leash? Why RLVR May or May Not Escape Its Origin](https://www.alphaxiv.org/abs/2507.14843)
- [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947) (Jun 2025)
- [Generalization of RLVR Using Causal Reasoning as a Testbed](https://www.alphaxiv.org/overview/2512.20760) (Dec 2025)
- [Breaking the Safety-Capability Tradeoff: RLVR Maintains Safety Guardrails in LLMs](https://arxiv.org/abs/2511.21050) (Nov 2025) — KL-constrained RLVR preserves safety

### 5.6 Extending to Non-Verifiable Domains

- [RLPR: Extrapolating RLVR to General Domains without Verifiers](https://arxiv.org/abs/2506.18254)
- [ENABLING TOOL USE OF REASONING MODELS WITHOUT VERIFIABLE REWARD VIA SFT-RL LOOP](https://openreview.net/pdf/1ca67ceed76207273bb57d9dc64f0ce06c209123.pdf)
- [EXTENDING RLVR TO OPEN-ENDED TASKS VIA VERIFIABLE MULTIPLE-CHOICE REFORMULATION](https://openreview.net/pdf?id=uZxyvmN72d)
- [Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](https://www.alphaxiv.org/abs/2510.00915)
- [Auditable-choice reframing unlocks RL-based verification for open-ended tasks](https://arxiv.org/abs/2511.02463) (Nov 2025) — Verifiable Multiple-Choice Reformulation for creative domains
- [Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards](https://arxiv.org/abs/2506.00103) — Pairwise GenRM (BRPO) for subjective domains

### 5.7 Domain Extension & Multi-Domain Learning

- [Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning](https://www.alphaxiv.org/abs/2507.17512)
- [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://www.alphaxiv.org/abs/2504.04736)

**Medical:**
- [Med-RLVR: Emerging Medical Reasoning from a 3B base model via RL](https://arxiv.org/abs/2502.19655) (Feb 2025)
- [Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain](https://arxiv.org/abs/2504.13950) (Apr 2025) — Difficulty-based filtering
- [Training LLMs for EHR-Based Reasoning Tasks via RL](https://arxiv.org/abs/2505.24105) (May 2025)

**Multimodal & Vision:**
- [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with RL](https://arxiv.org/abs/2503.05379) (Mar 2025)
- [Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards](https://arxiv.org/abs/2507.21745) (Jul 2025)
- [SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards](https://arxiv.org/abs/2505.19094) (May 2025)
- [MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM RL](https://arxiv.org/abs/2505.24871) (May 2025)
- [ManipLVM-R1: RL for Reasoning in Embodied Manipulation with Vision-Language Models](https://arxiv.org/abs/2505.16517) (May 2025)

**World Models & Forecasting:**
- [RLVR-World: Training World Models with Reinforcement Learning](https://arxiv.org/abs/2505.13934) (May 2025)
- [Outcome-based Reinforcement Learning to Predict the Future](https://arxiv.org/abs/2505.17989) (May 2025) — Binary/noisy reward forecasting

**Specialized Tasks:**
- [RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward](https://arxiv.org/abs/2505.10218) (May 2025)
- [Studying the Korean Word-Chain Game with RLVR: Mitigating Reward Conflicts via Curriculum Learning](https://arxiv.org/abs/2510.03394) (Oct 2025)

### 5.8 Agent-Based & Instruction-Following Approaches

- [Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards](https://arxiv.org/abs/2506.11425)
- [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://www.alphaxiv.org/abs/2511.14460)
- [Reinforcement Learning for Reasoning in LLMs with One Training Example](https://arxiv.org/abs/2504.20571) (Apr 2025) — Near-doubling of MATH500 with single curated example
- [IFDECORATOR: Wrapping Instruction Following RL with Verifiable Rewards](https://arxiv.org/abs/2508.04632) (Aug 2025) — Trap instructions + intent-alignment to mitigate reward hacking

### 5.9 Data Generation, Environments & Benchmarks

- [Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text](https://www.alphaxiv.org/abs/2601.22975) ⭐
- [SWE-Universe: Scale Real-World Verifiable Environments to Millions](https://www.alphaxiv.org/abs/2602.02361)
- [REASONING GYM: Reasoning Environments for RLVR](https://arxiv.org/abs/2505.24760) (May 2025) — 100+ procedural environments with verifiable checkers
- [Enigmata: Scaling Logical Reasoning in LLMs with Synthetic Verifiable Puzzles](https://arxiv.org/abs/2505.19914) (May 2025)
- [SHARP: Synthesizing High-quality Aligned Reasoning Problems for LRM RL](https://arxiv.org/abs/2505.14147) (May 2025) — Pipeline for diverse verifiable reasoning problems

### 5.10 Long-Context & Scaling

- [InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](https://www.alphaxiv.org/overview/2602.06960) ⭐
- [Learning to Reason in 13 Parameters](https://www.alphaxiv.org/abs/2602.04118) ⭐

### 5.11 Advanced Techniques & Miscellaneous

- [DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search](https://www.alphaxiv.org/abs/2509.25454) ⭐
- [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://www.alphaxiv.org/abs/2405.11106)
- [Reinforcement World Model Learning for LLM-based Agents](https://www.alphaxiv.org/abs/2602.05842)

### 5.12 Measurement, Evaluation & Systems

- [Position: The Hidden Costs and Measurement Gaps of RLVR](https://arxiv.org/abs/2509.21882) (Sep 2025) — "Tax-aware" evaluation: calibration, reliability, contamination
- [RL in the Wild: Characterizing RLVR Training in LLM Deployment](https://arxiv.org/abs/2509.25279) (Sep 2025) — PolyTrace benchmark for system-level optimization

### 5.13 Multi-Task & Hybrid Reward Frameworks

- [Multi-task RL in Reproducing Kernel Hilbert Spaces via Cross-learning](https://arxiv.org/abs/2008.11895) (2020) — Foundational multi-task RL with shared central policy
- [Multi Task Inverse Reinforcement Learning for Common Sense Reward](https://arxiv.org/abs/2402.11367) (Feb 2024) — Decomposes task-specific verifiable + shared common-sense rewards
- [Combining Reward Information from Multiple Sources](https://arxiv.org/abs/2103.12142) (2021) — Multitask Inverse Reward Design with uncertainty propagation
- [Learning Multi-Task Transferable Rewards via Variational Inverse RL](https://arxiv.org/abs/2206.09498) (Jun 2022) — Empowerment regularization for transferable rewards
- [Multi-Task Reward Learning from Human Ratings](https://arxiv.org/abs/2506.09183) (Jun 2025) — Adaptive weighting reflecting task uncertainty
- [Imperfect also Deserves Reward: Multi-Level and Sequential Reward Modeling for Dialog Management](https://arxiv.org/abs/2104.04748) (2021) — Hierarchical decomposition (domain/act/slot)

### 5.14 Theoretical Foundations: Symbolic, Compositional & Formal-Methods RL

- [Learning Intrinsic Symbolic Rewards in Reinforcement Learning](https://arxiv.org/abs/2010.03694) (Sheikh et al., 2020) — Interpretable symbolic trees as verifiable reward functions
- [Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification](https://arxiv.org/abs/2103.12656) (Eysenbach et al., 2021) — Learn directly from success examples
- [Verifiable and Compositional Reinforcement Learning Systems](https://arxiv.org/abs/2106.05864) (Neary et al., 2021) — Decompose global RL objectives into verifiable subsystems
- [Verifiable Reinforcement Learning Systems via Compositionality](https://arxiv.org/abs/2309.06420) (Neary et al., 2023) — Parametric MDPs with iterative adaptation
- [Omega-Regular Reward Machines](https://arxiv.org/abs/2308.07469) (Hahn et al., 2023) — Reward machines for safety/liveness/fairness specifications
- [Robustness Verification of Deep RL via Reward Martingales](https://arxiv.org/abs/2312.09695) (Zhi et al., 2023) — Provable cumulative-reward bounds under perturbations
- [Training Verifiably Robust Agents Using Set-Based Reinforcement Learning](https://arxiv.org/abs/2408.09112) (Wendl et al., 2024) — Zonotope-based reachability for certifiable safety

---

## 6. Supporting Resources & Reference Materials

### Implementation References

- **DAPO**: [An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- **R1-Zero RL Recipe**: [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)
- **Code Samples**: [reasoning-from-scratch GRPO implementations](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_original)

### Foundational Techniques

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [RLHF Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)

### Domain Adaptation Literature

- [Domain Specialization as the Key to Make Large Language Models Disruptive](https://arxiv.org/abs/2305.18703)
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://www.semanticscholar.org/paper/Don%E2%80%99t-Stop-Pretraining%3A-Adapt-Language-Models-to-Gururangan-Marasovi%C4%87/e816f788767eec6a8ef0ea9eddd0e902435d4271)
- [Domain-Adaptive Pre-Training (DAPT)](https://www.emergentmind.com/topics/domain-adaptive-pre-training-dapt-f56bf1f4-9ff0-4fb6-af76-587677db57c5)

### Frontier & Large-Scale Systems

- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [Olmo 3](https://arxiv.org/abs/2512.13961)

### Related Topics

- [The Era of Agentic Organization: Learning to Organize with Language Models](https://www.alphaxiv.org/overview/2510.26658)
- [Autonomous Agents for Scientific Discovery](https://www.alphaxiv.org/abs/2510.09901)

---

## Legend

⭐ — Highly relevant to your research (tool-use + search/reasoning integration)  
(Date) — Publication date for temporal context

---

# Appendices: Personal Annotations & Planning Notes

> The sections below preserve original research notes, personal commentary, paper-level summaries, status markers, and planning outlines from the unstructured working notes. Kept separate from the survey above to preserve thinking-as-it-happened.

---

## Appendix A: Per-Paper Annotations & Summaries

### A.1 Foundational Papers

**[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://www.alphaxiv.org/overview/2501.12948)** — Jan 4, 2025
- [DeepSeek-R1 (Supplementary)](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf)
- Foundational for RL and RLVR
- Showed that reasoning-like behaviour can be developed with RL — reasoning leads to better accuracy
- **Why RLVR?** Post-training (bottlenecked by requiring expensive written responses or preference labels) for SFT and RLHF
- **Reward model**:
  - The **accuracy reward** uses the LeetCode compiler to verify coding answers and a deterministic system to evaluate mathematical responses
  - The **format reward** relies on an LLM judge to ensure responses follow the expected format, such as placing reasoning steps inside `<think>` tags
- **PRM**: PRM demonstrates a good ability to rerank the top-N responses generated by the model or assist in guided search (Snell et al., 2024); its advantages are limited compared to the additional computational overhead it introduces during the large-scale reinforcement learning process in our experiments

**[DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://www.alphaxiv.org/abs/2511.22570)** — Nov 27, 2025
- ==READ==
- **Explanation scoring**: The way the explanations are currently being scored involves a second LLM. This leads to the other direction I am seeing for RLVR: an extension into other domains beyond math and code

### A.2 Domain-Specific Reproductions

**[MedRLVR](https://arxiv.org/abs/2502.19655)** — Feb 27, 2025
- RLVR with MCQ medical data as verifiable labels
- 3B base model
- Reward signals for: Correctness, Format compliance
- **Future direction?** Penalise short CoT

**[Crossing The Reward Bridge](https://arxiv.org/abs/2503.23829)** — Mar 31, 2025
- Generative soft model-based reward signals in unstructured free-form scenarios — z-score normalisation
- Uses objective reference answers
- Cross-domain reward model — ==CAN WE USE AN ENCODER AND SEMANTIC SIMILARITY?==
  - Created with SFT from data collected during RL exploration
- **RL**: Uses REINFORCE, REINFORCE++, RLOO
- **Future direction?** PRM, How to evaluate individual steps?

**[VerIF](https://arxiv.org/abs/2506.09942)** — June 11, 2025
- [VerIF: Verification Engineering for Reinforcement Learning in Instruction Following](https://www.alphaxiv.org/abs/2506.09942)
- Combines rule-based and LLM-based semantic verification
- ==Has an instruction-following dataset with signals==

**[Trust, But Verify: A Self-Verification Approach to RLVR](https://arxiv.org/abs/2505.13445)** — May 19, 2025
- Mathematics domain
- Simultaneously improves both problem-solving and self-verification (Integrated RL for both)
- Online-RL, Uses PPO
- Does problem-solving and self-verification in two steps
  - ==Why not combine both into one? Add a planning step too==
- **Future direction?** Expand to other domains, Use other policy-gradient algorithms, RAG or external-tool use for verification

**[Agent RLVR](https://arxiv.org/abs/2506.11425)** — Jun 13, 2025
- Inspired by human pedagogy
- The agent creates initial trajectories that are graded by code, and guidance is added
- Policy is updated based on RLVR on the rewards of guided trajectories
- Offline update with SFT, DPO

**==[ReSearch](https://www.alphaxiv.org/abs/2503.19470) — Sep 23, 2025== | AMAZING PLACE TO START**
- ==EXACTLY WHAT I WANTED TO DO==
- Uses GRPO and something similar to RLVR
- **Rule-based rewards**: Answer reward, Format reward
- Uses instruction-tuned models
- **Future directions?**
  - More reward signals other than F1 or exact match
  - More specialised databases
  - ==Not just search but tool-use?==
  - ==Maybe combine soft-rewards too?==
- Related GRPO/tool-use variants:
  - [RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents](https://alphaxiv.org/abs/2602.03025)
  - [Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](https://alphaxiv.org/abs/2601.15625)
  - [ToolRL: Reward is All Tool Learning Needs](https://alphaxiv.org/abs/2504.13958)
  - [Advancing SLM Tool-Use Capability using Reinforcement Learning](https://alphaxiv.org/abs/2509.04518)
  - [Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization](https://alphaxiv.org/abs/2511.14846)
  - [Acting Less is Reasoning More! Teaching Model to Act Efficiently](https://alphaxiv.org/abs/2504.14870)
  - [On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement](https://alphaxiv.org/abs/2512.04220)

### A.3 Mechanism & Understanding

**[On the Mechanism of Reasoning Pattern Selection in RL for LMs](https://arxiv.org/abs/2506.04695)** — Jun 5, 2025
- Optimises the selection of existing reasoning patterns, not creating new ones (distillation is better for new patterns)
- Slower optimisation of weaker models can be mitigated by SFT
- SFT + RLVR is amazing

**[Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)** — Apr 18, 2025
- Does not elicit new reasoning patterns
- Reasoning abilities originate from and are bounded by the base model
- Distillation can introduce new reasoning patterns
- **Future direction?** Improve RL paradigms — Continual scaling, ==Multi-turn agent-environment interaction==

**[Adaptive Guidance Accelerates RL of Reasoning Models](https://arxiv.org/abs/2506.13923)** — Jun 16, 2025
- ==LOOK INTO THIS==
- Adaptive guidance accelerates learning in RLVR, self-distillation and capability gain
- The guide-algorithm provides in-context guidance when all roll-outs fail, i.e. adaptive hints in prompts
- Off-policy importance-ratio correction (sampling) for stable training
- Guide-GRPO
- **Future direction?** Personalised guidance strategies

**[Spurious Rewards: Rethinking Training Signals in RLVR](https://www.alphaxiv.org/overview/2506.10947)** — Jun 12, 2025
- RLVR gains are dependent on the base model; even spurious rewards work with some models but not others
- High dependence on pre-training
- Code-reasoning may be the reason — ==maybe try a base code model==

**[Generalization of RLVR Using Causal Reasoning as a Testbed](https://www.alphaxiv.org/overview/2512.20760)** — Dec 23, 2025
- ==Says RLVR does not work for smaller models==
- Heavily dependent on the model capabilities
- Building strong reasoning foundations through pre-training or initial supervised fine-tuning may be essential before applying reinforcement learning techniques

### A.4 Reward-Free / Intrinsic Reward Approaches

**[Learning to Reason without External Feedback](https://www.alphaxiv.org/abs/2505.19590)** — May 26, 2025
- Uses internal self-certainty as a reward, uses the model's confidence (distributional verification)
- Self-certainty is the KL-divergence between the uniform distribution over the vocabulary and the predicted next token distribution
  - Metric is mode-seeking, encouraging concentrated probability distributions that indicate higher confidence, while being less biased toward longer sequences compared to perplexity-based measures
- Uses GRPO
- **Future direction?** ==A combination of internal and external rewards==

### A.5 Creative / Open-Ended Domains

**[Writing Zero](https://www.alphaxiv.org/overview/2506.00103)** — Jun 11, 2025
- Self-principled critique pairs (phase-dependent rewards) — Creates its own rules to judge the output
- For the creative writing domain
- **GenRM**: Made from SFT and rule-based RL to act as a judge
- **Bootstrapped Relative Policy Optimisation (BRPO)**: Pairwise comparison vs pointwise in GRPO, it bootstraps a reference (randomly) and compares all the group responses to this
- **Future direction?** Try expanding to new domains

### A.6 Infrastructure & Environments

**[REASONING GYM](https://arxiv.org/abs/2505.24760)** — May 30, 2025
- [REASONING GYM: Reasoning Environments for RLVR](https://www.alphaxiv.org/abs/2505.24760)
- Provides data generators and verifiers (over 100)
- Adjustable difficulty
- Curriculum learning is good

### A.7 Tool Use + Search (Annotated)

**[R1-Searcher: Incentivizing the Search Capability in LLMs via RL](https://www.alphaxiv.org/abs/2503.05592)**
- R1-Searcher utilises a **two-stage, outcome-based reinforcement learning** approach to autonomously incentivise LLMs to invoke external search systems during reasoning, without requiring supervised fine-tuning or distillation
- Stage-1 uses **retrieve-rewards** to teach the model how to invoke the search tool format, while Stage-2 uses **answer-rewards** to train it to integrate retrieved knowledge to solve complex questions correctly
- <mark style="background: #FF5582A6;">VERY IMPORTANT</mark>

**[Search-R1: Training LLMs to Reason and Leverage Search Engines with RL](https://www.alphaxiv.org/abs/2503.09516)**
- SEARCH-R1 extends reinforcement learning for reasoning by training LLMs to autonomously generate multi-turn search queries and process real-time retrieval results using an outcome-based reward function and retrieved token masking for stable optimization
- <mark style="background: #FF5582A6;">VERY IMPORTANT</mark>

### A.8 Data Generation

**[Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text](https://www.alphaxiv.org/abs/2601.22975)**
- Golden Goose synthesises unlimited RLVR tasks by transforming unverifiable internet text into multiple-choice "fill-in-the-middle" questions, where an LLM masks key reasoning steps and generates plausible distractors to create automatically verifiable training signals
- <mark style="background: #FF5582A6;">IMPORTANT</mark>

**[SWE-Universe: Scale Real-World Verifiable Environments to Millions](https://www.alphaxiv.org/abs/2602.02361)**
- SWE-Universe utilises an autonomous building agent powered by an efficient MoE model that employs iterative self-verification and in-loop hacking detection to automatically construct million-scale, multi-lingual, verifiable software engineering environments from GitHub pull requests
- <mark style="background: #ADCCFFA6;">LATER</mark>

### A.9 Long Context & Misc

**[InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via RL](https://www.alphaxiv.org/overview/2602.06960)**
- InftyThink+ is an end-to-end reinforcement learning framework that optimizes "infinite-horizon" iterative reasoning by teaching models to autonomously decide when to summarise intermediate thoughts and how to continue reasoning from those compressed states
- <mark style="background: #FF5582A6;">IMPORTANT</mark>

**[Learning to Reason in 13 Parameters](https://www.alphaxiv.org/abs/2602.04118)**
- <mark style="background: #FF5582A6;">IMPORTANT</mark>

**[Reinforcement World Model Learning for LLM-based Agents](https://www.alphaxiv.org/abs/2602.05842)**
- Reinforcement World Model Learning (RWML) is a self-supervised method that trains LLM agents to predict action-conditioned next states by minimising the "sim-to-real" gap in a pre-trained embedding space using reinforcement learning
- <mark style="background: #BBFABBA6;">LATER</mark>

---

## Appendix B: Domain Adaptation — Thesis Sketch

### Framing

- **Who?** Industry
- **What?** Specialised LLMs
- **Where?** Domain-specific tasks
- **Why?** Domain adaptation is crucial for private industries that can't use ChatGPT due to proprietary data and to avoid data leaks

### The Pipeline

- **Domain Adaptation?**
  - We do DAPT with LoRA for language understanding the jargon
  - We do SFT for understanding behaviour (we can do distillation based on a large model and domain data to create synthetic QA pairs, maybe an encoder that finds the answer in data)
  - Can we elicit domain-specific reasoning by RL?

### Citation Targets

- [Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey](https://arxiv.org/abs/2305.18703) — Check this to cite this information
- Pre-training is a bottleneck; find a way to cite
- DAPT will help:
  - [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://www.semanticscholar.org/paper/Don%E2%80%99t-Stop-Pretraining%3A-Adapt-Language-Models-to-Gururangan-Marasovi%C4%87/e816f788767eec6a8ef0ea9eddd0e902435d4271)
  - [DAPT topic overview](https://www.emergentmind.com/topics/domain-adaptive-pre-training-dapt-f56bf1f4-9ff0-4fb6-af76-587677db57c5)

### The Methodology

We then do SFT with synthetic data created from unstructured text. To understand expected behaviour.

**How?** We propose a method to elicit domain-specific reasoning through RL based on RLVR and GRPO on a small 3B foundational model (maybe instruction-tuned) by exploring how to create a hybrid verifier model from domain data.

### Guiding Quotes

> Domain Adaptation is not "Domain adaptation means the model must internalize the domain." but is "Domain adaptation means the model must learn how to _interact with_ the domain."

> Domain adaptation = policy adaptation over tools  
> Can we get domain-based reasoning by enabling tool use? Tool use + MCTS to learn domain reasoning

---

## Appendix C: Introduction Outline (Planning)

Outline for the paper's Introduction section.

### History
- [RLHF Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)
- Contains step-wise alternatives, starts with InstructGPT.

### RLHF
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3.
- This is how alignment started.

### DPO
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- Works like an upgrade from RLHF, easier to work with and more stable.
- This is the next step into alignment.

### LoRA
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- This is how we are going to fine-tune the model here.
- Why is pre-training so costly? Find more citations for why LoRA is used.

### DAPO — ==GRPO==
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- RL at scale framework with data and training code.
- Maybe a reference I can use.

### R1-Zero RL recipe, Dr.GRPO
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- Has code for basic recipe.

### DeepSeek-V3.2 — ==GRPO==
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- This shit seems to be sota. NEED TO CHECK OUT THE PAPER.
- Scalable Reinforcement Learning Framework
- Large-Scale Agentic Task Synthesis Pipeline: To integrate reasoning into tool-use scenarios, we developed a novel synthesis pipeline that systematically generates training data at scale.

### Olmo
- [Olmo 3](https://arxiv.org/abs/2512.13961)
- May have GRPO pipelines explained.
- This release includes the entire model flow, i.e., the full lifecycle of the family of models, including every stage, checkpoint, data point, and dependency used to build it.

### Code Sample for RLVR — GRPO Raschka
- https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_original

---

## Appendix D: Additional Resources & Reading Queue

### Reward Modelling Background
- [Reward Models — Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/reward-models)
- [A Comprehensive Survey of Reward Models](https://www.alphaxiv.org/abs/2504.12328)

### Datasets with Verifiable Signals
- [Facebook Research Plan Generation](https://huggingface.co/datasets/facebook/research-plan-gen)
  - Has Rubrics (may be useful)
  - Has reference solutions

### EmergentMind Topic Pages
- [ ] [Verifiable Rewards (RLVR) Framework](https://www.emergentmind.com/topics/verifiable-rewards-rlvr) — Dec 2025
- [ ] [RISE: Online Self-Verification for RL Models](https://www.emergentmind.com/topics/online-self-verification-rise) — Dec 2025
- [ ] [Reinforcement Learning with Verified Reward (RLVR)](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-reward-rlvr) — Aug 2025
- [ ] [Multi-tasking: Verifiable & Non-verifiable Rewards](https://www.emergentmind.com/topics/multi-tasking-with-verifiable-and-non-verifiable-rewards) — July 2025
- [ ] [Verifiable Rewards in Reinforcement Learning](https://www.emergentmind.com/topics/reinforcement-learning-with-verifiable-rewards-paradigm) — June 2025
- [ ] [Reinforcement Learning with Verifiable Rewards](https://www.emergentmind.com/topics/reinforcement-learning-with-verifiable-rewards-rlvr) — June 2025

### Connected Papers Graphs
- [Spurious Rewards: Rethinking Training Signals in RLVR — Connected Papers](https://www.connectedpapers.com/main/f3ec0ee1538796c8d5e5633958106fbe394a32ad/Spurious-Rewards%3A-Rethinking-Training-Signals-in-RLVR/graph)
  - <mark style="background: #FF5582A6;">CHECKOUT LATER</mark>

### Examples / Reproductions
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero/)
  - Exhibits some emergent self-verification abilities, supporting the idea that reasoning can emerge through pure RL, even in small models
  - Replicates the DeepSeek-R1-Zero approach
  - Costs less than $30 to train
  - 3B base LM develops self-verification and search abilities all on its own

---

## Appendix E: Open Problems (Original Form)

Original brain-dump of open problems before consolidation into Section 3.

- **Process Reward Models**
  - How to add rewards for each reasoning step?
  - The next logical step is to not only use the final answer's correctness as a reward signal but also judge the LLM's explanations during RLVR training.
  - This has been done before, for many years in the past, under the research label "process reward models" (PRMs).
- **Monte Carlo Tree Search**
  - How to incorporate MCTS in the RLVR training phase?
  - [DeepSearch: Overcome the Bottleneck of RLVR via MCTS](https://www.alphaxiv.org/overview/2509.25454)
- **Curriculum Learning**
  - Can curriculum learning make a difference? NEED TO VERIFY
- **Rhetoric**
  - Can games like Werewolf improve rhetoric?

### Original Questions

- **Reward modelling for non-deterministic domains?**
  - Rule-based: Unit tests, math checker
  - Semantic matching, encoder to compute similarity
    - [Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO](https://www.alphaxiv.org/overview/2509.13081)
  - Latent state comparison
  - Keyword matching
  - Evidence grounded with RAG
  - Constraints (predefined rules or on-the-fly generation of rules)
  - Schema compliance (format compliance) — Regex
  - Pair-wise ranking (Bradley-Terry style) — In PRM?
  - Hybrid setup?
- Can a small 1B model post-trained using RLVR and hybrid keyword, semantic reward modelling be used as a research assistant?
  - [Facebook Research Plan Generation](https://huggingface.co/datasets/facebook/research-plan-gen) — Has Rubrics (may be useful), Has reference solutions
- Can RLVR with PRM using semantic reward modelling be used to improve reasoning in a smaller model?
- Does negative sampling with SFT for cold-start help?
- ==NAIVE==
  - How can we bridge the gap between RLVR and non-verifiable domains?
  - Can we use Process Reward Modelling as a bridge between RLVR and non-verifiable domains?
  - How can we extend RLVR into non-verifiable domains?
