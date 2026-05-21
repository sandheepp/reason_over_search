---
title: ORIGINAL PLAN B
tags: [historical]
source: internal
created: 2026-05-04
updated: 2026-05-07
---

> **HISTORICAL (frozen Feb 2026, archived 2026-05-07).** RQ1–RQ4 (domain expansion, reward modeling, meta-reasoning, curriculum) were consolidated 2026-05-04 into a single reframed question — *"Is it feasible to post-train a small LM (Qwen3.5-2B) to Search-R1-level results under realistic resource constraints, and what is the optimised training recipe?"* — after Phase-2 wall-clock made the original sweep infeasible (11–17 days/run on 1× A100-80GB). Proposed recipe (E2H + S-GRPO + MC-GRPO + JustRL plain-GRPO control): see [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 5–6.

# Tool-use to bridge verifiable and non-verifiable domains

## Topic
Reinforcement Learning Verifiable Rewards (RLVR), along with the Group Relative Policy Optimisation (GRPO), has shown that we can elicit reasoning-like behaviour in Large Language Models (LLMs) in verifiable or deterministic domains solely through Reinforcement Learning [^1]. Reinforcement Learning Verifiable Rewards (RLVR) has been proven to work across models of various sizes. We choose to work with models ranging from 100M to 3B due to compute constraints and to achieve realistic reproducibility in a domain-specific setting. We model the non-verifiable domain as an environment and use tool-use as a surrogate objective to generate reward signals within it. Here, we explore the possibilities of tool-use as a means of expanding into non-verifiable domains.
## Problem 
Can we extend Reinforcement Learning Verifiable Rewards (RLVR) to non-verifiable domains by using tool use to model a surrogate objective?
RQ1: What are the current approaches being explored for domain expansion?
RQ2: How can we model reward functions to do this?
RQ3: Can we use meta-reasoning to improve the training process?
RQ4: Is it possible to set up a curriculum-based approach to streamline training?
## Methodology
1. Domain (data) and Evaluation Metrics Selection
2. Pipeline Design and Implementation
	1. Base Model Selection
	2. Model Implementation
	3. Training Algorithm Design
	4. Training Algorithm Implementation
3. Result Analysis
## Planning
1. Literature Review (February)
2. Pipeline Implementation (March)
3. Experimentation (April)
4. Analysis (May)
5. Writing (June)

[^1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)