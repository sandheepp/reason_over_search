---
title: ORIGINAL PLAN A
tags: [historical]
source: internal
created: 2026-05-04
updated: 2026-05-07
---

> **HISTORICAL (frozen Feb 2026, archived 2026-05-07).** This is the original Plan A draft (three threads: reward model design, meta-reasoning tags, curriculum + transfer learning). It was superseded on 2026-05-04 by a single reframed research question after Phase-2 wall-clock reality (11–17 days/run on 1× A100-80GB) made paired sweeps unaffordable. Active question and proposed recipe (E2H + S-GRPO + MC-GRPO + JustRL plain-GRPO control): see [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 5–6.

## Abstract

Large Language Models (LLMs) have demonstrated enhanced reasoning abilities solely through reinforcement learning (RL) in verifiable domains such as mathematics and programming. Recent studies have shed light on LLMs' capabilities for interacting with the environment through tool use to solve complex problems. In this study, we will explore whether integrating external tool use with reasoning is a suitable approach to achieve domain adaptation for smaller LLMs (1B to 3B). To achieve this, we wish to explore 1) reward model design (binary vs soft vs reward-free vs hybrid), 2) incorporating meta-reasoning like `<tool_use>...</tool_use>` along with `<thinking>...</thinking>` in the reasoning process, and 3) curriculum learning and transfer learning. We aim to verify if advanced domain-based reasoning, such as reflection, verification and self-correction, develops in small LLMs during the reinforcement learning (RL) process.

## Introduction

In recent years, Large Language Models (LLMs) have proved excellence in a wide range of tasks and have exhibited generalist capabilities. But the adoption of these generalist models in many domains is still slow because of many reasons, some of which are 1) proprietary data, 2) security and privacy, and 3) the knowledge cut-off problem [^1]. Pre-training or fine-tuning larger models is computationally expensive and may not be feasible for many domains, so one possible way forward is to align smaller models using Parameter-Efficient Fine-Tuning (PEFT) [^1] methods such as Low-rank Adaptation (LoRA) [^1]. Reinforcement learning (RL) has established itself as a crucial factor in aligning Large Language Models (LLMs) to human preferences. InstructGPT [^2] is an example of this, where a small model can outperform larger models in evaluations. Reinforcement Learning (RL) has come a long way since then, from Reinforcement Learning Human Feedback (RLHF), which used a complex reward model trained from costly annotated preference data, to Reinforcement Learning Verifiable Rewards (RLVR), which uses a much simpler deterministic reward model. DeepSeek-R1 was pivotal in many ways, the most important of which was demonstrating that reasoning-like behaviour can be elicited from pure reinforcement learning (RL) without reliance on human-annotated reasoning trajectories [^3]. The rule-based reward system worked wonders in mathematics and programming. Further work, such as MedRLVR [^4] and stochastic soft-reward mechanisms [^5], demonstrated that this paradigm could be extended to more domains. 

Reinforcement Learning Verifiable Rewards (RLVR) has seen significant advancements in the past year and is expected to see many more this year [^6]. We plan to explore Reinforcement Learning Verifiable Rewards (RLVR) as a methodology for efficient domain adaptation. Supervised Fine-Tuning (SFT) could be seen as a form of memorisation, and Reinforcement Learning (RL) as a means of aligning with preferences. In this context, we ask the question "Does domain adaptation mean learning a way to interact with the domain (eg, tool use) rather than internalising the domain?"

## Related Work

1) Reward model design (binary vs soft vs reward-free vs hybrid)
   Recent work has shown that we could use verifiable answers, such as multiple choice question answers (MCQA), to create reward models [^4]. There have also been studies that showed gold labels can be used to create soft rewards [^5]. This shows that reward modelling is a possible avenue to explore further. There have also been studies that use intrinsic properties such as internal token probabilities, for example, RLPR [^7]. Some use metrics such as self-certainty, calculated as the KL-divergence between a uniform distribution and the next-token distribution [^8]. Some studies use rubrics created for evaluation as a means to generate rewards [^9]. There have been several further advancements showing promise in reward model design and, hence, merit exploration as a possible avenue.
2) Incorporating meta-reasoning like `<tool_use>...</tool_use>` along with `<thinking>...</thinking>` in the reasoning process
   Past research has shown that the use of meta-reasoning improves reasoning in Large Language Models (LLMs) [^10]. "ReSearch" [^11] has shown that the use of meta-tags with search tools improves reasoning. This provides a strong precursor to our research and suggests we could obtain positive results.
3) Curriculum learning and transfer learning
   Pedagogically structured teaching has always proven to be effective. Step-wise curriculum approach has been shown to make significant results in Large Language Models [^12][^13]. Transfer learning is a phenomenon in which skills from one domain help with those in another. "REASONING GYM" [^14] provides use cases with verifiable rewards that can be used to cultivate skills that might translate to improvement in the specified domain.

[^1]: [Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey](https://arxiv.org/abs/2305.18703)
[^2]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
[^3]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
[^4]: [Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning](https://arxiv.org/abs/2502.19655)
[^5]: [Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains](https://arxiv.org/abs/2503.23829)
[^6]: [Multi-Step Reasoning with Large Language Models, a Survey](https://arxiv.org/abs/2407.11511)
[^7]: [RLPR: Extrapolating RLVR to General Domains without Verifiers](https://arxiv.org/abs/2506.18254)
[^8]: [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590)
[^9]: [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](https://arxiv.org/abs/2507.17746v1)
[^10]: [RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844)
[^11]: [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) - ==HIGHLY INFLUENTIAL==
[^12]: [Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards](https://arxiv.org/abs/2506.11425)
[^13]: [How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study](https://arxiv.org/abs/2504.00829v1)
[^14]: [REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards]()