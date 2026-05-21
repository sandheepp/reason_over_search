---
title: SURVEY OVERFLOW
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# SURVEY_OVERFLOW: Adjacent Papers Not Discussed in Main Prose

> Companion to [SURVEY.md](SURVEY.md) and [SURVEY_FOCUSED.md](SURVEY_FOCUSED.md). These 15 papers are in the bibliography but were not part of the thematic narrative of the main survey, nor are they directly project-critical for the small-LM search-augmented thesis. They are kept here as one-off cards for completeness — useful background, possible future citations, or alternative directions to explore later.
>
> Categories:
> - **Foundational priors** (LoRA, InstructGPT, DPO) — predate RLVR but underlie its scaffolding.
> - **Surveys** (LLM MARL, Reward Models) — useful pointers for related-work positioning.
> - **Adjacent techniques** (SWiRL, Semantic Reward, Noisy Verifiers, AsyncThink, RWML, TinyLoRA) — relevant in spirit but not directly applied in the thesis.
> - **Adjacent agents** (Autonomous Agents, BAPO Boundary, BAPO Buffer) — multi-agent / safety / off-policy variants beyond core scope.
> - **Adjacent training studies** (Multi-Domain RL) — would matter if the thesis expanded beyond search-augmented QA.

---

## 1. Foundational Priors

### 2106.09685 — LoRA: Low-Rank Adaptation

**Summary.** LoRA freezes a pretrained transformer and injects trainable low-rank update matrices into each layer, drastically cutting memory and parameter count for fine-tuning. The contribution is a parameter-efficient adapter that incurs no inference latency and matches full fine-tuning quality.

**Problem.** Full fine-tuning of large LMs is memory-prohibitive and produces a separate full-size copy per task.

**Method.** For each weight matrix W, LoRA learns a low-rank product BA (with rank r typically 4-64) and replaces W with W + BA during the forward pass; only B and A are trained, while merging them into W at deployment removes any latency.

**Result.** Reduces trainable parameters by 10,000x and GPU memory by 3x versus Adam fine-tuning of GPT-3 175B, while matching or beating full fine-tuning quality on RoBERTa, DeBERTa, GPT-2, GPT-3.

**Takeaway.** On a single A100 doing GRPO on a 1-3B model, LoRA (or QLoRA) is the default lever for fitting reference + actor + value heads in memory without sacrificing quality.

**ELI5.** Instead of repainting the whole house for every guest, slide a thin patterned overlay over the wallpaper that you can swap in seconds.

### 2203.02155 — InstructGPT (RLHF)

**Summary.** InstructGPT aligns GPT-3 to user intent by collecting labeler demonstrations for SFT, then preference rankings to train a reward model, then PPO against that reward model. The contribution is the canonical three-stage RLHF recipe, and the demonstration that a 1.3B aligned model can beat the 175B base.

**Problem.** Pretrained LMs produce untruthful, toxic, or unhelpful outputs; scaling parameters alone does not fix intent following.

**Method.** Stage 1 SFT on labeler demos; stage 2 reward model trained on pairwise preference rankings; stage 3 PPO fine-tunes the SFT policy against the reward model with a KL penalty back to SFT.

**Result.** Human labelers prefer 1.3B InstructGPT outputs over those of 175B GPT-3, despite a 100x parameter gap.

**Takeaway.** The KL-to-reference + RM + PPO scaffolding is the direct ancestor of GRPO/RLVR; verifiable rewards simply replace the learned RM, but the KL anchor and on-policy sampling discipline are still load-bearing.

**ELI5.** Like training a new intern by first showing examples, then having a senior rate their drafts, then letting them practise daily with that critic on their shoulder.

### 2305.18290 — DPO: Direct Preference Optimization

**Summary.** DPO derives a closed-form mapping from preference data to the optimal RLHF policy, replacing PPO with a simple classification loss on preference pairs. The contribution is a stable, sampling-free alternative to RLHF that matches or beats PPO without a separate reward model.

**Problem.** PPO-based RLHF is unstable, requires online sampling, and needs a separately trained reward model with careful KL tuning.

**Method.** Reparameterise the RLHF objective so the implicit reward is log(pi/pi_ref); maximise the Bradley-Terry likelihood of the preferred-over-dispreferred pair under this reward, yielding a supervised binary cross-entropy loss on the policy directly.

**Result.** Matches or outperforms PPO-RLHF on sentiment control, summarisation, and single-turn dialogue (no headline number in abstract).

**Takeaway.** When verifiable rewards are unavailable but pairwise preferences are, DPO is the cheap, stable default; for RLVR with binary outcomes you can also recast samples as pairs and use a DPO-style loss to avoid PPO machinery.

**ELI5.** Instead of training a critic, then training a chef to please the critic, just tell the chef "this dish was preferred to that one" and update them directly.

---

## 2. Surveys

### 2405.11106 — LLM-based Multi-Agent RL Survey

**Summary.** This is a survey/letter mapping current LLM-based single-agent RL frameworks to the multi-agent setting and proposing future directions. The contribution is a structured taxonomy covering coordination, communication, and human-in/on-the-loop scenarios for LLM MARL.

**Problem.** LLM-as-agent work is largely single-agent; extending to multi-agent systems requires new mechanisms for coordination and inter-agent communication that current RL frameworks do not provide.

**Method.** Literature survey: classify existing LLM-RL frameworks, identify the missing components (communication protocols, role assignment, joint credit assignment) needed for cooperative multi-agent extension, and outline open directions.

**Result.** No empirical results; surveys the space and proposes a research agenda.

**Takeaway.** If your thesis ever extends a single search-augmented agent to a planner+searcher+verifier setup, this is the canonical entry point for the open problems in LLM MARL.

**ELI5.** Like a review article asking "we know how to coach one chess player; how do we coach a whole team that has to talk to each other mid-game?"

### 2504.12328 — Survey of Reward Models

**Summary.** A comprehensive survey of reward models for LLMs, organized by preference collection, reward modeling, and downstream usage. Covers benchmarks, applications, and open challenges.

**Problem.** The reward-model literature is fragmented across RLHF, RLAIF, process supervision, and verifiable rewards with no unified taxonomy.

**Method.** Taxonomic survey across three axes (data collection, modeling, usage) with discussion of evaluation benchmarks and gaps.

**Result.** No headline metric (survey).

**Takeaway.** Use this as the canonical reference when positioning your verifiable-reward design against scalar/process/generative reward-model alternatives in your related-work section.

**ELI5.** Like the Michelin Guide for graders; it does not cook anything itself, it just maps every restaurant in town so you know which to try.

---

## 3. Adjacent Techniques

### 2504.04736 — SWiRL

**Summary.** SWiRL is a step-wise RL methodology for multi-step reasoning and tool-use trajectories. It synthesizes multi-step data, decomposes each trajectory into per-action sub-trajectories, filters them, and runs RL on the sub-trajectories.

**Problem.** RLHF/RLAIF treat generation as a single step, but agentic tool use requires many decisions whose intermediate quality must be credited individually.

**Method.** Generate multi-step rollouts, split into one sub-trajectory per action, filter by simple heuristics, then run RL on each sub-trajectory so each action gets its own gradient signal.

**Result.** Relative gains of +21.5% (GSM8K), +12.3% (HotPotQA), +14.8% (CofCA), +11.1% (MuSiQue), +15.3% (BeerQA); training only on HotPotQA boosts zero-shot GSM8K by +16.9%.

**Takeaway.** For a small multi-hop search agent, splitting trajectories into per-tool-call sub-trajectories before RL gives you both denser credit assignment and surprising cross-task transfer.

**ELI5.** Like reviewing a chess game move-by-move instead of just noting the final result; you learn what each individual move contributed even if the overall game was won or lost.

### 2509.13081 — Semantic Reward Modeling with Encoder Transformers for GRPO

**Summary.** Uses a small encoder-only transformer to score cosine similarity between generated and reference explanations as a dense GRPO reward. Demonstrated on Italian medical-school exam explanations after CPT and SFT.

**Problem.** ROUGE-style rewards miss semantics, while LLM-as-judge rewards are slow and expensive, especially for explanation-quality tasks.

**Method.** Replace the GRPO reward with cosine similarity between embeddings of the generated explanation and a ground-truth reference, using a lightweight encoder transformer.

**Result.** Improves explanation faithfulness and clarity over a strong SFT baseline on the Italian med-school task (no specific numbers).

**Takeaway.** For small-LM RLVR on explanation tasks, an encoder-embedding reward is the cheap middle ground between ROUGE and LLM-as-judge; fast enough to use online and semantic enough to actually steer the policy.

**ELI5.** Like swapping out a slow human grader for a small, fast similarity sensor that says "this answer means roughly the same thing as the answer key."

### 2510.00915 — RLVR with Noisy Verifiers: Bias Corrections

**Summary.** Models RLVR's binary verifier as a noisy channel with asymmetric false-positive and false-negative rates, and derives two lightweight corrections (backward and forward) for the policy gradient. An online appeals mechanism estimates the FN rate during training.

**Problem.** Real verifiers introduce false negatives (rejecting correct answers) and false positives (accepting wrong ones); naive RLVR optimizes a biased reward without acknowledging this.

**Method.** Treat reward as a stochastic channel with rates ρ₀, ρ₁; (i) backward correction yields an unbiased surrogate reward; (ii) forward correction reweights score-function terms to align gradient direction (needs only FN rate). Both implemented as hooks in GRPO; an LLM-verifier appeals loop estimates the FN rate online.

**Result.** Both corrections improve RLVR for math reasoning under synthetic and real verifier noise; forward correction is more stable under heavier noise (no specific numbers).

**Takeaway.** If your small-LM verifier has even modest false-negative rate (common with regex/EM judges on free-form answers), drop in the forward correction; it costs almost nothing and unbiases the gradient.

**ELI5.** Like a basketball ref who you know miscalls 10% of fouls; instead of arguing each call, you mathematically inflate everyone's score by the right amount so the standings stay fair.

### 2510.26658 — AsyncThink: Asynchronous Thinking for Agentic Orgs

**Summary.** AsyncThink treats internal LLM reasoning as a small organisation: an organiser dynamically dispatches sub-queries to worker threads, merges results, and is trained end-to-end with RL. The structure of thought itself becomes a learned, concurrent program.

**Problem.** Sequential chain-of-thought is slow, and naive parallel thinking wastes compute on independent branches that never coordinate.

**Method.** A learned organiser policy issues sub-queries to concurrent workers and merges intermediate knowledge; the dispatch/merge protocol is optimised by RL so the model learns when to fan out and when to synchronise.

**Result.** 28% lower inference latency than parallel thinking with improved math reasoning accuracy, and generalises zero-shot to unseen tasks.

**Takeaway.** Worth watching if you want test-time compute scaling beyond serial CoT; the organiser/worker protocol is a cleaner abstraction than ad-hoc tree search for RLVR rollouts.

**ELI5.** A chef who learns to call out tasks to multiple line cooks at once and only re-syncs when the dishes need plating, instead of cooking everything one pot at a time.

### 2602.04118 — TinyLoRA: Reasoning with 13 Parameters

**Summary.** TinyLoRA scales low-rank adapters far below rank-1, down to a single parameter, and trains them with RL to elicit reasoning. With only 13 trainable parameters, an 8B Qwen2.5 hits 91% on GSM8K.

**Problem.** Standard LoRA cannot scale below model dimension, and it is unclear whether reasoning gains require parameter updates of the size LoRA assumes.

**Method.** A new sub-rank-1 parameterization that allows scaling adapter size down to a single parameter; trained via RL (not SFT, which needs 100-1000x more parameters for the same quality).

**Result.** 91% GSM8K with 13 bf16 parameters (26 bytes total); recovers 90% of gains with 1000x fewer parameters across AIME/AMC/MATH500.

**Takeaway.** RLVR primarily nudges a tiny low-rank direction in parameter space; for small-LM thesis work this means RL elicits latent capability rather than installing new knowledge, and absurdly small adapters suffice if you use RL.

**ELI5.** Reasoning ability is already inside the model; RL just flicks a few hidden switches, and TinyLoRA shows there are literally only a handful of switches that matter.

### 2602.05842 — RWML: Reinforcement World Model Learning

**Summary.** RWML trains LLM agents to anticipate the next textual state by rewarding alignment between the model's simulated next-state and the actual environment next-state in an embedding space. This world-modeling signal is self-supervised and complements task-success rewards.

**Problem.** LLM agents struggle to anticipate action consequences, and next-state token prediction overfits to surface wording while LLM-as-a-judge invites reward hacking.

**Method.** The agent generates a predicted next textual state; a sim-to-real reward is computed from the embedding-space distance between the prediction and the realized state, then optimized via RL alongside or instead of task rewards.

**Result.** +6.9 points on ALFWorld and +5.7 points on tau2-Bench over direct task-success RL when combined with task rewards.

**Takeaway.** Embedding-space next-state prediction is a hack-resistant auxiliary reward for small agentic models when verifiable task rewards are sparse.

**ELI5.** Before each move in chess, predict what the board will look like; you get rewarded for being directionally right rather than for memorizing the exact pixel of every piece.

---

## 4. Adjacent Agents

### 2510.09901 — Autonomous Agents for Scientific Discovery

**Summary.** A position/survey paper sketching how LLM agents are reshaping the scientific discovery loop from hypothesis to refinement, by orchestrating scientists, language, code, and physics. It catalogues current methods, achievements, and open challenges for building robust scientific agents.

**Problem.** No unifying view exists for how heterogeneous LLM-agent capabilities (planning, code, lab control, paper writing) integrate across the discovery lifecycle.

**Method.** Conceptual framework + literature review covering hypothesis generation, experimental design, execution, and result analysis, with critical examination of limitations and open challenges (no new training method).

**Result.** No numerical result; this is a vision/survey paper.

**Takeaway.** Useful for framing related-work prose on agentic science; not directly relevant to RLVR algorithm design but handy for citing the broader agent-orchestration motivation.

**ELI5.** Less a recipe and more a city map showing where the labs, libraries, and notebooks of LLM-driven science currently sit and which streets still have potholes.

### 2601.11037 — BAPO: Boundary-Aware Policy Optimization

**Summary.** BAPO is an RL framework that teaches agentic search LLMs to say "I DON'T KNOW" when their evidence or reasoning runs out, rather than confidently hallucinating. It adds boundary-aware rewards to GRPO without sacrificing answer accuracy.

**Problem.** RL-trained search agents almost never abstain, producing plausible-but-wrong answers that are dangerous in real deployments.

**Method.** A group-based boundary-aware reward grants credit for IDK only when reasoning genuinely hits its limit, and an adaptive reward modulator suspends this signal during early exploration so the policy can't shortcut by always abstaining.

**Result.** Substantial reliability gains across four agentic-search benchmarks (no headline number in the abstract).

**Takeaway.** When training a small search-augmented model with GRPO, you can add an abstention action and a phased reward schedule to get calibration without losing EM; just don't reward IDK before the model has learned to try.

**ELI5.** Like teaching a student to leave a question blank only when they have honestly exhausted the textbook, while preventing them from learning that "blank" is the easy way to a passing grade.

### 2602.20722 — BAPO: Off-Policy Buffer for RLVR

**Summary.** This BAPO (distinct from the boundary-aware one) is an off-policy RLVR framework that maintains a replay buffer of historically difficult samples and re-evaluates them during training, with a lower-bound guarantee on policy improvement. It targets data efficiency and stuck-sample resolution.

**Problem.** On-policy RLVR wastes experience and sees uniform reward distributions on hard prompts, causing learning to stall on the samples that matter most.

**Method.** Dynamic batch selection re-evaluates historically difficult samples and reuses high-quality ones off-policy, with a theoretical lower bound preventing destructive updates.

**Result.** +12.5% average over GRPO across math/planning/visual-reasoning; resolves 40.7% of problems base models always fail.

**Takeaway.** A principled off-policy buffer can rescue the long tail of "always-failed" prompts that pure on-policy GRPO never learns from — useful when you cannot afford fresh rollouts every step.

**ELI5.** Keep a folder of the homework problems you keep getting wrong and revisit them periodically; you would otherwise just keep practicing the easy ones in your daily set.

---

## 5. Adjacent Training Studies

### 2507.17512 — Multi-Domain Reasoning via RL: Data-Centric Study

**Summary.** A systematic GRPO study on Qwen2.5-7B across math, code, and logic puzzles, examining single-domain training, cross-domain transfer, and combined training. The contribution is an empirical map of when domains help versus conflict under RLVR.

**Problem.** RLVR research is siloed per domain, leaving open how training on one reasoning skill transfers to or interferes with another.

**Method.** Train Qwen2.5-7B base and instruct with GRPO on each of {math, code, logic}, then on combinations, while ablating curriculum, reward design, language, and SFT-vs-base initialization.

**Result.** No single headline number; the paper reports both mutual enhancements and conflicts across domains, with curriculum and reward design materially affecting transfer.

**Takeaway.** For a small-LM RLVR mix, do not assume "more domains = more general"; expect interference and treat domain mixture and curriculum as first-class hyperparameters.

**ELI5.** Like a triathlete who learns that biking and running help each other but heavy swim training quietly steals from both, depending on the schedule.

---

## Bibliography

| Paper | Category | arXiv |
|-------|----------|-------|
| LoRA | Foundational | [2106.09685](https://arxiv.org/abs/2106.09685) |
| InstructGPT | Foundational | [2203.02155](https://arxiv.org/abs/2203.02155) |
| DPO | Foundational | [2305.18290](https://arxiv.org/abs/2305.18290) |
| LLM MARL Survey | Survey | [2405.11106](https://arxiv.org/abs/2405.11106) |
| Reward Models Survey | Survey | [2504.12328](https://arxiv.org/abs/2504.12328) |
| SWiRL | Adjacent technique | [2504.04736](https://arxiv.org/abs/2504.04736) |
| Semantic Reward (encoder) | Adjacent technique | [2509.13081](https://arxiv.org/abs/2509.13081) |
| Noisy Verifier Corrections | Adjacent technique | [2510.00915](https://arxiv.org/abs/2510.00915) |
| AsyncThink | Adjacent technique | [2510.26658](https://arxiv.org/abs/2510.26658) |
| TinyLoRA | Adjacent technique | [2602.04118](https://arxiv.org/abs/2602.04118) |
| RWML | Adjacent technique | [2602.05842](https://arxiv.org/abs/2602.05842) |
| Autonomous Agents Survey | Adjacent agents | [2510.09901](https://arxiv.org/abs/2510.09901) |
| BAPO Boundary-Aware | Adjacent agents | [2601.11037](https://arxiv.org/abs/2601.11037) |
| BAPO Off-Policy Buffer | Adjacent agents | [2602.20722](https://arxiv.org/abs/2602.20722) |
| Multi-Domain Reasoning RL | Adjacent training | [2507.17512](https://arxiv.org/abs/2507.17512) |
