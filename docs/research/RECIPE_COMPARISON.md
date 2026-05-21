---
title: Recipe comparison - Search-R1 vs R1-Searcher vs ReSearch
tags: [comparison, recipe, paper, rl, retrieval-augmented]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Recipe comparison: Search-R1 vs R1-Searcher vs ReSearch

Three papers, three recipes for "RL-train an LLM to interleave reasoning with retrieval". They differ on every load-bearing axis: algorithm, reward, curriculum, scale. This doc grounds each claim against the source paper or training script and notes what is **not stated**. No inference.

| | [Search-R1](../papers/2503.09516_search-r1.md) | [R1-Searcher](../papers/2503.05592_r1-searcher.md) | [ReSearch](../papers/2503.19470_research.md) |
|---|---|---|---|
| **arXiv id (version)** | 2503.09516v5 (2025-08-05) | 2503.05592v2 (2025-03-18) | 2503.19470v3 (2025-09-23, NeurIPS 2025) |
| **In our project as** | M1 reproduction target | Current-focus reference | Recipe behind [v0](../report/RESULTS_m0_a.md) / [v1](../report/RESULTS_m0_b.md) Phase-1 results |
| **Algorithm** | PPO (default), GRPO compared | Reinforce++ | GRPO |
| **Tag schema** | `<search>` / `<information>` / `<answer>` | `<begin_of_query>` / `<begin_of_documents>` | `<think>` / `<search>` / `<result>` + `\boxed{}` |
| **Reward shape** | Outcome-only EM | Stage 1: retrieval=0.5, format=0.5; Stage 2: F1 + format `+0/-2` | F1 if F1>0; 0.1 if F1=0 and format ok; else 0 |
| **Reward server** | in-process | separate process (`reward_server_qwen_zero.py`) | in-process |
| **Retrieved-token loss masking** | Yes; ablated in §5.4 with quantified ~25% average relative gain | Yes; called retrieve-mask loss; not ablated | Yes; asserted; not ablated |
| **Curriculum** | none | 2-stage (retrieval → answer) over difficulty buckets | none |
| **Train data** | NQ + HotpotQA merged; size not in our extract | HotpotQA + 2Wiki, ~8.1k samples (350 in stage 1, 8148 in stage 2), difficulty-bucketed | MuSiQue training set only, 19,938 samples |
| **Eval benchmarks** | NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle (7) | HotpotQA, 2Wiki, MuSiQue, Bamboogle (4) | HotpotQA, 2Wiki, MuSiQue, Bamboogle (4) |
| **Eval metric(s)** | EM | Cover-EM (`ACC_R`) and LLM-as-Judge with GPT-4o-mini (`ACC_L`) | EM and LLM-as-Judge with GPT-4o-mini |
| **Base models** | Qwen2.5-3B, 7B, 14B (Base + Instruct each) | Qwen2.5-7B-Base, Llama-3.1-8B-Instruct | Qwen2.5-7B and 32B (Base + Instruct each) |
| **GPUs** | "Single node, 8 H100 GPUs" (paper §4) | 8 GPUs (type not stated in paper or scripts) | "8×8 H800 GPUs" = 64 H800 (paper §4.2) |
| **Distributed config** | FSDP + CPU offload; vLLM TP=1; mem util 0.6 | ZeRO-2; actor 4 GPUs, ref 4 GPUs; vLLM TP=2; bf16 + FlashAttn + packed | full-param + grad-checkpoint; vLLM TP=2 |
| **Total batch (prompts)** | 512 | 256 | 256 |
| **Mini-batch** | 256 | 64 (rollout); 1 (micro train) | 64 (PPO mini) |
| **Rollouts per prompt (G)** | ablated 1, 3, 5 (Appendix H); headline G not in our extract | 16 | 5 |
| **Learning rate** | not in our extract | actor 2e-6, critic 9e-6 | 1e-6 |
| **KL coefficient** | not in our extract | `init_kl_coef=0.0` (no KL anchor) | reference-model KL on (vanilla GRPO) |
| **Max prompt len** | (combined) 4,096 total | 1,024 | 1,536 |
| **Max generate len** | (combined) 4,096 total | 29,000 | 6,656 |
| **Max retrieval calls / trajectory** | not stated in our extract | not stated | 5 |
| **Retriever** | E5, top-k=3, Wikipedia 2018 | BGE-large-en-v1.5, KILT 2019 Wikipedia | E5-base-v2, top-k=5, Wikipedia 2018 (FlashRAG) |
| **Schedule** | 500 steps; checkpoint every 100 | 1 epoch per stage; `num_episodes=10000` cap | 2 epochs |
| **Wall-clock** | not stated | not stated | not stated |
| **GitHub** | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | [RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | [Agent-RL/ReCall@re-search](https://github.com/Agent-RL/ReCall/tree/re-search) (formerly Agent-RL/ReSearch) |

## Headline numbers

Each paper reports on partly different benchmark sets and metrics. The four-benchmark intersection is HotpotQA, 2Wiki, MuSiQue, Bamboogle; only Search-R1 reports EM on all 7 of NQ + TriviaQA + PopQA on top.

### Search-R1 (Qwen2.5-3B GRPO, EM, paper Table 3 v5)

| | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.421 | 0.583 | 0.413 | 0.297 | 0.274 | 0.066 | 0.128 | 0.312 |
| Instruct | 0.397 | 0.565 | 0.391 | 0.331 | 0.310 | 0.124 | 0.232 | 0.336 |

### Search-R1 (Qwen2.5-7B GRPO, EM, paper Table 3 v5)

| | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.395 | 0.560 | 0.388 | 0.326 | 0.297 | 0.125 | 0.360 | 0.350 |
| Instruct | 0.429 | 0.623 | 0.427 | 0.386 | 0.346 | 0.162 | 0.400 | 0.396 |

### R1-Searcher (LJ via GPT-4o-mini, paper Table 3 v2)

| | HotpotQA | 2Wiki | Bamboogle (offline) |
|---|---:|---:|---:|
| Qwen2.5-7B-Base | 75.0 | 65.0 | 54.4 |
| Llama-3.1-8B-Instruct | 74.6 | 62.8 | 54.4 |

R1-Searcher does not headline EM, so direct comparison to Search-R1's EM column is not apples-to-apples. R1-Searcher's offline Bamboogle 54.4 (LJ) is on a different scale than Search-R1's 0.232 (EM, 3B-Instruct).

### ReSearch (Qwen2.5-32B-Instruct, paper Table 3 v3)

| | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
|---|---:|---:|---:|---:|
| EM | 0.4673 | 0.4490 | 0.2640 | 0.5680 |
| LJ | 0.6770 | 0.5030 | 0.3856 | 0.6720 |

ReSearch reports EM and LJ for both 7B and 32B sizes; the row above is the 32B-Instruct headline.

## Where the recipes disagree

- **Reward shape is the largest design disagreement.**
  - Search-R1: pure outcome EM. No format reward.
  - ReSearch: F1 + 0.1 partial-credit floor when format is correct but answer is wrong.
  - R1-Searcher: 2-stage; Stage-2 has asymmetric `+0 / -2` format penalty.
  - Implication: ReSearch's 0.1 floor lifts every trajectory's reward and may mask tool-use signal (we measured a 3-6 pp gap at our 0.6B scale; see [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md)). Search-R1 avoids this. R1-Searcher penalises format-collapse late.
- **Algorithm is a smaller disagreement than scale.**
  - All three are policy-gradient with retrieved-token masking; the differences (PPO vs GRPO vs Reinforce++) are smaller in effect than the scale and reward differences. Search-R1's §5.1 directly compares PPO and GRPO and reports both work.
- **Curriculum.**
  - Only R1-Searcher uses one (Stage 1 retrieval-only, Stage 2 answer-quality).
  - Search-R1 and ReSearch both train end-to-end without curriculum.
- **Group size G** is wildly different:
  - Search-R1: ablated {1, 3, 5}; headline G not in our extract.
  - R1-Searcher: G=16.
  - ReSearch: G=5.
  - Combined with ReSearch's 64-GPU stated compute, this is the most decision-relevant axis for our 1× A100 budget.
- **Generate-max length**:
  - Search-R1: 4,096 total (combined prompt + response).
  - R1-Searcher: 29,000 (response).
  - ReSearch: 6,656 (response).
- **Retrieval depth**:
  - Search-R1: top-k = 3.
  - ReSearch: top-k = 5.
  - R1-Searcher: top-k not in our extract.
- **Retriever**:
  - Search-R1 and ReSearch: E5 (different exact variants; Search-R1 says "E5", ReSearch specifies E5-base-v2).
  - R1-Searcher: BGE-large-en-v1.5.
- **Loss masking on retrieved tokens** is the only ablation across the three papers that has a quantified result: Search-R1 §5.4 reports ~25% average relative gain. The other two assert masking but do not measure it.

## Where the recipes agree

- All three: outcome reward only (no process reward, no neural reward model).
- All three: retrieved-token loss masking.
- All three: Wikipedia as the corpus (Search-R1 / ReSearch use 2018 dump; R1-Searcher uses KILT 2019).
- All three: HotpotQA, 2Wiki, MuSiQue, Bamboogle in the eval set.
- All three: published checkpoints on HuggingFace, training scripts on GitHub.

## Mapping to our work

- **M1 (eval reproduction)** is against [Search-R1](../papers/2503.09516_search-r1.md) v5; the published GRPO 3B Base/Instruct checkpoints. Plan B v1 reproduction is within ±2.5 pp paper average ([`../report/RESULTS_m1.md`](../report/RESULTS_m1.md)).
- **Phase-1 (Qwen3-0.6B, ALICE, sibling `research` repo)** ports the [ReSearch](../papers/2503.19470_research.md) recipe: GRPO with the F1 + 0.1 partial-credit reward. 29 runs total ([`../report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md)). Findings: recipe transfers but rewards cluster at 0.18-0.22; base model fails cold-start (5/5).
- **Phase-2 (Qwen3.5-2B, NeMo-RL, 1× A100)** is being designed against the [Search-R1](../papers/2503.09516_search-r1.md) shape (b=512 prompts, mb=256, mu=64, len=4096, 500 steps). The published training pipeline ([`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md)) is a direct port. Wall-clock extrapolation (smoke-anchored) is 11-17 days / run on 1× A100; that constraint is what drives the [recipe-search ablation plan](../todo/TODO_2026-05-04.md).
- **Tricks worth borrowing** (in priority order, given our 1× A100 budget):
  1. **R1-Searcher's 2-stage curriculum** to bypass the cold-start failure we hit on 0.6B base. Drop-in.
  2. **R1-Searcher's `+0 / -2` asymmetric format penalty** late in training to prevent format collapse.
  3. **R1-Searcher's `init_kl_coef=0.0`** to free reference-model GPU memory if stable. Compute win, not algorithmic.
  4. Replace ReSearch's 0.1 partial-credit floor with binary 0/1 (Search-R1 style) and measure the tool-use signal sharpening. This is the most ablation-worthy line in our recipe per [`../../claude/CLAUDE.md`](../../claude/CLAUDE.md) gotcha #4.
- **Tricks not worth borrowing**:
  - R1-Searcher's G=16 rollouts (compute-prohibitive at our scale).
  - R1-Searcher's 29k generate-max (no obvious gain at our context lengths).
  - LLM-as-Judge evaluation (introduces a closed-model dependency; we stay on EM/F1).

## Open questions

- What is Search-R1's headline GRPO group size G? (Appendix H ablates 1, 3, 5; not in our extract.)
- What is R1-Searcher's retriever top-k? (Not in our extract.)
- For each paper, what is wall-clock per run on the stated hardware? (None stated.)
- Cross-comparison on a single metric: re-score all three papers' published checkpoints under our EM scorer, on the 4-benchmark intersection (HotpotQA, 2Wiki, MuSiQue, Bamboogle). This would give the only apples-to-apples comparison; we already have the M1 pipeline for it.
- Does Search-R1's ~25% gain from retrieved-token masking hold at 0.6B / 2B?

## Provenance

- Sources: the three single-paper notes ([Search-R1](../papers/2503.09516_search-r1.md), [R1-Searcher](../papers/2503.05592_r1-searcher.md), [ReSearch](../papers/2503.19470_research.md)) and the linked PDFs in [`../raw/papers/`](../raw/papers/).
- Each row is grounded against the paper version cited in the corresponding note. Where a value is "not in our extract" or "not stated", we did not invent one.
- This document does not extrapolate or estimate. Comparisons that require an estimate (e.g. wall-clock) are listed in **Open questions** instead.
