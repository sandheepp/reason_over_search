---
title: <% tp.file.title %>
tags: [paper]
source: <arxiv URL>
created: <% tp.date.now("YYYY-MM-DD") %>
updated: <% tp.date.now("YYYY-MM-DD") %>
---

# <% tp.file.title %>

> **Authors**: \<names\> (\<affiliations\>)
> **Version read**: arXiv \<id\>v\<n\>, \<submission date\>; v\<latest\> is current at ingest.
> **Links**: [arxiv abs](https://arxiv.org/abs/<id>) - [arxiv html](https://arxiv.org/html/<id>v<n>) - [alphaxiv](https://www.alphaxiv.org/abs/<id>) - [wayback abs](https://web.archive.org/web/2026*/https://arxiv.org/abs/<id>) - [code](https://github.com/...) - [wayback code](https://web.archive.org/web/2026*/https://github.com/...) - [local PDF](../raw/papers/YYYY-MM-DD_<slug>.pdf)

## TL;DR

3 to 5 bullets. The version of this paper a tired reader needs at midnight.

## Problem

What gap does the paper claim to fill?

## Method

- **Algorithm**: e.g. GRPO / Reinforce++ / DPO; modifications.
- **Tag schema**: e.g. `<think> </think>`, `<search> </search>`, `<result> </result>`.
- **Reward**:
  - Answer reward: \<F1 / EM / shaped\>.
  - Format reward: \<value when correct, value when wrong\>.
- **Retrieval setup**: corpus, retriever model, top-k, framework.
- **Loss masking**: are retrieved tokens masked from the policy gradient?

## Training setup

(Be exhaustive. This is the section future-you reads most.)

- **Base model(s)**: e.g. Qwen2.5-7B-Base, Llama-3.1-8B-Instruct.
- **GPUs**: e.g. 8× H800 (inferred from `CUDA_VISIBLE_DEVICES=0..7` in `scripts/X.sh`; affiliation \<inst\> implies H800 by export restrictions).
- **Distributed config**: ZeRO stage; actor / ref GPU split; vLLM tensor-parallel size.
- **Batch sizes**: train batch \<N\>, micro-train \<n\>; rollout batch \<N\>, micro-rollout \<n\>; rollouts per prompt \<G\>.
- **Optimisation**: optimiser, learning rate, warmup ratio, KL coefficient, gradient clipping, precision (bf16 / fp16).
- **Sequence**: prompt-max \<L\>, generate-max \<L\>.
- **Schedule**: episodes / steps / epochs; save-every \<N\>.
- **Wall-clock**: \<reported, or back-of-envelope estimate from per-step time × steps\>.

## Data

- **Train**: \<dataset(s) and sizes\>; difficulty curriculum if any.
- **Eval**: \<benchmarks\>; in-domain vs OOD.

## Results

Headline numbers vs strongest baseline. Cite version and table.

| Benchmark | EM | LJ | vs baseline |
|---|---:|---:|---:|
| HotpotQA | | | |
| 2Wiki | | | |
| MuSiQue | | | |
| Bamboogle | | | |

## Takeaways for us

- What does this paper imply for our recipe?
- What can we copy directly? What cannot we afford?
- Which of our existing pages should this paper update? (Add cross-links.)

## Limitations

- What the authors admit.
- What we think they understate (compute, generalisation, eval split).

## Open questions / followups

- Things to chase next time we revisit this.

## Provenance

- Captured: \<YYYY-MM-DD\> from \<URL\> -> `../raw/papers/<file>`.
- Ingested: \<YYYY-MM-DD\>.
- Cross-links: \<list of pages this note should be linked from\>.
