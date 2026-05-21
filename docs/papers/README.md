---
title: docs/papers/ - paper ingest notes
tags: [meta, schema, papers]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# `docs/papers/` - paper ingest notes

One markdown file per paper we read seriously. The goal is not a faithful summary; it is a **distilled, queryable record** that future-you (or a fresh Claude session) can use to (a) recall what the paper actually claims, (b) compare its recipe to ours, and (c) find the source PDF if the answer is not in the note.

Companion to [`../research/SURVEY.md`](../research/SURVEY.md) and [`../research/LITERATURE_REVIEW.md`](../research/LITERATURE_REVIEW.md): those are wide-angle surveys; the files in here are deep, single-paper notes.

## Filename convention

`<arxiv-id>_<short-slug>.md`, e.g. `2503.05592_r1-searcher.md`.

- Arxiv ID first because it sorts chronologically and is unique.
- Slug is lowercase, hyphenated, derived from the project / model name.
- The PDF lives at `docs/raw/papers/YYYY-MM-DD_<short-slug>.pdf` (see [`../raw/README.md`](../raw/README.md)).

## How to ingest a paper (the workflow)

1. **Capture the source**: download the PDF to `docs/raw/papers/YYYY-MM-DD_<slug>.pdf`. Trigger a Wayback save of the abs page and the GitHub repo (Save Page Now at `https://web.archive.org/save/<url>`) so links survive linkrot.
2. **Start from the template**: copy [`../templates/paper.md`](../templates/paper.md) (or use Templater in Obsidian, "Insert template from..." → `paper`).
3. **Fill the cite block**: title, authors with affiliations (this matters; see "Compute archaeology" below), arxiv abs/HTML/PDF links, alphaxiv link, Wayback snapshot URLs, GitHub URL, and the local raw PDF path.
4. **Distil, do not transcribe**: keep prose tight. If you find yourself summarising more than 1-2 sentences per section, you are too close to the paper. Quote one or two lines verbatim only when wording matters (definitions, reward shapes).
5. **Go deep on the training setup** (this is the section that gets cited most often by future sessions): see "Compute archaeology" below.
6. **Always write a "Takeaways for us" section** that maps the paper's recipe to our project: what we already do, what we should try, what is off-limits because of compute.
7. **Append to [`../log.md`](../log.md)** with one line: source path → pages touched.
8. **Run the lint**: `python ../../scripts/wiki_lint.py docs/`. Fix broken links before moving on.

## Compute archaeology (the part that matters most)

Papers under-report compute. Authors say "we trained on 8 GPUs" and skip the type, the precision, the rollout shape, the wall-clock. For our project (1× A100, $1k budget) the compute claim is often the most decision-relevant number in the paper. Always pull it explicitly. Sources, in increasing order of trust:

1. **Abstract / intro** - usually silent.
2. **"Implementation details" or "Experimental setup" section** - sometimes gives GPU type and batch sizes; usually no wall-clock.
3. **Appendix** - GPU type and hours when the authors are diligent (not always).
4. **GitHub training scripts** (most reliable) - look for `CUDA_VISIBLE_DEVICES`, `--num_gpus`, `--actor_num_gpus_per_node`, `--ref_num_gpus_per_node`, `--vllm_tensor_parallel_size`, `--train_batch_size`, `--micro_train_batch_size`, `--n_samples_per_prompt`. Multiply through to recover the true rollout/training shape.
5. **Issues and PRs on the repo** - sometimes the only place a wall-clock number is admitted.

Things to record in the **Training setup** section of every paper note:
- GPU type and count, exactly as the paper or script states. If the paper says "8 GPUs" with no type, write "8 (type not stated)".
- Total training tokens or steps; epochs.
- Batch size convention. Is "batch size" prompts, prompt × rollouts, or tokens? GRPO papers in particular conflate these (see [`../edu/BATCH_MATH.md`](../edu/BATCH_MATH.md)).
- Rollouts per prompt (`n_samples_per_prompt`, `G`).
- Learning rate, KL coefficient (`init_kl_coef`), warmup ratio, gradient clipping.
- Maximum sequence length (`generate_max_len`, `prompt_max_len`).
- Reward shape: pure outcome? format bonus? if format, what value? what masking on retrieved tokens?
- Distributed config: ZeRO stage, vLLM tensor-parallel size, decolocation or colocation.
- Wall-clock if reported.

## No inference rule

Be true to paper and code. **Never infer** unstated facts: do not derive GPU type from author affiliation, do not estimate wall-clock from rollout shape, do not assume a missing hyperparameter from a sibling paper. When a field is not stated in the paper or scripts, write **"not stated"** and stop.

When citing a number, cite the row it came from: model size, table number, paper version. Do not say "0.46 EM at 7B" if the 0.46 was the 32B row.

If the user explicitly asks for an estimate, prefix with `**Estimate (not from paper)**:` and show the assumptions. The default is no estimate.

## Versioning gotcha

Arxiv papers are mutable. v1 and v3 of the same paper can have different numbers in Table 3. Always note **which version you read** in the cite block (`arXiv 2503.05592v2`, etc.) and which version is current at ingest time. If you cite a number from the paper, cite the version too.

## What does not belong in a paper note

- The full method explanation. Link to the paper for that. The note is for things you will need at 11 pm during a debugging session.
- Personal opinions disconnected from our project. Those go in [`../research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md).
- Highlights. The Obsidian highlight syntax (`<mark style=...>`) is not portable; the user does the highlighting manually after reading the note in Obsidian.
- Diagrams copied as images. Link the figure number; readers can open the PDF.

## Linking conventions inside paper notes

- Use relative markdown only: `[label](../research/SURVEY_FOCUSED.md)`. Never wikilinks (`[[...]]`); they break for non-Obsidian readers (Claude, GitHub, plain editors).
- When citing a paper's own claim, include section / equation / table number and the arxiv version: `(R1-Searcher arXiv 2503.05592v2 §3.2, Table 4)`.
- For our results that follow a paper's recipe, link the paper note from the result page **and** link the result page from the paper note's "Takeaways for us" section. Both directions; otherwise the link rots one-way.

## Index

(Append paper notes here as they are added.)

- [Search-R1](2503.09516_search-r1.md) - PPO/GRPO with outcome-only EM reward and retrieved-token loss masking on Qwen2.5-3B/7B/14B; the M1 reproduction target.
- [R1-Searcher](2503.05592_r1-searcher.md) - 2-stage Reinforce++ with RAG-rollout and retrieve-mask loss; current focus reference.
- [ReSearch](2503.19470_research.md) - GRPO with F1 + format-bonus reward on Qwen2.5; the recipe behind our [v0](../report/RESULTS_m0_a.md) and [v1](../report/RESULTS_m0_b.md) Phase-1 results.
