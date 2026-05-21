---
title: M5.1-prod-a2 rollout archive — stripped corpus, steps 1–15
tags: [report, m5, m5.1, rollouts, archive]
source: internal
created: 2026-05-12
---

# `exp_011_a2_archive.tar.gz`

What's in here: the per-step rollout corpus from `M5.1-prod-a2` (the second production run, W&B run `2b95h2fg`), steps 1 through 15. The run was killed at step 15 on a misdiagnosis (see [`RESULTS_SMOKE_m5.md §7.8`](../docs/report/RESULTS_SMOKE_m5.md#78-companion-postmortem--the-zombie-gpu-memory-misdiagnosis-2026-05-12)).

This archive is committed to git as defensive backup for the analysis corpus, after `a1`'s equivalent rollout corpus was permanently lost via accidental `rm -rf` ([`RESULTS_SMOKE_m5.md §7.8.1`](../docs/report/RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11)). It is a2's *partial* substitute — `a1`'s step 16-49 examples are not recoverable.

## Schema (per row)

The archive is `tar.gz` of 15 `.jsonl` files (one per step), one row per prompt-group (64 rows per step in production). Each row has been stripped to only the analysis-relevant fields:

| Field | Type | Note |
|---|---|---|
| `idx` | int | Prompt index within the step (0..319) |
| `content` | list[str] | Length = G = 5; the full chat-template-formatted rollout text for each of the 5 generations |
| `rewards` | list[float] | Length = G = 5; F1 score on `<answer>...</answer>` content vs gold |
| `input_lengths` | list[int] | Length = G = 5; total token count per rollout (incl. system + user + assistant + tool) |

Stripped (not present): `token_ids`, `token_loss_mask`, `sample_loss_mask`, `advantages`, `generation_logprobs`, `prev_logprobs`. These are GRPO-internal numerical training state and are not useful for trace-quality analysis. Stripping them dropped 1.77 GB → 104 MB before gzip → 22 MB committed.

## Provenance

- W&B run: [`2b95h2fg`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m5_1/runs/2b95h2fg) on project `reason_over_search_m5_1`.
- Local source dir at archive creation: `logs/exp_011/train_data_step{1..15}.jsonl`.
- Created 2026-05-12 by `python3` strip loop + `tar -czf`. Reproducer in commit message.
- Run config: [`training_m5_1/configs/m5_1_research_paper.yaml`](../training_m5_1/configs/m5_1_research_paper.yaml) (at `5895ad6` or earlier).
- Run name: `qwen3.5-0.8b-musique-m5_prod-seed42-20260511T2223Z`.
- Hardware: 1× A100-80GB on Vast, image `pantomiman/reason-over-search-v1:v1`.

## How to use

```bash
tar -xzf logs/exp_011_a2_archive.tar.gz -C /tmp/
ls /tmp/exp_011_a2_archive/   # train_data_step1.jsonl ... train_data_step15.jsonl
```

Per-row extraction example (Python):

```python
import json
rows = [json.loads(l) for l in open('/tmp/exp_011_a2_archive/train_data_step15.jsonl')]
# Find perfect-F1 rollouts:
for i, r in enumerate(rows):
    for j, rew in enumerate(r['rewards']):
        if rew >= 0.99:
            print(f"idx={i} g={j} reward={rew}")
            print(r['content'][j][:500])
```

Two specific traces are walked through in [`RESULTS_m5.md §4.2`](../docs/report/RESULTS_m5.md#42-reasoning-trace-evolution--concrete-examples-a2-kept-after-a2-termination):
- Step 1, idx=215, g=0 — reward 0.0, 10-turn truncation, fabricated acronyms ("NCAVC question").
- Step 15, idx=94, g=0 — reward 1.0, 3-turn "Russia/Lenin/Moscow" multi-hop with explicit plan.
- Step 15, idx=318, g=0 — reward 1.0, 3-turn "Fantasy Land Tour 2004 / Tony Daykin / S.H.E" 4-hop with plan-and-triangulate.
