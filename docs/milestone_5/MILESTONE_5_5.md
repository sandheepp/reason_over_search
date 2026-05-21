---
title: Milestone 5.5 — F1+format reward ablation
tags: [milestone, m5.5, reward, ablation, nemo-rl, qwen3.5]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: scaffolded (training_m5_5/ created; sbatch queued)
---

# Milestone 5.5 — F1+format reward ablation

> Sibling of [MILESTONE_5_6.md](MILESTONE_5_6.md) (EM-only). Together they
> form the **reward-shape ablation triad** with M5.1 (F1-only, the existing
> baseline at `training_m5_1/`). The triad answers the Phase-1 question
> [PHASE_1_SALVAGE.md Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal):
> does the ReSearch paper's 0.1 partial-credit floor mask tool-use signal at
> our scale?

## 1. Why this milestone exists

The Phase-1 v0 prompt sweep on Qwen3-0.6B
([RESULTS_m0_a.md §11.4](../report/RESULTS_m0_a.md#L472)) observed that the
ReSearch paper-faithful 3-tier reward (`0` if format broken; `0.1` if format
valid but F1=0; `F1` if F1>0) collapses tool-using and no-tool policies into a
**3 to 6 pp band of training reward**, while the prompt-design lever moves
rollout behaviour by ~9 pp. The 0.1 floor masks the tool-use signal — the
reward records less than what the policy is doing.

The ReSearch paper never ablates this floor. Per the paper-notes anchor
([2503.19470_research.md §"Takeaways for us" line 77](../papers/2503.19470_research.md#L77)),
the 0.1 floor is "the load-bearing detail [...] presented in the paper without
comment; it is the most ablation-worthy line in the loss." [Milestone 6
Candidate C](../milestone_6/MILESTONE_6.md#L96) elevates this from
"provisional candidate" to **likely pick #1** because it pairs an unfilled
literature gap with a Phase-1 motivation paragraph the project already owns.

**Question this milestone answers**: with all other Search-R1 / ReSearch knobs
held identical (model, data, GRPO settings, prompt, retrieval), does adding
the 0.1 floor on top of F1 reward change end-of-training behaviour and
held-out EM/F1, relative to F1-only (M5.1) and EM-only (M5.6)?

## 2. Spec — reward function

```
reward(rollout) =
  0.0    if not is_valid_sequence(rollout)            # format broken
  0.1    if extract_solution(rollout) is None         # format ok, no <answer>
  0.1    if f1(answer, gold) == 0                     # format ok, answer wrong
  f1     if f1(answer, gold) > 0                      # format ok, partial / full
```

Where `is_valid_sequence` is the state-machine format walker shared with the
M2 paper-faithful reward at
[training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py),
and `extract_solution` / `f1_check` / `normalize_answer` are re-exported
from the M4 eval pipeline
([evaluation_qwen35/flashrag/search_r1/reward.py](../../evaluation_qwen35/flashrag/search_r1/reward.py))
so the M5 train-time reward and M4 eval-time scorer share the same code path.

The floor value `0.1` matches the ReSearch paper's choice and the
Phase-1 v0 observation. It is NOT additive (F1=0.5 → reward=0.5, NOT 0.6); the
floor only applies when F1==0.

This is implemented at
[training_m5_5/src/rewards/search_r1.py](../../training_m5_5/src/rewards/search_r1.py).

## 3. Spec — what is held constant vs M5.1

| Knob | M5.1 (F1-only baseline) | M5.5 (F1+format) | M5.6 (EM-only) |
|---|---|---|---|
| Model | Qwen3.5-0.8B (hybrid) | same | same |
| Data | MuSiQue train (19,938 rows) | same | same |
| GRPO `num_prompts_per_step` | 64 | 64 | 64 |
| GRPO `num_generations_per_prompt` | 5 | 5 | 5 |
| GRPO `max_num_steps` | 622 | 622 | 622 |
| `max_total_sequence_length` | 8192 | 8192 | 8192 |
| `train_micro_batch_size` | 1 | 1 | 1 |
| Prompt | `qwen35_minimal` (M4.2 lock) | same | same |
| Retriever / top_n | IVF-SQ8 / 5 | same | same |
| **Reward** | **F1 only on `<answer>`** | **F1 + 0.1 format floor** | **EM 0/1** |

Every knob except the reward is byte-identical to
[training_m5_1/configs/m5_1_research_paper.yaml](../../training_m5_1/configs/m5_1_research_paper.yaml).

## 4. Success / failure semantics

This is a **mechanism ablation**, not a "is M5.5 better than M5.1" race.
Both outcomes are informative:

- **The 0.1 floor improves end-of-training behaviour** (e.g., tool-use rate
  stays > 0; reward trajectory smoother): the floor acts as a stabiliser and
  the Phase-1 observation is partially explained by reward shaping helping.
  Implication for the paper: ReSearch's choice has a reason; small-model
  retrieval benefits from a stabiliser.
- **The 0.1 floor degrades or is neutral** (tool-use rate equal or lower
  than M5.1; held-out EM/F1 equal or lower): the floor masks signal without
  helping training. Implication for the paper: the load-bearing line in the
  ReSearch loss does not survive a clean ablation at our scale.
- **Outcomes diverge between training reward and held-out EM/F1**: the
  floor over-fits the training signal (it shrinks the reward band but does
  not translate to held-out gain). This is the *strong* version of the
  Phase-1 hypothesis.

## 5. What the milestone produces

A frozen results doc once the run completes:

- `docs/report/RESULTS_m5_2.md` — per-step reward / tool-call / response-length
  trajectory plus the 7-dataset held-out eval against M5.1 + M5.6.
- Updated [MILESTONE_5.md](MILESTONE_5.md) §"Status" referencing both 5.2
  and 5.3.
- Updated [PHASE_1_SALVAGE.md §Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)
  with the "Limit of the claim" line resolved (Phase-1's missing
  {F1+0.1, F1-only, EM-only} ablation will then exist).

## 6. Wall-clock + cost

Same shape as M5.1 → ~10 to 13 days on 1× A100-80GB per the live M5.1
trajectory ([RESULTS_m5.md §4.1](../report/RESULTS_m5.md#L61)). Submitted to
ALICE `gpu-a100-80g` partition at the 7-day cap; if the run does not finish
within one sbatch slot, it will be resumed from the latest checkpoint in a
follow-up submission. Estimated queue wait at submission time: ~17 days
(see [todo/TODO_2026-05-11.md](../todo/TODO_2026-05-11.md) for the live ETA).

## 7. Pointers

- Sibling milestone (EM-only variant): [MILESTONE_5_6.md](MILESTONE_5_6.md).
- Baseline (F1-only): [MILESTONE_5.md](MILESTONE_5.md) (M5.1 in flight).
- Motivation paragraph: [PHASE_1_SALVAGE.md §Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal).
- Candidate-experiment context: [MILESTONE_6.md §Phase 2 candidate C](../milestone_6/MILESTONE_6.md#L96).
- Paper anchor (no ablation in the source): [2503.19470_research.md §"Takeaways"](../papers/2503.19470_research.md).
- Folder convention: [MILESTONE_5.md §"Folder layout"](MILESTONE_5.md#L96).
- Efficiency planning (orthogonal sibling, distinct milestone): [MILESTONE_5_3.md](MILESTONE_5_3.md).
