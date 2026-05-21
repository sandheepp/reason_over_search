---
title: Milestone 5.6 — EM-only reward ablation
tags: [milestone, m5.6, reward, ablation, nemo-rl, qwen3.5]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: scaffolded (training_m5_6/ created; sbatch queued)
---

# Milestone 5.6 — EM-only reward ablation

> Sibling of [MILESTONE_5_5.md](MILESTONE_5_5.md) (F1+format). Together they
> form the **reward-shape ablation triad** with M5.1 (F1-only, the existing
> baseline at `training_m5_1/`). M5.6 is the strictest leg: no partial credit
> at all, no format bonus — the policy gets `1` if and only if its extracted
> answer matches the gold answer exactly under `normalize_answer`.

## 1. Why this milestone exists

Three reasons EM-only earns a slot in the triad:

1. **Paper-faithful Search-R1**: the Search-R1 paper (arXiv 2503.09516 §3.4)
   explicitly specifies `rϕ(x, y) = EM(a_pred, a_gold)` and rejects format
   rewards on the basis that the trained model already shows strong
   structural adherence. The M2 reward module at
   [training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py)
   is byte-identical to the Search-R1 source-of-truth (`qa_em_format.py` with
   all shaping coefficients defaulted to 0.0). Running it at the M5.1 shape
   gives the paper-faithful Search-R1 reward as a leg of the triad.
2. **Strictest signal**: pure 0/1 reward is the highest-variance, lowest-bias
   teacher. If the M5.1 F1 baseline beats EM-only, that quantifies how much
   the partial-credit signal helps a sub-1B model; if EM-only matches or
   beats F1, the simpler reward wins.
3. **Closes the Phase-1 limit-of-the-claim line** in
   [PHASE_1_SALVAGE.md §Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal):
   *"Phase-1 was paper-faithful reward only. There is no Phase-1 ablation of
   {F1+0.1, F1-only, EM-only}."* M5.6 + M5.5 + M5.1 (F1-only existing) is
   exactly that ablation, at Qwen3.5-0.8B scale.

**Question this milestone answers**: with everything else held identical
to M5.1, does training under pure EM converge to the same held-out
EM / F1 / tool-use as under F1 or F1+format? If yes, simplest signal wins.
If no, the partial-credit gain is real and the M5.1 baseline is justified.

## 2. Spec — reward function

```
reward(rollout) =
  1.0    if extract_solution(rollout) is not None
         and em_check(answer, gold) == 1                   # exact match
  0.0    otherwise                                          # no answer, wrong, or
                                                            # format broken
```

The format-gate is implicit (no `<answer>` block extracted → no reward), but
there is **no format bonus and no F1 partial credit**. This matches
`compute_search_r1_reward(..., structure_format_score=0, final_format_score=0,
retrieval_score=0, score=1.0)` from
[training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py)
— the M2 paper-faithful reward in its "zero shaping coefficients"
configuration, which is the Search-R1 paper's `EM(a_pred, a_gold)`.

Implementation lives at
[training_m5_6/src/rewards/search_r1.py](../../training_m5_6/src/rewards/search_r1.py)
— a thin re-export of the M2 reward so the train-time and the M2-archive code
paths stay byte-identical. No new logic; the M2 module already implements EM.

## 3. Spec — what is held constant vs M5.1

Same table as [MILESTONE_5_5.md §3](MILESTONE_5_5.md#3-spec--what-is-held-constant-vs-m51).
**Reward** is the only variable across the triad:

| Variant | Folder | Reward formula |
|---|---|---|
| M5.1 | `training_m5_1/` | `f1(answer, gold)` (F1 only, no format gate, no floor) |
| M5.5 | `training_m5_5/` | `0 / 0.1 / f1` (format gate + 0.1 floor on F1=0) |
| M5.6 | `training_m5_6/` | `em(answer, gold)` (pure 0/1 EM) |

## 4. Success / failure semantics

Like M5.5, both outcomes are informative:

- **EM-only matches or beats F1-only**: the lower-bias / higher-variance
  signal is fine; the M5.1 baseline is a complexity-without-payoff choice.
  Implication: future recipes should default to EM.
- **EM-only lags F1-only meaningfully**: partial credit on the answer
  matters at this scale; the reward shaping survives ablation. This
  contradicts the JustRL "tricks may hurt" framing for retrieval-augmented
  tasks specifically. Implication: small-model retrieval needs the
  smoother reward landscape that F1 provides.
- **Same training-reward trajectory, divergent held-out EM/F1**: a strong
  signal that the train/eval gap depends on reward shape — interesting
  for the paper's threats-to-validity discussion.

## 5. What the milestone produces

- `docs/report/RESULTS_m5_3.md` once the run completes.
- A 3-row comparison table (M5.1 / M5.5 / M5.6) in
  [PHASE_1_SALVAGE.md §Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal),
  resolving the "Limit of the claim" line.
- Confirmation (or refutation) that the Search-R1 paper's EM choice is
  defensible at sub-1B scale on multi-hop QA.

## 6. Wall-clock + cost

Same shape as M5.1 / M5.5 → ~10 to 13 days on 1× A100-80GB. EM is faster to
compute per rollout than F1 (one normalised-string compare vs one Counter
intersection) but the policy / generation / logprob phases dominate so the
wall-clock differs negligibly. Queue ETA on ALICE at submission ~17 days
(see [todo/TODO_2026-05-11.md](../todo/TODO_2026-05-11.md)).

## 7. Pointers

- Sibling milestone (F1+format variant): [MILESTONE_5_5.md](MILESTONE_5_5.md).
- Baseline (F1-only, in flight): [MILESTONE_5.md](MILESTONE_5.md).
- Source-of-truth reward (M2 paper-faithful EM): [training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py).
- Search-R1 paper (EM specification): [arXiv:2503.09516 §3.4](https://arxiv.org/abs/2503.09516).
- Motivation paragraph: [PHASE_1_SALVAGE.md §Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal).
- Folder convention: [MILESTONE_5.md §"Folder layout"](MILESTONE_5.md#L96).
- Efficiency planning (orthogonal sibling, distinct milestone): [MILESTONE_5_3.md](MILESTONE_5_3.md).
