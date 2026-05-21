---
title: MILESTONE 3.1
tags: []
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Milestone 3.1: Evaluate the v0-best (no example) GRPO checkpoint

## Context

[MILESTONE_3](MILESTONE_3.md) closed 2026-05-07 by evaluating `p1_basic_w_ex_z7kcxfof` — the *with-example* run that converged on **heavy-tool 2-call / 4-turn** behaviour over 1046 steps, end-of-run rollout reward 0.190 (+28 % rel over 0.148 first-decile). That gave avg EM 0.102 → 0.155 on the 7 paper benchmarks (+0.053 abs, +52 % rel; full Plan A 51,713 items / variant).

But that run **was not the highest-reward run of the v0 block**. Per [`docs/report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md) §11, the highest-reward Phase-1 run was `p3_decide_no_ex_el6s2d2h`:

| Run | Prompt | Steps | First-decile | Last-decile | Δ rel | Behaviour |
|---|---|---:|---:|---:|---:|---|
| `z7kcxfof` (M3) | `p1_basic_w_ex` (with example) | 1046 | 0.148 | **0.190** | +28 % | heavy-tool 2/4, ~2050 tok |
| `el6s2d2h` (this milestone) | `p3_decide_no_ex` (no example, decision rules) | 2280 | 0.151 | **0.215** | **+43 %** | standard 1/3, ~1117 tok |

The *qualitative* finding from the Phase-1 ablations was that **decision-rule scaffolding can substitute for the few-shot example** (the only v0 prompt where removing the example was harmless). The *quantitative* question M3.1 answers: does that higher rollout reward translate to higher held-out EM, and if so, by how much vs. M3?

## Hypothesis

If the rollout-reward gap (0.215 vs 0.190) is meaningful, el6s2d2h should beat z7kcxfof on at least the single-hop datasets (where both checkpoints have headroom and the lift in M3 was concentrated). If the rollout reward gap is mostly the partial-credit floor (cf. `RESULTS_m0_a.md` finding 4), the gap may close at eval time. Either result is informative.

## Goal

Run the M3 evaluation pipeline against `eval/qwen_3_0.6b_v0_no_ex/` (HF-converted from the `el6s2d2h` step-2000 verl-FSDP archive on the user's training machine; the raw archive is not retained in this repo) on all 7 paper QA benchmarks at full Plan A; compare against the existing M3 numbers (pre-GRPO + z7kcxfof) populated in [`RESULTS_m3.md`](../report/RESULTS_m3.md).

## Setup

Reuses the M3 pipeline byte-for-byte except for the **prompt template**: `p3_decide_no_ex` (no Hamlet example; two extra decision-rule sentences). The rest of the alignment fixes ([`CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3 14-fix audit) carry over unchanged.

| Setting | M3 (z7kcxfof) | M3.1 (el6s2d2h) |
|---|---|---|
| Checkpoint | `eval/qwen_3_0.6b_v0/` (1046 steps; HF: [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0)) | `eval/qwen_3_0.6b_v0_no_ex/` (2000 steps; converted via `verl.model_merger merge --backend fsdp`; HF: [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1)) |
| Prompt template | `QWEN3_0_6B_TEMPLATE` (= `p1_basic_w_ex`) | **`P3_DECIDE_NO_EX_TEMPLATE`** (new; `evaluation_research/flashrag/search_r1/templates.py`) |
| `prompt_mode` | `qwen3` | **`qwen3_p3_decide_no_ex`** (new; falls in the `qwen3*` family — same retrieval format, same budgets, same `enable_thinking`) |
| Action format | `<search>` / `<result>` | same |
| Retrieval text | raw `{contents}\n\n` joined+stripped | same |
| `top_n` | 5 | 5 |
| `max_search_turns` | 5 | 5 |
| `step_limit` | 8192 (no per-step cap) | same |
| `max_obs_length` | 256 | same |
| `generator_max_input_len` | 4096 | 4096 |
| `enable_thinking` | True | True |
| Datasets | bamboogle, NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue (full Plan A, 51,713 items) | same |
| Hardware | ALICE 1× A100-80GB | same |
| Decoding | greedy (temperature=0.0) | same |
| Seed | 1 | 1 |

The new prompt template (verbatim from training, recovered from [`RESULTS_m0_a.md`](../report/RESULTS_m0_a.md) §10 `p3_decide_no_ex`):

```text
You are a helpful assistant who can answer questions using a Wikipedia search tool.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Use the information in the search results to determine the final answer.
After each search result, decide whether another search is needed or whether you can provide the final answer.
If a search result is incomplete, search again for the missing information.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
```

Difference vs `p1_basic_w_ex`: rules section has the two extra decision-guidance sentences ("After each search result, decide whether another search is needed…" / "If a search result is incomplete, search again…"); **no Hamlet example**.

## Pipeline changes (committed)

- [`evaluation_research/flashrag/search_r1/templates.py`](../../evaluation_research/flashrag/search_r1/templates.py): added `P3_DECIDE_NO_EX_TEMPLATE` and a `QWEN3_TEMPLATES` dict keyed on `prompt_mode`.
- [`evaluation_research/flashrag/pipeline/active_pipeline.py`](../../evaluation_research/flashrag/pipeline/active_pipeline.py): replaced `prompt_mode == 'qwen3'` checks with `prompt_mode.startswith('qwen3')` (so all qwen3 modes share retrieval format / budgets / `enable_thinking`); template selection now lookup-driven.
- [`evaluation_research/run_eval.py`](../../evaluation_research/run_eval.py): same `startswith('qwen3')` change for `retrieval_topk=5` gate; help text updated.
- [`scripts/run_m3.sh`](../../scripts/run_m3.sh): added `qwen3_0.6b_v0_no_ex` variant case → `prompt_mode=qwen3_p3_decide_no_ex`.
- [`scripts/sbatch_m3.sh`](../../scripts/sbatch_m3.sh): added `qwen3_0.6b_v0_no_ex` variant case (same SGLang flags as the other variants).

## How to run

```bash
# On ALICE login node:
cd /zfsstore/user/s4374886/omega/reason_over_search
sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0_no_ex
```

Same `gpu-short` partition, 4 h time limit, IVF-SQ8 × 8 retriever workers, 32 inference workers as M3. Expected wall-clock: ~2.5 h per variant (matches M3's 2 h 26 m for the v0 sbatch).

## Status

**2026-05-08 (COMPLETED)**: ran successfully on **sbatch 2150167** (gpu-short, A100-80GB on `node870`, COMPLETED at 08:41:06 in **1 h 32 m 15 s wall** — well under the 2.5 h M3 reference). Two prior attempts failed at infrastructure waits, not at training/eval logic:

| sbatch | Node | Failure | Fix |
|---|---|---|---|
| 2134645 | node875 | SGLang `/health` wait timed out at 300 s (M3 ref 260 s; cold cache crossed the cliff) | bumped 300 → 600 s in `scripts/sbatch_m3.sh` |
| 2134663 | node873 | Retriever `/health` wait timed out at 600 s (M3 ref 570 s; cold cache crossed *that* cliff) | bumped 600 → 1200 s |
| **2150167** | **node870** | — | **completed all 7 datasets** |

**Headline result**: simple-mean EM **0.169** vs M3's 0.155 (+0.014 abs, +9 % rel; +0.067 / +66 % rel vs pre-GRPO). 5 / 7 datasets improved over M3, 1 tied, 1 regressed (bamboogle, N = 125). Biggest wins on PopQA (+0.058), TriviaQA (+0.037), HotpotQA (+0.028). ACC / F1 widen the M3.1-vs-M3 gap to +12 % / +14 %. Full per-dataset breakdown + analysis: [`docs/report/RESULTS_m3.md`](../report/RESULTS_m3.md) §14.

**Training-curve comparison** (with-example z7kcxfof vs no-example el6s2d2h, generated by [`scripts/plot_m3_1_panel.py`](../../scripts/plot_m3_1_panel.py)):

![z7kcxfof vs el6s2d2h training curves](../report/archive/m0_a/comparison_z7kcxfof_vs_el6s2d2h.png)

The panel shows the structural divergence visually: same base model + algorithm + reward + data, only the prompt differs — yet z7kcxfof anchors at heavy-tool 2-call / 4-turn / ~2050-tok behaviour while el6s2d2h converges to standard 1-call / 3-turn / ~1100-tok at slightly higher reward. The no-example variant is a *pareto improvement* (lower compute, higher quality) on the same training budget. See [`docs/report/RESULTS_m3.md`](../report/RESULTS_m3.md) §14.7 for the full reading.

**Checkpoint also published to HF**: [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (parallel to [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0)).

## Deliverables

1. [x] Convert `el6s2d2h` global_step_2000 actor → HF (1.5 GB safetensors at `eval/qwen_3_0.6b_v0_no_ex/`).
2. [x] Add `P3_DECIDE_NO_EX_TEMPLATE` and `QWEN3_TEMPLATES` registry to `evaluation_research/flashrag/search_r1/templates.py`.
3. [x] Wire `prompt_mode='qwen3_p3_decide_no_ex'` through `active_pipeline.py` and `run_eval.py`; switch `qwen3` checks to family-prefix matching.
4. [x] Add `qwen3_0.6b_v0_no_ex` variant case to `scripts/run_m3.sh` and `scripts/sbatch_m3.sh`.
5. [x] Submit sbatch (2134645 failed on SGLang /health timeout → bumped 300 → 600 s; 2134663 failed on retriever /health timeout → bumped 600 → 1200 s; **2150167 completed in 1 h 32 m on node870**).
6. [x] Publish checkpoint to HF: [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) (parallel to [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0)).
7. [x] Job completes successfully (all 7 datasets — `evaluation_research/results/*/[...]_m3_qwen3_0.6b_v0_no_ex_seed1/`).
8. [x] Per-dataset EM / ACC / F1 added to [`docs/report/RESULTS_m3.md`](../report/RESULTS_m3.md) §14.4–§14.5.
9. [x] Side-by-side M3 vs M3.1 comparison (pre-GRPO / z7kcxfof / el6s2d2h) populated in [`RESULTS_m3.md`](../report/RESULTS_m3.md) §14.4–§14.5 and [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) §4 (parallel-reported across §4.1 headline / §4.2 per-dataset / §4.3 training runs / §4.6 wall-clock / §4.8 implications).
10. [x] Training-curve comparison panel built (`docs/report/archive/m0_a/comparison_z7kcxfof_vs_el6s2d2h.png`, generated by `scripts/plot_m3_1_panel.py`); embedded in this milestone, in [`RESULTS_m3.md`](../report/RESULTS_m3.md) §14.7, and in [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) §4.7.
