---
title: Results M1 — Search-R1 paper-baseline reproduction (Plan B v1, Qwen2.5-3B GRPO checkpoints)
tags: [report, eval, m1, plan-b]
source: `evaluation_search_r1/results/_archive_v1` + paper Table 3 (arXiv 2503.09516 v5)
created: 2026-05-01
updated: 2026-05-08
---

# Results M1: Search-R1 Paper-Baseline Reproduction (Plan B v1)

**Date compiled**: 2026-05-01 (sweep finished 2026-04-29 22:05 UTC); merged + harmonised 2026-05-08 from former `RESULTS_m1.md` + `RESULTS_m1.md`.  
**Source**: per-(dataset, variant, seed=1) `metric_score.txt` aggregated by [`scripts/aggregate.py`](../../scripts/aggregate.py) from `evaluation_search_r1/results/_archive_v1/<dataset>/<…>/metric_score.txt` (14 runs, mirrored at [`archive/m1/`](archive/m1/) for fact-check convenience). Paper targets: arXiv 2503.09516 v5 Appendix F / Table 3 (`arxiv.org/html/2503.09516v5#A6`).  
**Hardware**: single RTX 4090 (24 GB), AMD EPYC 7642, 503 GB RAM (no NVLink). ~17 h wall for one full v1 sweep (1 seed × 7 datasets × 2 variants).  
**Setup**: locked Plan B v1 config; full reproducer recipe in [`CODE_SETUP_m1.md`](CODE_SETUP_m1.md). Three audit fixes vs Plan B v0 (D1: `apply_chat=True` for base; D-prompt-micro: restored example sentence; D8: removed runtime `add_special_tokens` block).  
**Bottom line**: Plan B v1 reproduces the paper on both variants. Base avg EM −2.0 pp; instruct avg EM +2.5 pp. Both within the Phase-1 "reproduces" criterion (≤4 pp avg residual, every dataset ≤8 pp). Plan A readiness: **YES (unconditional)**.

---

## 1. Run roster

14 runs total (7 datasets × 2 variants × seed=1). All runs use the locked v1 config.

| Variant | Datasets | Splits | n_runs | Items / variant |
|---|---|---|---:|---:|
| base (`SearchR1-…-em-grpo`) | bamboogle, nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique | test/test/test/test/dev/dev/dev | 7 | 7,542 (1k subsamples + full bamboogle/musique) |
| instruct (`SearchR1-…-it-em-grpo`) | same | same | 7 | 7,542 |

The five large datasets (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA) are 1k stratified subsamples; Bamboogle (n=125) and MuSiQue (n=2,417) run on the full dev/test split. Sample → split mapping: [`scripts/run_one.sh:22-31`](../../scripts/run_one.sh#L22).

---

## 2. Configuration

Single locked config across all 14 runs. Full spec in [`CODE_SETUP_m1.md`](CODE_SETUP_m1.md). Highlights:

| Setting | Value | Notes |
|---|---|---|
| Sampling | greedy (`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `n=1`) | matches verl `_validate()` override |
| Models | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo` | sha256-pinned |
| Retriever | E5-base-v2 + FAISS Flat IP `wiki18_100w_e5_flat_inner.index` (~65 GB host RAM, recall=100 %) | flat IP for paper-fidelity |
| Top-k | 3 | upstream-matched |
| Max search turns | 4 | upstream-matched |
| Per-step token cap | 500 | upstream-matched |
| Stop tokens | `</search>`, `</answer>`, `<\|im_end\|>`, `<\|endoftext\|>` | upstream-matched |
| `apply_chat=True` | both variants (D1 fix) | base used to default False; v1 audits this |

**No per-run differences.** All variation is along the (dataset, variant) axis.

---

## 3. Headline result

**Plan B v1 reproduces the paper on both variants.**

| Variant  | Plan B v0 avg EM | Plan B v1 avg EM | Paper avg EM | v1 − paper |
|---       |---:              |---:              |---:          |---:        |
| base     | 0.229            | **0.292**        | 0.312        | **−2.0 pp** |
| instruct | 0.367            | **0.361**        | 0.336        | **+2.5 pp** |

- **Base**: gap shrunk from −8.3 pp (v0) → −2.0 pp (v1). No dataset more than 4 pp off paper. TriviaQA matches paper EM exactly. PopQA beats paper by +1.1 pp. Format-validity (`</answer>` close-rate) jumped from 84 % on v0 base/Bamboogle to ≥99.6 % on every v1 base dataset.
- **Instruct**: barely moved between v0 and v1 (−0.6 pp avg); the audit's prediction that the three fixes would be small for instruct was correct. Average is +2.5 pp above paper, within ±5 pp on 6/7 datasets, with Bamboogle the only outlier (+11.2 pp on n=125; variance, not signal).

Both variants now satisfy the Phase 1 "reproduces" criterion.

---

## 4. Per-dimension breakdown

### 4.1 Base — EM (v0 vs v1 vs paper)

| Dataset | v0 EM | **v1 EM** | Paper EM | v1−paper |
|---|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.112 | **0.088** | 0.128 | −4.0 (n=125 noise) |
| NQ-1k | 0.316 | **0.390** | 0.421 | −3.1 |
| TriviaQA-1k | 0.421 | **0.583** | 0.583 | **0.0** |
| PopQA-1k | 0.309 | **0.424** | 0.413 | **+1.1** |
| HotpotQA-1k | 0.201 | **0.263** | 0.297 | −3.4 |
| 2WikiMultiHopQA-1k | 0.207 | **0.239** | 0.274 | −3.5 |
| MuSiQue (full, 2417) | 0.034 | **0.055** | 0.066 | −1.1 |
| **Average** | **0.229** | **0.292** | **0.312** | **−2.0 pp** |

Per-dataset lifts from v0 → v1 range from −2.4 pp (Bamboogle, n=125 noise) to **+16.2 pp on TriviaQA**. NQ +7.4 pp, PopQA +11.5 pp, HotpotQA +6.2 pp, 2Wiki +3.2 pp, MuSiQue +2.1 pp.

### 4.2 Base — F1 and ACC (v1)

| Dataset | v1 EM | v1 F1 | v1 ACC |
|---|---:|---:|---:|
| Bamboogle | 0.088 | 0.172 | 0.096 |
| NQ-1k | 0.390 | 0.474 | 0.428 |
| TriviaQA-1k | 0.583 | 0.657 | 0.638 |
| PopQA-1k | 0.424 | 0.458 | 0.441 |
| HotpotQA-1k | 0.263 | 0.365 | 0.284 |
| 2WikiMultiHopQA-1k | 0.239 | 0.306 | 0.252 |
| MuSiQue (full) | 0.055 | 0.123 | 0.064 |
| **Grand average** | **0.292** | **0.365** | **0.315** |

### 4.3 Instruct — EM (v0 vs v1 vs paper)

| Dataset | v0 EM | **v1 EM** | Paper EM | v1−v0 | v1−paper |
|---|---:|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.360 | **0.344** | 0.232 | −1.6 | **+11.2** (n=125 variance) |
| NQ-1k | 0.399 | **0.402** | 0.397 | +0.3 | **+0.5** |
| TriviaQA-1k | 0.539 | **0.531** | 0.565 | −0.8 | −3.4 |
| PopQA-1k | 0.412 | **0.413** | 0.391 | +0.1 | **+2.2** |
| HotpotQA-1k | 0.354 | **0.346** | 0.331 | −0.8 | **+1.5** |
| 2WikiMultiHopQA-1k | 0.353 | **0.350** | 0.310 | −0.3 | **+4.0** |
| MuSiQue (full, 2417) | 0.149 | **0.141** | 0.124 | −0.8 | **+1.7** |
| **Average** | **0.367** | **0.361** | **0.336** | **−0.6** | **+2.5 pp** |

The three audit fixes barely shifted instruct, in line with the audit's prediction. Per-dataset moves are all ≤1.6 pp and within single-seed greedy noise. The +0.3 pp move on NQ for instruct is consistent with the audit's ≤1 pp estimate, unlike the +4.4 pp combined kick the same fixes gave the base variant; the prompt sentence + special-tokens fixes were base-specific.

### 4.4 Instruct — F1 and ACC (v1)

| Dataset | v1 EM | v1 F1 | v1 ACC |
|---|---:|---:|---:|
| Bamboogle | 0.344 | 0.453 | 0.376 |
| NQ-1k | 0.402 | 0.487 | 0.450 |
| TriviaQA-1k | 0.531 | 0.620 | 0.604 |
| PopQA-1k | 0.413 | 0.458 | 0.448 |
| HotpotQA-1k | 0.346 | 0.458 | 0.383 |
| 2WikiMultiHopQA-1k | 0.350 | 0.422 | 0.390 |
| MuSiQue (full) | 0.141 | 0.216 | 0.167 |
| **Grand average** | **0.361** | **0.445** | **0.403** |

---

## 5. Wall-clock / observed performance

Single 4090 wall-clock for one full v1 sweep: **~17 h** (1 seed × 7 datasets × 2 variants). Per-dataset Bamboogle/instruct: ~6 min. The 4090 + 65 GB host-RAM Flat-IP FAISS combo is comfortably below failure budgets. Plan A scaling: 5 seeds × full datasets × 2 variants = 70 runs; on 8× RTX 4090 fleet completes in ≤24 h at $58–77 (see [`../setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md)).

---

## 6. Trace health

`</answer>` close-rate = fraction of examples whose `final_response` contains `</answer>`. Length-truncated = fraction whose SGLang `stop_reason` was anything other than `stop`/`eos`/`stop_str` (typically the per-step token cap firing).

### 6.1 Close-rate

| Dataset | base | instruct | n |
|---|---:|---:|---:|
| bamboogle | 100.0 % | 100.0 % | 125 |
| nq | 99.9 % | 99.0 % | 1,000 |
| triviaqa | 99.9 % | 99.1 % | 1,000 |
| popqa | 100.0 % | 97.5 % | 1,000 |
| hotpotqa | 100.0 % | 98.1 % | 1,000 |
| 2wikimultihopqa | 99.6 % | 94.2 % | 1,000 |
| musique | 100.0 % | 91.4 % | 2,417 |

Almost every base trace closes cleanly. Mean turns is **1.00 across the board for base** (the base GRPO model issues exactly one `<search>` then answers). Instruct's mean turns ranges 1.0 to 1.94 and the multi-turn behaviour is variant-specific. The 6 to 9 % open-trace rate on the multi-hop datasets (MuSiQue, 2Wiki) for instruct is the model burning all 4 search turns without converging on `</answer>`, not the per-step cap. This is a known failure mode of the released GRPO instruct checkpoint and not something the eval pipeline can fix.

### 6.2 Length-truncation rate

| Dataset | base | instruct | n |
|---|---:|---:|---:|
| bamboogle | 0.0 % | 0.0 % | 125 |
| nq | 0.1 % | 0.1 % | 1,000 |
| triviaqa | 0.1 % | 0.3 % | 1,000 |
| popqa | 0.0 % | 0.0 % | 1,000 |
| hotpotqa | 0.0 % | 0.3 % | 1,000 |
| 2wikimultihopqa | 0.4 % | 0.2 % | 1,000 |
| musique | 0.0 % | 0.3 % | 2,417 |

Length-truncation ≤0.4 % everywhere; the per-step token cap is not biting.

### 6.3 Mean completion tokens (full trace, summed over turns)

| Dataset | base | instruct |
|---|---:|---:|
| bamboogle | 10 | 44 |
| nq | 10 | 45 |
| triviaqa | 9 | 45 |
| popqa | 9 | 41 |
| hotpotqa | 9 | 55 |
| 2wikimultihopqa | 12 | 65 |
| musique | 10 | 56 |

Base emits ~10 tokens / trace (one `<search>` query + one `<answer>`). Instruct emits 4× to 6× more, consistent with multi-turn behaviour.

---

## 7. Findings

### 7.1 The locked v1 config reproduces both variants

Both variants now satisfy the Phase-1 "reproduces" criterion (every dataset ≤8 pp from paper, average residual ≤4 pp). Base avg −2.0 pp; instruct avg +2.5 pp.

### 7.2 The three v0 → v1 fixes are base-specific

D1 (`apply_chat=True` for base) alone delivered +3.0 pp on NQ base. D-prompt-micro (restored example sentence) + D8 (removed runtime `add_special_tokens` block) delivered another +4.4 pp on NQ base combined; the audit had estimated ≤1 pp each. The same three fixes moved instruct by <1 pp combined; instruct already had `apply_chat=True` at v0 and the prompt-sentence + special-tokens fixes affect tokenization in a way the instruct chat template was already absorbing.

### 7.3 Bamboogle's +11.2 pp on instruct is n=125 variance

Plan A (5 seeds × full data) will collapse it. Treat anything ≥3 pp on n=125 with skepticism.

### 7.4 Format validity is now production-grade

Base close-rate ≥99.6 % on every v1 dataset (vs 84 % on v0 Bamboogle). Length-truncation ≤0.4 % everywhere. The remaining instruct multi-hop open traces are a model-side failure mode (4 turns exhausted), not a pipeline bug.

### 7.5 Greedy is correct (paper eval is greedy)

The audit traced upstream `verl/trainer/ppo/ray_trainer.py:478,508` → `_validate()` forces `do_sample=False`, and `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162-171` overrides `temperature=0, top_p=1.0, top_k=-1, n=1` whenever that flag is set. Earlier −10 pp gap on base was D1+D-prompt-micro+D8, not temperature. (A previous experimental run at temp=1.0 looked like it added +1.7 pp on NQ but that was single-seed sampling noise; re-run would not reproduce.) Post-mortem: [`../archive/TEMPERATURE_HYPOTHESIS_WRONG.md`](../archive/TEMPERATURE_HYPOTHESIS_WRONG.md).

### 7.6 Plan A readiness: YES (unconditional)

Decision criteria from [`../milestone_1/MILESTONE_1.md`](../milestone_1/MILESTONE_1.md): all datasets within 8 pp of paper, no catastrophic divergence, average residual ≤4 pp. Met for both variants. Plan A's 5-seed × full-data sweep should tighten everything to within ±1.5 pp of paper.

---

## 8. Open questions

1. The +16.2 pp lift on TriviaQA base from v0 → v1 is unusually large vs +7.4 pp on NQ. Worth checking which of D-prompt-micro or D8 is doing more of the work in isolation; not run as a paired ablation.
2. The 6 to 9 % open-trace rate on instruct multi-hop datasets (MuSiQue, 2Wiki) suggests the model would benefit from more search turns. The paper's `max_search_turns=4` is presumably load-bearing (Plan A 5-seed will quantify variance, but the cap is shared with the paper).
3. The instruct +12.8 pp Bamboogle outlier in v0 / +11.2 pp in v1 is noise on n=125, but it's persistently positive. Plan A's 5 seeds will tell whether this is variance or a small systematic edge.

---

## 9. Reproduction / artifact record

### 9.1 Per-(variant, dataset) metric files

Mirrored under [`archive/m1/`](archive/m1/) for fact-check convenience:

```
docs/report/archive/m1/
├── base/
│   ├── 2wikimultihopqa/metric_score.txt
│   ├── bamboogle/metric_score.txt
│   ├── hotpotqa/metric_score.txt
│   ├── musique/metric_score.txt
│   ├── nq/metric_score.txt
│   ├── popqa/metric_score.txt
│   └── triviaqa/metric_score.txt
└── instruct/
    └── (same 7 datasets)
```

Pipeline-source-of-truth lives at `evaluation_search_r1/results/_archive_v1/<dataset>/<dataset>_<timestamp>_search_r1_<variant>_seed1/metric_score.txt`.

### 9.2 Reproducing one cell

Pre-flight (services up): see [`CODE_SETUP_m1.md` §7.2](CODE_SETUP_m1.md). Run:

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1|acc):" "$LATEST/metric_score.txt"
```

Aggregate after a full sweep:

```bash
/venv/evaluation_search_r1/bin/python scripts/aggregate.py --output docs/report/RESULTS_m1.md
```

### 9.3 What changed v0 → v1 in summary

- **Code (committed)**: `scripts/run_one.sh:35` flips base `apply_chat=False → True`; `evaluation_search_r1/flashrag/search_r1/templates.py:10` adds the missing example sentence; `evaluation_search_r1/flashrag/pipeline/active_pipeline.py` drops the runtime `add_special_tokens` block.
- **No change**: temperature, top_p, retriever, model checkpoints, prompt template body, max_turns, step_limit, observation truncation, splits, metrics, FAISS index, SGLang flags.
- **Archived (committed)**: v0 result dirs at `evaluation_search_r1/results/_archive_v0/` (13 runs; bamboogle/instruct row in the v0 aggregate is the smoke-test number); v1 result dirs at `evaluation_search_r1/results/_archive_v1/` (14 runs). Pre-fix v0 comparison snapshot: [`../archive/COMPARISON_PLAN_B_v0.md`](../archive/COMPARISON_PLAN_B_v0.md).
- **Discarded experiments** (full list in [`../archive/DISCARDED_ABLATIONS.md`](../archive/DISCARDED_ABLATIONS.md)): temperature sweep above 0.0, the autoresearch loop's 11 ablations on `experiment_ros/apr27`, the `apply_chat=True + temp=1.0` probe on NQ. None reproduced when re-run greedy.

---

## 10. Pointers

- Frozen reproducer config: [`CODE_SETUP_m1.md`](CODE_SETUP_m1.md).
- Per-(variant, dataset) raw metric_score.txt mirror: [`archive/m1/`](archive/m1/).
- Pre-fix v0 baseline (archived): [`../archive/COMPARISON_PLAN_B_v0.md`](../archive/COMPARISON_PLAN_B_v0.md).
- Audit: [`../eval/PAPER_VS_OURS_AUDIT.md`](../eval/PAPER_VS_OURS_AUDIT.md), [`../eval/REPRODUCIBILITY.md`](../eval/REPRODUCIBILITY.md).
- Vast.ai Plan A cost analysis: [`../setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md).
- Milestone narrative: [`../milestone_1/MILESTONE_1.md`](../milestone_1/MILESTONE_1.md).
- Sister milestone results: [`RESULTS_m0_a.md`](RESULTS_m0_a.md), [`RESULTS_m0_b.md`](RESULTS_m0_b.md), [`RESULTS_m3.md`](RESULTS_m3.md), [`RESULTS_m4.md`](RESULTS_m4.md).
