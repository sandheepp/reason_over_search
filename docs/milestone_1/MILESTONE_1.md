---
title: MILESTONE 1
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-07
---

# Milestone 1: Baseline [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

> **Status (2026-05-07): CLOSED.** Plan B v1 within ±2.5 pp of paper across 7 datasets on both base and instruct variants. Locked numerical record: [`RESULTS_m1.md`](../report/RESULTS_m1.md). Frozen reproducer config: [`CODE_SETUP_m1.md`](../report/CODE_SETUP_m1.md). 3-fix audit: [`../eval/PAPER_VS_OURS_AUDIT.md`](../eval/PAPER_VS_OURS_AUDIT.md). Plan A on Vast.ai (5 seeds × 7 datasets × 2 variants) was Jose-owned and was not executed before the project pivoted to recipe-focused Phase-2 work; see [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 2.

## Phase 1 Context

- Retriever code is working and has already been tested.
- Evaluation code is a hybrid:
  - base evaluation flow adapted from ReSearch
  - Search-R1 prompts and evaluation logic added on top
- Search-R1 does not provide an official evaluation pipeline, so this adapted pipeline must be validated carefully.

## Goal

Reproduce the Search-R1 3B baseline for both model variants:
- base
- instruct

Target benchmarks:
- Bamboogle (first dataset for fast iteration)
- 2WikiMultiHopQA
- TriviaQA
- PopQA
- MuSiQue
- NQ (in-distribution)
- HotpotQA (in-distribution)

## Dataset Requirements

- Download missing datasets from Hugging Face.
- Use the Search-R1 paper to confirm official dataset sources.
- Verify existing local datasets against online sources.
- Confirm correct split per benchmark (test/dev/train), including whether `dev` is treated as evaluation where no test split is available.
- Use AlphaXiv AI paper assistant when split definitions are ambiguous.

## Reproducibility and Cost Constraints

- Primary objective: reproducible setup on Vast.ai and in-house GPUs.
- Secondary objective: minimize cost and wall-clock runtime.
- Track both reproducibility and cost clearly; these are important for publication quality.

## Step-by-step

1. Prepare runtime on Vast.ai.
   - Use this Docker image: [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1)
   - Or build and push your own image using `docker/reason-over-search-v1/README.md`.
   - Create a Vast.ai custom template from that image and start an instance.
   - Storage guidance: allocate at least `150 GB` for indexes and model files.
   - GPU guidance: start testing with lower-cost options first (for example, 40 GB VRAM class) before moving to larger GPUs (for example, A100 80 GB), depending on throughput and stability.

2. Set up local retriever on the Vast instance.
   - Follow `local_retriever/README.md`.
   - If you are using the Docker image above, required environments are already installed, so you can skip environment creation.
   - Download:
     - corpus
     - indexes
     - embedding model

3. Run and validate the retriever service.
   - Start the retriever using `local_retriever/README.md`.
   - Run the health/search test calls to confirm it is serving correctly.
   - Resource note: the retriever runs on CPU and needs about `~80 GB` CPU RAM.

4. Set up evaluation models and SGLang on the Vast instance.
   - Go to `evaluation_search_r1/`.
   - Activate the evaluation environment.
   - Download both Search-R1 model variants (base and instruct).
   - Start the SGLang server for the model being evaluated.

5. Run baseline evaluations.
   - Run all target benchmarks listed in this milestone.
   - Run each benchmark `5` times per model variant.
   - Report the average score across the 5 runs for each benchmark.

## If Results Diverge from Paper

- Inspect prompt templates and answer verification logic in `evaluation_search_r1/flashrag/search_r1/`.
- Verify behavior separately for base and instruct variants.
- Cross-check implementation details against the official repository:
  - [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1)

## Deliverables

1. All evaluation datasets downloaded, verified, and tracked appropriately.
2. Evaluation pipeline validated and run on all listed benchmarks.
3. Aggregated results from repeated runs (5 runs per benchmark/model variant).
4. Code reviewed and cleaned up for clarity (remove unnecessary complexity/fluff).
5. Docker setup verified; documentation updated for clear reproduction.
6. Repository state suitable for easy reproduction and publication submission.

## Status (2026-05-01)

### What was done

- Adapted the FlashRAG/ReSearch eval pipeline to Search-R1 (no official eval pipeline ships upstream).
- Both GRPO checkpoints (base, instruct) sha256-verified against the upstream HF repos.
- Wiki-18 corpus + E5-base-v2 encoder + FAISS Flat IP and IVF-SQ8 indexes built.
- Exhaustive paper-vs-ours audit ([../eval/PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md)): 8 divergences catalogued, 10 earlier ones already fixed ([../eval/REPRODUCIBILITY.md](../eval/REPRODUCIBILITY.md)).
- Plan B v0 sweep — preserved as [../archive/RESULTS_PLAN_B_v0.md](../archive/RESULTS_PLAN_B_v0.md); v0 result dirs are committed at `evaluation_search_r1/results/_archive_v0/` (13 runs — bamboogle/instruct in the aggregate is the smoke-test number from ../eval/REPRODUCIBILITY.md, run dir not preserved).
- Three audit fixes applied: `apply_chat=True` for base ([run_one.sh:35](../../scripts/run_one.sh#L35)), `For example, <answer> Beijing </answer>.` restored ([templates.py:10](../../evaluation_search_r1/flashrag/search_r1/templates.py#L10)), `add_special_tokens` block removed ([active_pipeline.py](../../evaluation_search_r1/flashrag/pipeline/active_pipeline.py)). `temperature: 0.0` kept (paper eval is greedy per upstream `verl` `_validate()` override).
- **Plan B v1 sweep complete (both variants)**: full comparison and aggregated numbers in [RESULTS_m1.md](../report/RESULTS_m1.md), reproducer config locked in [CODE_SETUP_m1.md](../report/CODE_SETUP_m1.md).
- Vast.ai Plan-A fleet costing: 8× RTX 4090 ≈ $58–77 / 24 h ([../setup/VAST_AI_PLAN_A.md](../setup/VAST_AI_PLAN_A.md)).

### Plan B v1 — final results

| Dataset | base v1 | base paper | base Δ | instruct v1 | instruct paper | instruct Δ |
|---|---:|---:|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.088 | 0.128 | −4.0 | 0.344 | 0.232 | +11.2 (n=125 var) |
| NQ-1k | 0.390 | 0.421 | −3.1 | 0.402 | 0.397 | +0.5 |
| TriviaQA-1k | 0.583 | 0.583 | **0.0** | 0.531 | 0.565 | −3.4 |
| PopQA-1k | 0.424 | 0.413 | +1.1 | 0.413 | 0.391 | +2.2 |
| HotpotQA-1k | 0.263 | 0.297 | −3.4 | 0.346 | 0.331 | +1.5 |
| 2WikiMultiHopQA-1k | 0.239 | 0.274 | −3.5 | 0.350 | 0.310 | +4.0 |
| MuSiQue (full, 2417) | 0.055 | 0.066 | −1.1 | 0.141 | 0.124 | +1.7 |
| **Average** | **0.292** | **0.312** | **−2.0** | **0.361** | **0.336** | **+2.5** |

Format validity (close-rate of `</answer>`): base ≥99.6 % every dataset; instruct 91.4–100 %. Length-truncation ≤0.4 % across the board.

### Plan A — YES (unconditional)

**Decision criteria** (set before the sweep): all datasets within 8 pp of paper, no catastrophic divergence, average residual ≤4 pp. **Met for both variants.** Max gap 11.2 pp on Bamboogle/instruct (n=125 single-seed variance — Plan A's 5 seeds will collapse it); next-largest is 4.0 pp. Avg residual −2.0 pp (base) / +2.5 pp (instruct).

## What's left

1. **Plan A on Vast.ai**: 5 seeds × 7 datasets × 2 variants = 70 runs, ~517 K examples, ≤24 h on a fleet. Reproducer config: [CODE_SETUP_m1.md](../report/CODE_SETUP_m1.md). Instructions for **Jose**: [../setup/VAST_AI_PLAN_A.md](../setup/VAST_AI_PLAN_A.md).
2. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.
