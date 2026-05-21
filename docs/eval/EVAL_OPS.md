---
title: EVAL OPS
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-02
---

# Evaluation Operations

How to run the Search-R1 evaluation sweeps and where the wall-clock goes. For *what* the sweep validates and the per-dataset numbers, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [../report/RESULTS_m1.md](../report/RESULTS_m1.md). Hardware reference is in [../setup/HARDWARE_COMPARISON.md](../setup/HARDWARE_COMPARISON.md) (4090 dev-box snapshot at [../setup/HARDWARE_4090.md](../setup/HARDWARE_4090.md)).

## Three sweep plans

The originally-requested 5 seeds × 7 datasets × 2 variants = 70 runs covers ~517 K example evaluations. Smoke profiling on Bamboogle base (slowest, multi-hop) gave **2.85 s/example** end-to-end with 8-worker concurrency on a single 4090.

| Plan | Runs | Examples | Wall-clock |
|---|---:|---:|---|
| **A** Full (5 seeds × 7 × 2)               | 70 | 517,130 | ~410 h (~17 days) |
| **C** 1 seed × 7 × 2 (full data)           | 14 | 103,426 | ~82 h (~3.4 days) |
| **B** 1 seed, large datasets subsampled to 1k | 14 | ~15,084 | ~12–18 h (~1 day) |

Bamboogle is on the slow end; NQ/TriviaQA/PopQA are factoid (~1 turn, ~1 s/example), so totals will skew toward the lower end of each band.

**Recommendation: Plan B first.** Finishes overnight; produces paper-comparable means (1k-row subsample SE ~1.5 pp on factoid datasets, small enough to detect any >3 pp deviation from paper). Plan A buys tighter error bars on already-correct numbers — useful for the final write-up only after Plan B confirms reproduction.

## Scripts (`/scripts`)

| Script | Purpose |
|---|---|
| `manage_sglang.sh` | Stop / start / wait / **switch** SGLang for `base`\|`instruct`. The `switch` subcommand stops, restarts, and waits for `/get_model_info`. |
| `subsample.sh` | Build `data_subsample/` with deterministic 1k samples of NQ/TriviaQA/PopQA/HotpotQA/2Wiki and full copies of Bamboogle/MuSiQue. Idempotent. |
| `run_one.sh` | Runs one `(variant, dataset, seed)` evaluation. Resume-aware: skips if a `metric_score.txt` already exists. |
| `run_variant_sweep.sh` | Iterates `seeds × datasets` for one variant. Refuses to run if SGLang is serving the wrong variant. |
| `sweep_a_full.sh` | Plan A — 5 × 7 × 2. |
| `sweep_b_reduced.sh` | Plan B — 1 × 7 × 2 on `data_subsample/`. |
| `sweep_c_one_seed.sh` | Plan C — 1 × 7 × 2 on full data. |
| `aggregate.py` | Walks `evaluation_search_r1/results/`, parses `metric_score.txt`, writes a markdown report grouped by `(dataset, variant, seed)`. |

### Save-note convention

```
evaluation_search_r1/results/<dataset>/<dataset>_<YYYY>_<MM>_<DD>_<HH>_<MM>_search_r1_<variant>_seed<N>/
```

`aggregate.py` parses the trailing `search_r1_<variant>_(seed|run)<N>` to group runs.

## How to run

Prereqs:

- Retriever up on `127.0.0.1:3005` (`curl /health` returns "healthy"). See [/local_retriever/README.md](../../local_retriever/README.md).
- SGLang launchable on `127.0.0.1:3000`; sweep scripts will switch models as needed.
- Eval venv at `/venv/evaluation_search_r1` (override with `PY=...`).

> Each plan kills and restarts SGLang twice (once per variant). Make sure nothing else needs the running process.

```bash
# Plan B (recommended)
nohup scripts/sweep_b_reduced.sh > /tmp/sweep_b.log 2>&1 &
disown
tail -f /tmp/sweep_b.log
# produces docs/report/RESULTS_m1.md

# Plan C — full datasets, 1 seed (~3.4 days)
nohup scripts/sweep_c_one_seed.sh > /tmp/sweep_c.log 2>&1 &
disown

# Plan A — full sweep (~17 days)
nohup scripts/sweep_a_full.sh > /tmp/sweep_a.log 2>&1 &
disown
```

Aggregate manually any time:

```bash
scripts/aggregate.py --output docs/RESULTS_NOW.md
```

## Where the wall-clock goes (smoke profile)

Per example on Bamboogle base, 8 concurrent workers, current SGLang flags:

```
GPU  ──  prefill ~600–1500 token prompt × 2 calls       ~0.4–0.6 s
GPU  ──  decode ~50–150 tokens × 2 calls (~80 tok/s)    ~0.8–1.6 s
CPU  ──  FAISS Flat IP over 21M × 768-dim (1 worker)    ~0.1–0.3 s
HTTP/coord overhead (gen ↔ retrieve ↔ gen)              ~0.2–0.4 s
                                                  ─────  ~1.5–2.9 s
```

Matches the observed 2.85 s.

### Bottleneck: GPU decode (~70–80 % of wall clock)

- 3B in bf16 on a 4090 ceilings at ~80–120 tok/s single-stream; ~3–5× that with batching. Memory-bandwidth-bound, not compute-bound. Adding more concurrent examples doesn't scale linearly.
- Original SGLang launch flags `--disable-radix-cache --disable-overlap` (in [/local_retriever/README.md](../../local_retriever/README.md)) cost ~50 % of multi-turn prefill (re-prefilling the entire prompt every turn instead of reusing the encoded prefix) and ~10–20 % decode throughput respectively. Worth a controlled A/B to see if our pipeline is OK with them re-enabled.

### Secondary: 1-worker FAISS (~5–10 % today, more on big datasets)

`IndexFlat` is 100–300 ms/query on CPU. With 8 concurrent eval threads, queries serialize behind a single worker. On PopQA (14,267 examples) that's 30–60 min of pure retrieval queueing per run.

Mitigations available:

- `--num_retriever 4` for parallel CPU FAISS workers (read-only, fully safe).
- IVF-SQ8 index — default in [retriever_config.yaml](../../local_retriever/retriever_config.yaml); lives at [`/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index`](../../local_retriever/indexes/) (16 GB, ~3-10× faster than flat, <1 % recall hit). Download from HF: [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index).
- GPU FAISS — wired up in the retriever, but cannot co-exist with SGLang on the same 24 GB 4090 (16 GB index + 22 GB SGLang > 24 GB). Useful only when SGLang is stopped, or for offline batch retrieval.

### Speedup ranking (no model/data changes)

| Change | End-to-end speedup | Effort | Risk |
|---|---|---|---|
| Drop `--disable-radix-cache` and `--disable-overlap` | **1.5–2×** | flag change | low — needs A/B to confirm EM unchanged |
| Bump `INFERENCE_MAX_WORKERS` 8 → 16/32 | 1.2–1.5× | one-line | medium — too high causes KV thrashing |
| `--num_retriever 4` (CPU FAISS) | 1.0–1.1× today, 1.3–1.5× on PopQA/2Wiki | trivial | low |
| FAISS Flat → IVF-SQ8 | 1.05–1.10× end-to-end, ~5× on retrieval alone | already built | low — ~1 % recall |
| Move to A100 80 GB / H100 | 2–3× | hardware swap | none |

Realistic sweet spot for this 4090: re-enable radix-cache + overlap, bump workers to 16, run retriever with 4 CPU workers. Plan B drops from ~12–18 h to ~6–10 h, Plan C from ~3.4 days to ~1.5–2 days, Plan A from ~17 days to ~7–10 days.

## Operational risks

| Risk | Mitigation |
|---|---|
| Mid-run crash (OOM, SGLang hang, retriever crash) | `run_one.sh` is resume-aware; restart the same `sweep_*.sh`. |
| SGLang fails to load on switch | `manage_sglang.sh switch` waits up to 10 min for `/get_model_info` and exits non-zero on timeout. |
| Disk fill (intermediate JSONs hundreds of MB per run) | Clean old runs in `evaluation_search_r1/results/`. |
| Retriever bottleneck on big datasets | Use `--num_retriever 4` or IVF-SQ8 (see above). |
| Determinism | SGLang at `temperature=0` (greedy, matches paper eval per [PAPER_VS_OURS_AUDIT.md D3](PAPER_VS_OURS_AUDIT.md)); `seed` in `save_note` is a label only (SGLang ignores FlashRAG's `seed`). |
