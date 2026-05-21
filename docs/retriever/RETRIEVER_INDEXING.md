---
title: RETRIEVER INDEXING
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-02
---

# Indexing & Retrieval Internals

How dense retrieval is wired up here, what the index choices cost in RAM/VRAM/recall/latency, and the experiments worth running before committing to a swap.

## Basics — how it works

Dense retrieval needs two artifacts kept row-aligned:

- **Corpus** ([`corpus/wiki18_100w.jsonl`](corpus/)) — raw text. Each line is `{"id": ..., "contents": ...}`. This is what gets returned to the caller.
- **Index** ([`indexes/wiki18_100w_e5_*.index`](indexes/)) — a FAISS index of pre-computed E5 embeddings for every passage. This is what actually gets searched.

**The invariant:** row `i` in the FAISS index is the embedding of row `i` in the corpus jsonl.

`search(query, top_n)` flows:

1. **Embed** the query with E5 (prefix `"query: "`, mean pool, L2 normalize).
2. **FAISS** `index.search(emb, k=top_n)` → integer row IDs + scores. No text.
3. **Corpus lookup** — `corpus[i]` for each returned ID fetches `{id, contents}`.
4. Wrap into Pydantic `Document` models and return.

`batch_search` is the same pipeline vectorized: one encoder pass over `(N, dim)`, one FAISS call, one bulk corpus lookup.

The corpus is memory-mapped via HuggingFace Datasets (~14 GB RSS, OS page-cache), so its cost is mostly evictable. The **index** is what dominates the resident memory bill.

## Options for wiki-18 @ top-10

What you can swap in via `--index` / yaml. All numbers are for E5-base-v2 at top-10, `nprobe=64` on IVF variants. CPU column is the local 96-thread EPYC 7642; GPU columns assume the index is fully resident in VRAM with `useFloat16=True` for flat-style indexes.

| Index | RAM/VRAM | Recall@10 | Score | CPU (EPYC 96t) | RTX 4090 | A100 80 | H100 80 |
|---|---|---|---|---|---|---|---|
| `Flat,IP` | 65 GB | 100 % | exact | 100–300 ms | n/a — 32.5 GB fp16 > 24 GB | 10–25 ms | 5–12 ms |
| `HNSW32,Flat` | ~70 GB | 97–99 % | exact | 0.5–3 ms | n/a — no FAISS GPU HNSW | n/a | n/a |
| `IVF4096,SQ8` *(default; download from HF [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index))* | ~16 GB | 96–99 % | very close | 30–100 ms | 5–15 ms | 3–8 ms | 2–6 ms |
| `IVF65536,SQ8` | ~16 GB | 96–99 % | very close | 20–80 ms | 3–10 ms | 2–6 ms | 1–4 ms |
| `IVF65536,PQ96` | ~3 GB | 88–95 % | approximate | 5–30 ms | 1–4 ms | 0.5–2 ms | 0.5–1.5 ms |
| `IVF65536,PQ32` | ~1 GB | 75–88 % | coarse | 2–15 ms | <1–3 ms | <1 ms | <1 ms |

Caveats:
- Single-query latencies. Encoder time (E5 forward pass) adds ~3–8 ms on a 4090 / ~1–3 ms on H100; corpus lookup adds ~0.1 ms (mmap'd page cache).
- IVF latency scales near-linearly with `nprobe`; dropping to 16 roughly quarters the FAISS time for a few percent of recall.
- A100/H100 numbers are from public FAISS benchmarks at this corpus size; not measured on this box. Treat as ranges, not commitments.
- HNSW on GPU isn't supported by FAISS — stay on CPU for that index.

### Why we built `IVF4096,SQ8` specifically

FAISS's `min_points_per_centroid=39` floor needs ≥ 2.55 M training rows for `nlist=65536` but only 160 K for `nlist=4096`. With the 1 M training sample drawn from the existing flat index, 4096 cells is comfortably above the floor; 65 536 would have trained on a thin sample with degraded centroids. Bumping the training sample to ~5 M would unlock the 65k variant for marginally better latency at the same memory cost.

`nprobe` is the dominant accuracy-vs-latency knob on any IVF index. Sweet spot for SQ8 is `nprobe=64` — already wired up in [retriever_config.yaml](../../local_retriever/retriever_config.yaml) and applied automatically in [`load_index`](../../local_retriever/flashrag/retriever/retriever.py).

## Impact on a Search-R1-style RAG pipeline

Three axes quantization affects:

1. **Recall@k** — fraction of the flat top-k that the quantized index also returns.
2. **Scores** when `return_score=true` — PQ/SQ reconstruct approximate vectors; absolute values aren't comparable across index types, only rankings.
3. **Latency** — almost always *improves*. Smaller vectors fit in cache; IVF scans only `nprobe` cells out of `nlist` instead of all 21 M vectors.

End-to-end answer quality is less sensitive than raw recall numbers suggest, because:

- Wiki-18 has many near-duplicate passages (same article, adjacent chunks). Missing the "true" #3 and getting a near-twin at rank #4 rarely hurts.
- E5 produces peaky similarity distributions — the top few are clearly separated, so quantization mostly reshuffles among strong candidates.
- LLMs reading the docs don't care about millisecond-scale score differences.

**Concrete expectations** for the Search-R1 eval pipeline:

| Index | EM impact (NQ / TriviaQA) | Latency impact (end-to-end) | Verdict |
|---|---|---|---|
| `HNSW32,Flat` | indistinguishable | 1.2–2× faster | safe swap if RAM permits |
| `IVF4096,SQ8` (CPU) | indistinguishable | 1.05–1.10× faster | safe swap, recommended default for sweeps |
| `IVF4096,SQ8` (GPU) | indistinguishable | 1.10–1.20× faster | only when SGLang is offline |
| `IVF65536,PQ96` | 1–3 pp drop | 1.10–1.15× faster | noticeable in eval, rarely in production |
| `IVF65536,PQ32` | measurable regression | fastest | only if RAM-starved |

## Experiments to run

This retriever is the kind of backend used by **agentic-search reasoning models** — systems where the LLM decides mid-chain-of-thought when to call `/search`, reads the results, and continues reasoning. A concrete example is [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366) (Li et al., EMNLP 2025), which wraps retrieval in a "Reason-in-Documents" (RiD) module that filters retrieved passages before they re-enter the reasoning context.

Each question triggers multiple retrieval calls; cumulative latency matters more than per-call latency, and RiD may absorb recall losses a vanilla top-k pipeline can't tolerate. Below are experiments this repo is well-positioned to run.

### A. Index quality vs end-to-end answer accuracy

Sweep `index_path` through increasingly lossy indexes; measure downstream QA on Search-o1-style benchmarks via [/evaluation_search_r1/](../../evaluation_search_r1/).

- **Variables:** `Flat,IP` → `HNSW32,Flat` → `IVF4096,SQ8` → `IVF65536,PQ96` → `IVF65536,PQ32`.
- **Metrics:** EM, F1, passage recall@10, passage recall@5.
- **Hypothesis:** SQ8 and HNSW are near-lossless; PQ96 is the break-point where EM drops 1–3 pp; PQ32 is noticeably worse.

### B. Does RiD (or any doc-refiner) absorb recall loss?

The compelling claim in Search-o1 is that RiD filters noise. If true, a lossier index returning more noisy-but-relevant docs should hurt less than recall@10 suggests.

- Fix the retriever to `IVF65536,PQ96`. Run with **and without** RiD. Compare EM delta vs `Flat,IP` with/without RiD.
- **Hypothesis:** `(Flat − PQ96) with RiD < (Flat − PQ96) without RiD`. Quantization is cheaper when a refiner exists downstream.

### C. Latency compounding under agentic loops

Agentic systems issue `N` retrieval calls per question (often 2–6). Wall-clock scales with `N × per_call_latency`.

- For each index type, measure median per-call latency at `top_n ∈ {3, 5, 10, 20}`.
- Sample Search-o1 traces on HotpotQA to get the empirical `N` distribution.
- Plot end-to-end response time vs index type, broken into (encoding, FAISS, corpus lookup, RiD).
- **Hypothesis:** For `N ≥ 3`, HNSW beats Flat by a wide margin and nearly matches IVF-PQ on latency while keeping exact scores.

### D. `nprobe` sweep for IVF indexes

`nprobe` is the dominant accuracy-vs-latency knob once you're on IVF. First experiment to do after swapping.

- Fix index = `IVF4096,SQ8`. Sweep `nprobe ∈ {1, 4, 16, 32, 64, 128, 256, 512}`.
- For each: recall@10 vs `Flat` on 1 K held-out queries + per-call latency.
- Also report downstream EM on NQ to check whether recall@10 is a faithful proxy.

### E. Top-k sensitivity under lossy indexes

Classic compensation for lossy retrieval: "retrieve more, let the reader filter."

- For each `(index_type, top_n)` in `{Flat, SQ8, PQ96} × {3, 5, 10, 20, 50}`, measure EM.
- **Hypothesis:** PQ96 @ top_n=20 matches or beats Flat @ top_n=10 on EM at lower memory cost.

### F. Behavioral feedback — does a worse index trigger more searches?

Search-o1 lets the model decide when to issue another query. A worse retriever may push it to re-query.

- Log `/search` calls per question per index type on HotpotQA / Musique.
- **Hypothesis:** `mean(N_calls | PQ32) > mean(N_calls | Flat)`. Lossy retrieval shifts work onto the reasoning loop, partially offsetting latency savings.

### G. Multi-hop recall propagation

Multi-hop questions chain retrievals: hop-2 uses the output of hop-1. Small recall losses compound.

- Run HotpotQA / 2Wiki / Musique with oracle hop-1 (gold first-hop doc) vs retrieved hop-1.
- For each index type, measure the delta between oracle and retrieved hop-1 + its effect on hop-2 recall.
- **Hypothesis:** Multi-hop benchmarks are disproportionately sensitive to recall loss vs single-hop NQ.

### H. Corpus chunk size × index choice

The 100-word chunking is a design choice, not a law. Larger chunks shrink the index and change recall dynamics.

- Rebuild the corpus at 256-word and 512-word chunks (requires re-embedding with E5 passage prefix).
- For each chunk size: index size, embedding build time, recall@10, downstream EM.
- **Hypothesis:** 256-word chunks + IVF-SQ8 ≈ 100-word chunks + Flat on EM, at a fraction of RAM.

### I. Score calibration across index types

`return_score=true` returns raw FAISS scores. Inner products of L2-normalized vectors for Flat, but approximate reconstructions for PQ/SQ.

- Collect score distributions on a fixed query set per index type.
- Fit a per-index affine calibration `s' = a·s + b` mapping PQ scores onto the Flat distribution.
- Or: use *ranks* in downstream RiD thresholding to avoid the problem entirely.

### J. Throughput under concurrency

Independent of compression: how well does `--num_retriever` actually scale?

- Fix index = `Flat,IP` (CPU). Benchmark QPS at `--num_retriever ∈ {1, 2, 4}` with `top_n=10` and steady concurrent load.
- Report GPU util for the encoder and CPU util for FAISS separately.
- **Hypothesis:** Encoder is the bottleneck on GPU; more retrievers past N=2 only help if encoder is on CPU or queries are long. Validates whether paying the 2× index RAM cost for `--num_retriever 2` is worth it.

### Minimum useful version

If you only have time for one experiment, run **D (`nprobe` sweep on `IVF4096,SQ8`)** with recall@10 on 1 K held-out queries. Tells you whether the cheap-and-almost-free swap from 65 GB → 16 GB actually costs anything meaningful for your workload, without needing the full agentic eval pipeline.

## Sources

- [Search-o1: Agentic Search-Enhanced Large Reasoning Models (arXiv:2501.05366)](https://arxiv.org/abs/2501.05366)
- [Search-o1 project page](https://search-o1.github.io/)
- [Search-o1 code](https://github.com/sunnynexus/Search-o1)
