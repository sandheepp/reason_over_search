# Indexing & Retrieval Internals

Background on how the retriever actually works, why the FAISS index file is so large, and what your options are if you want to trade a little recall for a lot of RAM.

## How search works end-to-end

Dense retrieval needs two artifacts that must be built together and kept aligned:

- **`corpus_path`** (`corpus/wiki18_100w.jsonl`) — the raw text. Each line is `{"id": ..., "contents": ...}`. This is what gets returned to the caller.
- **`index_path`** (`indexes/wiki18_100w_e5_flat_inner.index`) — a FAISS index of pre-computed dense embeddings for every passage in the corpus, produced offline with the E5 model. This is what actually gets searched.

**The invariant:** row `i` in the FAISS index is the embedding of row `i` in the corpus jsonl.

A `search(query, top_n)` call flows:

1. **Embed** the query with E5 (prefix `"query: "`, mean/CLS pool, L2 normalize).
2. **FAISS `index.search(emb, k=top_n)`** → returns integer row IDs + scores. No text.
3. **Corpus lookup** — `corpus[i]` for each returned ID fetches the actual `{id, contents}` dict.
4. Wrap into Pydantic `Document` models and return.

So the pipeline is: `text → vector → row numbers → text`, with the model, FAISS, and the corpus each doing one arrow.

`batch_search` does the same thing but vectorized — a single encoder pass over all N queries, one FAISS call with an (N, dim) matrix, one bulk corpus lookup, reshape.

## Why both files exist

- The **index** answers "which documents look like this query?" via dense vector search. It's huge, fixed-shape, and benefits from being a single binary file that FAISS can load into a contiguous array (optionally on GPU).
- The **corpus** answers "what does row 12345 actually say?" It's variable-length text, only read on hits, and is memory-mapped via HuggingFace Datasets so cost is mostly OS page cache.

Keeping them separate lets you rebuild the index with a different embedding model without touching the corpus, or swap corpora while reusing infrastructure.

## RAM requirements

| Component | RAM | Notes |
|---|---|---|
| **FAISS flat index** | **~65 GB** | Dominates; fully resident in process heap |
| Corpus (Arrow mmap) | ~14 GB RSS | OS page cache, shared/evictable |
| E5-base-v2 model | ~0.5 GB CPU (+ ~0.5 GB GPU) | Small |
| FastAPI / Python / libs | ~1–2 GB | Negligible |
| **Total (single retriever, CPU)** | **~80 GB** | Plan for 96 GB RAM |

### Why the index is ~65 GB

Wiki-18 has ~21 M passages. The filename says `_flat_inner` — a **flat** index: raw float32 vectors, no quantization.

```
21,000,000 × 768 dims × 4 bytes = ~64.5 GB
```

And [flashrag/retriever/retriever.py](flashrag/retriever/retriever.py) loads it with plain `faiss.read_index(path)` — no `IO_FLAG_MMAP`, so the whole file goes into heap.

### Two footguns

**1. `--num_retriever N` multiplies the FAISS cost.** Each retriever gets its own `read_index` copy — with `N=2` you need ~145 GB. Only the corpus (Arrow mmap) and on-disk model weights are shared via page cache; FAISS arrays are not.

**2. `faiss_gpu: True` moves the cost to VRAM.** 65 GB fits on A100-80 / H100-80, but not on 40 GB cards without sharding.

## Flat vs quantized indexes

The flat index is the full 65 GB because it stores every vector exactly and scans them all on every query. Quantized indexes compress vectors and/or prune the search, trading a little recall for a lot of memory and speed.

### The three axes quantization affects

1. **Recall@k** — fraction of the flat index's top-k that the quantized index also returns. Measured on a held-out query set.
2. **Scores in `return_score=true`** — PQ/SQ reconstruct approximate vectors from codebooks, so scores shift slightly. Absolute values aren't comparable across index types; only rankings are.
3. **Latency** — usually *improves*. Smaller vectors fit in CPU caches, and IVF scans only a few clusters instead of all 21 M vectors.

### Options for wiki-18 @ top-10

| Index | RAM | Recall@10 vs flat | Latency vs flat | Score fidelity |
|---|---|---|---|---|
| `Flat,IP` (current) | 65 GB | 100% | 1× baseline | exact |
| `HNSW32,Flat` | ~70 GB | 97–99% | 10–50× faster | exact |
| `IVF4096,SQ8` | ~16 GB | 96–99% | 3–10× faster | very close |
| `IVF65536,PQ96` | ~3 GB | 88–95% | 10–100× faster | approximate |
| `IVF65536,PQ32` | ~1 GB | 75–88% | fastest | coarse |

Ranges depend on `nprobe` (how many IVF cells to scan) and `k`. Common sweet spot: `nprobe=32` or `64` with `IVF65536`.

### End-to-end impact on a RAG pipeline

With `top_n=3`, downstream answer quality is usually less sensitive than raw recall numbers suggest:

- Wiki-18 has many near-duplicate passages (same article, adjacent chunks). Missing the "true" #3 and getting a near-twin at rank #4 rarely hurts.
- E5 produces peaky similarity distributions — the top few are clearly separated, so quantization mostly reshuffles among strong candidates.
- LLMs reading the docs don't care about millisecond-scale score differences.

**Concrete expectations:**
- **HNSW** or **IVF-SQ8** — answer quality indistinguishable from flat. Safe swap.
- **IVF-PQ96** — 1–3% drop on QA benchmarks (NQ / TriviaQA). Noticeable in eval, rarely in production.
- **IVF-PQ32** — measurable regression. Only worth it if RAM-starved.

## Practical recommendations

**If RAM is the problem → `IVF65536,SQ8`** is the best balance: ~16 GB, <1% recall loss, faster queries.

```python
import faiss
quantizer = faiss.IndexFlatIP(768)
index = faiss.IndexIVFScalarQuantizer(
    quantizer, 768, 65536,
    faiss.ScalarQuantizer.QT_8bit,
    faiss.METRIC_INNER_PRODUCT,
)
index.train(training_vectors)   # ~1M sampled embeddings is enough
index.add(all_embeddings)
faiss.write_index(index, "wiki18_100w_e5_ivf_sq8.index")
```

**If RAM is not the problem and you just want speed → `HNSW32,Flat`.** Same memory as flat, exact scores, 10–50× faster queries. No training, no `nprobe` tuning.

## Swapping the index in this repo

Point `index_path` in [retriever_config.yaml](retriever_config.yaml) at the new file — the serving code uses `faiss.read_index`, which handles any FAISS index type.

For IVF-based indexes, you'll also want to set `nprobe` after loading. That isn't exposed in the current config; it takes a two-line patch to `DenseRetriever.load_index` in [flashrag/retriever/retriever.py](flashrag/retriever/retriever.py):

```python
self.index = faiss.read_index(self.index_path)
if hasattr(self.index, "nprobe"):
    self.index.nprobe = self._config.get("faiss_nprobe", 64)
```

Then add `faiss_nprobe: 64` to `retriever_config.yaml`.

### Rebuild cost

Swapping the index means getting the embeddings again. You have two choices:

- **Re-embed** the 21 M passages with E5 from scratch — 1–6 hour GPU job depending on hardware.
- **Extract** the existing vectors from the current flat index with `index.reconstruct_n(0, ntotal)` — avoids the embedding pass entirely, just a disk-to-disk reshuffle. Much faster.

Either way, the serving code and corpus are unchanged.

## Experiments to run

This retriever is the kind of backend used by **agentic-search reasoning models** — systems where the LLM decides mid-chain-of-thought when to call `/search`, reads the results, and continues reasoning. A concrete example is [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366) (Li et al., EMNLP 2025), which wraps retrieval in a "Reason-in-Documents" (RiD) module that refines/filters retrieved passages before they re-enter the reasoning context.

That setting changes what "good indexing" means. Each question triggers multiple retrieval calls, cumulative latency matters more than per-call latency, and RiD may absorb recall losses that a vanilla top-k RAG pipeline can't tolerate. Below are experiments this repo is well-positioned to run.

### A. Index quality vs end-to-end answer accuracy

Swap `index_path` through increasingly lossy indexes and measure downstream QA metrics on Search-o1-style benchmarks (NQ, TriviaQA, HotpotQA, 2Wiki, Musique, Bamboogle — single- and multi-hop). The [evaluation_search_r1/](../evaluation_search_r1/) harness in this repo is the natural driver.

- **Variables:** `Flat,IP` (baseline) → `HNSW32,Flat` → `IVF65536,SQ8` → `IVF65536,PQ96` → `IVF65536,PQ32`.
- **Metrics:** Exact Match, F1, passage recall@10, passage recall@5.
- **Hypothesis:** SQ8 and HNSW are near-lossless; PQ96 is the break-point where EM drops ~1–3 pts; PQ32 is noticeably worse.

### B. Does RiD (or any doc-refiner) absorb recall loss?

The compelling claim in Search-o1 is that RiD filters noise. If true, a lossier index that returns more noisy but still-relevant docs should hurt less than raw recall@10 suggests.

- Fix the retriever to `IVF65536,PQ96`. Run the pipeline **with** and **without** RiD.
- Compare EM delta vs running `Flat,IP` with and without RiD.
- **Hypothesis:** `(Flat − IVF-PQ96) with RiD` < `(Flat − IVF-PQ96) without RiD`. Quantization is cheaper when a refiner exists downstream.

### C. Latency compounding under agentic loops

Agentic systems issue `N` retrieval calls per question (often 2–6). Wall-clock scales with `N × per_call_latency`.

- For each index type, measure median per-call latency at `top_n ∈ {3, 5, 10, 20}`.
- Sample Search-o1 traces on HotpotQA (multi-hop) to get the empirical `N` distribution per question.
- Plot end-to-end response time vs index type, broken down into (encoding, FAISS, corpus lookup, RiD).
- **Hypothesis:** For `N ≥ 3`, HNSW beats Flat by a wide margin and nearly matches IVF-PQ on latency while keeping exact scores.

### D. `nprobe` sweep for IVF indexes

Once on an IVF index, `nprobe` is the dominant accuracy-vs-latency knob. This is the first experiment to do after swapping.

- Fix index = `IVF65536,SQ8`. Sweep `nprobe ∈ {1, 4, 16, 32, 64, 128, 256, 512}`.
- For each: measure recall@10 vs `Flat` ground truth on 1,000 held-out queries, plus per-call latency.
- Also report downstream EM on NQ to check whether recall@10 is a faithful proxy for end accuracy.
- Requires the two-line patch in the "Swapping the index" section above to expose `faiss_nprobe` in config.

### E. Top-k sensitivity under lossy indexes

A classic compensation for lossy retrieval is "retrieve more, let the reader filter."

- For each (index_type, top_n) pair in `{Flat, SQ8, PQ96} × {3, 5, 10, 20, 50}`, measure EM.
- **Hypothesis:** PQ96 @ top_n=20 matches or beats Flat @ top_n=10 on EM but at lower memory cost.

### F. Behavioral feedback: does a worse index trigger more searches?

Search-o1 lets the model decide when to issue another query. A worse retriever may push the model to re-query.

- Log the number of `/search` calls per question for each index type on HotpotQA / Musique.
- **Hypothesis:** `mean(N_calls | PQ32) > mean(N_calls | Flat)` — lossy retrieval shifts work onto the reasoning loop, partially offsetting latency savings.

### G. Multi-hop recall propagation

Multi-hop questions chain retrievals: hop-2 uses the output of hop-1. Small recall losses compound.

- Run HotpotQA / 2Wiki / Musique with oracle hop-1 context (gold first-hop doc) vs retrieved hop-1.
- For each index type, measure the delta between oracle and retrieved hop-1, and its effect on hop-2 recall.
- **Hypothesis:** Multi-hop benchmarks are disproportionately sensitive to recall loss vs single-hop NQ.

### H. Corpus chunk size × index choice

The 100-word chunking in `wiki18_100w.jsonl` is a design choice, not a law. Larger chunks shrink the index and change recall dynamics.

- Rebuild the corpus at 256-word and 512-word chunks (requires re-embedding with E5 passage prefix).
- For each chunk size, report: index size, embedding build time, recall@10, downstream EM.
- **Hypothesis:** 256-word chunks + IVF-SQ8 ≈ 100-word chunks + Flat on EM, at a fraction of the RAM.

### I. Score calibration across index types

`return_score=true` returns raw FAISS scores. These are inner products of L2-normalized vectors for Flat, but approximate reconstructions for PQ/SQ.

- Collect score distributions on a fixed query set for each index type.
- Fit a per-index affine calibration (`s_calibrated = a·s_raw + b`) that maps PQ scores onto the Flat distribution.
- Check whether using *ranks* instead of raw scores in downstream RiD thresholding avoids the problem entirely.

### J. Throughput under concurrency

Independent of index compression: how well does the `--num_retriever` pool actually scale?

- Fix index = `Flat,IP`. Benchmark QPS at `--num_retriever ∈ {1, 2, 4}` with `top_n=10` and a steady concurrent-request load.
- Report GPU utilization for the encoder and CPU utilization for FAISS separately.
- **Hypothesis:** Encoder is the bottleneck on GPU; more retrievers past N=2 only help if the encoder is on CPU or if queries are long. Validates whether paying the 2× index RAM cost for `--num_retriever 2` is actually worth it.

### Minimum useful version

If you only have time for one experiment, run **D (nprobe sweep) on `IVF65536,SQ8`** and measure recall@10 on 1k held-out queries. It tells you whether the cheap-and-almost-free swap from 65 GB to 16 GB actually costs anything meaningful for your workload, without needing the full agentic evaluation pipeline.

### Sources

- [Search-o1: Agentic Search-Enhanced Large Reasoning Models (arXiv:2501.05366)](https://arxiv.org/abs/2501.05366)
- [Search-o1 project page](https://search-o1.github.io/)
- [Search-o1 code](https://github.com/sunnynexus/Search-o1)
