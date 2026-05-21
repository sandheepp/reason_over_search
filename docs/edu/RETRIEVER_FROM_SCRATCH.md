---
title: RETRIEVER FROM SCRATCH
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-02
---

# Retriever from Scratch — Embeddings, FAISS, IVF, SQ8

A top-down walkthrough of how this repo's retriever works: from "what problem are we solving" to the bit-level FAISS machinery. Companion to [INDEX_ARCHITECTURE.md](../retriever/INDEX_ARCHITECTURE.md), which is the reference doc; this is the teaching version.

---

## 0. The job, in one sentence

You have **21 million** Wikipedia passages sitting on disk. The model asks `<search>who directed Inception</search>`. You need to return the **3 most relevant passages** — in <500 ms, on CPU. The retriever is the box that does that.

---

## 1. Why this is hard (the naive approach fails)

The naive way: scan all 21M passages, score each one against the query, return top 3.

- "Score" how? Substring match misses paraphrases. BM25 (TF-IDF style) works but is brittle: "who directed Inception" won't strongly match a passage that says "Christopher Nolan's 2010 film".
- Even if scoring is cheap, 21M comparisons per query × hundreds of queries per eval is too slow.

We need two things:
1. A **scoring function that captures meaning** (not just word overlap).
2. A **search method that doesn't visit all 21M passages**.

The retriever in this repo solves (1) with **embeddings + E5**, and (2) with **FAISS indexes**.

---

## 2. Embeddings: turning text into a list of numbers that carries meaning

### ELI5
An embedding is a fixed-length list of numbers (here, **768 floats**) that represents a piece of text in a way where *similar meaning ≈ similar numbers*.

Imagine a 768-dimensional space. Every passage is a point. Passages about Christopher Nolan films cluster in one region; passages about quantum mechanics cluster in another. To find "Inception director" you compute a vector for the query, then look for nearby points.

### A bit deeper
The 768 dimensions don't correspond to human-readable concepts ("dim 42 = 'is about a movie'"). They emerge from training. The model is trained so that:
- A query and a relevant passage end up with vectors pointing roughly the same direction.
- Unrelated text ends up pointing in different directions.

### "Same direction" = cosine similarity
Two vectors **u** and **v** can be compared by the cosine of the angle between them:

$$\cos(\theta) = \frac{u \cdot v}{\|u\|\|v\|}$$

`u · v` is the **inner product** (sum of element-wise products). Identical direction → 1.0; orthogonal → 0; opposite → −1.

**Trick used in this repo (and most retrieval systems):** if every vector is **L2-normalized** (its length is forced to 1), then `‖u‖ = ‖v‖ = 1`, so:

$$\cos(\theta) = u \cdot v$$

Cosine similarity becomes a plain dot product. That matters because **inner-product search is what FAISS does fastest**.

You can see this happen in the codebase at [encoder.py:63](../../local_retriever/flashrag/retriever/encoder.py#L63):
```python
query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
```
That line forces every query vector to length 1.

---

## 3. E5 specifically (the encoder used here)

E5-base-v2 is the model Search-R1 uses. It's a 110M-parameter BERT-style transformer. It encodes both queries and passages into the *same* 768-d space (a "dual encoder"). Two quirks worth knowing:

1. **Prefix matters.** E5 was trained with a `"query: "` prefix on questions and `"passage: "` prefix on documents. This is why [encoder.py:40](../../local_retriever/flashrag/retriever/encoder.py#L40) calls `parse_query(...)` — it prepends the right prefix. Without it, retrieval drops noticeably.
2. **Mean pooling.** A transformer outputs one vector per token. To get a single vector for the whole text, E5 **averages** the token vectors (weighted by the attention mask, so padding tokens don't count). That's `pooling(...)` at [encoder.py:59-61](../../local_retriever/flashrag/retriever/encoder.py#L59-L61).

### What happens at encode time, step by step
For a query like `"who directed Inception"`:

1. Tokenize with prefix: `"query: who directed Inception"` → `[101, 23435, 102, ...]` (token IDs)
2. Forward pass through the transformer → `(seq_len, 768)` float matrix
3. Mean-pool over `seq_len` → `(768,)` vector
4. L2-normalize → unit vector
5. Cast to `float32` → ready for FAISS

That's the **3–8 ms "Encode" row** in the lifecycle table at [INDEX_ARCHITECTURE.md:67](../retriever/INDEX_ARCHITECTURE.md#L67).

---

## 4. The corpus side (offline, done once)

The same encoder ran over **all 21M Wikipedia passages**, once, ahead of time. That produced:
- A `(21M, 768)` matrix of float32 vectors → **the FAISS index file**
- The original passages in a parallel `wiki18_100w.jsonl` file → **the corpus**

These two files are **row-aligned**: row `i` of the FAISS index is the embedding of line `i` of the jsonl. FAISS only deals in row IDs — it doesn't know any text. When you search, FAISS hands back integer IDs; the service then opens the jsonl at those line numbers to fetch the actual passage text.

Re-embedding 21M passages takes 6–10 hours on a 4090, so this is done once and the file is shipped pre-built.

Sizes:
- `21,000,000 × 768 × 4 bytes` ≈ **65 GB** — that's the flat fp32 index.

---

## 5. Search problem: find the top-k nearest neighbors

You now have 21M unit vectors and one query unit vector. You want the top-k by inner product. This is **k-NN (k nearest neighbors) search**.

### Approach A — Flat IP (exact, slow)

"IP" = inner product. The simplest index: store every vector, and for each query compute the inner product against all 21M, then sort.

- **Pros**: exact. The top-3 is the *true* top-3.
- **Cons**: 21M × 768 multiplications per query. On CPU, 100–300 ms. Memory: 65 GB resident in RAM.

This is the repo's exact-recall fallback (`wiki18_100w_e5_flat_inner.index`, ~65 GB) — useful for paper-fidelity eval. (The training-time *default* is the faster IVF-SQ8 variant — see Approach D.) The trick: 65 GB sounds huge, but the box has 503 GB RAM, and a flat index is just one big matrix, so vectorized BLAS makes it surprisingly fast.

### Approach B — IVF (approximate, faster)

**IVF = Inverted File Index.** The big idea: don't compare against all 21M; only compare against vectors *likely* to be near the query.

How:

1. **Offline (once):** run k-means clustering on 1M sampled vectors → get **4096 cluster centroids**. Each of the 21M vectors gets assigned to its nearest centroid. Now you have 4096 "buckets" with ~5,000 vectors each.

2. **At query time:**
   - Compare the query vector to all 4096 centroids (cheap: 4096 dot products)
   - Pick the `nprobe` closest centroids (e.g. nprobe=32)
   - Only compare the query against vectors *in those 32 buckets* (~160K vectors instead of 21M)

That's ~130× fewer comparisons. Speedup: **3–10×** in practice (overhead, cache effects).

**Tradeoff: accuracy.** If the *true* nearest neighbor sits in a bucket whose centroid wasn't in the top 32, you miss it. `nprobe` is the recall/speed knob:
- nprobe=1 → fastest, lowest recall
- nprobe=4096 → equivalent to flat (search every bucket)
- Typical: 32–128

In this repo's config that's `faiss_nprobe` at [retriever.py:357-358](../../local_retriever/flashrag/retriever/retriever.py#L357-L358).

### Approach C — SQ8 (compression)

**SQ = Scalar Quantization. 8 = 8 bits.** Idea: instead of storing each of the 768 floats as fp32 (4 bytes), store it as a uint8 (1 byte). 4× smaller.

How: for each dimension, find the min and max value across the corpus, then linearly map `[min, max]` → `[0, 255]`. Decoding multiplies back by `(max-min)/255 + min`. Loses some precision but for L2-normalized embeddings the loss is small (recall typically drops <1%).

`65 GB / 4 ≈ 16 GB` — that's where the IVF-SQ8 index size comes from.

### Approach D — IVF4096 + SQ8 combined

This is the alternative index this repo ships: `wiki18_100w_e5_ivf4096_sq8.index`. It applies *both* tricks:
- IVF cuts the number of vectors visited per query
- SQ8 quantizes the vectors that *are* visited

Result: 16 GB instead of 65 GB **and** 3–10× faster than flat. Cost: a tiny recall hit, paid for many times over by speed.

---

## 6. Why nlist=4096 specifically?

Footnote in [INDEX_ARCHITECTURE.md:99](../retriever/INDEX_ARCHITECTURE.md#L99) explains: FAISS won't train k-means with fewer than 39 points per centroid. With 1M training samples:

- `1,000,000 / 4096 ≈ 244` points per centroid → fine
- `1,000,000 / 65,536 ≈ 15` points per centroid → fails the floor

To use 65,536 cells you'd need 2.55M training samples, which costs more memory and build time. So 4096 is the largest viable nlist with the chosen sample size. (Using more cells = smaller cells = fewer vectors per probe = faster, up to a point.)

---

## 7. The build pipeline (one-time, offline)

The flat 65 GB index already exists (it ships pre-built). Building the IVF-SQ8 index reuses the flat one — no need to re-embed the 21M passages, which would cost 6–10 GPU-hours.

From [INDEX_ARCHITECTURE.md:76-95](../retriever/INDEX_ARCHITECTURE.md#L76-L95):

1. **Load** the flat index into RAM (~60-90s, mostly disk read)
2. **Sample** 1M random row indices for k-means training
3. **Reconstruct** their fp32 vectors with `faiss.reconstruct_n()` — pulls them out of the flat index
4. **Train** k-means on GPU with nlist=4096 (under 1 minute)
5. **Add** all 21M vectors to the trained IVF index in chunks of 1M (~5–10 min). FAISS quantizes to SQ8 on the way in.
6. **Write** the resulting 16 GB file to disk

---

## 8. The full request lifecycle, end to end

Now stitch it all together. The mermaid sequence at [INDEX_ARCHITECTURE.md:38-61](../retriever/INDEX_ARCHITECTURE.md#L38-L61) corresponds to:

1. **Client** (the eval pipeline) hits `POST /search {query: "...", top_n: 3}`.
2. **FastAPI** acquires the semaphore — caps in-flight searches to N.
3. **Worker** is popped from a deque (each worker is its own DenseRetriever with its own encoder + FAISS handle).
4. **Encoder** (CPU or GPU): tokenize → forward pass → mean-pool → L2-normalize → `(1, 768)` fp32 vector. **3–8 ms.**
5. **FAISS search** with `index.search(vec, k=3)`:
   - Flat: 21M dot products. **100–300 ms** on CPU.
   - IVF-SQ8 CPU: nprobe×~5K dot products. **30–100 ms.**
   - IVF-SQ8 GPU: same algorithm but on CUDA. **5–15 ms.**
   Returns `[id_0, id_1, id_2]` and corresponding scores.
6. **Corpus lookup**: open `wiki18_100w.jsonl` at those row IDs (mmap'd, so it's a page-cache hit after warmup). **<0.1 ms.**
7. **Worker** returns `List[Document]`, gets pushed back to the deque, semaphore released.
8. **JSON response** to client.

---

## 9. Memory layout (why the 4090 forces CPU FAISS)

Three big consumers of memory:

| Component | Size | Where it can live |
|---|---|---|
| Flat FAISS | 65 GB | RAM only (won't fit in 24 GB VRAM) |
| IVF-SQ8 FAISS | 16 GB | RAM, *or* VRAM (if room) |
| SGLang (the 3B policy model) | 22 GB bf16 | VRAM |
| E5 encoder | 0.5 GB | RAM or VRAM |
| Corpus jsonl | 14 GB | RAM (mmap, evictable) |

On the 4090 (24 GB VRAM), SGLang takes 22 GB → only 2 GB free → **no FAISS index of any size fits in VRAM alongside it**. So on this box, FAISS lives in RAM, encoder runs on whichever device has room.

On an H100 (80 GB), 22 (SGLang) + 16 (IVF-SQ8) + 0.5 (encoder) ≈ 38 GB → fits comfortably with headroom → GPU FAISS is viable, ~10× faster than CPU FAISS.

That's why the doc says *"GPU FAISS and SGLang cannot share the 4090."*

---

## 10. Concurrency model (semaphore + worker pool)

The service is `--num_retriever N`, where N is the number of independent worker objects. Each worker holds its own `DenseRetriever` (its own encoder + its own FAISS handle, but they read the **same** index data — FAISS index reads are read-only and parallel-safe).

The semaphore at `retriever_serving.py:14-35` (referenced in the doc) ensures at most N concurrent searches. A request:
1. Acquires the semaphore (blocks if N searches already running).
2. Pops a worker idx off the deque (constant time).
3. Runs encode + FAISS search.
4. Pushes worker back, releases semaphore.

FAISS releases the GIL during its C++ search code, so workers really do run on different CPU cores in parallel — *provided* the FastAPI handler doesn't hold the event loop synchronously. The doc flags that current handlers do block the event loop ([RETRIEVER_CONCURRENCY.md](../retriever/RETRIEVER_CONCURRENCY.md)) — that's the known issue.

---

## 11. Putting the abstractions on a number line

If you only remember one thing: **the retriever turns text into 768-d vectors, then asks FAISS for nearest neighbors.** The choices are:

| Choice | What | Why pick it |
|---|---|---|
| Flat IP | Compare against all 21M | Exact. Default. 65 GB RAM. Slow-ish on CPU. |
| IVF (4096 cells, nprobe=32) | Compare against ~160K nearby | 3–10× faster. ~99% recall. |
| SQ8 quantization | 4-byte → 1-byte per dim | 4× smaller. <1% recall hit. |
| IVF + SQ8 | Both | What `wiki18_100w_e5_ivf4096_sq8.index` uses. |
| GPU FAISS | Same algorithm on CUDA | 10× faster than CPU. Needs 16 GB free VRAM. |

And the four files that matter:
- `wiki18_100w_e5_ivf4096_sq8.index` (16 GB, default; HF dataset [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index))
- `wiki18_100w_e5_flat_inner.index` (65 GB, optional, exact recall)
- `wiki18_100w.jsonl` (14 GB raw passages, mmap'd, row-aligned with index)
- E5-base-v2 weights (~0.5 GB, loaded into the encoder)

---

That's the full picture: text → vector → ANN search → row IDs → text. Everything in [INDEX_ARCHITECTURE.md](../retriever/INDEX_ARCHITECTURE.md) is one of those steps or a memory/concurrency consequence of choosing where each piece lives.
