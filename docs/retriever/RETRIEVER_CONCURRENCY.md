---
title: RETRIEVER CONCURRENCY
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Retriever Concurrency Audit

Audit of the FastAPI retriever service ([`local_retriever/retriever_serving.py`](../../local_retriever/retriever_serving.py)) for concurrent-read correctness and throughput. For the architectural overview see [INDEX_ARCHITECTURE.md](INDEX_ARCHITECTURE.md); for index/RAM/recall tradeoffs see [RETRIEVER_INDEXING.md](RETRIEVER_INDEXING.md).

**TL;DR**: the semaphore + N-worker pool is correct but provides **no actual parallelism** because both endpoint handlers call blocking sync code from inside `async def`. `--num_retriever 4` is a placebo on the current code. One-line fix per handler. Plus one RAM-duplication issue worth fixing for hosts with <128 GB.

## Findings

| # | Issue | Severity | Effort to fix |
|---|---|---|---|
| 1 | Sync `retriever.search()` blocks the asyncio event loop | 🔴 Critical | 5 min |
| 2 | Each worker loads its own RAM copy of the FAISS index | 🟡 Medium | 15 min |
| 3 | `/batch_search` holds one worker for the whole batch | 🟡 Medium | by-design |
| 4 | No backpressure / connection limit on uvicorn | 🟢 Low | 1 min |
| 5 | Single uvicorn worker process | 🟢 Low | n/a — correct for this workload |

## 1. Sync calls inside async handlers serialize the event loop

[`retriever_serving.py:82-101`](../../local_retriever/retriever_serving.py#L82):

```python
@app.post("/search", ...)
async def search(request: QueryRequest):
    ...
    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].search(query, top_n, return_score)
            #          ^^ blocking sync call inside async handler
```

`async def` handlers run on a **single event loop thread**. `retriever.search()` is a synchronous Python call — encoder forward (`model(...)`) plus FAISS `index.search(...)` — that holds the loop until it returns. While it runs, no other coroutine progresses. With 8 concurrent client requests, all 8 serialize.

The semaphore + deque protect against two coroutines using the same retriever object, which is real correctness — but they don't introduce parallelism, because nothing forces concurrent execution onto separate threads.

### Verification

With the service running and `--num_retriever 4`:

```bash
time (
  for i in 1 2 3 4; do
    curl -sS -X POST http://127.0.0.1:3005/search \
      -H 'Content-Type: application/json' \
      -d '{"query":"test '"$i"'","top_n":3}' > /dev/null &
  done
  wait
)
```

If parallel: total ≈ single-call latency (100–300 ms on flat IP).
If serialized (current state): total ≈ 4 × single-call latency.

### Fix

Wrap each blocking call with `asyncio.to_thread` (Python 3.9+). FAISS releases the GIL during `search()`; PyTorch CPU forward also releases it. Threads give true parallelism.

```python
import asyncio  # already implicitly available; add explicit import if needed

async with retriever_semaphore:
    retriever_idx = available_retrievers.popleft()
    try:
        if return_score:
            results, scores = await asyncio.to_thread(
                retriever_list[retriever_idx].search,
                query, top_n, return_score,
            )
            ...
        else:
            results = await asyncio.to_thread(
                retriever_list[retriever_idx].search,
                query, top_n, return_score,
            )
            ...
```

Apply the same pattern to `batch_search`. After the change, `--num_retriever N` actually delivers N-way parallel CPU FAISS.

### Why this matters for Plan A

The [EVAL_OPS bottleneck note](../eval/EVAL_OPS.md#secondary-1-worker-faiss-510-today-more-on-big-datasets) attributes ~5–10% of 4090 wall-clock to single-worker FAISS. On H100 PCIe (3.35× faster GPU decode), that share rises to ~25%. The recommended fix in the H100 setup is `--num_retriever 4` — but **without fix 1, that flag does nothing**. Both fixes are required together.

## 2. Per-worker FAISS index duplication

[`flashrag/retriever/retriever.py:356`](../../local_retriever/flashrag/retriever/retriever.py#L356):

```python
def load_index(self):
    ...
    self.index = faiss.read_index(self.index_path)   # full in-memory load, no mmap
```

Each `DenseRetriever` instance in [`init_retriever`](../../local_retriever/retriever_serving.py#L29-L33) calls this independently, so RAM scales linearly with `--num_retriever`:

| Index | 1 worker | 4 workers | 8 workers |
|---|---:|---:|---:|
| Flat IP (65 GB) | 65 GB | 260 GB | 520 GB |
| IVF-SQ8 (16 GB) | 16 GB | 64 GB | 128 GB |

**On the local 503 GB box:** flat × 4 fits, flat × 8 does not.
**On a typical Vast/RunPod 4090 host (64–128 GB RAM):** flat × 1 doesn't fit at all — you'd need IVF-SQ8.
**On a RunPod H100 PCIe (1–2 TB RAM):** flat × 8 fits comfortably.

### Verification

```bash
ps -o pid,rss,cmd -C python | grep retriever_serving
# RSS scales linearly with --num_retriever in the current code
```

### Fix

Use FAISS's mmap flag so all workers share the OS page-cache copy of the index:

```python
self.index = faiss.read_index(
    self.index_path,
    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
)
```

Caveats:
- `IO_FLAG_MMAP` is supported for `IndexFlat` and most IVF variants.
- For GPU FAISS (`--gpu`), the index gets cloned to VRAM via `index_cpu_to_all_gpus`, so the CPU mmap copy is freed after clone — no benefit on GPU. Keep the mmap flag anyway; the CPU code path benefits, and GPU is unaffected.
- Backed by the OS page cache, so the *first* search is cold-cache slower (one-time cost). Subsequent searches are full-speed.

After the fix: RAM is constant regardless of `--num_retriever`. Enables the IVF-SQ8 + 8-worker config recommended for H100 in the Plan A setup.

## 3. `/batch_search` holds one worker for the whole batch

[`retriever_serving.py:111-141`](../../local_retriever/retriever_serving.py#L111) acquires the semaphore once, runs `retriever.batch_search(K queries)`, and releases. Other workers sit idle while one worker processes K queries sequentially via FAISS's vectorized `batch_search`.

This is correct but inefficient when K > 1 and N > 1. The eval pipeline doesn't hit this path (it fires per-example `/search` calls via its own thread pool), so this is theoretical for the current workload. Worth knowing if anyone reaches for batch endpoints.

A proper fix is non-trivial: split the K-query batch across N workers, then merge results — easy to get wrong on score ordering and worker assignment. Skip unless someone actually starts using `/batch_search`.

## 4. No backpressure on uvicorn

[`retriever_serving.py:162`](../../local_retriever/retriever_serving.py#L162):

```python
uvicorn.run(app, host="0.0.0.0", port=args.port)
```

Defaults: no `--limit-concurrency`, no max queue. Under a burst of 1000 requests, all 1000 sit in the asyncio waitqueue holding HTTP connection state — could OOM the host. Eval workload caps at ~8 in-flight, so not a real risk for Plan A. Worth setting defensively if this service is ever exposed beyond the eval pipeline:

```python
uvicorn.run(app, host="0.0.0.0", port=args.port, limit_concurrency=64)
```

## 5. Single uvicorn worker process — correct for this workload

`uvicorn.run(app, ...)` runs one worker process. Multi-process scaling via `workers=N` would multiply RAM by N (each process loads its own retriever pool), and combined with `--num_retriever 4` you'd get N×4 retriever instances using N×4× the index RAM — way past what's useful.

The right scaling axis here is **threads inside one process** (via fix 1), not processes. Single uvicorn worker is correct.

## What is OK

- **No race on the deque.** `asyncio.Semaphore(N)` + N items + `popleft`/`append` guarantees one-coroutine-per-worker. CPython deque ops are atomic; the semaphore enforces the bound.
- **FAISS reads are thread-safe.** Multiple threads can call `index.search()` on the same `IndexFlat` or IVF index. Confirmed by FAISS docs and the threaded fix above relies on this.
- **`/health`** is async-clean — no blocking calls, returns instantly.
- **Encoder per-worker isolation.** Each worker has its own `Encoder` object, so no shared-tensor / shared-tokenizer-state contention.

## Priority for Plan A

Apply in order:

1. **Fix 1 (`asyncio.to_thread` wrap)** — load-bearing for the H100 throughput claim. Without it, `--num_retriever 4` is a placebo. **5 min.**
2. **Fix 2 (`IO_FLAG_MMAP`)** — required for the 8× 4090 marketplace option in [../setup/VAST_AI_PLAN_A.md](../setup/VAST_AI_PLAN_A.md), where hosts have 64–128 GB RAM. Optional on RunPod H100 PCIe (lots of RAM). **15 min.**
3. **Fix 4 (`limit_concurrency=64`)** — defensive, 1 min, free.

After (1) and (2), the [INDEX_ARCHITECTURE.md concurrency model](INDEX_ARCHITECTURE.md#concurrency-model) actually behaves as drawn. Before, the diagram is aspirational.

## Files to change

| File | Change |
|---|---|
| [`local_retriever/retriever_serving.py`](../../local_retriever/retriever_serving.py) | Wrap `.search()` and `.batch_search()` calls in `asyncio.to_thread`; add `limit_concurrency=64` to `uvicorn.run` |
| [`local_retriever/flashrag/retriever/retriever.py`](../../local_retriever/flashrag/retriever/retriever.py) | Change `faiss.read_index(self.index_path)` to `faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP \| faiss.IO_FLAG_READ_ONLY)` |

After the edits, re-run the parallelism check from §1 and confirm 4 concurrent requests complete in ~1× single-call time, not ~4×.
