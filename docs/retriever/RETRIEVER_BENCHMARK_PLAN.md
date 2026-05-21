---
title: RETRIEVER BENCHMARK PLAN
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-01
---

# Retriever Benchmark Plan — pre/post concurrency fixes

> **Status (2026-05-01): plan only, not executed.** Written while the Plan B v1 instruct sweep was in flight (now finished — see [../report/RESULTS_m1.md](../report/RESULTS_m1.md)). The "don't disturb the in-flight sweep" constraint below is no longer active. Re-read before kicking this off so the plan doesn't reference stale runtime state.

Plan to measure the throughput impact of the [RETRIEVER_CONCURRENCY.md](RETRIEVER_CONCURRENCY.md) fixes (`asyncio.to_thread` wrap + `IO_FLAG_MMAP`). Two runs: one against the current (unfixed) code, one against the fixed code. Same workload, same machine, side-by-side comparison.

## Goals & non-goals

**Goals**
- Quantify per-concurrency-level throughput and tail latency (p50/p95/p99) for the retriever service.
- Confirm that `--num_retriever > 1` actually scales after the fix (and confirm it doesn't before).
- Confirm `IO_FLAG_MMAP` makes per-process RSS independent of `--num_retriever`.

**Non-goals**
- Measuring end-to-end EM impact (the fixes are throughput-only, not behavioural — EM should be byte-identical).
- Comparing index variants (Flat vs IVF-SQ8) — that's [RETRIEVER_INDEXING.md](RETRIEVER_INDEXING.md) experiment D, separate scope.
- Touching the production retriever on port 3005 or the in-flight Plan B v1 instruct sweep on port 3000.

## Constraints — don't disturb the in-flight sweep

Current state on this box (`free -g`, `ps`):

| Resource | In use | Budget for benchmark |
|---|---:|---:|
| RAM total | 503 GB | — |
| RAM free | ~198 GB | 64 GB headroom for benchmark |
| Prod retriever RSS | ~130 GB (2× flat IP, the duplication bug) | unchanged |
| SGLang VRAM | ~22 GB | unchanged |
| GPU util | ~99% (instruct sweep) | benchmark uses **0% GPU** (CPU-only retriever) |
| CPU cores | 96 threads, prod uses ~2 | benchmark can use 4 cores |

**Hard rules:**
- Benchmark retriever runs on **port 3006** (prod is 3005).
- Benchmark uses **IVF-SQ8 index** (~16 GB × N workers), not Flat IP — caps RAM impact at ~64 GB even with N=4.
- Benchmark uses **`taskset -c 80-95`** to pin to 16 idle cores so we don't steal cycles from prod's FAISS workers.
- **Do not edit source files** until the in-flight sweep finishes (running Python processes don't reload on file edit, but a crash + restart would pick up new code mid-sweep — avoidable risk).

## Methodology

### Workload

- 1000 real queries sampled deterministically from `data/nq/test.jsonl` (seed=42).
- `top_n=3` (matches eval pipeline).
- `return_score=false` (matches eval pipeline).
- Queries cycle in order; each request is independent.

### Concurrency sweep

For each retriever configuration, fire requests at concurrency levels:

```
1, 2, 4, 8, 16, 32
```

Per level:
- 5 s warmup (results discarded).
- 30 s measurement window.
- Record: completed requests, p50/p95/p99 latency, throughput (RPS).

Why 30 s: long enough that startup jitter and FAISS internal cache warmup don't dominate; short enough to keep total wall-clock under 30 min for a full sweep × 2 conditions.

### Configurations to test

| Config | `--num_retriever` | Code state | Purpose |
|---|---:|---|---|
| **A** Pre-fix N=1 | 1 | current (sync blocking) | Baseline single-worker latency |
| **B** Pre-fix N=4 | 4 | current (sync blocking) | Show that N=4 doesn't help today |
| **C** Post-fix N=1 | 1 | fixed (asyncio.to_thread) | Confirm N=1 unchanged |
| **D** Post-fix N=4 | 4 | fixed | Show that N=4 now scales |

A vs B: proves the bug. C vs D: proves the fix. B vs D at concurrency 8: the headline number.

### Metrics

Per (config, concurrency level):

| Metric | Source |
|---|---|
| RPS (requests/second) | `completed / measurement_window` |
| p50 / p95 / p99 latency (ms) | per-request elapsed time, sorted |
| Process RSS (GB) | `/proc/<pid>/status` VmRSS, sampled mid-run |
| CPU% (process) | `ps -p <pid> -o %cpu`, sampled mid-run |
| Errors | non-200 responses + timeouts |

Tabulated as a markdown table in `docs/RETRIEVER_BENCHMARK_RESULTS.md` (created by the benchmark script).

### Tool

Custom async Python script using `httpx.AsyncClient` — gives clean per-request timing and avoids `wrk`'s POST-JSON friction. ~80 LoC, lives at `scripts/bench_retriever.py`.

Pseudo:

```python
async def worker(client, queries, port, results):
    while time.monotonic() < deadline:
        q = queries[next_idx()]
        t0 = time.monotonic()
        r = await client.post(f"http://127.0.0.1:{port}/search",
                              json={"query": q, "top_n": 3, "return_score": False})
        results.append((time.monotonic() - t0, r.status_code))

async def run(port, concurrency, duration):
    async with httpx.AsyncClient(timeout=30) as client:
        # warmup
        await asyncio.gather(*[worker(...) for _ in range(concurrency)])
        # measure
        ...
```

## Step-by-step execution

### Phase 1 — pre-fix baseline (run NOW, in parallel with prod sweep)

```bash
# 1. Start benchmark retriever on port 3006, IVF-SQ8 index, N=1 (config A)
cd /workspace/reason_over_search/local_retriever
taskset -c 80-95 nohup python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 1 \
  --port 3006 \
  --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
  > /tmp/bench_retriever_3006.log 2>&1 &
disown

# 2. Wait for it to be ready
until curl -sS http://127.0.0.1:3006/health 2>/dev/null | grep -q healthy; do sleep 2; done

# 3. Run benchmark sweep for config A
python scripts/bench_retriever.py --port 3006 --label "A_pre_n1" \
  --concurrency 1,2,4,8,16,32 --warmup 5 --duration 30

# 4. Stop, restart with N=4 (config B)
pkill -f 'retriever_serving.*--port 3006'
sleep 5
taskset -c 80-95 nohup python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 4 \
  --port 3006 \
  --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
  > /tmp/bench_retriever_3006.log 2>&1 &
disown
until curl -sS http://127.0.0.1:3006/health 2>/dev/null | grep -q healthy; do sleep 2; done

# 5. Run benchmark sweep for config B
python scripts/bench_retriever.py --port 3006 --label "B_pre_n4" \
  --concurrency 1,2,4,8,16,32 --warmup 5 --duration 30

# 6. Stop benchmark retriever, free RAM
pkill -f 'retriever_serving.*--port 3006'
```

Total Phase 1 wall-clock: ~30 min (load index ~2 min × 2 + 6 levels × 35 s × 2 + restart overhead).

### Phase 2 — apply fixes (after instruct sweep finishes)

Per [RETRIEVER_CONCURRENCY.md "Files to change"](RETRIEVER_CONCURRENCY.md#files-to-change):

1. Edit [`local_retriever/retriever_serving.py`](../../local_retriever/retriever_serving.py):
   - Wrap both `retriever.search()` and `retriever.batch_search()` calls in `await asyncio.to_thread(...)`.
   - Add `limit_concurrency=64` to `uvicorn.run(...)`.
2. Edit [`local_retriever/flashrag/retriever/retriever.py:356`](../../local_retriever/flashrag/retriever/retriever.py):
   - Change `faiss.read_index(self.index_path)` → `faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)`.

Smoke test before benchmark:

```bash
# Start a fresh single-worker on 3006 to confirm the service still serves correct results
taskset -c 80-95 nohup python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 1 --port 3006 \
  --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
  > /tmp/bench_retriever_3006.log 2>&1 &
disown
until curl -sS http://127.0.0.1:3006/health 2>/dev/null | grep -q healthy; do sleep 2; done

# Sample query — output should be identical to pre-fix
curl -sS -X POST http://127.0.0.1:3006/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"who wrote The Lord of the Rings?","top_n":3}' | jq .
```

Compare the doc IDs returned to a saved pre-fix sample (capture during Phase 1). Identical → fix is behavior-preserving.

### Phase 3 — post-fix benchmark

Same as Phase 1, but with the fixed code:

```bash
# Config C: post-fix N=1
python scripts/bench_retriever.py --port 3006 --label "C_post_n1" \
  --concurrency 1,2,4,8,16,32 --warmup 5 --duration 30

# Restart with N=4
pkill -f 'retriever_serving.*--port 3006'
sleep 5
taskset -c 80-95 nohup python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 4 --port 3006 \
  --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
  > /tmp/bench_retriever_3006.log 2>&1 &
disown
until curl -sS http://127.0.0.1:3006/health 2>/dev/null | grep -q healthy; do sleep 2; done

# Config D: post-fix N=4
python scripts/bench_retriever.py --port 3006 --label "D_post_n4" \
  --concurrency 1,2,4,8,16,32 --warmup 5 --duration 30

# Cleanup
pkill -f 'retriever_serving.*--port 3006'
```

### Phase 4 — write up

`scripts/bench_retriever.py` appends each run's results to `docs/RETRIEVER_BENCHMARK_RESULTS.md`. Final doc has four tables (A/B/C/D), one comparison chart, and a one-paragraph headline.

## Expected results (hypothesis)

| Config | Concurrency 1 | Concurrency 4 | Concurrency 8 | Concurrency 16 |
|---|---:|---:|---:|---:|
| A pre N=1 | ~30 RPS | ~30 RPS (serialized) | ~30 RPS | ~30 RPS |
| B pre N=4 | ~30 RPS | **~30 RPS (placebo!)** | ~30 RPS | ~30 RPS |
| C post N=1 | ~30 RPS | ~30 RPS (only one worker) | ~30 RPS | ~30 RPS |
| D post N=4 | ~30 RPS | **~110 RPS (~3.7×)** | ~110 RPS | ~110 RPS |

The headline confirmation is **B and D both at concurrency 4**: same flag, vastly different throughput. If B ≈ A everywhere, the audit was right. If D > B by ~3-4× at concurrency 4–8, the fix works.

RSS expectation:
- Pre-fix N=4: ~64 GB (4 × 16 GB IVF-SQ8 copies).
- Post-fix N=4 with mmap: ~16 GB (shared via page cache).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Benchmark steals CPU from prod retriever's 2 workers, slowing the sweep | `taskset -c 80-95` pins benchmark to cores 80–95; prod uses cores 0–1 by default (no overlap) |
| Loading the IVF-SQ8 index takes 1–2 min × 4 restarts | Acceptable, factored into wall-clock |
| Benchmark traffic disturbs production by accident (wrong port) | Hard-code `--port 3006`; never use 3005 |
| OS page cache eviction pushes prod retriever's flat index to disk | IVF-SQ8 is 16 GB, much smaller than the 65 GB flat — minimal cache pressure |
| Source edits applied before sweep finishes; prod retriever crashes and restarts with new code | Don't edit until sweep is done; if absolutely needed earlier, edit in a copy under `/tmp/` and run benchmark from there |

## Deliverables

After all 4 phases:

1. `scripts/bench_retriever.py` — reusable benchmark tool, committed.
2. `docs/RETRIEVER_BENCHMARK_RESULTS.md` — auto-generated table, four configs × six concurrency levels, RSS column.
3. Update [RETRIEVER_CONCURRENCY.md](RETRIEVER_CONCURRENCY.md) §Priority for Plan A with the empirical numbers (replace "5 min fix" effort claim with actual measured throughput delta).
4. If results match hypothesis, the fixes are committed to a single PR with the benchmark numbers in the description.

## Open questions

- **Does the encoder forward pass dominate at small `top_n`?** If yes, FAISS-side parallelism only helps so much; encoder-side batching becomes the next bottleneck. The benchmark will show this if D doesn't scale linearly past N=4.
- **Should we also test on Flat IP?** The bug is index-agnostic, but Flat is what prod actually uses. After Phase 1–3 prove the IVF-SQ8 case, optional Phase 5 re-runs B and D with `--index ./indexes/wiki18_100w_e5_flat_inner.index` (RAM-permitting; 4 × 65 GB = 260 GB needed for pre-fix N=4 — only do post-mmap-fix).
- **`--limit-concurrency 64` interaction**: at concurrency=32 the cap is not hit; at concurrency=128 it would be. Not in current sweep, but worth noting in the results doc.
