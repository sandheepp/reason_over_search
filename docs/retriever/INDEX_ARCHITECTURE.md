---
title: INDEX ARCHITECTURE
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-02
---

# Index Architecture

How the retriever service is wired up: components, query lifecycle, build pipeline, memory layout. For the *what-to-pick* side (RAM/VRAM/recall/latency tradeoffs across Flat/HNSW/IVF/PQ) see [RETRIEVER_INDEXING.md](RETRIEVER_INDEXING.md). For known concurrency issues in the current implementation (sync calls in async handlers, RAM duplication) see [RETRIEVER_CONCURRENCY.md](RETRIEVER_CONCURRENCY.md).

## Components

```mermaid
graph LR
    Client["Eval client<br/>(active_pipeline.py)"]
    subgraph Service["FastAPI service — retriever_serving.py"]
        API["POST /search<br/>POST /batch_search<br/>GET  /health"]
        Sema["asyncio.Semaphore<br/>(num_retriever)"]
        Pool["Worker pool<br/>deque[0..N-1]"]
    end
    subgraph Worker["DenseRetriever worker (×N)"]
        Enc["E5 encoder<br/>(GPU or CPU)"]
        FAISS["FAISS index<br/>(RAM or VRAM)"]
        Corpus["Corpus<br/>(HF Datasets, mmap)"]
    end
    Client -->|HTTP JSON| API
    API --> Sema
    Sema --> Pool
    Pool --> Worker
    Enc -->|"(N, 768) fp32"| FAISS
    FAISS -->|"row IDs + scores"| Corpus
    Corpus -->|"{id, contents}"| API
    API -->|HTTP JSON| Client
```

Three things to know:

1. **One service, N workers, one shared FAISS read path per worker.** `--num_retriever N` instantiates N independent `DenseRetriever` objects, each holding its own encoder + FAISS handle. Read-only, fully parallel-safe.
2. **The semaphore + deque pattern** in [`retriever_serving.py:14-35`](../../local_retriever/retriever_serving.py) hands one worker to one in-flight request. This caps concurrent FAISS searches at N.
3. **Two artifacts are kept row-aligned**: the FAISS index and the corpus jsonl. Row `i` in FAISS = embedding of row `i` in `corpus/wiki18_100w.jsonl`. FAISS returns integer row IDs, the corpus lookup returns the text.

## Query lifecycle

```mermaid
sequenceDiagram
    participant C as Eval client
    participant API as FastAPI
    participant W as Worker (1 of N)
    participant E as E5 encoder
    participant F as FAISS index
    participant Corp as Corpus (mmap)

    C->>API: POST /search {query, top_n=3}
    API->>API: acquire semaphore
    API->>W: pop worker idx from deque
    W->>E: tokenize "query: {q}"
    E->>E: forward pass → mean-pool → L2-normalize
    E-->>W: (1, 768) fp32 vector
    W->>F: index.search(vec, k=3)
    Note over F: Flat: scan 21M × 768 IPs<br/>IVF-SQ8: scan nprobe cells × ~5K vectors
    F-->>W: top_k row IDs + scores
    W->>Corp: corpus[ids]  (page-cache hit)
    Corp-->>W: list of {id, contents}
    W-->>API: List[Document]
    API->>API: push worker back, release semaphore
    API-->>C: JSON response
```

Wall-clock per stage (single 4090, [../eval/EVAL_OPS.md](../eval/EVAL_OPS.md) profile):

| Stage | Flat IP CPU | IVF-SQ8 CPU | IVF-SQ8 GPU |
|---|---:|---:|---:|
| Encode | 3–8 ms | 3–8 ms | 3–8 ms |
| FAISS search | 100–300 ms | 30–100 ms | 5–15 ms |
| Corpus lookup | <0.1 ms | <0.1 ms | <0.1 ms |
| HTTP overhead | ~10 ms | ~10 ms | ~10 ms |

## Index build pipeline

The flat IP index ships pre-built. The IVF-SQ8 index is reconstructed *from* the flat index — no re-embedding.

```mermaid
flowchart TB
    subgraph Source["Source assets (downloaded)"]
        Wiki["wiki-18.jsonl.gz<br/>21M passages, ~13 GB"]
        Flat["wiki18_100w_e5_flat_inner.index<br/>21M × 768 fp32, ~65 GB"]
    end
    subgraph Build["Build (build_ivf_sq8.py)"]
        Load["Load flat index<br/>~60–90s"]
        Sample["Random sample<br/>1M training vectors"]
        Reconstruct["faiss.reconstruct_n()<br/>~10–30s"]
        Train["Train k-means on GPU<br/>nlist=4096, <1 min"]
        Add["Add all 21M vectors<br/>chunks of 1M, ~5–10 min"]
        Write["Write to disk"]
    end
    Out["wiki18_100w_e5_ivf4096_sq8.index<br/>~16 GB"]
    HFDataset["HF: pantomiman/reason-over-search<br/>retriever/wiki18_100w_e5_ivf4096_sq8.index"]
    Wiki -.->|served as text| Service["Retriever service"]
    Flat --> Load --> Sample --> Reconstruct --> Train --> Add --> Write --> Out
    HFDataset -.->|curl download<br/>(default path)| Out
    Out -.->|default| Service
    Flat -.->|optional swap<br/>(exact recall)| Service
```

The IVF-SQ8 index is the retriever's default. The fastest way to obtain it is to download the prebuilt artifact from this project's HF dataset:

```bash
curl -L -o local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
```

If you want to rebuild it locally instead (~1 hour), the recipe is to reconstruct vectors from the flat index via `faiss.reconstruct_n()` and train + add to a fresh `IndexIVFScalarQuantizer(QT_8bit)` with `nlist=4096`, then write to disk. This avoids re-embedding the 21 M passages (6 to 10 hours on a single 4090). The HF download above is the supported path; the standalone `index_creation/` workspace that previously held a build script has been retired.

`nlist=4096` is constrained by FAISS's `min_points_per_centroid=39` floor: with 1 M training samples, 4096 cells gives ~244 points per centroid. Going to 65 536 cells would need ≥ 2.55 M training samples to clear the floor, so `4096` is the largest viable choice with the current sample size.

## Memory layout

Where each artifact lives at runtime, by index choice:

```mermaid
graph TB
    subgraph Host["Host: RAM (503 GB on this box)"]
        FlatRAM["FAISS flat IP<br/>65 GB resident"]
        IVFRAM["FAISS IVF-SQ8<br/>16 GB resident"]
        CorpusRAM["Corpus (mmap'd)<br/>~14 GB page-cache, evictable"]
        Encoder["E5 encoder<br/>~0.5 GB model on CPU,<br/>or VRAM if --gpu"]
    end
    subgraph GPU["GPU: VRAM (24 GB on 4090, 80 GB on H100)"]
        SGLang["SGLang 3B model<br/>~22 GB bf16"]
        IVFVRAM["FAISS IVF-SQ8 (--gpu)<br/>16 GB VRAM"]
        EncVRAM["E5 encoder (GPU)<br/>~0.5 GB"]
    end
    note1["Flat IP CPU: only host RAM used<br/>safe alongside any SGLang model"]
    note2["IVF-SQ8 GPU: 16 + 22 = 38 GB<br/>doesn't fit on 4090, fits on H100"]
    FlatRAM -.-> note1
    IVFVRAM -.-> note2
```

The hard constraint on a 24 GB 4090: SGLang's 22 GB footprint leaves 2 GB free. Neither the 65 GB flat index nor the 16 GB IVF-SQ8 index fits on GPU alongside SGLang. So on the 4090, the retriever is **CPU-only by default**.

On 80 GB H100: 22 GB (SGLang) + 16 GB (IVF-SQ8 in VRAM) + 0.5 GB (E5 encoder) = ~38 GB, fits comfortably with headroom. GPU FAISS becomes viable. See [RETRIEVER_INDEXING.md](RETRIEVER_INDEXING.md) for the speedup ranking on each option.

## Concurrency model

```mermaid
sequenceDiagram
    participant C1 as Client req 1
    participant C2 as Client req 2
    participant C3 as Client req 3
    participant Sem as Semaphore (N=4)
    participant W as Worker pool

    par 3 concurrent requests
        C1->>Sem: acquire
        C2->>Sem: acquire
        C3->>Sem: acquire
    end
    Sem->>W: pop worker 0 → req 1
    Sem->>W: pop worker 1 → req 2
    Sem->>W: pop worker 2 → req 3
    Note over W: 3 FAISS searches run<br/>in parallel on 3 CPU cores
    W-->>Sem: worker 0 done, push back
    W-->>Sem: worker 1 done
    W-->>Sem: worker 2 done
```

Each worker is a thread-isolated `DenseRetriever` instance. FAISS index reads are GIL-released (the heavy lifting is in C++), so the workers *would* do truly parallel CPU work — except that the current handlers call into them synchronously from `async def`, blocking the event loop. See [RETRIEVER_CONCURRENCY.md](RETRIEVER_CONCURRENCY.md) for the audit and the one-line fix that makes this diagram match reality.

## Files involved

| Path | Role |
|---|---|
| [`local_retriever/retriever_serving.py`](../../local_retriever/retriever_serving.py) | FastAPI service, semaphore pool, CLI |
| [`local_retriever/retriever_config.yaml`](../../local_retriever/retriever_config.yaml) | Default index path, encoder path, nprobe |
| [`local_retriever/flashrag/retriever/retriever.py`](../../local_retriever/flashrag/retriever/retriever.py) | `DenseRetriever`: index load, search, batch_search |
| [`local_retriever/flashrag/retriever/encoder.py`](../../local_retriever/flashrag/retriever/encoder.py) | E5 encoder: tokenize, forward, pool, normalize |
| [`local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index`](../../local_retriever/indexes/) | 16 GB IVF-SQ8 index (default; download from HF: [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index)) |
| [`local_retriever/indexes/wiki18_100w_e5_flat_inner.index`](../../local_retriever/indexes/) | 65 GB Flat IP index (optional, exact recall) |
| [`local_retriever/corpus/wiki18_100w.jsonl`](../../local_retriever/corpus/) | Raw passages, mmap'd |
