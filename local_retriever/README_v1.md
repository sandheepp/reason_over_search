# Retriever — v1 (May 2026)

> **Status:** v1, current default. Replaces the flat-IP-by-default setup that shipped with the initial repo (still documented in [`README.md`](README.md) for reference).
>
> **Why this exists:** training rollouts hammer the retriever (≈80 concurrent `/batch_search` calls per smoke step on 4 prompts × 5 generations × ≤4 turns; ≈2k+ on a real-config step). The flat IP index with `num_retriever=2` timed out on most queries (`Read timed out (read timeout=30.0)`), which collapsed nearly all rollout rewards to 0. v1 switches the **default** to the IVF-SQ8 quantized index + 8 workers and verified throughput holds under load.

## Defaults at a glance

| Knob | Value | Where |
|---|---|---|
| Index | `./indexes/wiki18_100w_e5_ivf4096_sq8.index` (~16 GB; HF: [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index)) | `retriever_config.yaml: index_path` |
| Engine | CPU FAISS (`faiss-cpu`) | `retriever_config.yaml: faiss_gpu: false` |
| nprobe | 64 | `retriever_config.yaml: faiss_nprobe` |
| Worker count | **8** (recommended for training rollouts) | `--num_retriever 8` on the CLI |
| Port | 3005 | `--port` on the CLI |
| Encoder | `intfloat/e5-base-v2` (local clone) | `retriever_config.yaml: retrieval_method` |

## Run

Single canonical command for training-time use on a 1× A100 / 80 GB Vast box:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate retriever
cd /workspace/reason_over_search/local_retriever

nohup python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 8 \
  --port 3005 \
  > /tmp/retriever.log 2>&1 &
disown
```

Cold start: ~30–60 s (8 workers each load the IVF index into heap; total RSS ~134 GB on first launch). Watch for `Uvicorn running on http://0.0.0.0:3005` in `/tmp/retriever.log`.

Smoke check:

```bash
curl -sS http://127.0.0.1:3005/health
# {"status":"healthy","retrievers":{"total":8,"available":8}}

curl -sS -X POST http://127.0.0.1:3005/batch_search \
  -H "Content-Type: application/json" \
  -d '{"query":["Tolkien Lord of the Rings","capital of France","largest planet"],"top_n":3,"return_score":false}'
```

Expected: ~1–2 s for a 3-query batch.

## When to deviate from v1

| Scenario | Index | Workers | Why |
|---|---|---|---|
| **Training rollouts** (default) | IVF-SQ8 | 8 | Concurrency-safe, ≤2 s p99 batch latency. < 1 % recall hit is acceptable for online RL. |
| **Validation / final eval / paper-fidelity reproduction** | Flat IP | 2 | Exact float32 inner-product. ~65 GB resident per worker, so 2 workers max. Slower under concurrency, fine for serial eval. |
| **GPU FAISS** (only when SGLang is OFF) | IVF-SQ8 | 1 | ~16 GB VRAM per worker. Co-existing with SGLang's 22 GB 3B-model footprint on a 24 GB 4090 doesn't fit; on an 80 GB A100 there's slack but no real win. |

CLI examples:

```bash
# Validation / paper fidelity:
python retriever_serving.py --config retriever_config.yaml --num_retriever 2 --port 3005 \
  --index ./indexes/wiki18_100w_e5_flat_inner.index

# GPU IVF (requires the GPU venv built by setup_gpu_venv.sh):
./.venv/bin/python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005 \
  --gpu
```

## Capacity sanity (1× A100 80 GB Vast box)

- IVF-SQ8 index: 16 GB on disk, ~16 GB loaded into heap per worker → 8 workers ≈ 128–134 GB RSS total (one process). Requires ≥ 150 GB host RAM; comfortably met by the 503 GB Vast A100 image.
- Flat IP index: 65 GB on disk and per-worker resident — at 2 workers that's 130 GB, still fits, but each retrieval is full-corpus exact search → seconds per query.

## Get the index

```bash
mkdir -p /workspace/reason_over_search/local_retriever/indexes
curl -L -o /workspace/reason_over_search/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
```

(Build-from-source path: see [/workspace/index_creation/README.md](../../index_creation/README.md).)

## Endpoints (unchanged from v0)

`GET /health`, `POST /search`, `POST /search` (with scores), `POST /batch_search`, `POST /batch_search` (with scores). See [`README.md`](README.md) for request/response schemas. The contract did not change in v1.

## Changelog

- **v1 (2026-05-02):** default index switched to IVF-SQ8; `retriever_config.yaml: index_path` updated; recommended `--num_retriever 8` for training. Verified end-to-end with all four GRPO smoke combos (no retrieval timeouts).
- **v0:** flat IP default, `--num_retriever 1` documented. Kept for reference; not recommended for training rollouts on this hardware.
