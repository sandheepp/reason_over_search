# Local Retriever

A FAISS-backed dense passage retriever served over HTTP. It encodes queries with the `intfloat/e5-base-v2` model and retrieves passages from the Wikipedia-18 (100-word chunk) corpus, exposing `/search` and `/batch_search` endpoints on `127.0.0.1:3005`.

The default index is a quantized IVF4096-SQ8 FAISS index held in host RAM (CPU FAISS); a flat exact-recall index and a GPU FAISS build are both opt-in.

## Prerequisites

- Python 3.10
- The CPU venv installs `faiss-cpu`, `sentence-transformers`, `transformers`, `fastapi`, and `uvicorn` (full list in `requirements.txt`).
- For the default IVF-SQ8 index, allow roughly 16 GB of free host RAM (the index loads into heap). For the optional flat index, allow roughly 65 GB.

## Install (CPU, default)

Create the environment and install the CPU dependencies:

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements.txt
```

## Install (GPU, optional)

GPU FAISS holds the index in VRAM (roughly 16 GB for IVF-SQ8 plus about 1 GB for the encoder). Build a separate venv at `local_retriever/.venv` with `faiss-gpu-cu12` and a CUDA torch build:

```bash
bash setup_gpu_venv.sh
```

The script creates the venv, installs `requirements-gpu.txt`, and verifies that FAISS and torch both see a GPU. On a single 24 GB card the GPU index does not co-exist with a large model server on the same GPU; run GPU FAISS only when that GPU is otherwise free.

## Download required assets

Run these from the `local_retriever` directory. The encoder, corpus, and index are downloaded into `models/`, `corpus/`, and `indexes/` respectively; the config expects them at those paths.

### Encoder

```bash
mkdir -p models
huggingface-cli download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2 \
  --local-dir-use-symlinks False
```

### Corpus

```bash
mkdir -p corpus
huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset \
  --include "wiki-18.jsonl.gz" \
  --local-dir corpus \
  --local-dir-use-symlinks False

gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl
```

### Index (IVF-SQ8, default)

The default index is the quantized IVF4096-SQ8 build (roughly 16 GB; faster than flat with a recall hit under 1%). Download it from the public Hugging Face dataset:

```bash
mkdir -p indexes
curl -L -o indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
```

### Index (flat, optional, exact recall)

The flat inner-product index (`wiki18_100w_e5_flat_inner.index`) gives exact recall at the cost of roughly 65 GB resident RAM and slower search. Download it only if you need exact retrieval:

```bash
huggingface-cli download PeterJinGo/wiki-18-e5-index \
  --repo-type dataset \
  --local-dir indexes \
  --local-dir-use-symlinks False

# If the index ships split into parts, merge and extract:
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz
```

## Run the service

### CPU + IVF-SQ8 (default)

```bash
python retriever_serving.py --config retriever_config.yaml --port 3005
```

Or use the launch helper, which starts a background process and writes a log under `/tmp`:

```bash
./launch_ivfsq8.sh start 3005
./launch_ivfsq8.sh wait  3005
./launch_ivfsq8.sh stop  3005
```

### CPU + flat index (exact, slower)

```bash
python retriever_serving.py --config retriever_config.yaml --port 3005 \
  --index ./indexes/wiki18_100w_e5_flat_inner.index
```

### GPU + IVF-SQ8 (requires the GPU venv)

```bash
./.venv/bin/python retriever_serving.py --config retriever_config.yaml --port 3005 \
  --gpu --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index
```

## Health check

Once the service is up, confirm it is healthy:

```bash
curl -sS http://127.0.0.1:3005/health
# {"status":"healthy","retrievers":1}
```

A combined health and concurrency smoke check is available:

```bash
./smoke_concurrent.sh 3005 8
```

## API

### Search

```bash
curl -X POST "http://127.0.0.1:3005/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who wrote The Lord of the Rings?", "top_n": 3, "return_score": false}'
```

Set `"return_score": true` to also receive per-document scores.

### Batch search

```bash
curl -X POST "http://127.0.0.1:3005/batch_search" \
  -H "Content-Type: application/json" \
  -d '{"query": ["capital of France", "largest planet in our solar system"], "top_n": 2, "return_score": false}'
```

## Configuration

Defaults live in `retriever_config.yaml`; `retriever_config_smoke.yaml` is a minimal variant for quick checks. Key knobs:

| Setting | Default | Purpose |
|---|---|---|
| `index_path` | `./indexes/wiki18_100w_e5_ivf4096_sq8.index` | FAISS index to load |
| `retrieval_method` | `./models/e5-base-v2/` | path to the query encoder |
| `corpus_path` | `./corpus/wiki18_100w.jsonl` | passage corpus (jsonl) |
| `faiss_gpu` | `False` | hold the index in VRAM (CPU FAISS by default) |
| `faiss_nprobe` | `64` | IVF probe count; ignored by the flat index |

Command-line flags on `retriever_serving.py`:

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `./retriever_config.yaml` | config file to load |
| `--port` | `80` | HTTP port to serve on |
| `--index` | (from config) | override `index_path` without editing the config |
| `--gpu` | off | force GPU FAISS (needs the GPU venv) |
