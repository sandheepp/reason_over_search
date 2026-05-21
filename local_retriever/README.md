# Retriever

> **How the search actually works + RAM costs + swapping in a quantized index:** see [docs/RETRIEVER_INDEXING.md](../docs/retriever/RETRIEVER_INDEXING.md).
>
> TL;DR — default is **CPU + IVF-SQ8** (~16 GB, ~3-10× faster than flat, <1 % recall hit; runs in the existing `/venv/retriever` faiss-cpu venv). Download with one curl from this project's HF dataset (see [Index](#index) below). For GPU FAISS, run [`setup_gpu_venv.sh`](setup_gpu_venv.sh) and pass `--gpu` to `retriever_serving.py`. The flat IP index (~65 GB, exact float32) is opt-in via `--index ./indexes/wiki18_100w_e5_flat_inner.index` for paper-quality eval.

## Environment Setup — CPU (default)

The default install path uses `faiss-cpu`. The standard repo setup creates a conda env (or use the prebuilt `/venv/retriever` if you're on the docker image):

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements.txt
```

Then download the corpus / index / model (steps below).

## Environment Setup — GPU (opt-in)

A separate venv at `local_retriever/.venv` with `faiss-gpu-cu12` + `torch+cu130`. Created by [`setup_gpu_venv.sh`](setup_gpu_venv.sh):

```bash
bash local_retriever/setup_gpu_venv.sh
# verifies: faiss.get_num_gpus() >= 1 and torch.cuda.is_available()
```

What the script does:

1. `python3.10 -m venv local_retriever/.venv`
2. `pip install -r requirements-gpu.txt` with `TMPDIR=/workspace/tmp` (the root filesystem is small on this box; pip wheel extraction would OOM on `/tmp` otherwise).
3. Imports `faiss` and `torch`, asserts the GPU is visible.

Final size of the venv: ~6.3 GB (mostly nvidia-cudnn / torch / triton wheels).

### VRAM caveat (single 4090)

GPU FAISS holds the IVF-SQ8 index in VRAM (~16 GB) plus the E5 encoder (~1 GB). That fits on an idle 4090 (24 GB) but **does not co-exist with SGLang's 22 GB 3B-model footprint** on the same card. On this single-GPU box, run GPU FAISS only when SGLang is stopped (or use `--num_retriever 1`; each worker would clone its own 16 GB into VRAM).

For sweeps where SGLang must be live, stay on CPU with the IVF-SQ8 default. The flat IP index is the slower fallback; pass `--index ./indexes/wiki18_100w_e5_flat_inner.index` only when exact recall matters.

## Download steps

### Corpus

```bash
cd local_retriever
mkdir -p corpus

pip install -U "huggingface_hub[cli]"   # if not already installed

huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset \
  --include "wiki-18.jsonl.gz" \
  --local-dir corpus \
  --local-dir-use-symlinks False

gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl
```

### Index

The retriever's default is the IVF-SQ8 quantized index (~16 GB, ~3-10× faster than flat, <1 % recall hit). Download from this project's HF dataset:

```bash
cd local_retriever
mkdir -p indexes
curl -L -o indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
# ~16 GB, ~2-4 min on a fast link
```

The flat IP index (`wiki18_100w_e5_flat_inner.index`, ~65 GB after merge + gunzip) is optional — only needed for full retrieval-quality eval (exact recall). To download:

```bash
huggingface-cli download PeterJinGo/wiki-18-e5-index \
  --repo-type dataset \
  --local-dir indexes \
  --local-dir-use-symlinks False

# If files are split into part_aa / part_ab, merge and extract:
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz
```

To rebuild the IVF-SQ8 index locally from the flat index instead of downloading, see [/workspace/index_creation/README.md](../../index_creation/README.md).

### Embedding Model

```bash
cd local_retriever
mkdir -p models
huggingface-cli download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2 \
  --local-dir-use-symlinks False
```

## Run Retriever

### CPU + IVF-SQ8 (default)

```bash
cd /workspace/reason_over_search/local_retriever
python retriever_serving.py --config retriever_config.yaml --num_retriever 4 --port 3005
```

Defaults read from [retriever_config.yaml](retriever_config.yaml): `faiss_gpu: False`, `index_path: ./indexes/wiki18_100w_e5_ivf4096_sq8.index`, `faiss_nprobe: 64`. CPU FAISS is read-only and parallel-safe — bump `--num_retriever` for higher QPS on big datasets.

### CPU + flat IP (exact, slower fallback)

```bash
python retriever_serving.py --config retriever_config.yaml --num_retriever 4 --port 3005 \
  --index ./indexes/wiki18_100w_e5_flat_inner.index
```

`--index` overrides the yaml's `index_path`. Use this only when paper-quality exact recall matters.

### GPU + IVF-SQ8 (fastest, requires the GPU venv)

```bash
./.venv/bin/python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005 \
  --gpu --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index
```

`--gpu` forces `faiss_gpu=True`, overriding both the yaml and the FlashRAG `Config._init_device` autodetection. Use `--num_retriever 1` (each worker would clone its own 16 GB into VRAM).

### CLI reference

| Flag | Default | What it does |
|---|---|---|
| `--config` | `./retriever_config.yaml` | yaml config file |
| `--num_retriever` | `1` | number of in-process retriever workers (one FAISS index per worker) |
| `--port` | `80` | HTTP port |
| `--gpu` | off | hold the FAISS index in VRAM; needs the `.venv` GPU build |
| `--index` | (from yaml) | override `index_path`; useful for switching flat ↔ IVF-SQ8 without editing yaml |

## API Endpoints

### Health

```bash
curl -X GET "http://127.0.0.1:3005/health"
```

### Search

```bash
curl -X POST "http://127.0.0.1:3005/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": false
  }'
```

### Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3005/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": true
  }'
```

### Batch Search

```bash
curl -X POST "http://127.0.0.1:3005/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": false
  }'
```

### Batch Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3005/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": true
  }'
```