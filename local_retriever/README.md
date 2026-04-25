# Retriever

> **How the search actually works + RAM costs + swapping in a quantized index:** see [INDEXING.md](INDEXING.md).
>
> TL;DR — the flat FAISS index is ~65 GB because it stores exact 768-dim float32 vectors for all 21 M wiki-18 passages. A quantized index (e.g. `IVF65536,SQ8`) cuts that to ~16 GB with <1% recall loss and faster queries; only `index_path` in the config changes.

## Environment Setup

1. Create a conda environment

```
conda create -n retriever python=3.10 -y
conda activate retriever
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Download Corpus

```
cd local_retriever
mkdir -p corpus

# Install once (if needed)
pip install -U "huggingface_hub[cli]"

# Download from Hugging Face dataset:
# https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/tree/main
hf download PeterJinGo/wiki-18-corpus \
  --repo-type dataset \
  --include "wiki-18.jsonl.gz" \
  --local-dir corpus

# Extract and rename to match retriever_config.yaml
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl
```

4. Download Index

```
cd local_retriever
mkdir -p indexes

# Download from Hugging Face dataset:
# https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/tree/main
hf download PeterJinGo/wiki-18-e5-index \
  --repo-type dataset \
  --local-dir indexes

# Merge the split parts into the final index (~64 GB, not gzipped):
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index
rm indexes/part_aa indexes/part_ab
```

5. Download Embedding Model

```
cd local_retriever
mkdir -p models

# Download model from Hugging Face:
# https://huggingface.co/intfloat/e5-base-v2/tree/main
hf download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2
```

## Run Retriever

```
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005
```

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