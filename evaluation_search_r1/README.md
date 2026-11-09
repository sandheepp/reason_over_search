# Search-R1 Evaluation

A FlashRAG-based evaluation harness for search-augmented question answering. It drives a `<search>` / `<answer>` style LLM through a retrieval loop, queries a FAISS Wikipedia retriever, and scores the final answer with EM, F1, and accuracy across standard QA benchmarks (Bamboogle, 2WikiMultiHopQA, TriviaQA, PopQA, MuSiQue, NQ, HotpotQA).

## Prerequisites

- Python 3.11
- An NVIDIA GPU to serve the model (the harness itself is CPU-side; the model server needs the GPU)
- Two running services (see [Services](#services) below): a FAISS retriever and an OpenAI-compatible LLM server
- Per-benchmark dataset files in JSONL form (see [Datasets](#datasets))

## Installation

Create an environment and install the package in editable mode:

```bash
conda create -n evaluation_search_r1 python=3.11 -y
conda activate evaluation_search_r1
python setup.py develop --no-deps
pip install -r requirements.txt
```

## Models

The harness evaluates a served model; it does not load weights itself. Download a checkpoint into a local directory, then serve it (next section). Example checkpoints:

```bash
mkdir -p search_r1_base_model search_r1_instruct_model
pip install -U "huggingface_hub[cli]"

huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model \
  --local-dir-use-symlinks False

huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model \
  --local-dir-use-symlinks False
```

## Services

Both services must be up before running any evaluation.

### Retriever (FAISS)

A FAISS retriever serves Wikipedia passages over HTTP, expected at `127.0.0.1:3005`. The retriever lives in the sibling `local_retriever/` directory; follow its README to download the index, corpus, and embedding model, then start it:

```bash
cd ../local_retriever
python retriever_serving.py --config retriever_config.yaml --num_retriever 4 --port 3005
```

Health check:

```bash
curl -sS http://127.0.0.1:3005/health
```

### LLM server (OpenAI-compatible)

Serve the model under test on `127.0.0.1:3000` with any OpenAI-compatible backend. With SGLang:

```bash
python -m sglang.launch_server \
  --served-model-name search_r1_base \
  --model-path search_r1_base_model \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code
```

Swap `--served-model-name` and `--model-path` for the instruct checkpoint to evaluate that variant. vLLM works equally well; point the harness at its OpenAI-compatible endpoint on the same port.

## Datasets

Benchmark data is read from a sibling `data/` directory, one subdirectory per benchmark, each holding split files in JSONL form (for example `data/bamboogle/test.jsonl`, `data/hotpotqa/dev.jsonl`). These files are not shipped; supply them yourself. The harness picks the split via `--split` (`test`, `dev`, or `train`).

## Running an evaluation

### Convenience driver

The `scripts/run_one.sh` helper (in the repo root `scripts/` directory) runs one `(variant, dataset, seed)` combination, looks up the correct split and chat setting per dataset, writes under `results/`, and skips if a result already exists:

```bash
../scripts/run_one.sh base bamboogle 1
../scripts/run_one.sh instruct hotpotqa 1
```

Arguments are `<base|instruct> <dataset> <seed> [data_dir]`. Override the LLM server port with `SGL_PORT` if needed.

### Direct invocation

`run_eval.py` is the entry point. Pass the dataset, split, model, and service URLs directly:

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name bamboogle \
  --split test \
  --save_dir results/bamboogle \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat True
```

Change `--dataset_name` and `--split` to target a different benchmark (for example `--dataset_name hotpotqa --split dev`). For the instruct checkpoint, point `--generator_model` at the instruct directory.

### Key arguments

| Flag | Purpose |
|---|---|
| `--config_path` | Base YAML config (`flashrag/config/basic_config.yaml`) |
| `--method_name` | Pipeline to run; use `search-r1` |
| `--data_dir` | Parent directory holding per-benchmark dataset folders |
| `--dataset_name` | Benchmark name (matches a subdirectory of `--data_dir`) |
| `--split` | `test`, `dev`, or `train` |
| `--save_dir` | Output directory for this run |
| `--save_note` | Label used in the output folder name |
| `--sgl_remote_url` | Host:port of the OpenAI-compatible LLM server |
| `--remote_retriever_url` | Host:port of the FAISS retriever |
| `--generator_model` | Served model directory or name |
| `--apply_chat` | Apply the chat template (`True` for chat/instruct models) |
| `--enable_thinking` | Enable thinking traces if the model supports them |

Common retrieval and generation knobs (top-k documents, query length, sampling temperature, max tokens, metrics) live in `flashrag/config/basic_config.yaml` and can be edited there.

## Output

Results are written under `results/<dataset>/<dataset>_<timestamp>_<save_note>/`, which ships empty. Each run directory contains:

- `metric_score.txt`: EM, F1, and accuracy
- `intermediate_data.json`: per-sample traces (prompts, retrievals, responses)

## Aggregating results

The `scripts/aggregate.py` helper walks the `results/` tree, groups runs by dataset and variant, and writes a Markdown summary with per-seed scores and the mean over seeds:

```bash
python ../scripts/aggregate.py --output RESULTS.md
```

## Tests

```bash
pytest
```
