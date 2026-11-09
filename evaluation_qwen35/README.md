# Qwen3.5 Search-Augmented Evaluation

A FlashRAG-based harness for evaluating a small Qwen3.5 policy on open-domain QA with tool-augmented retrieval. The model interleaves reasoning with `search` tool calls, retrieves Wikipedia passages from a local retriever, and emits a final answer. It uses the Qwen3.5 native tool-call format (`<tool_call>` / `<tool_response>`) rather than the plain `<search>` / `<information>` tags, and scores predictions with EM, F1, and accuracy.

## Prerequisites

You need two services running before evaluation, plus the model weights:

1. A local retriever exposing a search HTTP endpoint (see `../local_retriever`).
2. An OpenAI-compatible LLM server (for example SGLang) serving the Qwen3.5 policy under test.
3. The Qwen3.5 policy weights, fetched and validated (see Model setup below).

A CUDA GPU is recommended for the LLM server. The retriever can run on CPU or GPU.

## Installation

Create an environment and install the package in develop mode:

```bash
conda create -n evaluation_qwen35 python=3.11 -y
conda activate evaluation_qwen35
python setup.py develop --no-deps
pip install -r requirements.txt
```

## Model setup

`download_and_test_models.py` downloads the Qwen3.5-0.8B base and instruct weights into the Hugging Face cache, renders the tool-call prompt via `apply_chat_template`, runs a short CPU generation on a few sample questions, and reports how many `<tool_call>` and `<answer>` blocks each model produced:

```bash
python3 download_and_test_models.py
```

This is a quick sanity check that the weights load and the tool-call prompt renders correctly. Weights are cached under `~/.cache/huggingface/` by default. To serve a model you trained yourself, point your LLM server at its local checkpoint directory instead.

## Retriever

The harness queries a retriever over a Wikipedia FAISS index. Download the default index and start the service from `../local_retriever` (see its README for full instructions):

```bash
curl -L -o ../local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
```

Note the host and port the retriever listens on (for example `127.0.0.1:3005`); you pass it to the eval via `--remote_retriever_url`.

## Serving the model

Start an OpenAI-compatible server for the policy under test. With SGLang:

```bash
python -m sglang.launch_server \
  --served-model-name qwen35_policy \
  --model-path /path/to/qwen35_checkpoint \
  --tp 1 \
  --context-length 8192 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code
```

Note the host and port (for example `127.0.0.1:3000`); you pass it to the eval via `--sgl_remote_url`.

## Running an evaluation

With the retriever and LLM server up, run `run_eval.py`. Set `--prompt_mode qwen35` to use the Qwen3.5 native tool-call template:

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --prompt_mode qwen35 \
  --data_dir /path/to/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir ./results/bamboogle \
  --save_note qwen35 \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model qwen35_policy \
  --apply_chat True
```

`--data_dir` is the parent directory that contains per-dataset subfolders (for example `data/bamboogle/test.jsonl`); pass `--dataset_name bamboogle` and `--split test` to select the file. Datasets include `bamboogle`, `hotpotqa`, `musique`, `2wikimultihopqa`, `nq`, `popqa`, and `triviaqa`; switch with `--dataset_name` and `--split` (`test`, `dev`, or `train` as available).

Results, including per-sample outputs and a `metric_score.txt` with EM, F1, and accuracy, are written under `--save_dir`.

## Key options

- `--prompt_mode`: prompt template. `qwen35` (alias `qwen35_native`) uses the Qwen3.5 nested-XML tool-call format; the search-tool schema is auto-injected into the system area by the chat template.
- `--apply_chat`: apply the model's chat template (`True` for instruct-style checkpoints).
- `--enable_thinking`: enable the model's thinking mode in the chat template (default `False`).
- `--test_sample_num`: subsample N items from the split for a fast smoke run; pair with `--random_sample True` for a random subset (otherwise the first N are taken).
- `--seed`: seed for deterministic subsampling.
- `--gpu_id`: override the eval-process GPU id; set `0` on a single-GPU box.

## Config knobs

Defaults live in `flashrag/config/basic_config.yaml` and can be overridden there:

- `retrieval_topk`: number of passages returned per search call.
- `retrieval_method`: retrieval model (default `e5`).
- `generator_max_input_len`: max prompt length fed to the policy (default `8192`).
- `generation_params`: sampling settings for the policy (`max_tokens`, `temperature`).
- `metrics`: scoring metrics computed over predictions (`em`, `acc`, `f1`).

## Tests

A small test suite covers the parser, reward, and run-eval wiring:

```bash
pytest
```
