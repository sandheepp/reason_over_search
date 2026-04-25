# Evalutation Search R1

## Environment Setup

1. Create a conda environment

```
conda create -n evaluation_search_r1 python=3.11 -y
conda activate evaluation_search_r1
python setup.py develop --no-deps
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Fetch Evaluation Datasets (Git LFS)

The Bamboogle / HotpotQA / MuSiQue / 2WikiMultihopQA files under `data/` are tracked with Git LFS. On a fresh clone they are 1-line pointer files until you pull them.

```bash
cd /workspace/reason_over_search
git lfs pull

# Verify — expect real JSON, not a "version https://git-lfs..." line
head -1 data/bamboogle/test.jsonl
```

4. Download Models

```
cd evaluation_search_r1
mkdir -p search_r1_base_model search_r1_instruct_model

# Install once (if needed)
pip install -U "huggingface_hub[cli]"

# Base model:
# https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo/tree/main
hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model

# Instruct model:
# https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo/tree/main
hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model
```

## Run SG Lang

#### Base Model

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
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

#### Instruct Model

```bash
python -m sglang.launch_server \
  --served-model-name search_r1_instruct \
  --model-path search_r1_instruct_model \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

## Run Evalution

### Reproducibility Reference

Sanity-check on Bamboogle test (Qwen2.5-3b-Instruct, GRPO checkpoint, `apply_chat=True`, `generator_max_input_len=4096`):

| Metric | Search-R1 paper (Table 3) | This repo |
|---|---|---|
| EM | 0.232 | 0.328 |
| format_valid_rate | — | 0.952 |
| retrieval_hit_rate | — | 0.456 |

If your numbers come in much lower, check (in order): retriever health at `127.0.0.1:3005`, `generator_max_input_len` matches sglang's `--context-length`, and that `--apply_chat True` is set for instruct checkpoints.

### Search-R1 Reward-Compatible Evaluation

To compute Search-R1-format reward signals together with standard EM/F1/Acc metrics, add:

```bash
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

When enabled, output files include per-sample fields such as:
- `search_r1_reward`
- `search_r1_format_valid`
- `search_r1_retrieval_hit`
- `search_r1_extracted_answer`

And `metric_score.txt` will also include:
- `search_r1_reward`
- `search_r1_format_valid_rate`
- `search_r1_retrieval_hit_rate`

##### Bamboogle (`data/bamboogle/test.jsonl`)

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
  --apply_chat False \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir results/hotpotqa \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name musique \
  --split dev \
  --save_dir results/musique \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir results/2wikimultihopqa \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` for `train.jsonl`.

#### Instruct Model

##### Reasoning (optional)

##### Bamboogle (`data/bamboogle/test.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name bamboogle \
  --split test \
  --save_dir results/bamboogle \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir results/hotpotqa \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name musique \
  --split dev \
  --save_dir results/musique \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir ../data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir results/2wikimultihopqa \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True \
  --search_r1_mode True \
  --search_r1_structure_format_score 0.2 \
  --search_r1_final_format_score 0.1 \
  --search_r1_retrieval_score 0.1
```

Use `--split train` for `train.jsonl`.