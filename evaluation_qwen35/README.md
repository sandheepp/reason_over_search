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

3. Download Models

```
cd evaluation_search_r1
mkdir -p search_r1_base_model search_r1_instruct_model

# Install once (if needed)
pip install -U "huggingface_hub[cli]"

# Base model:
# https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo/tree/main
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model \
  --local-dir-use-symlinks False

# Instruct model:
# https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo/tree/main
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model \
  --local-dir-use-symlinks False
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
  --trust-remote-code
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
  --trust-remote-code
```

## Run Evalution

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
  --data_dir /app/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /app/evaluation_search_r1/results/bamboogle \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/hotpotqa \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name musique \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/musique \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/2wikimultihopqa \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False
```

Use `--split train` for `train.jsonl`.

#### Instruct Model

##### Reasoning (optional)

##### Bamboogle (`data/bamboogle/test.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /app/evaluation_search_r1/results/bamboogle \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/hotpotqa \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name musique \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/musique \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir /app/evaluation_search_r1/results/2wikimultihopqa \
  --save_note search_r1_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_instruct_model \
  --apply_chat True
```

Use `--split train` for `train.jsonl`.