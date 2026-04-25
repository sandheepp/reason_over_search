# Remote Dev Scripts

Utilities for working with the vast.ai (or any SSH) remote that runs the GPU workloads. All scripts read connection details from the repo-root `.env` file.

## Setup

```bash
cp .env.example .env
# edit .env with your remote host / port / user / path
```

`.env` is gitignored — it contains credentials and the per-instance vast.ai host that rotates each time you spin up a new box.

Required keys:

| Key | Example | Notes |
|---|---|---|
| `REMOTE_HOST` | `ssh1.vast.ai` | Hostname from the vast.ai instance card |
| `REMOTE_PORT` | `18659` | SSH port from the same card |
| `REMOTE_USER` | `root` | Almost always `root` on vast.ai |
| `REMOTE_PATH` | `/workspace/reason_over_search` | Where the repo lives on the remote |

## sync_to_remote.sh

Pushes local code to the remote. Never deletes anything on the remote (no `--delete`). Skips datasets, indexes, model weights, eval results, caches, and `.git/`.

```bash
# Preview what would transfer
./scripts/sync_to_remote.sh -n

# Real run
./scripts/sync_to_remote.sh
```

What gets excluded (kept local-only):

- `data/` — Git-LFS QA datasets
- `evaluation_search_r1/results/` — eval outputs
- `evaluation_search_r1/search_r1_{base,instruct}_model/` — HF-downloaded model weights
- `local_retriever/{corpus,indexes,models}/` — Wiki18 corpus, FAISS index, e5 model
- `.git/`, `.env`, `.venv/`, `__pycache__/`, `.DS_Store`, `node_modules/`
- Binary patterns: `*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, `*.pth`, `*.parquet`, `*.index`, `*.index.gz`, `*.jsonl.gz`, `*.tar`, `*.tar.gz`

If you add a new heavy artifact, also add it to the exclude list in [sync_to_remote.sh](sync_to_remote.sh) (and ideally to `.gitignore`).

## ssh_remote.sh

Opens an interactive shell on the remote, landing in `$REMOTE_PATH`, or runs a single command and exits.

```bash
# Interactive
./scripts/ssh_remote.sh

# One-off
./scripts/ssh_remote.sh "nvidia-smi"
./scripts/ssh_remote.sh "cd evaluation_search_r1 && ls results/bamboogle"
```

## Typical loop

```bash
# Edit code locally, then push
./scripts/sync_to_remote.sh

# Run on the remote
./scripts/ssh_remote.sh "cd evaluation_search_r1 && python run_eval.py --config_path flashrag/config/basic_config.yaml ..."
```

## Switching vast.ai instances

When you stop and re-create a vast.ai box, `REMOTE_HOST` and `REMOTE_PORT` change. Update those two lines in `.env` — the rest stays the same.
