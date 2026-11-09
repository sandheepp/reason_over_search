# reason-over-search v1 / v2 (Docker)

> **Tags on Docker Hub**: `pantomiman/reason-over-search-v1:v1` (original) and `pantomiman/reason-over-search-v1:v2` (= v1 + `transformers==5.7.0`, adds Qwen3.5 `model_type=qwen3_5` AutoConfig support). **Use `:v2` for new Vast instances.** v1 is kept for reproducing the M1/M2 smoke-results runs on which the M2 numbers were anchored.
>
> The image-rebuild path (`Dockerfile`) currently hits an upstream NeMo-RL conflict on a fresh build (`automodel`+`vllm` extras became mutually exclusive after 2026-05-02). For now, `Dockerfile.v2` is a thin layer FROM `:v1` that adds the transformers upgrade — see "Build v2" below.

## Hybrid paradigm

| Concern | Pattern | Why |
|---|---|---|
| `retriever` + `evaluation_search_r1` | **Code + env baked into the image.** | Stable surface; no need to clone the repo on Vast just to run them. |
| `training` (NeMo-RL) | **Code cloned at runtime; env materialised from a pre-warmed wheel cache.** Image ships `uv` + `/.uv/cache/` (~13 GB, NeMo-RL v0.6.0 + `vllm` extra, including built-from-source `deep_ep` / `deep_gemm`). The repo is `git clone`d onto Vast; `uv sync --extra vllm` reads straight from the cache (fast — no PyPI download). | Active iteration surface — pushing edits doesn't require rebuilding the image. |

## What's in the image

- **Conda env `retriever`** (Python 3.10) — `local_retriever/requirements.txt` installed; source at `/app/local_retriever/`.
- **Conda env `evaluation_search_r1`** (Python 3.11) — `evaluation_search_r1/requirements.txt` installed; source at `/app/evaluation_search_r1/`; flashrag editable-installed at build time (`python setup.py develop --no-deps`).
- **`uv` (pin 0.11.8) at `/usr/local/bin/uv` + pre-warmed wheel cache at `/.uv/cache/`** (~13 GB). Covers NeMo-RL v0.6.0 deps with the `vllm` extra, including the git+source `deep_ep` / `deep_gemm` wheels (built during the docker build with `libibverbs-dev` from apt). `UV_NO_CACHE=0` is set in the image env so `uv sync` on Vast actually reads the cache (the base image's default `UV_NO_CACHE=1` would silently bypass it).
- **Vast SSH boot hook** at `/etc/vast_boot.d/10-fix-ssh-perms.sh` — normalizes `/root/.ssh/authorized_keys` perms on startup.

## Build (from repo root)

```bash
docker build -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

The `Dockerfile` uses `FROM --platform=linux/amd64` so the result matches **Vast.ai** and typical cloud GPUs (**x86_64**), even when you build on an **Apple Silicon** machine. The first such build on a Mac can be slower (QEMU) while it runs the `RUN` steps.

If a remote builder still complains about base platform, use Buildx and load into the local engine:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/reason-over-search-v1/Dockerfile \
  -t reason-over-search-v1:v1 --load .
```

### Build args (training)

| Arg | Default | Purpose |
|---|---|---|
| `NEMO_RL_REF` | `v0.6.0` | NeMo-RL ref. Currently a passthrough — only used when wheel pre-warming is enabled. Bump alongside `training/nemo_rl/`. |
| `UV_EXTRAS` | `vllm` | NeMo-RL extras (e.g. `vllm,nemo_gym`). Currently a passthrough; controls which extras the wheel cache covers if pre-warming is re-enabled. |

```bash
docker build \
  --build-arg NEMO_RL_REF=v0.6.0 \
  --build-arg UV_EXTRAS=vllm \
  -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

### Build VM disk requirement

The wheel pre-warm step extracts torch (~3 GB unpacked) plus other CUDA libs while populating `/.uv/cache/`. The whole training layer needs **≥15 GB free in the build VM** during that step. Docker Desktop on Apple Silicon allocates ~58 GB by default which is too tight once the conda envs are in place.

Bump it: **Settings → Resources → "Disk image size" → 120 GB → Apply & Restart**. Verify with:

```bash
docker run --rm alpine df -h /
```

### Build OOMs on pip / conda

Raise Docker / Colima / Desktop **RAM (8G+)** and try again.

## Build v2 (recommended; thin layer on `:v1`)

```bash
docker build --platform linux/amd64 \
  -f docker/reason-over-search-v1/Dockerfile.v2 \
  -t pantomiman/reason-over-search-v1:v2 .
```

Dockerfile.v2 is `FROM pantomiman/reason-over-search-v1:v1` + `pip install transformers==5.7.0` (in the eval venv) + `LABEL version=2`. The new layer is ~500 MB; the rest is shared with v1.

## Push to Docker Hub

```bash
docker tag reason-over-search-v1:v1 pantomiman/reason-over-search-v1:v1
docker login
docker push pantomiman/reason-over-search-v1:v1   # v1
docker push pantomiman/reason-over-search-v1:v2   # v2 (only the new layer is uploaded; rest dedups)
```

## Run an interactive shell

The Vast base image has its own `ENTRYPOINT`. For a normal shell, override it:

```bash
docker run --rm -it --entrypoint /bin/bash \
  -p 3005:3005 \
  -v "$(pwd)/local_retriever/models":/app/local_retriever/models:ro \
  -v "$(pwd)/local_retriever/indexes":/app/local_retriever/indexes:ro \
  -v "$(pwd)/local_retriever/corpus":/app/local_retriever/corpus:ro \
  reason-over-search-v1:v1
```

## Run retriever (inside container)

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate retriever
cd /app/local_retriever

python retriever_serving.py --config retriever_config_mini.yaml --num_retriever 4 --port 3005
```

In another shell on the host, check health:

```bash
curl -sS http://127.0.0.1:3005/health
```

## Run evaluation (inside container)

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate evaluation_search_r1
cd /app/evaluation_search_r1

# Example:
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

## Run training (inside container, after cloning the repo on Vast)

The training env is **not** baked with code. On a fresh Vast.ai instance:

```bash
# 1. Clone the repo (once per instance)
cd /workspace
git clone https://github.com/<your-user>/reason_over_search.git
cd reason_over_search

# 2. Materialize the venv against the committed NeMo-RL source.
#    Fast — wheels come from the pre-warmed cache at /.uv/cache/, no PyPI download.
cd training/nemo_rl
uv sync --extra vllm

# 3. Activate
source .venv/bin/activate

# 4. Sanity check
python -c "import nemo_rl; print(nemo_rl.__version__)"   # 0.6.0

# 5. Run an example
python examples/run_grpo.py --config=examples/configs/grpo_math_1B.yaml
```

Or use `uv` directly (no activation):

```bash
cd /workspace/reason_over_search/training/nemo_rl
uv run python examples/run_grpo.py --help
```

See [`training/README.md`](../../training/README.md) for setup-script knobs and bumping the pinned NeMo-RL version.

## Vast SSH troubleshooting

If you see:

- `No ED25519 host key is known ... and you have requested strict checking`
- `Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys`

use:

```bash
# First connect once and accept/add host key (or use StrictHostKeyChecking=accept-new)
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai
```

Then your tunnel command works:

```bash
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai -L 8080:localhost:8080
```
