# Training

> **How the training paradigm works + Search-R1 vs Qwen3.5 differences:** see [docs/training/](../docs/training/).
>
> TL;DR — `pantomiman/reason-over-search-v1` ships `uv` + a **pre-warmed NeMo-RL wheel cache** at `/.uv/cache/` (~13 GB; covers v0.6.0 with the `vllm` extra including the built-from-source `deep_ep` and `deep_gemm` wheels). NeMo-RL source is committed at [`nemo_rl/`](nemo_rl/) (pinned to `v0.6.0`). On Vast: clone the repo, `cd training/nemo_rl`, `uv sync --extra vllm` — fast (wheels pulled from cache, no PyPI download). Same pattern as `local_retriever/` and `evaluation_search_r1/` (env from image, code from repo).

## Environment Setup

`uv` is preinstalled in the docker image at `/usr/local/bin/uv`, and the wheel cache at `/.uv/cache/` is pre-warmed. Inside the docker container:

```bash
cd training/nemo_rl
uv sync --extra vllm                  # fast — wheels read from /.uv/cache/
source .venv/bin/activate
```

The `uv sync` creates the venv at `training/nemo_rl/.venv/` (Python 3.13). On the Vast docker image this is fast — wheels come from the pre-warmed cache, no PyPI round-trip.

> **How the pre-warm works.** The Dockerfile shallow-clones NeMo-RL @ v0.6.0, runs `uv sync --extra vllm --no-install-project` (downloads + builds all transitive deps including the git+ source `deep_ep` / `deep_gemm`), and leaves the populated cache at `/.uv/cache/`. We override `UV_NO_CACHE=0` (the vastai base image sets it to `1` by default — that would silently disable caching). The base image also requires `libibverbs-dev` for `deep_ep` to compile (provided by the apt step in the Dockerfile). Build VM needs ≥120 GB disk for torch's unpack — bump Docker Desktop's "Disk image size" if building locally on a Mac.

If `uv` is missing entirely (e.g. running outside the docker image and outside any other env that has it), use the helper script:

```bash
bash training/setup.sh
```

`setup.sh` installs `uv` if needed, then runs the same `uv sync --extra ${UV_EXTRAS:-vllm}` against the committed NeMo-RL source.

### Adding optional dependency groups

NeMo-RL exposes extras: `vllm`, `fsdp`, `automodel`, `mcore`, `nemo_gym`, `sglang`, `nvrx`. Default is `vllm`. Add more:

```bash
UV_EXTRAS="vllm,nemo_gym" bash training/setup.sh
```

`UV_EXTRAS` is also exposed at docker build time (currently a passthrough; if/when the wheel pre-warm is re-enabled, the build arg controls which extras get cached):

```bash
docker build --build-arg UV_EXTRAS=vllm,nemo_gym \
  -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

### Bumping the pinned NeMo-RL version

```bash
NEMO_RL_REF=v0.7.0 FORCE_RECLONE=1 bash training/setup.sh
git add training/nemo_rl/
git commit -m "training: bump NeMo-RL to v0.7.0"
# (optional) rebuild the docker image; only matters if/when wheel pre-warm is enabled:
docker build --build-arg NEMO_RL_REF=v0.7.0 \
  -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

`FORCE_RECLONE=1` wipes any local edits to `nemo_rl/`. If you have local changes you want to preserve across an upstream version bump, save them to a feature branch first; `setup.sh` no longer auto-replays patches.

## Download steps

### Training dataset

```bash
training/scripts/prepare_dataset.py
```

`prepare_dataset.py` is a `uv` inline-script (idempotent). It pulls the [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) parquets via `hf_hub_download` (NQ + HotpotQA mix, 169,615 train + 51,713 test) and **strips the prebaked Search-R1 template** — replaces each row's `prompt[0].content` with the bare `question`. Output: `data/training/nq_hotpotqa_train/{train,test}.parquet`. Schema in [docs/training/TRAINING_DATA.md](../docs/training/TRAINING_DATA.md).

```bash
training/scripts/prepare_dataset.py --force   # regenerate even if outputs exist
```

### Models

Download both Qwen3.5-2B variants we train:

- **base** → [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (pure LM, no post-training)
- **hybrid** → [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (post-trained with soft-switch reasoning toggle)

Each is ~5 GB on disk (bf16 weights + tokenizer + config). Run the downloads on the Vast instance — they live on persistent storage so they survive instance restarts.

```bash
cd training
mkdir -p models

# Install once if needed
pip install -U "huggingface_hub[cli]"

# Base
huggingface-cli download Qwen/Qwen3.5-2B-Base \
  --local-dir models/Qwen3.5-2B-Base \
  --local-dir-use-symlinks False

# Hybrid (default soft-switch reasoning)
huggingface-cli download Qwen/Qwen3.5-2B \
  --local-dir models/Qwen3.5-2B \
  --local-dir-use-symlinks False
```

> **Note:** The launch scripts default to passing the HF repo IDs (`Qwen/Qwen3.5-2B-Base` / `Qwen/Qwen3.5-2B`) directly to vLLM, which resolves them via the HF cache populated by the downloads above. To pin to a specific local copy instead, pass `policy.model_name=$PWD/training/models/Qwen3.5-2B-Base` (absolute path) on the CLI.

## Configure W&B (one-time per Vast instance)

```bash
cat > training/.env <<'EOF'
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=reason-over-search
EOF
source training/.env
```

`training/.env` is gitignored.

## Run Training

### Sanity check the env

```bash
cd training/nemo_rl
python -c "import nemo_rl; print(nemo_rl.__version__)"   # → 0.6.0
python examples/run_grpo.py --help
```

### Run NeMo-RL's bundled GRPO recipe (smoke test)

Useful to confirm the env works before wiring our Search-R1 retrieval env:

```bash
cd training/nemo_rl
uv run python examples/run_grpo.py --config=examples/configs/grpo_math_1B.yaml
```

### Run Search-R1 GRPO

**Pre-flight (one-time per Vast instance):**

1. Retriever live at `127.0.0.1:3005` (see [`local_retriever/README.md`](../local_retriever/README.md)). Verify: `curl -sS http://127.0.0.1:3005/health` → `"healthy"`.
2. `training/.env` populated (`cp training/.env.example training/.env`, then fill `WANDB_API_KEY`). Optional: `CHECKPOINT_DIR_BASE=/workspace/persistent/checkpoints` for survival across instance restarts.
3. Training venv materialized at `training/nemo_rl/.venv/` (run `bash training/setup.sh` if not).
4. Datasets pulled: `data/training/nq_hotpotqa_train/{train,test}.parquet` present (Git LFS — `git lfs pull`).

**Launch (each command runs one seed of one variant):**

```bash
# 1× A100 80GB
bash training/scripts/run_grpo_1xa100.sh --variant {base|hybrid} --seed N [--arm {qwen_native|paper}]

# 2× A100 80GB
bash training/scripts/run_grpo_2xa100.sh --variant {base|hybrid} --seed N [--arm {qwen_native|paper}]
```

| Arg | Values | Default | Effect |
|---|---|---|---|
| `--variant` | `base`, `hybrid` | `base` | Selects `Qwen/Qwen3.5-2B-Base` or `Qwen/Qwen3.5-2B`; `hybrid` adds `enable_thinking=true`. |
| `--seed` | int | `42` | RNG seed for the whole run — controls dataset-shuffle order, vLLM rollout sampling, and PyTorch RNG. Different seeds give error bars on EM (Phase 2 = 3 seeds × 2 variants for variance estimation). Sets `grpo.seed`; also goes into the W&B run name and the checkpoint dir. |
| `--arm` | `qwen_native`, `paper` | `qwen_native` | Chat-template arm. `paper` wires [`training/src/prompts/search_r1_paper.txt`](src/prompts/search_r1_paper.txt); `qwen_native` registers the `search` tool via `tokenizer.apply_chat_template`. |

**6-run plan** (matches Phase-2 success criteria — 3 seeds × 2 variants):

```bash
for v in base hybrid; do
  for s in 42 7 1337; do
    bash training/scripts/run_grpo_1xa100.sh --variant $v --seed $s
  done
done
```

**Under the hood:** the bash wrapper sources `training/.env`, picks the config at [`training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`](configs/), assembles Hydra overrides (model name, seed, arm, W&B run name, checkpoint dir, plus hybrid `enable_thinking` and paper `prompt_file` when relevant), and execs [`training/scripts/run_grpo.py`](scripts/run_grpo.py) — a thin overlay launcher that imports [`training.src.registry`](src/registry.py) (populates DATASET / PROCESSOR / ENV registries) before calling NeMo-RL's `examples.run_grpo.main()`.

**Hyperparameter rationale + verified mapping vs upstream verl:** [docs/training/README.md](../docs/training/README.md).
**Vast.ai end-to-end sequence:** [docs/milestone_2/PHASE_2_RUNBOOK.md](../docs/milestone_2/PHASE_2_RUNBOOK.md).

## Folder layout

```
training/
├── README.md                    # this file (operational; how to run things)
├── setup.sh                     # installs uv + runs uv sync (local dev / docker fallback)
├── .env                         # W&B key (gitignored)
├── nemo_rl/                     # vendored NeMo-RL source @ v0.6.0 (committed; edit-in-place)
│   └── .venv/                   # uv-managed Python 3.13 venv (gitignored, materialized on Vast)
├── configs/                     # full GRPO YAMLs per GPU layout
│   ├── grpo_qwen3.5_2b_1xa100.yaml
│   └── grpo_qwen3.5_2b_2xa100.yaml
├── scripts/
│   ├── prepare_dataset.py       # download + strip + reshape training data
│   ├── run_grpo.py              # overlay launcher: register + hand off to NeMo-RL main()
│   ├── run_grpo_1xa100.sh       # bash wrapper (variant, seed, arm)
│   └── run_grpo_2xa100.sh       # bash wrapper (variant, seed, arm)
├── src/                         # Search-R1 overlay — registered into NeMo-RL at launch
│   ├── chat_template/tools.py
│   ├── datasets/search_r1.py
│   ├── environments/parsers.py
│   ├── environments/search_r1_env.py
│   ├── processors/search_r1.py
│   ├── prompts/search_r1_paper.txt
│   ├── rewards/search_r1.py
│   └── registry.py              # populates registries on import (side effect)
└── tests/                       # pytest — reward parity, parser dispatch, env step (mocked retriever)
```

## See also

Operational guide is here. The **why** lives in [docs/training/](../docs/training/) — start there:

- [docs/training/README.md](../docs/training/README.md) — landing page, end-to-end view, step-5 audit summary, overlay architecture
- Per-topic deep dives: [TRAINING_DATA.md](../docs/training/TRAINING_DATA.md), [CHAT_TEMPLATE.md](../docs/training/CHAT_TEMPLATE.md), [PAPER_VS_OURS_TRAINING.md](../docs/training/PAPER_VS_OURS_TRAINING.md), [VERL_REFERENCE.md](../docs/training/VERL_REFERENCE.md), [VALIDATION.md](../docs/training/VALIDATION.md), [NEMO_RL_KNOBS.md](../docs/training/NEMO_RL_KNOBS.md)
- [docs/milestone_2/MILESTONE_2.md](../docs/milestone_2/MILESTONE_2.md) — overall milestone scope
