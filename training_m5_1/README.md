# training_m5_1

GRPO training scaffold for the **ReSearch paper recipe on Qwen3.5-0.8B / MuSiQue / F1-only reward** (milestone M5 + M5.1). Self-contained sibling of [`training/`](../training/); the M5.1 sub-experiment runs out of this directory without touching the M2 source tree.

Narrative + design decisions: [docs/milestone_5/MILESTONE_5.md](../docs/milestone_5/MILESTONE_5.md). Paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md). Audit of what differs from M2: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md).

## What's different from M2

The whole point: be byte-aligned with the M4 eval pipeline ([`evaluation_qwen35/`](../evaluation_qwen35/)) so a trained checkpoint is directly evaluable.

| | M2 ([`training/`](../training/)) | M5.1 (this directory) |
|---|---|---|
| Target model | Qwen3.5-2B | **Qwen3.5-0.8B** |
| Training data | NQ + HotpotQA mix | **MuSiQue only** |
| Reward | EM + format shaping (Search-R1 paper-faithful) | **F1-only on `<answer>X</answer>`**; no format reward, no `\boxed{}` wrap |
| Parsers / scorer / tool schema | local copies of FlashRAG functions | **re-exported from `evaluation_qwen35`** so train and eval can't drift |
| Prompts | local `.txt` files (M2) | materialised from `evaluation_qwen35.flashrag.search_r1.templates` via `scripts/sync_m4_prompts.py` |
| W&B project | `reason-over-search` (shared) | `reason_over_search_m5_1` (per-experiment isolation) |

## Environment setup

`nemo_rl/` here is a **symlink** to `../training/nemo_rl/` (disk-constrained; the upstream vendor is 23 GB and `/workspace` only has ~25 GB free). The `setup.sh` flow follows the symlink and creates the venv inside it, which means **M5.1 and M2 share the same venv state**. That's fine while we're on the same NeMo-RL version (v0.6.0); if you ever bump NeMo-RL in one experiment, you must break the symlink first (otherwise the bump leaks to the other experiment).

```bash
cd training_m5_1
bash setup.sh                       # uv sync --extra vllm against ../training/nemo_rl/
source nemo_rl/.venv/bin/activate   # resolves through the symlink
```

The pre-warmed wheel cache at `/.uv/cache/` makes this fast on Vast (`pantomiman/reason-over-search-v1`).

## Data prep

```bash
# 1. MuSiQue train split (~20k rows, ~10 MB parquet).
python scripts/prep_musique.py
#   ↳ writes data/training/musique/train.parquet at the repo root.

# 2. M4 prompt template. Currently pre-staged at the M4.2 canonical
#    (`qwen35_minimal`). After M4.4 locks a new winner, re-run with --mode.
python scripts/sync_m4_prompts.py --list
python scripts/sync_m4_prompts.py --mode qwen35_minimal
#   ↳ writes src/prompts/m5_qwen35_user.txt (+ optional m5_qwen35_system.txt).
```

Both scripts are idempotent: re-running skips if outputs are already there.

## Configure W&B (one-time per Vast instance)

```bash
cp training_m5_1/.env.example training_m5_1/.env
# edit training_m5_1/.env, fill WANDB_API_KEY (+ optional CHECKPOINT_DIR_BASE).
```

`training_m5_1/.env` is gitignored.

## Run

**Pre-flight**: retriever live at `127.0.0.1:3005` (see [`local_retriever/README.md`](../local_retriever/README.md)); `data/training/musique/train.parquet` present; `src/prompts/m5_qwen35_user.txt` present; `nemo_rl/.venv/` materialized.

```bash
# 50-step end-to-end smoke (20 traj/step). Target: <= 25 s/step on 1x A100-80GB.
bash scripts/smoke.sh
bash scripts/smoke.sh --seed 7

# Production run (M5.1 paper-faithful recipe).
# configs/m5_1_research_paper.yaml is committed (paper-faithful + M5.2
# system gains O1 fused AdamW + R2 vLLM async_engine). Schedule: 622 steps
# × 320 trajectories. Live anchor: ~10-11 min/step steady-state on 1x A100-80GB
# (= ~4.5 d, ~$130 on Vast). See docs/report/RESULTS_SMOKE_m5.md §6 for the
# live trajectory and docs/setup/HARDWARE_COMPARISON.md for per-GPU estimates.
bash scripts/run.sh --mode prod --seed 42

# Hydra overrides:
bash scripts/smoke.sh -- policy.train_micro_batch_size=8 grpo.max_num_steps=20
```

W&B run name: `qwen3.5-0.8b-musique-m5_<mode>-seed<N>-<TS>`. Checkpoints land under `${CHECKPOINT_DIR_BASE:-results/grpo}/m5_<mode>/seed<N>/` every 50 steps (not timestamped, so resumes work).

## Folder layout

```
training_m5_1/
├── README.md                       # this file
├── setup.sh                        # uv venv + uv sync against nemo_rl/ symlink
├── .env, .env.example, .gitignore
├── nemo_rl/ -> ../training/nemo_rl/   # SYMLINK; shared with M2
├── src/                            # overlay; see docs/report/CODE_SETUP_m5.md §3.2 for the per-file diff
│   ├── chat_template/tools.py      # re-exports QWEN35_SEARCH_TOOL from evaluation_qwen35
│   ├── datasets/search_r1.py       # dataset-agnostic parquet adapter
│   ├── environments/parsers.py     # qwen_native parse_query delegates to evaluation_qwen35
│   ├── environments/search_r1_env.py # M5.1: F1 fallback + near_em_rate metric + kwarg-dispatch fix
│   ├── processors/search_r1.py     # docstring updated; dataset-agnostic
│   ├── prompts/m5_qwen35_user.txt  # pre-staged from M4.2 canonical
│   ├── prompts/_archive_m2/        # M2 prompt files preserved for ablation
│   ├── rewards/search_r1.py        # M5.1: F1-only; re-exports scorer from evaluation_qwen35
│   └── registry.py                 # `training_m5_1.src.*` import paths
├── configs/
│   ├── m5_smoke.yaml               # M5 pipeline-validation smoke (20 traj/step × 50 steps)
│   ├── m5_1_research_paper.yaml    # M5.1 paper-faithful + M5.2 gains (LIVE; 622 steps, 320 traj/step)
│   └── _archive_m2/                # M2 2B configs
├── scripts/
│   ├── run.sh                      # --mode smoke|prod --seed N
│   ├── smoke.sh                    # alias for `run.sh --mode smoke`
│   ├── run_grpo.py                 # imports training_m5_1.src.registry, hands off to NeMo-RL examples/run_grpo.py
│   ├── prep_musique.py             # MuSiQue train -> data/training/musique/train.parquet
│   ├── sync_m4_prompts.py          # materialises M4 prompt mode into src/prompts/
│   ├── bootstrap.sh                # M2-shape Vast bootstrap; rewrite for M5.1 when needed
│   └── _archive_m2/                # M2 wrappers preserved
├── tests/                          # 23 pure-Python tests (reward + parser + format-helper)
└── fix/CHANGES.md                  # M2 changelog; M5.1 changes summarized in docs/report/CODE_SETUP_m5.md
```

`docs/training/CONVERSATION_CONTEXT.md` is the per-strand bootstrap doc for the training side.

## Subsequent experiments (M5.2 onwards)

Once M5.1 starts full-training on the box, copy this directory:

```bash
cp -a training_m5_1/ training_m5_2/
# edit configs/m5_2_<variant>.yaml + the one or two src/ files that change.
```

The W&B project key in the new copy's yaml (`logger.wandb.project: reason_over_search_m5_<N>`) auto-isolates the new run from M5.1's metrics. See [docs/milestone_5/MILESTONE_5.md §"Parallel experiments"](../docs/milestone_5/MILESTONE_5.md) for the candidate variants.

## Pointers

- M5 milestone narrative: [docs/milestone_5/MILESTONE_5.md](../docs/milestone_5/MILESTONE_5.md)
- M5 paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md)
- M5 code-setup audit: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md)
- M5 smoke results log: [docs/report/RESULTS_SMOKE_m5.md](../docs/report/RESULTS_SMOKE_m5.md)
- M2 NeMo-RL operational guide (still the foundation): [training/README.md](../training/README.md), [docs/training/README.md](../docs/training/README.md)
- M4 eval pipeline (the rollout shape we're aligned to): [docs/milestone_4/MILESTONE_4.md](../docs/milestone_4/MILESTONE_4.md), [docs/report/CODE_SETUP_m4.md](../docs/report/CODE_SETUP_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470); official code: [Agent-RL/ReSearch](https://github.com/Agent-RL/ReSearch)
