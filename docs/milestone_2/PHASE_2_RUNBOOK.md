---
title: PHASE 2 RUNBOOK
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-06
---

# Milestone 2 Phase 2 — Runbook

Concrete sequence to run Search-R1 GRPO training on Vast.ai. Phase 1 (build the training pipeline) is documented in [MILESTONE_2.md](MILESTONE_2.md) §"Step-by-step"; this runbook covers Phase 2 (actually run training + smoke-eval).

> **TL;DR** — boot the docker image on Vast.ai, run [`bash training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) (idempotent: pulls LFS, downloads Qwen weights + v2 venv tarball, starts retriever), then `bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 -- policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true policy.train_micro_batch_size=2`. Eval on Bamboogle through the Milestone-1 pipeline.

> **Plan pivot (2026-05-04 onwards).** The original "3 seeds × {base, hybrid} = 6 runs" plan is **superseded by the recipe-ablation plan in [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)**. With smoke-anchored wall-clock at 11–17 d / run on 1× A100 80GB and a $1000 USD budget, that supports ~2–3 runs (the JustRL plain-GRPO control + the E2H+S-GRPO+MC-GRPO stack), not 6. The "6-run plan" subsection below is kept for reference but is no longer the live target.

---

## Pre-flight checklist

Run through this before booting the Vast instance. Prevents wasted GPU-hours.

### Local-side

- [ ] Latest code on `main` branch (or whatever branch you intend to run): `git status` clean, `git push` done.
- [ ] Datasets on Git LFS: `git lfs ls-files` shows `data/training/nq_hotpotqa_train/{train,test}.parquet`.
- [ ] W&B account ready; API key copied to clipboard.
- [ ] Docker Hub credentials ready (only needed if pushing a new image).

### Verify the overlay still passes its tests

```bash
uv run --no-project --with pytest pytest training/tests/ -v
# Expected: 19 passed, 2 skipped (env/dataset modules need the training venv).
```

### Verify the docker image is current

The latest image on Docker Hub should match the committed `training/nemo_rl/` ref. If you bumped NeMo-RL recently, rebuild:

```bash
# Build v2 (recommended; thin layer on :v1 with transformers 5.7.0 for Qwen3.5)
docker build --platform linux/amd64 \
  -f docker/reason-over-search-v1/Dockerfile.v2 \
  -t pantomiman/reason-over-search-v1:v2 .
docker push pantomiman/reason-over-search-v1:v2

# Build v1 from scratch (broken right now: NeMo-RL upstream made automodel+vllm
# extras mutually exclusive; needs UV_EXTRAS=vllm only or a NEMO_RL_REF SHA pin):
# docker build \
#   --build-arg NEMO_RL_REF=v0.6.0 \
#   --build-arg UV_EXTRAS=vllm \
#   -f docker/reason-over-search-v1/Dockerfile -t pantomiman/reason-over-search-v1:v1 .
# docker push pantomiman/reason-over-search-v1:v1
```

---

## Vast.ai setup (per fresh instance)

### 1. Boot

- **Image**: `pantomiman/reason-over-search-v1:v1` (or your fork).
- **GPU**: 1× or 2× A100 80GB. Match the launch script you intend to use.
- **Persistent storage**: ≥ **150 GB** for indexes (~25 GB) + model weights (~15 GB) + checkpoints (~10–30 GB per run).
- **Disk**: container disk ≥ 60 GB (for the venv at `training/nemo_rl/.venv/` + uv cache + working set).
- **CPU/RAM**: ≥ 16 cores / **≥ 150 GB RAM**. The retriever uses 8 workers × the 16 GB IVF-SQ8 index ≈ **128 GB resident** (each worker calls `flashrag.utils.get_retriever()` independently → separate FAISS handles, separate index loads). Bootstrap.sh warns at < 150 GB. Do not use the flat IP index for training — it needs ~65 GB per worker × 8 = unworkable, and times out under rollout HTTP load even at 1 worker.

### 2. Clone repo + materialize the training venv

```bash
cd /workspace
git clone https://github.com/<your-user>/reason_over_search.git
cd reason_over_search

# Pull LFS objects (training parquets + eval jsonls)
git lfs install
git lfs pull

# Materialize the NeMo-RL venv. The docker image carries a pre-warmed uv
# wheel cache (~5 GB at /root/.cache/uv/), so this completes in ~30s–2min
# rather than re-downloading. If you build the image without the wheel
# pre-warm step, expect ~10–20 min on Vast's network.
cd training/nemo_rl
uv sync --extra vllm
cd ../..
```

Smoke-check the venv:
```bash
training/nemo_rl/.venv/bin/python -c "import nemo_rl; print(nemo_rl.__version__)"
# → 0.6.0
```

### 3. Set up the retriever

Follow [`local_retriever/README.md`](../../local_retriever/README.md). Summary:

```bash
# Use the docker image's pre-built /venv/retriever (faiss-cpu).
# Download corpus + index + encoder model (~25 GB total):
cd local_retriever
mkdir -p corpus indexes models

huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset --include "wiki-18.jsonl.gz" \
  --local-dir corpus --local-dir-use-symlinks False
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

# IVF-SQ8 index (~16 GB; required for training — flat IP times out under rollout load).
curl -L "https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index" \
  -o indexes/wiki18_100w_e5_ivf4096_sq8.index

huggingface-cli download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2 --local-dir-use-symlinks False
cd ..
```

**Move heavy artifacts to persistent storage** if `/workspace` isn't already mounted persistent — checkpoint persistence assumes the run survives instance restarts.

### 4. Validate retriever

Start it in a `tmux` / `screen` session so it survives:

```bash
tmux new -s retriever
/venv/retriever/bin/python local_retriever/retriever_serving.py \
    --config local_retriever/retriever_config.yaml \
    --num_retriever 8 \
    --port 3005
# Ctrl-b d to detach
```

Health + sanity-search:
```bash
curl -sS http://127.0.0.1:3005/health
# → "healthy"

curl -sS -X POST http://127.0.0.1:3005/batch_search \
    -H "Content-Type: application/json" \
    -d '{"query":["who founded SpaceX"],"top_n":3,"return_score":false}' \
  | head -c 400
# Expect a list[list[{id, contents}]] with 3 docs.
```

If retriever startup fails: check that `indexes/wiki18_100w_e5_ivf4096_sq8.index` exists (~16 GB on disk), confirm `retriever_config.yaml` points to it, and verify ≥ 16 GB free RAM. Do not use the flat IP index for training — it needs ~65 GB RAM and times out under rollout load.

### 5. Download model weights

```bash
mkdir -p training/models
huggingface-cli download Qwen/Qwen3.5-2B-Base \
  --local-dir training/models/Qwen3.5-2B-Base \
  --local-dir-use-symlinks False

# Hybrid (only if you'll run the hybrid variant):
huggingface-cli download Qwen/Qwen3.5-2B \
  --local-dir training/models/Qwen3.5-2B \
  --local-dir-use-symlinks False
```

The launch scripts pass model names as HF repo IDs (`Qwen/Qwen3.5-2B-Base`), not local paths — vLLM resolves via the HF cache. If you want to use a local copy, override `policy.model_name=/workspace/.../Qwen3.5-2B-Base` on the CLI.

### 6. Configure W&B

```bash
cp training/.env.example training/.env
$EDITOR training/.env
# Fill in WANDB_API_KEY (and optionally WANDB_ENTITY, CHECKPOINT_DIR_BASE).
```

The launch scripts source this file. `training/.env` is gitignored.

For checkpoint persistence, set:
```bash
CHECKPOINT_DIR_BASE=/workspace/persistent/checkpoints
```

---

## Smoke run (recommended before the real thing)

Run a tiny truncated training to verify all the pipes connect — retriever, dataset, env, GRPO loop, W&B logging.

```bash
# 5-step training run, single seed, 1× A100.
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 999 \
    grpo.max_num_steps=5 \
    logger.wandb.name=smoke-test
```

Expected outcome:
- W&B picks up the run; you see `train/reward_mean`, `train/kl`, `train/policy_loss`, GPU utilization.
- The retriever logs receive `/batch_search` calls during rollouts.
- No NaN losses or KL spikes.
- The 5 steps complete and the process exits cleanly.

> **Validation + checkpointing are disabled in the current configs** — see [`VALIDATION.md`](../training/VALIDATION.md) and the `[DISABLED for first-pass training]` comments in the YAMLs. The smoke run won't write any checkpoint and won't run any val rollouts; that's expected. Re-enable in a follow-up once training mechanics are verified.

If anything blows up here, fix it before committing real GPU-hours.

---

## Real runs — recipe ablation plan (current target)

The active plan: **C-minimal control + the optimised stack** (~2 runs, fits the $1000 budget). See [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) for full ordering. Each run is one launch of [`run_grpo_1xa100.sh`](../../training/scripts/run_grpo_1xa100.sh) with the always-required Qwen3.5 overrides:

```bash
# JustRL plain-GRPO control (no tricks; recipe baseline)
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 \
  -- policy.sequence_packing.enabled=false \
     policy.dynamic_batching.enabled=true \
     policy.train_micro_batch_size=2

# Optimised stack: E2H curriculum + S-GRPO + MC-GRPO (overrides depend on the
# stack-block code; placeholders here — fill in once each block lands)
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 \
  -- policy.sequence_packing.enabled=false \
     policy.dynamic_batching.enabled=true \
     policy.train_micro_batch_size=2 \
     <stack-specific Hydra overrides>
```

Each run:
- 1005 steps × 510 trajectories/step. Smoke-anchored wall-clock: **11–17 d on 1× A100 80GB SXM** (~$300–490 / run at $1.20/h Vast median); see [`PAPER_VS_OURS_TRAINING.md §7`](../training/PAPER_VS_OURS_TRAINING.md#7-compute) and [`SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config) for the derivation.
- **No validation, no checkpoints written** in the current first-pass configs (see [`VALIDATION.md`](../training/VALIDATION.md) and the YAML's `[DISABLED for first-pass training]` blocks). Re-enable per VALIDATION.md §7 before kicking off long ablations — final weights live only in W&B run artifacts until then.

> **Step-100 gate**: launch ONE run first (`--variant base --seed 42`, no stack overrides → C-minimal). At step 100 (~2.5 h on 1× A100 if going well; on 1× H100 ~1 h), check the W&B `gpu_monitoring` panel + step rate + `train/reward_mean` curve. If wall-clock projects ≤17 d and `train/reward_mean` is climbing (not flat at 0), continue. If flat at ~0 (model not finding any rewardable trajectories), abort and debug — cheap is better than complete.

### Hardware choice (smoke-anchored, see [`SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config))

| Hardware | Wall-clock / run | $ / run | Notes |
|---|---|---|---|
| **1× A100 80 GB SXM** *(current default)* | 11–17 d | $300–490 | viable; slow |
| **1× H100 80 GB SXM** *(recommended)* | 5–8.5 d | $240–410 | best $/run; same single-GPU config |
| **2× A100 80 GB SXM** | 6.5–9.5 d | $370–550 | use [`run_grpo_2xa100.sh`](../../training/scripts/run_grpo_2xa100.sh); decolocates vLLM/DTensor |
| 1× H200 141 GB SXM | 4–7 d | $270–470 | fastest; small marginal $$ over H100 |

**Training params are byte-identical between 1× and 2× A100 configs** — only `cluster.gpus_per_node`, `vllm.tensor_parallel_size`, and `gpu_memory_utilization` differ ([diff](../../training/configs/)). Switching mid-experiment doesn't change the GRPO trajectory.

### Original 6-run plan (superseded)

Kept for reference. Replaced by the recipe-ablation plan above (≤3 runs total) once the smoke-anchored wall-clock made 6 paired runs unaffordable.

```bash
# Three seeds × {base, hybrid} × qwen_native arm — NOT current plan
for v in base hybrid; do
  for s in 42 7 1337; do
    bash training/scripts/run_grpo_1xa100.sh --variant $v --seed $s \
      -- policy.sequence_packing.enabled=false \
         policy.dynamic_batching.enabled=true \
         policy.train_micro_batch_size=2
  done
done
```

### Sequential vs concurrent

On a single Vast instance, runs are **sequential** (one process at a time per GPU). To run multiple seeds concurrently, spin up separate Vast instances or split GPUs. Concurrent runs share retriever traffic — the retriever handles batched requests but watch CPU saturation.

---

## Monitoring during training

### W&B

Validation is off for first-pass training, so the only signal is from training-side metrics:

- `train/reward_mean` — should rise from ~0 to ~0.3 over the first 100 steps for `paper` arm; from ~0 to ~0.5 for `qwen_native` (different reward scale, see [§7d of CHAT_TEMPLATE.md](../training/CHAT_TEMPLATE.md#7d-final-state--model-emits-answerxanswer)).
- `train/kl` — spikes followed by recovery are normal early. Sustained KL > 0.5 is a warning.
- `train/policy_loss`, `train/clip_fraction` — standard PPO/GRPO sanity.
- `train/grad_norm`, `train/lr` — should track the warmup schedule (linear ramp over 286 steps from 0.1× to 1× of `1e-6`, then constant).
- Per-rollout-batch metrics from our env's `global_post_process_and_metrics`: `train/accuracy`, `train/em_hit_rate`, `train/fraction_of_samples_properly_ended`, `train/generation_lengths`. These are computed on training rollouts (since validation is off) and are the closest signal to "is the model learning the format" without paying for a val pass.
- GPU utilization via `gpu_monitoring`.

### Retriever

```bash
tmux attach -t retriever
# Watch /batch_search call rate during rollouts.
```

CPU saturation on the retriever node is the most likely throughput bottleneck. If it pegs 100 %, the GPU sits idle waiting.

### Failure modes + recovery

Configs ship with conservative defaults (`train_micro_batch_size=2` — required for Qwen3.5-2B with `sequence_packing: false`; `gpu_memory_utilization=0.6` on 1× A100 / `0.7` on 2× A100, `activation_checkpointing=true`) to make the first-run launch survivable. If something still goes wrong:

| Symptom | Likely cause | Fix (override on the CLI) |
|---|---|---|
| OOM in policy forward / backward | activations too tight even at micro=2 | `policy.train_micro_batch_size=1`. If still OOM: `policy.dtensor_cfg.cpu_offload=true` |
| OOM at vLLM init or first rollout | `gpu_memory_utilization` reserves more than the policy + adam + grad + ref leaves free | Drop by 0.05: `policy.generation.vllm_cfg.gpu_memory_utilization=0.55` (1× A100) or `0.65` (2× A100) |
| Retriever HTTP timeout / connection errors | retriever overloaded (CPU pegged) or down | Check `tmux attach -t retriever`. Bump `env.search_r1.request_timeout_s=60`. If retriever is healthy but slow, drop concurrent rollout pressure: `grpo.num_prompts_per_step=64` (lower trajectories/step temporarily). |
| `val/accuracy` stays at 0 after step 100 | format-validity ≈ 0 (model not emitting `<answer>`) | Inspect a few val rollouts in W&B (`logger.num_val_samples_to_print: 5`). Likely needs longer warmup or a slightly higher LR (`policy.optimizer.kwargs.lr=2.0e-6`). |
| KL explodes (>5) | LR too high or KL coef too low | Verify `loss_fn.reference_policy_kl_penalty=0.001`. If correct, halve LR. |
| Hybrid variant emits `<think>` blocks but no `<answer>` | enable_thinking=true broke the chat-template termination | Verify the bash wrapper added `++policy.tokenizer.chat_template_kwargs.enable_thinking=true`. If the model loops in `<think>...</think>` without converging, raise `policy.generation.max_new_tokens` from 500 to 750 or 1000 for the hybrid runs. |

---

## After Phase 2 (first-pass goals)

The first-pass training run is **mechanics verification only** — checkpointing and validation are off. After the first 1005-step run finishes cleanly:

1. **Confirm `train/reward_mean` climbed** above 0 over the run, and the loss/KL/grad-norm curves are sane. That's the first-pass success criterion.
2. **Update §7 of [PAPER_VS_OURS_TRAINING.md](../training/PAPER_VS_OURS_TRAINING.md#7-compute) "Ours — observed"** with measured wall-clock, GPU-hours, $/run from the W&B run summary. Replaces the linear/sub-linear smoke extrapolation with a real anchor.
3. **Re-enable validation + checkpointing** — flip the `[DISABLED for first-pass training]` blocks in both YAMLs back to `enabled: true` / `val_period: 100` / `val_at_start: true` / restore `data.validation`. See [`VALIDATION.md §7`](../training/VALIDATION.md#7-re-enabling-validation-planned-not-active) for the planned re-enable.
4. **Run the recipe-ablation plan** ([`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)): JustRL plain-GRPO control + the optimised stack (E2H curriculum + S-GRPO + MC-GRPO). 2 to 3 runs total — the original 6-run plan is superseded by smoke-anchored wall-clock making it unaffordable. Smoke-eval each finished checkpoint on Bamboogle through the M1 eval pipeline ([`MILESTONE_1.1_QWEN_BASELINES.md`](../milestone_1/MILESTONE_1.1_QWEN_BASELINES.md) gives you the untrained baseline to beat).
5. **Aggregate** across runs, side-by-side vs. paper Table 3 + the JustRL control, write up.

---

## Quick reference

| Need | Path |
|---|---|
| Launch script (1× A100) | [`training/scripts/run_grpo_1xa100.sh`](../../training/scripts/run_grpo_1xa100.sh) |
| Launch script (2× A100) | [`training/scripts/run_grpo_2xa100.sh`](../../training/scripts/run_grpo_2xa100.sh) |
| Hyperparameter audit | [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) |
| W&B metric set | [`docs/training/VALIDATION.md`](../training/VALIDATION.md#4-metrics-logged-to-wb) (validation off in first-pass — train-side metrics only) |
| Tuning knobs | [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) |
| Overlay architecture | [`docs/training/README.md`](../training/README.md#trainingsrc-overlay-architecture) |
| Retriever setup | [`local_retriever/README.md`](../../local_retriever/README.md) |
| M1 eval pipeline | [`docs/report/CODE_SETUP_m1.md`](../report/CODE_SETUP_m1.md) |
