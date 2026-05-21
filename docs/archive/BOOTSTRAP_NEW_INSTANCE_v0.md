---
title: BOOTSTRAP NEW INSTANCE (v0, archived)
tags: [archive, setup, m1]
source: internal
created: 2026-05-01
updated: 2026-05-02
archived: 2026-05-09
---

> **ARCHIVED 2026-05-09.** This is the original M1-era manual-download bootstrap walkthrough.
> Successors:
> - [`docs/vast/SETUP_VAST.md`](../vast/SETUP_VAST.md) — bootstrap-script-driven Vast.ai setup (M2 + M4).
> - [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../setup/BOOTSTRAP_NEW_INSTANCE.md) — thin host-agnostic recipe (`docker pull / docker run / docker build`) that delegates the actual setup steps to SETUP_VAST.md.
>
> What changed:
> - Manual `huggingface-cli download` steps replaced by one-shot [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) (idempotent).
> - Default retriever index switched from flat IP (~65 GB, paper-fidelity) to IVF-SQ8 (~16 GB); flat IP retired.
> - Scope expanded from M1 eval-only to also cover M2 training + M4 Qwen3.5-0.8B eval-models setup.
> - Search-R1 GRPO checkpoint downloads (Qwen2.5-3B, M1) dropped; replaced by Qwen3.5-0.8B (M4).
>
> Kept here for historical reference; not maintained.

# Bootstrap a new instance (v0, archived)

Recreate the full `reason_over_search` runtime on a fresh box (Vast.ai, in-house GPU, or anywhere else) from scratch.

The fastest path is **the prebuilt docker image + HF re-fetch**. Everything below assumes you start with no local state.

## Step 0 — host requirements

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1× 24 GB (RTX 4090) | 1× 80 GB (H100 / H200) | SGLang holds the 3B Qwen2.5 in bf16 (~22 GB on 4090). H100 PCIe gives 3.35× decode bandwidth → 24 h Plan A. |
| Host RAM | 32 GB (with IVF-SQ8 only) | 80–128 GB (or 503 GB for flat FAISS) | Flat FAISS = ~65 GB resident. IVF-SQ8 = ~16 GB. |
| Disk | 60 GB (IVF-SQ8 only) | 150 GB (full setup) | 14 GB corpus + 16 or 60 GB index + 26 GB checkpoints + 2 GB encoder + headroom. |
| GPU/host coexistence | — | — | GPU FAISS + SGLang **cannot share** a single 4090 (16 GB index + 22 GB SGLang > 24 GB VRAM). On 24 GB cards, run FAISS on CPU. On 80 GB H100, GPU FAISS + SGLang fit comfortably. |

See [HARDWARE_4090.md](../setup/HARDWARE_4090.md) for the full accelerator comparison (current comparison: [HARDWARE_COMPARISON.md](../setup/HARDWARE_COMPARISON.md)).

## Step 1 — get the image

The published image at [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1) bundles two conda envs (`retriever` + `evaluation_search_r1`) with all pip dependencies pre-installed, and the app code under `/app`. Source: [`docker/reason-over-search-v1/`](../../docker/reason-over-search-v1/).

### On Vast.ai

Create a custom template:

- **Image**: `pantomiman/reason-over-search-v1:v1`
- **Disk**: ≥150 GB
- **GPU filter**: 1× of your chosen class (4090 / H100 PCIe / H200)
- **On-start**: nothing (the image has no auto-start services; you'll launch retriever + SGLang manually)
- **Open ports**: 3000 (SGLang), 3005 (retriever) if you need them externally

Boot the instance, SSH in.

### On any other host

```bash
docker pull pantomiman/reason-over-search-v1:v1
docker run --rm -it --gpus all \
  -p 3000:3000 -p 3005:3005 \
  -v "$HOME/ros":/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v1
```

### Or build from source

```bash
git clone <repo-url> reason_over_search && cd reason_over_search
docker build -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

See [`docker/reason-over-search-v1/README.md`](../../docker/reason-over-search-v1/README.md) for buildx + cross-arch notes.

## Step 2 — clone the repo

The image has the app code at `/app`, but the Vast template typically clones into `/workspace/reason_over_search` so you can pull updates and use git normally:

```bash
cd /workspace
git clone <repo-url> reason_over_search
cd reason_over_search
git checkout <branch>   # e.g. main or experiment_ros/<tag>
```

## Step 3 — verify the venvs

The image exposes two pre-built environments. The Vast.ai template surfaces them at `/venv/retriever` and `/venv/evaluation_search_r1`; the underlying conda envs are at `/opt/miniforge3/envs/retriever` and `/opt/miniforge3/envs/evaluation_search_r1`. Either path works.

```bash
/venv/retriever/bin/python -c "import faiss, flask; print('retriever ok')"
/venv/evaluation_search_r1/bin/python -c "import flashrag, sglang; print('eval ok')"
```

If the `/venv/...` symlinks are missing on a non-Vast host, fall back to `conda activate retriever` / `conda activate evaluation_search_r1` after sourcing `/opt/miniforge3/etc/profile.d/conda.sh`.

## Step 4 — download data + indexes + checkpoints

All artifacts are sha256-identical to upstream HF; total ~110 GB.

```bash
huggingface-cli login   # only needed if you push your own dataset; downloads are public

cd /workspace/reason_over_search/local_retriever
mkdir -p corpus indexes models

# Corpus (~14 GB)
huggingface-cli download PeterJinGo/wiki-18-corpus --repo-type dataset \
  --include "wiki-18.jsonl.gz" --local-dir corpus
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

# IVF-SQ8 FAISS index (~16 GB) — the retriever default
curl -L -o indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index

# Flat FAISS index (~60 GB after merge + gunzip) — optional, for paper-quality eval (exact recall)
huggingface-cli download PeterJinGo/wiki-18-e5-index --repo-type dataset --local-dir indexes
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz
rm -f indexes/part_aa indexes/part_ab

# Encoder (~2 GB)
huggingface-cli download intfloat/e5-base-v2 --local-dir models/e5-base-v2

# GRPO checkpoints (2 × 13 GB)
cd ../../evaluation_search_r1
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model
```

### Verify checkpoint identity (sha256)

```bash
cd /workspace/reason_over_search/evaluation_search_r1
sha256sum search_r1_base_model/model-00001-of-00003.safetensors
# Expected: 7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf
sha256sum search_r1_instruct_model/model-00001-of-00003.safetensors
# Expected: 3d787062256210d1cc6c7c666a0ab0ac83a7a5d0296281b4811df72c968ccd35
```

The full identity table (sizes + eos_token_id) is in [../eval/REPRODUCIBILITY.md#models—confirmed-grpo](../eval/REPRODUCIBILITY.md).

## Step 5 (optional) — set up the GPU FAISS venv

Only useful on hosts where SGLang doesn't need the same GPU (e.g. multi-GPU boxes; or single H100 80 GB where 16 GB index + 22 GB SGLang fit).

```bash
cd /workspace/reason_over_search
bash local_retriever/setup_gpu_venv.sh
# Creates local_retriever/.venv with faiss-gpu-cu12 + torch+cu130 (~6 GB)
```

## Step 6 — start the retriever

CPU + IVF-SQ8 (the default — fast, ~16 GB RAM):

```bash
cd /workspace/reason_over_search/local_retriever
nohup /venv/retriever/bin/python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 4 --port 3005 \
  > /tmp/retriever.log 2>&1 &

# Wait + sanity check
until curl -sf http://127.0.0.1:3005/health; do sleep 5; done
echo
```

CPU + flat IP (exact, slower, ~65 GB RAM — only when you specifically need exact recall):

```bash
nohup /venv/retriever/bin/python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 4 --port 3005 \
  --index ./indexes/wiki18_100w_e5_flat_inner.index \
  > /tmp/retriever.log 2>&1 &
```

GPU + IVF-SQ8 (only when SGLang doesn't share the GPU):

```bash
nohup ./.venv/bin/python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 1 --port 3005 \
  --gpu --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
  > /tmp/retriever.log 2>&1 &
```

Search smoke test:

```bash
curl -sS -X POST http://127.0.0.1:3005/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Who wrote The Lord of the Rings?", "top_n": 3, "return_score": false}'
```

## Step 7 — start SGLang

```bash
cd /workspace/reason_over_search
scripts/manage_sglang.sh switch instruct   # or `base`
# Logs: /tmp/sglang_<variant>.log

# Sanity:
curl -sS http://127.0.0.1:3000/get_model_info | python -c \
  'import sys, json; print(json.load(sys.stdin)["model_path"])'
```

The script kills any existing SGLang, launches the requested variant on `127.0.0.1:3000` with the canonical flags (`--context-length 8192 --dtype bfloat16 --tp 1 --trust-remote-code`), and waits for `/get_model_info` to come up (≤10 min).

## Step 8 — single-run smoke test

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > /tmp/smoke.log 2>&1
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

Expected on Bamboogle/instruct (n=125, greedy): EM ≈ 0.36, F1 ≈ 0.45 — see [../eval/REPRODUCIBILITY.md#smoke-validation](../eval/REPRODUCIBILITY.md). A run takes ~6 min on a 4090, ~2 min on H100 PCIe.

If EM drops to 0.05–0.10 territory the eval is broken — check the `apply_chat`/template/parser surface listed in [../eval/PAPER_VS_OURS_AUDIT.md](../eval/PAPER_VS_OURS_AUDIT.md) before touching anything else.

## Step 9 (optional) — restore prior results

The Plan B v0 + v1 result archives (`evaluation_search_r1/results/_archive_v0/`, `_archive_v1/`) are tracked in git and present after a normal clone — no separate restore step needed. If you bring a tarball of additional results from another box, untar into `evaluation_search_r1/results/`:

```bash
cd /workspace/reason_over_search
tar -xf /path/to/results_bundle.tar -C evaluation_search_r1/results/
```

`run_one.sh` is resume-aware — any `(variant, dataset, seed)` cell with a `metric_score.txt` will be skipped on the next sweep.

## Common pitfalls

- **`/venv/...` paths missing** — the image's conda envs are at `/opt/miniforge3/envs/...`. The `/venv/retriever` and `/venv/evaluation_search_r1` paths exist on the Vast template but not on a vanilla `docker run`. Either symlink them or use `conda activate` directly.
- **GPU FAISS + SGLang fight on a 4090** — 16 GB index + 22 GB model > 24 GB VRAM. On a 4090, keep FAISS on CPU.
- **`huggingface-cli upload` requires login** — the public downloads above don't, but if you ever push artifacts back, run `huggingface-cli login` first.
- **Datasets ship in the repo** at `data/<dataset>/<split>.jsonl` (full eval splits) and `data_subsample/<dataset>/<split>.jsonl` (deterministic 1 k subsamples used by the v1 sweep). Total ~140 MB, tracked in git (not LFS). A normal clone gets them. Source: [`RUC-NLPIR/FlashRAG_datasets`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets); to re-pull, see the dataset re-pull snippet in [`VAST_INSTANCE_SETUP_v0.md` § 4d](VAST_INSTANCE_SETUP_v0.md).
- **SGLang cold start is slow** — first launch JIT-compiles CUDA kernels (~3–5 min). Subsequent launches are ~30 s.
- **Resume hazard for re-runs** — `run_one.sh` is resume-aware: if a `metric_score.txt` exists for `(variant, dataset, seed)` it skips silently. To force a re-run, `rm -rf` the matching `evaluation_search_r1/results/<dataset>/<dataset>_*_search_r1_<variant>_seed<N>` dir first.

## Total bootstrap time

| Phase | Wall-clock |
|---|---|
| Vast.ai instance boot | ~5 min |
| HF downloads (Step 4) | ~25–60 min @ 1 Gbps |
| (Optional) IVF-SQ8 build | ~60 min |
| Retriever + SGLang first start | ~10 min |
| Smoke test (Step 9) | ~6 min |
| **Cold-to-eval-running** | **~45–90 min** |
