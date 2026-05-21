---
title: BOOTSTRAP NEW INSTANCE
tags: [setup, runbook, docker]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# Bootstrap a new instance — non-Vast hosts

Bring up the full `reason_over_search` runtime on **any** host that can run the docker image (in-house GPU box, lab cluster, cloud VM that isn't Vast.ai). For Vast.ai / Verda B300 / RunPod-class hosts specifically, use [`docs/setup/SETUP_INSTANCE.md`](SETUP_INSTANCE.md) directly; it's tuned to those template + persistent-volume layouts. For ALICE HPC use [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh) (Apptainer-based).

This doc is the **thin wrapper**: it covers the host-specific steps that SETUP_INSTANCE.md doesn't (pulling the image, running the container, the `/venv/...` symlink caveat). After the container is up and you've cloned the repo into it, you run [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) the same way SETUP_INSTANCE.md does, and it provisions everything (retriever assets, training venvs, M4 eval models).

## Step 0 — host requirements

Same accelerator + RAM + disk envelope as the provider templates; pick a row in [SETUP_INSTANCE.md § 0 disk-budget table](SETUP_INSTANCE.md#0-pre-flight-do-this-before-booting-the-instance) for the disk size.

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1× 24 GB | 1× 80 GB (A100 / H100 / H200) | Training needs ≥40 GB; eval is fine on 24 GB |
| Host RAM | 32 GB | 150 GB (8 retriever workers under training rollout load) | 8 IVF-SQ8 workers each load ~16 GB index |
| Disk | 100 GB (M2/M4 on 0.8B) | 130 GB (M2/M4 on 2B) | Image ~30 + retriever assets ~30 + venvs ~13 + models ~3.5–10 + headroom |
| Docker | 24+ | latest | needs `--gpus` flag (NVIDIA Container Toolkit installed on host) |

**Constraint**: GPU FAISS + SGLang **cannot share a 24 GB 4090** (16 GB index + 22 GB SGLang > 24 GB VRAM). On 4090 keep the retriever on CPU FAISS (the bootstrap default). On 80 GB cards GPU FAISS + SGLang fit, but CPU FAISS is fine and one less moving part.

## Step 1 — get the image

```bash
docker pull pantomiman/reason-over-search-v1:v2   # recommended: v1 + transformers 5.7.0 (Qwen3.5 qwen3_5 arch)
# docker pull pantomiman/reason-over-search-v1:v1 # original
```

Or build from source (rebuild instructions in [`docker/reason-over-search-v1/README.md`](../../docker/reason-over-search-v1/README.md)):

```bash
git clone <repo-url> reason_over_search && cd reason_over_search
docker build --platform linux/amd64 \
  -f docker/reason-over-search-v1/Dockerfile.v2 \
  -t pantomiman/reason-over-search-v1:v2 .
```

The image bundles two conda envs (`retriever`, `evaluation_search_r1`) at `/opt/miniforge3/envs/`, the NeMo-RL pre-warmed uv wheel cache at `/.uv/cache`, and the boot-hook for SSH perms (Vast-only; no-op on other hosts).

## Step 2 — run the container

```bash
mkdir -p ~/ros               # persistent volume; pick anywhere with enough disk
docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -p 3000:3000 -p 3005:3005 \
  -v "$HOME/ros":/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v2
```

Notes:
- `--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.
- `--shm-size=16g` matters for SGLang and Ray (default 64 MB will OOM under load).
- Port mappings are only needed if you want the retriever / SGLang reachable from outside the container.
- The bind mount makes `/workspace` survive `docker rm`; everything below assumes the repo lives at `/workspace/reason_over_search`.

## Step 3 — clone, bootstrap, done

Inside the container:

```bash
cd /workspace
git clone <repo-url> reason_over_search
cd reason_over_search
git checkout <branch>          # research_v2 / main / etc.

bash training/scripts/bootstrap.sh
```

This is the **same script SETUP_INSTANCE.md uses**. It provisions retriever assets (corpus + IVF-SQ8 index + e5-base-v2), training venvs (`uv sync` + the v2/automodel worker venv), Qwen3.5-2B training weights, M4 eval models (Qwen3.5-0.8B hybrid + base), and starts the retriever on `127.0.0.1:3005`. See [SETUP_INSTANCE.md § 4](SETUP_INSTANCE.md#4-bootstrap-25-min-cold-hf-path-50-min-compile-fallback-1-min-warm) for the per-step breakdown and skip flags (`SKIP_V2_BUILD=1`, `SKIP_RETRIEVER=1`, `SKIP_M4_MODELS=1`).

After bootstrap finishes the box is ready. Launch your actual experiments per the milestone runbooks; this doc has no run commands by design.

## Non-Vast caveats

- **`/venv/...` symlinks don't exist on non-Vast hosts.** The Vast template surfaces `/venv/retriever` and `/venv/evaluation_search_r1` for convenience; on a vanilla `docker run` they're absent. The underlying conda envs at `/opt/miniforge3/envs/{retriever,evaluation_search_r1}/` are what bootstrap.sh and the helper scripts actually use, so nothing breaks. If a doc you're following references `/venv/...`, either symlink (`ln -s /opt/miniforge3/envs/retriever /venv/retriever`) or `source /opt/miniforge3/etc/profile.d/conda.sh && conda activate retriever`.
- **No persistent volume by default.** With `--rm`, the container is destroyed on exit and only data under `/workspace` (your bind mount) survives. Don't put outputs anywhere else.
- **W&B credentials.** `training/.env` lives inside the bind-mounted repo, so it persists across container restarts. Set `WANDB_API_KEY` there once.
- **SGLang first-launch JIT compile** is ~3–5 min on cold flashinfer caches; subsequent launches are ~30 s. The cache lives under `$HOME/.cache/flashinfer/` inside the container, which means it's lost on container destroy unless you bind-mount it too.
- **Ray cluster state** is at `/tmp/ray` inside the container. If a previous training run died unclean, `rm -rf /tmp/ray` before relaunching.

## What this doc does NOT cover

- **The actual setup steps** (asset downloads, venvs, retriever start). Those live in `bootstrap.sh` and are documented in [`docs/setup/SETUP_INSTANCE.md`](SETUP_INSTANCE.md).
- **Eval pipelines** (M1 GRPO checkpoint reproduction, M3 Qwen3-0.6B, M4 Qwen3.5-0.8B). Those live in their respective milestone docs.
- **Training runs.** See [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md).
- **Smoke tests.** Deliberately omitted — once `bootstrap.sh` exits 0, the box is ready; testing happens at the experiment layer.

## See also

- [`docs/setup/SETUP_INSTANCE.md`](SETUP_INSTANCE.md) — full setup steps (the "actual" runbook this doc delegates to)
- [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) — the bootstrap script
- [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh) — ALICE HPC equivalent (Apptainer-based)
- [`local_retriever/README.md`](../../local_retriever/README.md) — retriever asset download steps + index choices (bootstrap.sh mirrors these)
- [`docs/setup/HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md) — accelerator comparison
- [`docker/reason-over-search-v1/README.md`](../../docker/reason-over-search-v1/README.md) — image rebuild instructions
- [`docs/archive/BOOTSTRAP_NEW_INSTANCE_v0.md`](../archive/BOOTSTRAP_NEW_INSTANCE_v0.md) — original M1-era manual-download walkthrough (archived 2026-05-09)
