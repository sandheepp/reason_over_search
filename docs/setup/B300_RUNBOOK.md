---
title: B300 / Verda bring-up runbook (M5.5 GRPO)
tags: [setup, b300, verda, runbook, lessons]
source: internal
created: 2026-05-15
updated: 2026-05-15
---

# B300_VERDA_RUNBOOK.md — how to bring up M5.5 training on a fresh Verda B300

> Written from the actual 2026-05-15 bring-up session on Verda B300 (host
> `cosmic-matrix-fin-03`, IP `95.133.252.173`, 2× B300 SXM6 / 275 GB each,
> Ubuntu 24.04, 60 vCPU / 550 GB RAM / ~160 GB free disk, $5.94/h). Every
> footgun in this doc actually bit us. Skip to **§Quickstart** for the
> next-time-fast path; keep reading the rest if you want to understand why
> each step is in there.
>
> Pair with: [SETUP_INSTANCE.md §10](SETUP_INSTANCE.md#10-variant-verda-b300-fresh-ubuntu-no-docker-image) (the high-level Verda variant) and [training_m5_5/scripts/start_b300.sh](../../training_m5_5/scripts/start_b300.sh) (the one-shot launcher this runbook prepares the box for).

## What you actually have on a fresh Verda B300

| Field | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.8 |
| GPUs | 2× NVIDIA B300 SXM6 AC, **sm_120**, 275 GB HBM3e each |
| Driver / CUDA host | 580.126.09 / CUDA 13.0 runtime |
| **Default `/usr/local/cuda` symlink** | **CUDA 13.0** (this is a problem — see §1) |
| Python | system `/usr/bin/python3` (no `uv`, no conda) |
| Build tools | `git`, basic `apt`; **no** ninja, no cmake (in the version we need), no nvcc 12.9 |
| cuDNN | not installed |
| Persistent mount | none separate — everything under `/` (193 GB total) |

## The eight things that bit us

This is the *reason* the runbook has the steps it does. Each item below was a build failure we hit in real time.

### 1. CUDA toolkit mismatch (13.0 default vs torch built for 12.9)

PyTorch wheels in the NeMo-RL lock are **`torch==2.10.0+cu129`** — built against CUDA 12.9. Verda's default `nvcc` is **CUDA 13.0**. When `uv sync` builds `deep-ep` or `transformer-engine`, torch's `_check_cuda_version()` aborts because `nvcc` version (13.0) ≠ torch's `cu129`.

**Fix**: install CUDA toolkit 12.9 *alongside* 13.0, then swap the default symlink to 12.9. The 580 driver supports both.

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9

# Swap default
rm -f /etc/alternatives/cuda /usr/local/cuda
ln -s /usr/local/cuda-12.9 /etc/alternatives/cuda
ln -s /usr/local/cuda-12.9 /usr/local/cuda
/usr/local/cuda/bin/nvcc --version | tail -1   # should say 12.9
```

### 2. InfiniBand dev headers missing (deep-ep needs `mlx5dv.h`)

NeMo-RL's `vllm` extra pulls in `deep-ep` (DeepSeek expert parallel). Its CUDA sources `#include <infiniband/mlx5dv.h>`.

**Fix**:
```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libibverbs-dev libmlx5-1 libnuma-dev rdma-core
```

### 3. `uv` and `ninja` not on system PATH (Ray actor subprocesses can't find them)

NeMo-RL spawns env-builder Ray actors that call `uv` and `ninja` as subprocesses. Ray actors inherit a **stripped PATH** — `~/.local/bin` (where `uv` from the Astral installer lands) is **not** on it. Same with venv-local `ninja`.

**Fix**:
```bash
# After uv installer puts it in ~/.local/bin:
ln -sf /root/.local/bin/uv  /usr/local/bin/uv
ln -sf /root/.local/bin/uvx /usr/local/bin/uvx
apt-get install -y -qq ninja-build tmux cmake build-essential
```

### 4. CMake too old for sm_120 (CMake 3.28 doesn't know B300)

`apt-get install cmake` on Ubuntu 24.04 ships **CMake 3.28**. The `CUDA_ARCHITECTURES=120` literal was added in **CMake 3.31+**. Without it, every `transformer-engine` CMake call fails with "CUDA_ARCHITECTURES is empty for target transformer_engine".

**Fix** (install latest CMake via `uv tool`):
```bash
uv tool install --force cmake             # gets cmake 4.3+
mv /usr/bin/cmake /usr/bin/cmake.apt      # back up the old one
ln -sf /root/.local/bin/cmake /usr/bin/cmake
cmake --version | head -1                 # should say 4.x
```

### 5. `nv-grouped-gemm` can't `torch.cuda.init()` inside a GPU-less Ray actor

NeMo-RL's `_env_builder()` Ray actor has no GPU assigned. `nv-grouped-gemm`'s `setup.py` calls `torch.cuda.init()` at build time. Result: `RuntimeError: No CUDA GPUs are available`.

**Fix**: **build the V2 worker venv from the host shell, not from inside Ray**. The host has GPUs visible.

```bash
cd /root/reason_over_search
V2_VENV=training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
UV_PROJECT_ENVIRONMENT="$V2_VENV" \
  uv sync --locked --extra automodel --directory training_m5_5/nemo_rl
```

**Faster alternative — fast-path via pre-built tarball**: `bootstrap_b300.sh` tries multiple per-arch tarballs in order and falls back to source compile only if all fail. Currently published:

| GPU family | SM | Repo | File | Notes |
|---|---|---|---|---|
| Hopper (H100/H200) | sm_70/80/89/90 | `pantomiman/reason-over-search-v1-venvs` | `dtensor_policy_worker_v2.tar.gz` | Hopper-built; SASS for sm_70/80/89/90, PTX for sm_90 → will NOT run on Blackwell |
| **Blackwell-Ultra (B300)** | **sm_103** | **`sandheepp/reason-over-search-venvs` (public)** | **`dtensor_policy_worker_v2_sm103.tar.gz`** | **Built 2026-05-16 on Verda B300 (commit `907af71`); torch 2.10+cu129, TE 2.14+71bbefbf** |

The bootstrap auto-discovers your private tarballs by querying `hf auth whoami` against `HF_TOKEN` — no env-var plumbing needed. Lookup order:
1. `<your_hf_user>/reason-over-search-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz`
2. `<your_hf_user>/reason-over-search-venvs:dtensor_policy_worker_v2.tar.gz` (unversioned legacy)
3. `pantomiman/reason-over-search-v1-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz`
4. `pantomiman/reason-over-search-v1-venvs:dtensor_policy_worker_v2.tar.gz`
5. Source compile (last resort)

Total fast-path savings: ~20-30 min vs source compile per fresh B300 box.

**To bake your own tarball** after a successful bootstrap (one-time per SM):
```bash
bash training_m5_5/scripts/package_v2_venv.sh
# uploads to <your_hf_user>/reason-over-search-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz
```

Each tarball only works on its **exact SM** (sm_103 binaries can't run on sm_90 etc. — different SASS, and PTX forward-compat is only within a major arch family). Build once per SM you ever provision.

Original framing in this runbook ("Verda has no such tarball, so we compile") was wrong — I copied it from `SETUP_INSTANCE.md` without checking. Pantomiman's tarball is publicly downloadable from any host but it's Hopper-only, hence the per-SM tarballs above.

### 6. transformer-engine compiles for **7 GPU architectures by default**

This is the single biggest time-waster. With CUDA 12.9, `transformer-engine/build_tools/utils.py:cuda_archs()` defaults to `"70;80;89;90;100;120"`. Each `.cu` file gets compiled 7 times. On the M5.5 v2 build we measured ~6 obj/min before fixing this; **~36 obj/min** after.

**Fix**: set `NVTE_CUDA_ARCHS` env. But a quirk in TE's CMakeLists.txt means **you cannot pass `"120"` alone** — TE removes "100/101/110/120" from `CMAKE_CUDA_ARCHITECTURES` into `NVTE_SPECIFIC_ARCHS`, and if the original list is just "120" the post-removal list is empty → CMake error "CUDA_ARCHITECTURES is empty for target transformer_engine".

**Working value**: `NVTE_CUDA_ARCHS="90;120"` (Hopper + Blackwell Ultra). One non-stripped arch keeps CMake happy; the binary still runs on B300 because we still emit sm_120 code.

```bash
export NVTE_CUDA_ARCHS="90;120"
export TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX"   # affects deep-ep, causal-conv1d, mamba
```

### 7. transformer-engine pytorch ext needs **system cudnn headers**

TE's CMake build (the `common` library) finds cuDNN via the `nvidia-cudnn-cu12` wheel in the v2 venv. **But TE's pytorch-binding extension compile** uses plain `c++ -I/usr/local/cuda-12.9/include` — that doesn't include the venv's cuDNN, and Verda has no system cuDNN by default.

**Fix**:
```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cudnn9-cuda-12 libcudnn9-dev-cuda-12
# headers land at /usr/include/x86_64-linux-gnu/cudnn.h ; the system compiler search path picks it up automatically
```

If apt cuDNN install fails (older versions or repo limitations), the alternative is to symlink the venv's cuDNN into a system include dir:
```bash
V2_SITE=/root/reason_over_search/training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/lib/python3.13/site-packages
ln -s "$V2_SITE/nvidia/cudnn/include/"*.h /usr/local/include/
ln -s "$V2_SITE/nvidia/cudnn/lib/"*.so* /usr/local/lib/
ldconfig
```

### 8. Slow anonymous HF downloads (Qwen3.5-0.8B weights)

vLLM's GenerationWorker downloads `Qwen/Qwen3.5-0.8B` (~1.7 GB) on first launch. Anonymous HF traffic is rate-limited and can take 10+ min — even though the file *itself* is tiny.

**Fix**: pre-cache it before launching anything. With `hf_transfer` it lands in ~5 s:
```bash
source training_m5_5/nemo_rl/.venv/bin/activate
pip install -U "huggingface_hub[hf_xet,cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.5-0.8B   # caches to ~/.cache/huggingface/hub
```

Optionally set `HF_TOKEN` to bypass anon rate limits altogether.

---

## Cheap validation on a 4090 (or any GPU) before paying for B300

The bootstrap auto-detects local GPU compute capability and picks
`NVTE_CUDA_ARCHS` accordingly, so you can validate the install pipeline
on a $0.30/hr Vast 4090 before spinning up the $5.94/hr B300.

```bash
# On a fresh Vast.ai 4090 instance (bare Ubuntu, no docker image):
ssh root@<4090_host>
cd /root && git clone https://github.com/<your-fork>/reason_over_search.git
cd reason_over_search
git checkout m5.5_b300

# Write your HF + WANDB tokens before running so no interactive prompts fire
cat > training_m5_5/.env <<EOF
WANDB_API_KEY=<your_key>
HF_TOKEN=<your_hf_token>
EOF
chmod 600 training_m5_5/.env

# One command: bootstrap (if needed) → retriever → smoke → prod (only if smoke passes)
# On a 4090/A100 validation box, just stop at --mode smoke:
bash training_m5_5/scripts/start_b300.sh --mode smoke

# On B300 (production), chain smoke → prod in one tmux session:
bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_b300
```

**What this proves**:
- ✅ apt prereqs install cleanly (ninja, cmake, tmux, IB headers, cudnn)
- ✅ CUDA 12.9 toolkit + symlink swap works
- ✅ `uv` symlinks correctly so Ray actors find it
- ✅ V2 venv builds (compile path; tarball fast-path would skip this)
- ✅ Retriever assets download
- ✅ Qwen3.5-0.8B HF download succeeds (with token, no rate limit)
- ✅ Full GRPO loop runs end-to-end through 4 smoke steps

**What this does NOT prove** (only B300 can):
- B300-specific memory headroom at micro=4/seq=8192 (will OOM on 24 GB)
- sm_120 SASS / PTX correctness in V2 venv kernels
- TP=2 across two B300s
- Wall-clock projections for full 622-step prod run

**Cost math**: 4090 on Vast spot ~$0.30-0.60/hr × ~30-40 min ≈ **~$0.20-0.40**.
Compare to first B300 bring-up: ~2.5 h × $5.94/hr ≈ ~$15. Even if you only
validate once, the 4090 catches every system-level failure mode for ~3% of
the cost.

The launcher (`start_b300.sh`) refuses to run `--mode prod_b300*` on a GPU
with <80 GB VRAM (would OOM immediately), so there's no risk of accidentally
running prod on the small box — only smoke.

## Quickstart — next time, run these in this order

Assumes a fresh Verda B300 box, repo synced to `/root/reason_over_search/`. **Total runtime: ~30-45 min cold** (most of it is the V2 venv compile).

```bash
# 0. apt prereqs (one-shot)
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ninja-build tmux cmake build-essential \
    libibverbs-dev libmlx5-1 libnuma-dev rdma-core

# 1. install CUDA 12.9 toolkit alongside 13.0, swap default
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9
rm -f /etc/alternatives/cuda /usr/local/cuda
ln -s /usr/local/cuda-12.9 /etc/alternatives/cuda
ln -s /usr/local/cuda-12.9 /usr/local/cuda

# 2. cuDNN dev headers
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cudnn9-cuda-12 libcudnn9-dev-cuda-12

# 3. install uv, symlink it system-wide for Ray actors
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
ln -sf $HOME/.local/bin/uv  /usr/local/bin/uv
ln -sf $HOME/.local/bin/uvx /usr/local/bin/uvx

# 4. install newer cmake (Ubuntu's is too old for sm_120)
uv tool install --force cmake
mv /usr/bin/cmake /usr/bin/cmake.apt
ln -sf $HOME/.local/bin/cmake /usr/bin/cmake

# 5. main NeMo-RL venv (parent venv, ~5 min from clean uv cache)
cd /root/reason_over_search
bash training_m5_5/setup.sh

# 6. V2 worker venv (run from host, NOT inside Ray) — slow tail kernels still
#    dominate; budget ~15-25 min
V2_VENV=training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
rm -rf "$V2_VENV"
env \
  PATH=/root/.local/bin:/usr/local/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  CUDA_HOME=/usr/local/cuda-12.9 \
  LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  NVTE_CUDA_ARCHS="90;120" \
  TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX" \
  MAX_JOBS=32 \
  CMAKE_BUILD_PARALLEL_LEVEL=32 \
  UV_PROJECT_ENVIRONMENT="$V2_VENV" \
  uv sync --locked --extra automodel --directory training_m5_5/nemo_rl

# 7. retriever assets — see SETUP_INSTANCE.md §10c
mkdir -p local_retriever/{corpus,indexes,models}
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir /tmp/wiki18 \
    --include "wiki-18.jsonl.gz"
gunzip -c /tmp/wiki18/wiki-18.jsonl.gz > local_retriever/corpus/wiki18_100w.jsonl
curl -L --fail -o local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index \
    https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index
hf download intfloat/e5-base-v2 --local-dir local_retriever/models/e5-base-v2

# 8. retriever venv (cpu faiss)
uv venv local_retriever/.venv_cpu --python 3.10
uv pip install --python local_retriever/.venv_cpu/bin/python -r local_retriever/requirements.txt

# 9. training data + prompt
source training_m5_5/nemo_rl/.venv/bin/activate
python training_m5_5/scripts/prep_musique.py
python training_m5_5/scripts/sync_m4_prompts.py --mode qwen35_minimal

# 10. pre-cache Qwen3.5-0.8B weights
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.5-0.8B

# 11. .env with WANDB key
cat > training_m5_5/.env <<EOF
WANDB_API_KEY=<your_key>
EOF

# 12. one-shot launch (pre-flights everything + brings up retriever + tmux-launches training)
bash training_m5_5/scripts/start_b300.sh --mode smoke   # validate first
bash training_m5_5/scripts/start_b300.sh                # prod_b300, seed 42
```

## Time budget by step (observed on the 2026-05-15 run)

| Step | Wall-clock |
|---|---|
| 0–4 apt prereqs + CUDA 12.9 toolkit + cuDNN + cmake/uv symlinks | ~5-8 min |
| 5 main NeMo-RL venv (`setup.sh`) | ~3-5 min |
| 6 V2 worker venv compile | **~15-25 min** (dominated by TE common kernels) |
| 7 retriever assets (corpus 14 GB + index 15 GB + encoder) | ~5-10 min in parallel |
| 8 retriever venv | ~30 s |
| 9 training data + prompt | ~10 s |
| 10 pre-cache Qwen3.5-0.8B | ~5-10 s with hf_transfer |
| 11 .env | instant |
| 12 smoke test (4 steps, smoke config) | ~5-10 min |
| **Total cold-to-smoke-pass** | **~35-60 min** |

After this, `bash training_m5_5/scripts/start_b300.sh` is your single trigger for production training. The launcher pre-flights everything; if a check fails it prints exactly what to do.

## V2 venv build speed levers (in order of impact)

| Lever | Speedup | Notes |
|---|---|---|
| `NVTE_CUDA_ARCHS="90;120"` (vs default 7-arch) | **~3.5×** | The single biggest knob. Must include at least one non-stripped arch (i.e. NOT just "120"). |
| `MAX_JOBS=32` / `CMAKE_BUILD_PARALLEL_LEVEL=32` | ~2× until single-file tail | 60 vCPUs → up to 32 parallel nvcc procs. Tail-limited. |
| `TORCH_CUDA_ARCH_LIST="12.0+PTX"` | ~1.5× | Affects `deep-ep`, `causal-conv1d`, `mamba-ssm` (not TE). |
| Pre-built per-arch tarball (default fast-path) | ~10× | Bootstrap tries `<your_hf>/reason-over-search-venvs:…_sm${CC}.tar.gz` first, then pantomiman's Hopper tarball, then source compile. For B300 specifically: `sandheepp/reason-over-search-venvs:dtensor_policy_worker_v2_sm103.tar.gz`. See §5 above. |

The remaining slow tail is dominated by a handful of large TE kernels (`cast_transpose_fusion`, `ln_fwd_cuda_kernel`, etc) that take 1-3 min each per arch and can't parallelize across files because `--threads 1`.

## What's in this run's repo state

- New B300 config: [training_m5_5/configs/m5_5_research_paper_b300.yaml](../../training_m5_5/configs/m5_5_research_paper_b300.yaml) (micro=4, gpu_mem=0.85, act-ckpt off)
- New launcher: [training_m5_5/scripts/start_b300.sh](../../training_m5_5/scripts/start_b300.sh) (preflight → retriever → tmux train)
- `--mode prod_b300` wired into [training_m5_5/scripts/run.sh](../../training_m5_5/scripts/run.sh)
- README has the "Start training" trigger documented
- Companion doc: [SETUP_INSTANCE.md §10](SETUP_INSTANCE.md#10-variant-verda-b300-fresh-ubuntu-no-docker-image)

## Common gotchas during runtime (not setup)

- **Smoke needs ~25 GB of GPU RAM** at seq=4096; prod_b300 needs ~100 GB at micro=4/seq=8192. B300's 275 GB is well above either.
- **2nd GPU is unused** by the default prod_b300 config (`cluster.gpus_per_node: 1`). For 2× B300, write a TP=2 variant mirroring `m5_5_research_paper_2xa100.yaml`.
- **Anonymous HF Hub during a long run** can hit rate limits if vLLM re-downloads metadata. Set `HF_TOKEN` in `.env` to avoid.
- **Disk pressure during V2 build**: peak ~150 GB of 193 GB used. If you're squeezed, delete `/usr/local/cuda-13.0` (~5 GB freed) after the toolkit swap and prune `/root/.cache/uv` between attempts.

## Check the rollouts every 10 steps (REQUIRED for unattended runs)

The 2026-05-16 B300 run silently trained for 107 steps with a dead retriever: every `<tool_response>` was `Errno 111 Connection refused`, and the model learned a zero-tool policy. The launcher's `/health` probe didn't catch this because FastAPI stayed up while the FAISS workers were OOM-killed. We only discovered the failure in a post-mortem.

The fix in [`watch_resources.sh`](../../training_m5_5/scripts/watch_resources.sh) auto-runs [`check_trace.py`](../../training_m5_5/scripts/check_trace.py) on every 10th rollout step landed. The digest catches:

- **Tool-call collapse** — `<50%` of rollouts call search means the model is abandoning retrieval
- **Retriever broken** — `>10%` of `<tool_response>` tags are connection errors
- **Floor dominance** — `>90%` of rollouts at exact `reward=0.1` (signal-compression by the 3-tier shape)
- **Degenerate generation** — mean gen tokens below 100 (model outputting padding)
- **Generic-answer guessing** — `>20%` of answers are short country names (reward-hacking)

Output lands in `/root/logs/m5_5_traces_seed*.log`; red flags also raise a `WARN` line in the resource log so they're visible without scrolling.

Manual invocation any time:

```bash
python training_m5_5/scripts/check_trace.py            # latest step
python training_m5_5/scripts/check_trace.py --step 50  # specific step
python training_m5_5/scripts/check_trace.py --no-samples  # digest only
```

Exit `0` if healthy, `1` if any flag tripped, `2` if rollouts unreadable. Full doc + thresholds: [training_m5_5/README.md §6.1](../../training_m5_5/README.md).

**Rule of thumb**: scan the trace log at least every 100 steps on a multi-day run, or every 10 steps if you've changed config / hardware. The 5-minute cost of a quick scan saves the multi-day cost of a poisoned policy.

## Build artifacts that should NOT be re-created on a re-launch

These are cached on disk and reused unless explicitly cleaned. Don't `rm -rf` them just to "start fresh" — you'll pay the cost again.

- `~/.cache/uv/` (Python wheel + git checkout cache, ~22 GB)
- `~/.cache/huggingface/hub/` (Qwen + datasets, ~3 GB)
- `local_retriever/{corpus,indexes,models}/` (~30 GB)
- `training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/` (~10 GB, the V2 venv itself)
- `training_m5_5/nemo_rl/.venv/` (~13 GB, the parent venv via symlink)

## Persistent volume (Spheron / virtiofs) — survive instance termination

Long M5.5 runs (~300 steps × 5-7 min = 1-2 days) on a spot box will fill the
root disk and lose everything if the instance dies. If the provider attaches a
persistent volume (Spheron offers virtiofs shares that survive instance
termination), mount it **before launching training** and point the checkpoint
dir at it. A dead instance then just means re-bootstrapping a fresh box and
pointing it at the same volume — every `step_N` is intact.

Do this **once, before `start_b300.sh`**:

```bash
# 1. Mount the virtiofs share (name comes from the provider; "milestone55" here)
sudo mkdir -p /mnt/milestone55
sudo mount -t virtiofs milestone55 /mnt/milestone55
# persist across reboots
echo "milestone55 /mnt/milestone55 virtiofs nofail 0 0" | sudo tee -a /etc/fstab
df -h /mnt/milestone55                                          # confirm capacity

# 2. Tell run.sh to write ckpts to the volume (run.sh reads CHECKPOINT_DIR_BASE
#    and lands ckpts under ${CHECKPOINT_DIR_BASE}/m5_<MODE>/seed<N>/).
#    Persist it in training_m5_5/.env so every launch picks it up.
echo "CHECKPOINT_DIR_BASE=/mnt/milestone55/ckpts" \
    | sudo tee -a /root/reason_over_search/training_m5_5/.env

# 3. (optional) Pre-stage retriever assets on the volume so future bootstraps
#    on a fresh box skip the ~10 min download. bootstrap_b300.sh checks
#    /mnt/milestone55/retriever_assets/ before going to HF.
sudo mkdir -p /mnt/milestone55/retriever_assets
```

That's it. Now launch:

```bash
bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_h200
```

Checkpoints land directly on the volume from step 10 onward. No symlink dance,
no rsync loop, no race window. If the instance is killed, mount the same
volume on the replacement and re-launch with
`checkpointing.resume_from_checkpoint=latest` — NeMo-RL picks up at the last
`step_<N>` on the volume.

### Triple-backup invariant

With `CHECKPOINT_DIR_BASE=/mnt/milestone55/ckpts` set, each new checkpoint
lands in three independent places:

1. **Volume** (`/mnt/milestone55/ckpts/m5_<mode>/seed<N>/step_<N>/`) — survives instance death
2. **HF Hub** (`<hf_user>/reason-over-search-m5_5_<mode>-seed<N>-step<step>`) — `upload_ckpts_watcher.sh` polls and uploads
3. **Local disk** — *only* if `CHECKPOINT_DIR_BASE` is unset; not used in the volume setup

### If you mounted the volume *after* launching training

You can't change `CHECKPOINT_DIR_BASE` mid-run (NeMo-RL reads the path at
launch). But you can rsync existing local ckpts to the volume, then atomically
swap the local `seed<N>` directory for a symlink to the volume — NeMo-RL keeps
writing to the same path; the kernel resolves it through the symlink to
virtiofs. Do this between ckpt saves (every 10 steps, ~30 min window) so no
`tmp_step_N/` exists mid-rename.

```bash
MODE=prod_h200; SEED=42
SRC=/root/reason_over_search/results/grpo/m5_5_${MODE}/seed${SEED}
DST=/mnt/milestone55/ckpts/m5_5_${MODE}/seed${SEED}

sudo rsync -a "$SRC/" "$DST/"                    # mirror current ckpts
sudo mv "$SRC" "${SRC}.local_backup"             # atomic rename
sudo ln -sfn "$DST" "$SRC"                       # symlink in its place
sudo cat "$SRC/.uploaded_steps"                  # verify HF uploader still resolves
sudo rm -rf "${SRC}.local_backup"                # frees ~32 GB (5 × 6.4 GB)
```

Prefer the fresh-start `CHECKPOINT_DIR_BASE` path above — the symlink swap is
only here for "oops I forgot to mount the volume first."
