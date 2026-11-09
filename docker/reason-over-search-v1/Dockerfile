# reason-over-search v1.
#
# Hybrid paradigm:
#   - retriever  + evaluation_search_r1: code AND env baked into image (stable,
#     no need to clone the repo on Vast for these to work).
#   - training (NeMo-RL): only the env (uv wheel cache + uv installed) is
#     baked. The repo is cloned on Vast and the venv is materialized at
#     training/nemo_rl/.venv via `uv sync`. Reason: training is the active
#     iteration surface; rebuilding the image on every code edit is wasteful.
#
# Pin base to linux/amd64: Vast hosts and "expected amd64" builds are x86_64.
# Without this, a `docker build` on Apple Silicon produces an arm64 image,
# which will not run as a base (or on Vast) for amd64-only workflows.

ARG VAST_BASE_TAG=cuda-12.9.1-auto
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} vastai/base-image:${VAST_BASE_TAG}

LABEL org.opencontainers.image.title="reason-over-search-v1"
LABEL org.opencontainers.image.version="1"
LABEL org.opencontainers.image.description="Conda envs for retriever + evaluation_search_r1 (with code baked in); uv + pre-warmed NeMo-RL wheel cache for training (code cloned at runtime)."

ENV APP_ROOT=/app
WORKDIR ${APP_ROOT}

# === 1. Conda env `retriever` (Python 3.10) — code + env baked in ===
ENV PIP_NO_CACHE_DIR=1
COPY local_retriever/requirements.txt /tmp/retriever-requirements.txt
RUN /opt/miniforge3/bin/conda create -n retriever python=3.10 -y && \
    /opt/miniforge3/bin/conda run -n retriever \
      python -m pip install --no-cache-dir -r /tmp/retriever-requirements.txt && \
    rm -f /tmp/retriever-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# === 2. Conda env `evaluation_search_r1` (Python 3.11) — code + env baked in ===
# eval requirements pin ipython==9.10.1, which requires Python >=3.11.
COPY evaluation_search_r1/requirements.txt /tmp/evaluation-requirements.txt
RUN /opt/miniforge3/bin/conda create -n evaluation_search_r1 python=3.11 -y && \
    /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      python -m pip install --no-cache-dir -r /tmp/evaluation-requirements.txt && \
    rm -f /tmp/evaluation-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# Source trees for retriever + eval (data dirs excluded via .dockerignore;
# mount or download at runtime). Editable install of the eval package so
# `import flashrag` works without re-running setup at container start.
COPY local_retriever ${APP_ROOT}/local_retriever
COPY evaluation_search_r1 ${APP_ROOT}/evaluation_search_r1
RUN /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      /bin/bash -lc "cd /app/evaluation_search_r1 && python setup.py develop --no-deps"

# === 3. Training env: uv + pre-warmed NeMo-RL wheel cache (NO repo code) ===
# The training/nemo_rl/ source is committed to the repo and cloned at runtime
# on Vast. The image needs (a) uv installed, and (b) NeMo-RL's wheels pre-
# downloaded into uv's cache so the on-Vast `uv sync` is fast (~30s instead
# of ~10–20 min).
#
# Build VM requirement: ~15 GB free during torch's unpack. Docker Desktop on
# Apple Silicon allocates ~58 GB by default which is too small with the conda
# envs taking ~33 GB; bump to ≥120 GB (Settings → Resources → Disk image size)
# before building. Confirm with: docker run --rm alpine df -h /
ARG NEMO_RL_REF=v0.6.0
# v0 shipped UV_EXTRAS=vllm only. v1 also pre-warms `automodel` so the
# DTensorPolicyWorkerV2 (default in our config) doesn't have to download
# wheels on Vast first-boot. The `automodel` extra still has to *compile*
# transformer-engine + nv-grouped-gemm + deep_ep on first launch (these are
# not pure wheels — they pull from git+ sources and build CUDA kernels);
# but the wheel deps for them ARE cached, which saves ~10 min on Vast.
# The full compile (~25 min) happens once via training/scripts/bootstrap.sh,
# which gracefully reuses any pre-built venv on subsequent invocations.
ARG UV_EXTRAS=vllm,automodel

# vastai/base-image already ships a `uv` at /usr/local/bin/uv; we still install
# our own copy via the Astral installer to pin a known version, then `ln -sf`
# to make sure the symlinks point at /root/.local/bin/* (not the base image's).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

# Install build deps for the git+ packages NeMo-RL pulls in.
# - libibverbs-dev: required for `deep-ep` (DeepSeek's MoE expert-parallelism
#   kernel) which vLLM transitively pulls in. Without this, the compile fails
#   with "fatal error: infiniband/mlx5dv.h: No such file or directory".
#   This is documented in NeMo-RL's README under bare-metal install notes.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-warm uv's wheel cache at /.uv/cache/ (the path UV_CACHE_DIR points to in
# vastai/base-image). Subsequent `uv sync` against the user's cloned
# training/nemo_rl/ on Vast hits this cache and skips the 5+ GB download.
# The cache is content-addressed by hash, so it works across project paths.
#
# IMPORTANT: vastai/base-image sets UV_NO_CACHE=1 which would silently disable
# caching. We override it here so the wheels actually persist into /.uv/cache/.
# (Vast users running `uv sync` later: the same env carries UV_NO_CACHE=1 by
# default — that's fine for them because uv falls back to UV_FIND_LINKS-style
# resolution against the populated cache directory listed below as a find-link.)
#
# Approach: shallow-clone NeMo-RL, run `uv sync` to populate the cache (this
# also creates a temp venv we delete afterwards), then drop the clone.
# --no-install-project skips the editable nemo_rl install — that happens on
# Vast against the user's clone.
RUN UV_NO_CACHE=0 \
    git clone --depth=1 --branch ${NEMO_RL_REF} --recursive \
        https://github.com/NVIDIA-NeMo/RL.git /tmp/nemo-rl-deps && \
    cd /tmp/nemo-rl-deps && \
    EXTRA_FLAGS="" && \
    for e in $(echo ${UV_EXTRAS} | tr ',' ' '); do EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${e}"; done && \
    UV_NO_CACHE=0 uv venv .venv && \
    UV_NO_CACHE=0 uv sync ${EXTRA_FLAGS} --no-install-project && \
    rm -rf /tmp/nemo-rl-deps && \
    echo "uv cache size after pre-warm:" && du -sh /.uv/cache 2>/dev/null

# Tell `uv sync` on Vast to use the cache from the image even when UV_NO_CACHE=1
# is still set in the environment (the base image's default). Setting both
# UV_NO_CACHE=0 here AND keeping UV_CACHE_DIR pointed at /.uv/cache so that the
# pre-warmed wheels are found.
ENV UV_NO_CACHE=0
ENV UV_CACHE_DIR=/.uv/cache

# === 4. Vast boot hook — fix SSH key permissions so OpenSSH accepts auth ===
COPY docker/reason-over-search-v1/10-fix-ssh-perms.sh /etc/vast_boot.d/10-fix-ssh-perms.sh
RUN chmod +x /etc/vast_boot.d/10-fix-ssh-perms.sh

# === Runtime defaults ===
# v1 retriever defaults: 8 workers + IVF-SQ8 index (path is set in
# local_retriever/retriever_config.yaml as the default). The flat IP index
# at 2 workers timed out on >80% of /batch_search calls under the training
# rollout load; IVF-SQ8 + 8 workers brought 3-query batch latency from
# 30 s timeout → ~1.7 s. Recall hit < 1 % (acceptable for online RL).
# See local_retriever/README_v1.md for the full rationale.
ENV RETRIEVER_PORT=3005
ENV NUM_RETRIEVER=8
# HF cache lives on the persistent /workspace volume (large), not the
# overlay root (small). bootstrap.sh overrides if the user prefers a
# different location.
ENV HF_HOME=/workspace/hf_cache
EXPOSE 3005

RUN env-hash > /.env_hash
