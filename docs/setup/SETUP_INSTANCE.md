---
title: SETUP INSTANCE
tags: [setup, vast, verda, runpod, h200, b200, b300, runbook]
source: internal
created: 2026-05-09
updated: 2026-05-15
---

# SETUP_INSTANCE.md — total setup guide for a fresh GPU instance

> **Audience**: a human operator OR a Claude agent given SSH access to a freshly
> booted GPU instance. Covers multiple providers / hardware shapes we actively
> use:
>
> - **§1–§9 Vast.ai path** — instance booted with `pantomiman/reason-over-search-v1:v2`
>   (use v2; v2 = v1 + transformers 5.7.0 baked in, which Qwen3.5 needs). The
>   image bundles conda envs + the pre-warmed `uv` wheel cache.
> - **§10 Verda B300 path** — bare Ubuntu 24.04, no docker image, 1× or 2× B300
>   SXM6 with 275 GB HBM3e each. No pre-warmed cache; you install `uv` and
>   materialize venvs yourself.
> - **RunPod / other docker-capable hosts (H200, B200, …)** — see the
>   "Porting this runbook to a non-Vast host" section below; the same image
>   works, only the persistent-volume mount path may differ.
>
> Follow the right section end-to-end and the box ends up with retrieval, eval
> (M4 Qwen3.5-0.8B), and training (M5.1+ GRPO) all set up and ready to run.
> After bootstrap finishes you don't run anything in this doc; you launch your
> actual experiments per the milestone runbooks.
>
> Sister doc for the ALICE HPC cluster: [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh)
> (Apptainer-based, same image, ZFS-aware paths).

What this guide provisions, in order:
1. Conda envs for **retrieval** + **eval** + **training** (baked into the image).
2. **Retriever assets**: wiki-18 corpus (~14 GB), IVF-SQ8 index (~16 GB), e5-base-v2 encoder (~0.5 GB) → `local_retriever/`.
3. **Training weights**: Qwen3.5-2B + Qwen3.5-2B-Base → `$HF_HOME`; NeMo-RL `.venv` + v2/automodel worker venv.
4. **M4 eval models**: Qwen3.5-0.8B hybrid + base → `eval/`.
5. Live retriever on port 3005 ready for both eval and training rollouts.

Total cold-to-ready: ~25 min on the HF fast path (dominated by the ~30 GB retriever-asset pull); ~50 min on the compile fallback. The flat IP retriever index (~65 GB, paper-fidelity exact recall) is **not used anymore**; we run IVF-SQ8 only.

---

## 0. Pre-flight (do this before booting the instance)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1× 24 GB (4090) | 1× 80 GB (A100 / H100 / H200) or 1× B200/B300 SXM6 | Training needs ≥40 GB; eval is fine on 24 GB. B300 (288 GB) unlocks `train_micro_batch_size=4` for M5.5; see §10. |
| Host RAM | 32 GB (IVF-SQ8 only, 1 worker) | 150 GB (IVF-SQ8, 8 workers for training) | 8 retriever workers each load ~16 GB index; flat IP needs ~65 GB |
| Disk | 60 GB (eval-only, 0.8B) | 150 GB (M1 reproduction or 2B training) | See per-scenario table below |
| Public ports | none required | 3000, 3005 | Optional for external SGLang / retriever; default workflow is local-only |

### Disk budget by scenario

Pick the row that matches your run; the headline 150 GB is conservative for the M1-reproduction path and **not required for Qwen3.5-0.8B training + eval**. Vast.ai charges per GB-hour, so sizing for the actual scenario beats sizing for "all of the above".

| Scenario | Recommended | Tight | Notes |
|---|---|---|---|
| **Qwen3.5-0.8B training + eval (M2 / M4 on 1× A100)** | **100 GB** | 80 GB | image 30 + retriever assets ~30 (corpus 14 + IVF-SQ8 16 + e5 0.5) + 2× 0.8B ~3.5 + venvs ~13 + data ~2 + W&B/logs ~5 + headroom ≈ 85 GB. Bump to 120 GB if you re-enable `checkpointing.enabled: true` or run multiple variants on the same box. |
| **Qwen3.5-2B training + eval** | 130 GB | 100 GB | 2× 2B ~10 GB instead of 0.8B ~3.5 GB; otherwise same as 0.8B row |
| **M1 reproduction (Qwen2.5-3B GRPO checkpoints)** | 150 GB | 120 GB | adds 27 GB for base + instruct GRPO checkpoints |
| **+ flat IP index (paper-fidelity exact recall)** | +60 GB | — | only if you specifically need exact recall; M4 uses IVF-SQ8 |
| **Eval-only (no training; no GRPO checkpoints)** | 60 GB | 50 GB | drop NeMo-RL `.venv` + v2 venv (~13 GB) and W&B/checkpoint headroom |

Have ready:
- A Vast.ai account with billing set up
- Your SSH public key registered with Vast (Account > Keys)
- (Optional) A `WANDB_API_KEY` for live training curves
- (Optional) A `HF_TOKEN` if you plan to push artifacts back to HF (downloads are public)

See [`docs/setup/HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md) for the full accelerator comparison (and [`docs/setup/HARDWARE_4090.md`](HARDWARE_4090.md) for the historical 4090 dev-box snapshot).

---

## 1. Launch the instance

Vast template fields:

| Field | Value |
|---|---|
| Image | `pantomiman/reason-over-search-v1:v2` (v1 also works; v2 adds transformers 5.7.0 for Qwen3.5 `model_type=qwen3_5`) |
| Disk space | **100 GB for Qwen3.5-0.8B training + eval**; 130 GB for 2B; 150 GB for M1 reproduction; 60 GB for eval-only (see § 0 disk-budget table) |
| Launch mode | Default Vast SSH/Jupyter mode |
| GPU filter | A100-80GB or H100-80GB for training; any 24 GB+ for eval |
| On-start script | leave blank; we bootstrap manually |
| Open ports | 3000 (SGLang), 3005 (retriever); only if you need external access |

The image bundles:
- Two conda envs: `retriever` (faiss-cpu, FastAPI) and `evaluation_search_r1` (sglang, FlashRAG)
- App code at `/app` (read-only baseline; we work under `/workspace`)
- A boot hook at `/etc/vast_boot.d/10-fix-ssh-perms.sh` that fixes `/root/.ssh` permissions before sshd starts (avoids "bad ownership" SSH errors)
- A pre-warmed uv wheel cache for the NeMo-RL training venv (~13 GB at `/.uv/cache`); the bootstrap script uses this

Source: [`docker/reason-over-search-v1/`](../../docker/reason-over-search-v1/). Rebuild instructions in [`docker/reason-over-search-v1/README.md`](../../docker/reason-over-search-v1/README.md).

---

## 2. Connect

```bash
# Vast UI gives the line; format is:
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai
```

If you see `bad ownership or modes for file /root/.ssh/authorized_keys`, the boot hook didn't run. Open Vast's web shell and run:

```bash
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```

---

## 3. Clone the repo + sanity (≈30 s)

The image stages app code at `/app`, but `/workspace` is the persistent volume; clone there so your work survives image rebuilds.

```bash
cd /workspace
git clone https://github.com/<your-org>/reason_over_search.git
cd reason_over_search
git checkout <branch>   # research_v2, develop, main, or whichever is current

# Sanity
pwd && which uv && conda env list 2>/dev/null | head && \
  nvidia-smi --query-gpu=name --format=csv,noheader && \
  df -h /workspace
```

Expect: cwd is the repo root; `uv` resolves; conda envs include `retriever` and `evaluation_search_r1`; one A100/H100/etc.; ≥30 GB free.

If anything is missing, **STOP**. Don't try to repair the image; it's a docker-rebuild job, not a runtime fix. Confirm `pantomiman/reason-over-search-v1:v2` (or `:v1`) was used.

---

## 4. Bootstrap (≈25 min cold HF path; ≈50 min compile fallback; <1 min warm)

One command. Idempotent: re-running prints "already present" for steps that are.

```bash
bash training/scripts/bootstrap.sh
```

Watch for the final line `▶ Bootstrap complete.`. If it errors out, read the error and stop; `bootstrap.sh` checks for the conditions it needs and fails loud.

What it does, in order (so you know what to expect):

1. **Sanity** (envs, disk, RAM, GPU). Instant.
2. **Git LFS pull** if `data/training/nq_hotpotqa_train/train.parquet` is missing (or still an LFS pointer). ~30 s.
3. **Download Qwen3.5-2B-Base + Qwen3.5-2B** training weights to `$HF_HOME=/workspace/hf_cache` if not cached (~4 min, ~8 GB).
4. **`uv sync --extra vllm`** to materialize `training/nemo_rl/.venv` (~2 min from the pre-warmed wheel cache).
5. **Download the pre-built v2/automodel uv venv** (~5 GB) from `pantomiman/reason-over-search-v1-venvs` and extract to `training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/` (~3 min fast path; ~25 min compile fallback). **This step cannot run inside a Ray actor** because `nv-grouped-gemm`'s setup.py calls `torch.cuda.init()` at install time and the actor has no GPU.
6. **Download retriever assets**: wiki-18 corpus from `PeterJinGo/wiki-18-corpus` + gunzip + rename → `local_retriever/corpus/wiki18_100w.jsonl` (14 GB after gunzip); IVF-SQ8 index from `pantomiman/reason-over-search` → `local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index` (16 GB); `intfloat/e5-base-v2` encoder → `local_retriever/models/e5-base-v2/` (~0.5 GB). ~10 min on a fast link. Mirrors [`local_retriever/README.md`](../../local_retriever/README.md) "Download steps".
7. **Start the IVF-SQ8 retriever** with 8 workers on port 3005, wait until `Uvicorn running` lands in `/tmp/retriever.log`, smoke-check the `/health` endpoint.
8. **Download M4 eval models** (Qwen3.5-0.8B hybrid + base) → `eval/qwen3.5_0.8b/` and `eval/qwen3.5_0.8b_base/` (~3.4 GB total). Invokes [`scripts/m4_download_models.sh`](../../scripts/m4_download_models.sh).

Override flags:

```bash
SKIP_V2_BUILD=1 bash training/scripts/bootstrap.sh         # skip the long compile/download (eval-only)
SKIP_RETRIEVER=1 bash training/scripts/bootstrap.sh        # don't auto-start retriever
SKIP_M4_MODELS=1 bash training/scripts/bootstrap.sh        # skip Qwen3.5-0.8B eval-model downloads
V2_BUILD_FROM_SOURCE=1 bash training/scripts/bootstrap.sh  # force compile, skip HF tarball
```

After bootstrap finishes, sanity:

```bash
# Retriever healthy?
curl -sS http://127.0.0.1:3005/health
# {"status":"healthy"}

# Training weights cached?
ls $HF_HOME/hub/ | grep Qwen3.5-2B

# v2 venv exists (unless SKIP_V2_BUILD=1)?
ls training/nemo_rl/venvs/*/DTensorPolicyWorkerV2/bin/python

# Retriever assets all present?
ls -lh local_retriever/{corpus/wiki18_100w.jsonl,indexes/wiki18_100w_e5_ivf4096_sq8.index,models/e5-base-v2/config.json}

# M4 eval models present?
ls eval/qwen3.5_0.8b{,_base}/config.json
```

---

## 5. Configure W&B (optional but recommended for training)

If you have a `WANDB_API_KEY`:

```bash
echo "WANDB_API_KEY=<your_key>" >> training/.env
echo "WANDB_PROJECT=reason_over_search_2b_v1" >> training/.env  # or your project name
```

If you don't, prepend `WANDB_MODE=disabled` to any training launch command (or set it once in `training/.env`).

---

## 6. M4 eval — what's ready, where to go next

After bootstrap finishes the box has everything M4 (Qwen3.5-0.8B baseline eval) needs:

- Retriever live on `127.0.0.1:3005` (IVF-SQ8 × 8 workers).
- Both Qwen3.5-0.8B variants on disk at `eval/qwen3.5_0.8b/` and `eval/qwen3.5_0.8b_base/`.
- Eval pipeline at [`evaluation_qwen35/`](../../evaluation_qwen35/) using the `evaluation_search_r1` conda env (already in the docker image).

What's **not** done by bootstrap (because Vast doesn't have SLURM and the launch shape is run-specific):

- SGLang server on port 3000. Launch it for the variant you're testing (mirror the launch lines in [`scripts/sbatch_m4.sh`](../../scripts/sbatch_m4.sh) §SGLang server).
- Per-`(variant, dataset, seed)` invocation. Use [`scripts/run_m4.sh`](../../scripts/run_m4.sh) for single cells; ALICE-only `scripts/sbatch_m4.sh` for the full 7-dataset sweep.

Runbook + design (action format, prompt templates, datasets, expected wall-clock): [`docs/milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md).

**Constraint reminder**: GPU FAISS + SGLang **cannot share a 24 GB 4090** (16 GB index + 22 GB SGLang > 24 GB VRAM). Bootstrap leaves the retriever on CPU FAISS, which is what we want for any GPU under 80 GB. On 80 GB cards CPU FAISS is still fine and one less moving part.

---

## 7. Training path (M2 GRPO + recipe ablations)

Pre-reqs: bootstrap finished without `SKIP_V2_BUILD=1` (you need the `DTensorPolicyWorkerV2` venv) and the retriever is up on 3005.

### 7a. Pick a combo

| Goal | Variant | Arm | Notes |
|---|---|---|---|
| Smoke-test the full pipeline (~5 min × 4 combos) | base + hybrid | qwen_native + paper | 2 outer steps × 4 prompts × group=5 |
| Recipe-ablation (`C-minimal`, `+MC-GRPO`, etc.) | base | qwen_native | 1005 steps; ~11–17 d on 1× A100, ~5–8.5 d on 1× H100 |
| One-off `full custom` | (you choose) | (you choose) | compose your own Hydra args |

The recipe-ablation plan supersedes the original 6-run plan. See [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) for the prioritised list (systems-only → C-minimal → +MC-GRPO → +S-GRPO → +E2H curriculum) and [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) for the full operational runbook.

### 7b. Required Hydra overrides

**Smoke knobs** (add to any smoke run):

```
grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
```

**Always-required Qwen3.5 overrides** (smoke OR full):

```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```

Without these, GatedDeltaNet/Mamba layers crash with `CUDA illegal memory access` during `get_logprobs`. See [`training/fix/CHANGES.md`](../../training/fix/CHANGES.md) for the diagnostic.

### 7c. Launch

Single combo (smoke):

```bash
cd /workspace/reason_over_search
export HF_HOME=/workspace/hf_cache
ulimit -n 65536
rm -rf /tmp/ray   # clear stale Ray cluster state

bash training/scripts/run_grpo_1xa100.sh \
  --variant base --seed 42 --arm qwen_native \
  -- \
  policy.sequence_packing.enabled=false \
  policy.dynamic_batching.enabled=true \
  policy.train_micro_batch_size=2 \
  grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
```

Single combo (full 1005-step run):

```bash
nohup bash training/scripts/run_grpo_1xa100.sh \
  --variant base --seed 42 --arm qwen_native \
  -- \
  policy.sequence_packing.enabled=false \
  policy.dynamic_batching.enabled=true \
  policy.train_micro_batch_size=2 \
  > /workspace/logs/m2_full.log 2>&1 &
disown
tail -f /workspace/logs/m2_full.log
```

Watch for `wandb: 🚀 View run at https://wandb.ai/...` in the launcher output; that's your live curve.

For all 4 smoke combos sequentially:

```bash
for variant in base hybrid; do
  for arm in qwen_native paper; do
    bash training/scripts/run_grpo_1xa100.sh \
      --variant $variant --seed 42 --arm $arm \
      -- \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=true \
      policy.train_micro_batch_size=2 \
      grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
    # Rename trace dir so the four don't collide
    mv logs/exp_$(ls -1t logs/ | grep -E '^exp_[0-9]+$' | head -1 | sed 's/exp_//') \
       logs/smoke_${variant}_${arm}
  done
done
```

### 7d. Health checks

- A successful step prints `========================= Step N/M =========================` followed by `Logged data to logs/exp_NNN/train_data_step{N}.jsonl`.
- For smokes, after step 2 the run exits cleanly with `EXIT=0`.
- For full runs, **checkpointing is disabled by default** in the YAML (`checkpointing.enabled: false`). Re-enable in `training/configs/grpo_qwen3.5_2b_1xa100.yaml` if you want to resume; see [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md).
- **Step-100 tripwire**: at step 100 (~1 h H100, ~2.5 h A100), check W&B `train/reward_mean`. If flat at ~0, abort and debug rather than burning the rest of the run.

### 7e. Sample extraction (smoke only)

After running all 4 smoke combos, consolidate samples and reward distributions:

```bash
python3 training/scripts/extract_smoke_samples.py
# Writes docs/training/SMOKE_RESULTS.md
# Rename to SMOKE_RESULTS_<UTC-DATE>.md before committing; previous dated runs live under docs/archive/training/
```

---

## 8. Background process discipline

Vast SSH connections drop. Always run long jobs under `nohup … & disown` (logs to `/tmp/` or `/workspace/logs/`) or under `tmux`.

```bash
mkdir -p /workspace/logs

# Option A: nohup (long training run example)
nohup bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 --arm qwen_native \
  > /workspace/logs/m2_full.log 2>&1 & disown
tail -f /workspace/logs/m2_full.log

# Option B: tmux
tmux new -s work
# inside tmux: run anything
# Ctrl-b d   to detach
# tmux attach -t work   to reattach
```

---

## 9. Tear-down checklist (before destroying the instance)

If results matter, exfiltrate first:

```bash
# M4 eval results
tar czf /tmp/results-$(date +%F).tar.gz \
  /workspace/reason_over_search/evaluation_qwen35/results
scp -P <PORT> root@ssh<HOST>.vast.ai:/tmp/results-*.tar.gz .

# Training traces
tar czf /tmp/training-traces-$(date +%F).tar.gz \
  /workspace/reason_over_search/logs

# Or sync to object storage
rclone sync /workspace/reason_over_search/evaluation_qwen35/results \
            r2:reason-over-search/results-$(date +%F)
```

W&B keeps the live curves; per-step JSONL on disk has the raw trajectories. Both are worth pulling before destroy.

Then destroy from the Vast UI; persistent-volume billing stops on destroy.

---

## Time + cost expectations (level-set the user)

### Bootstrap + smoke (any milestone)

| Phase | Wall-clock | Cost @ $1.20/h on 1× A100 |
|---|---|---|
| Bootstrap, fresh box, HF fast path | ~25 min | ~$0.50 |
| Bootstrap, fresh box, compile fallback | ~50 min | ~$1.00 |
| Bootstrap, re-used box | <1 min | ~$0.02 |
| Smoke combo (2 steps × 4 prompts) | ~5 min | ~$0.10 |

### M5.1 production training (current live experiment)

Qwen3.5-0.8B GRPO on MuSiQue, ReSearch paper recipe, 622 steps × 320 trajectories. Anchored on the live run at step 17 (`exp_010`, see [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)).

| Hardware | Wall-clock | Cost |
|---|---|---|
| **1× A100-80GB (live)** | **~4.5 d (~109 h)** | **~$130** (Vast @ $1.20/h) |
| 1× H100-80GB SXM | ~2 d (~48 h) | ~$90 (Vast @ $1.87/h) |
| 1× H200-141GB | ~1.2 d (~29 h) | ~$104 (RunPod spot @ $3.59/h) |
| 2× H100-80GB SXM (TP=2) | ~22 h | ~$82 (Vast 2× $1.87/h) |
| **1× B200-192GB** | **~14–16 h** | **~$90** (RunPod spot @ $5.98/h) |
| **1× B300-288GB (M5.5 b300 config)** | **~9–11 h** (projected; micro=4 + act-ckpt off vs B200 micro=2 + act-ckpt on) | **~$55–65** (Verda @ $5.94/h, see §10) |

Full per-config table + the choice criteria (Pareto pick: 1× H100 SXM on Vast at ~2 d/$90; fastest single-GPU: 1× B200 on RunPod) live in [`HARDWARE_COMPARISON.md` §3-§6](HARDWARE_COMPARISON.md#3-m51-wall-clock--cost-estimates-by-hardware). The M5.1-specific per-step trajectory (which dropped from 58 min/step at step 1 to 10 min/step at step 17 as the model learned shorter rollouts) is in [`../report/RESULTS_SMOKE_m5.md` §6.2](../report/RESULTS_SMOKE_m5.md#62-per-step-trajectory-live-refresh-as-steps-land).

### M2 Phase-2 reference (historical)

The earlier M2 sketch (Qwen3.5-2B, NQ+HotpotQA, 1005 steps × 510 trajectories) projected **~11–17 d** on 1× A100 / **~5–8.5 d** on H100. Those numbers are M2-shape, not M5.1, and predate the per-step time collapse observed in M5.1. Source: [`docs/training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md). For any new M5-derived experiment, scale from the [HARDWARE_COMPARISON anchor](HARDWARE_COMPARISON.md#1-live-anchor--what-were-measuring-against), not from the M2 numbers.

### Porting this runbook to a non-Vast host (e.g. RunPod)

This doc hardcodes `/workspace` as the persistent-volume mount (line 108: `cd /workspace`; bootstrap.sh's `HF_HOME=/workspace/hf_cache` and disk-check paths). RunPod's default container mount is also `/workspace`, so the doc works as-is on RunPod with two caveats:

1. **Image tag stays the same** (`pantomiman/reason-over-search-v1:v2`) but **Blackwell sm_100 (B200) is untested** — the v2 worker venv was built against Hopper-era CUDA. Run a v6-equivalent 10-step smoke ([`docs/report/RESULTS_SMOKE_m5.md` §2](../report/RESULTS_SMOKE_m5.md#2-v6--m5-smoke-pipeline-validation-smoke-shape--success)) before committing a multi-day run on a B200.
2. **On a host with a different persistent mount path** (not `/workspace`), override before bootstrap: `export HF_HOME=<your-mount>/hf_cache && export TMPDIR=<your-mount>/tmp_build` then `cd <your-mount>/reason_over_search && bash training/scripts/bootstrap.sh`. Verify the disk-free check in `bootstrap.sh:50` against the actual mount.

Generic (non-Vast) bootstrap delegates to this doc plus the wrapper in [`BOOTSTRAP_NEW_INSTANCE.md`](BOOTSTRAP_NEW_INSTANCE.md).

---

## 10. Variant: Verda B300 (fresh Ubuntu, no docker image)

The Verda B300 SXM6 instance type ships as **bare Ubuntu 24.04** with no docker
image, no conda envs, no `/workspace` mount, and no pre-warmed `uv` cache.
Everything §1–§9 assumes (the `pantomiman/reason-over-search-v1:v2` image) is
**absent**. You have to install `uv`, materialize venvs, and download retriever
assets yourself — but in exchange you get 1× or 2× B300 SXM6 (275 GB HBM3e
each), ~3.6× A100's memory, and ~50% more compute than B200.

### TL;DR — one-shot bootstrap (recommended)

```bash
# 1. Sync repo to /root/reason_over_search/ (see §10a)
# 2. Then run:
bash training_m5_5/scripts/bootstrap_b300.sh   # ~30-45 min cold; idempotent
bash training_m5_5/scripts/start_b300.sh       # auto-runs bootstrap if needed
```

`bootstrap_b300.sh` bakes in every fix learned from the 2026-05-15 bring-up:
CUDA 12.9 toolkit swap, InfiniBand headers, ninja, cmake 4.x (Ubuntu 24.04
ships 3.28 which doesn't know sm_103/sm_120), cuDNN dev headers, `uv` symlinked
system-wide, V2 worker venv built from host shell (not Ray actor), TE compile
narrowed to `NVTE_CUDA_ARCHS="90;100"` (B300 cutlass needs the Blackwell
family-key, not raw `103`), and Qwen3.5-0.8B HF cache pre-warmed. If anonymous
HF download fails, the script prompts for an `HF_TOKEN` interactively and
persists it to `training_m5_5/.env`. Root-cause for each fix: [B300_RUNBOOK.md](B300_RUNBOOK.md).

### Pre-built V2 venv tarballs (skip source compile)

The bootstrap auto-discovers per-arch tarballs in your HF account (via `hf auth whoami` on `HF_TOKEN`) before falling back to pantomiman's Hopper-only legacy. Published tarballs:

| GPU family | SM | Repo | File |
|---|---|---|---|
| Hopper (H100/H200) | sm_90 | `pantomiman/reason-over-search-v1-venvs` | `dtensor_policy_worker_v2.tar.gz` |
| **Blackwell-Ultra (B300)** | **sm_103** | **`sandheepp/reason-over-search-venvs` (public)** | **`dtensor_policy_worker_v2_sm103.tar.gz`** |

Each tarball is built on its native SM and **only runs on that SM** (no cross-arch fallback). After your first successful bootstrap on a new SM, run `bash training_m5_5/scripts/package_v2_venv.sh` to publish your own tarball to `<your_hf_user>/reason-over-search-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz`; future bootstraps on the same SM auto-discover it. See [B300_RUNBOOK.md §5](B300_RUNBOOK.md#5-nv-grouped-gemm-cant-torchcudainit-inside-a-gpu-less-ray-actor) for the lookup chain.

The §10b–§10h sections below are the **manual** equivalent — useful only if
you want to understand or debug what the bootstrap is doing. Skip them
otherwise.

Reference box (observed 2026-05-15):

| Field | Value |
|---|---|
| Hostname | `cosmic-matrix-fin-03` |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-100-generic |
| GPUs | 2× NVIDIA B300 SXM6 AC, 275040 MiB each, driver 580.126.09 / CUDA 13.0 |
| vCPU / RAM | 60 cores / 550 GB |
| Disk (`/`) | 193 GB total, ~160 GB free (no separate `/workspace`) |
| Cost | $5.94/h (Verda spot) |

### 10a. Code sync (from your local checkout)

```bash
# From local repo root — tar-stream to skip the bloat:
tar --exclude='./.venv' --exclude='./.git' --exclude='./results' \
    --exclude='./logs' --exclude='./wandb' --exclude='./__pycache__' \
    --exclude='*.pyc' --exclude='*.parquet' \
    --exclude='./training/nemo_rl/.venv' \
    -czf - . | ssh root@<HOST> 'mkdir -p /root/reason_over_search && \
    tar -xzf - -C /root/reason_over_search'
```

The repo lands at `/root/reason_over_search/` (~275 MB). Verda has no
`/workspace` mount, so paths in this doc that say `/workspace/reason_over_search`
become `/root/reason_over_search` here.

### 10b. Install uv + materialize the NeMo-RL venv

```bash
ssh root@<HOST>
cd /root/reason_over_search

# 1. uv (Astral installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. M5.5 venv (works for M5.1 too — shared NeMo-RL symlink). ~13 GB, ~5-15 min.
bash training_m5_5/setup.sh
ls training_m5_5/nemo_rl/.venv/bin/python   # sanity
```

There is **no pre-warmed wheel cache** on Verda; `uv sync --extra vllm` pulls
~5 GB of wheels on first run. Cache at `~/.cache/uv` after that.

### 10c. Retriever assets (corpus + IVF-SQ8 index + e5-base-v2)

The retriever stack and download URLs are unchanged from §4 step 6, but you
have to drive the download manually. From `/root/reason_over_search`:

```bash
mkdir -p local_retriever/{corpus,indexes,models}

# Hugging Face downloader (uses hf_transfer for speed)
pip install --upgrade huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1. wiki-18 corpus (~14 GB after gunzip)
hf download PeterJinGo/wiki-18-corpus --repo-type dataset \
    --local-dir /tmp/wiki18 --include 'wiki-18.jsonl.gz'
gunzip -c /tmp/wiki18/wiki-18.jsonl.gz > local_retriever/corpus/wiki18_100w.jsonl

# 2. IVF-SQ8 index (~16 GB)
hf download pantomiman/reason-over-search --repo-type model \
    --local-dir local_retriever/indexes \
    --include 'wiki18_100w_e5_ivf4096_sq8.index'

# 3. e5-base-v2 encoder (~0.5 GB)
hf download intfloat/e5-base-v2 --local-dir local_retriever/models/e5-base-v2
```

Disk budget: corpus 14 + index 16 + encoder 0.5 + venv 13 + Qwen3.5-0.8B model
~1.7 + ~12 ckpts × 3.2 GB ≈ 84 GB. Fits comfortably in 160 GB free.

### 10d. Training data (MuSiQue)

```bash
source training_m5_5/nemo_rl/.venv/bin/activate
python training_m5_5/scripts/prep_musique.py
# Writes data/training/musique/train.parquet (~10 MB).
```

### 10e. Stand up the retriever

```bash
# from a screen / tmux session so it survives SSH drops
tmux new -s retriever
cd /root/reason_over_search/local_retriever
# 8 workers fits in 550 GB host RAM (each loads the ~16 GB index)
# follow local_retriever/README.md "Run" section; same as the Vast path
# Detach: Ctrl-b d. Verify:
curl -sS http://127.0.0.1:3005/health    # → {"status":"healthy"}
```

### 10f. W&B + HF tokens

```bash
cat > training_m5_5/.env <<EOF
WANDB_API_KEY=<your_key>
# optional, for upload_ckpts_watcher.sh:
HF_TOKEN=<your_hf_token>
HF_REPO_PREFIX=<your_hf_user>/qwen3.5-0.8b-grpo-musique
EOF
```

### 10g. Launch M5.5 production on B300

The B300-specific config bumps `train_micro_batch_size=4`,
`gpu_memory_utilization=0.85`, and turns `activation_checkpointing=false` — see
the fix table in [`training_m5_5/configs/m5_5_research_paper_b300.yaml`](../../training_m5_5/configs/m5_5_research_paper_b300.yaml)
for the per-knob rationale.

```bash
cd /root/reason_over_search
tmux new -s train
bash training_m5_5/scripts/run.sh --mode prod_b300 --seed 42 \
    > /root/logs/m5_5_b300.log 2>&1 &
disown
tail -f /root/logs/m5_5_b300.log
```

Per-step wall-clock projection vs the B200 anchor (B200 = ~1000 s/step at
micro=2 with activation_checkpointing on, per RESULTS_M5_1_B200.md §6.1):

| Knob change | Expected speedup vs B200 |
|---|---|
| B300 vs B200 compute (~1.5× SMs, same bandwidth class) | ~1.3× |
| `train_micro_batch_size: 2 → 4` (fewer DTensor roundtrips, bigger matmul efficiency) | ~1.15× |
| `activation_checkpointing: true → false` (skip recomputation) | ~1.25× |
| **Cumulative** | **~1.9×** → ~525 s/step → 622 steps ≈ **91 h ≈ 3.8 d** |

Tighten with smoke first: `bash training_m5_5/scripts/run.sh --mode smoke` runs
50 steps at the small shape (~20 traj/step) and confirms the loop end-to-end
before committing the 4-day prod run.

### 10h. 2× B300 (TP=2) — `prod_b300_2xgpu`

For boxes with two B300s (the standard Verda B300 SKU), there's now a TP=2
variant ([`m5_5_research_paper_b300_2xgpu.yaml`](../../training_m5_5/configs/m5_5_research_paper_b300_2xgpu.yaml))
that uses both GPUs:

```bash
bash training_m5_5/scripts/start_b300.sh --mode prod_b300_2xgpu
```

The launcher refuses if only one GPU is visible, and warns (without
blocking) if you launch the 1× config on a 2-GPU box. Differences from
the 1× config: `cluster.gpus_per_node: 2`,
`policy.dtensor_cfg.tensor_parallel_size: 2` with the
`training_m5_5.src.parallel_plan_qwen35.custom_parallel_plan` (Qwen3.5
vocab-parallel FSDP bug workaround, identical hook used by 2× A100), and
`policy.generation.vllm_cfg.tensor_parallel_size: 2`.

Expected speedup vs 1× B300: ~1.7× per step (TP=2 cuts training/logprobs
phases ~1.7× and rollout ~1.8×, with ~15% all-reduce overhead). Full-run
projection: ~52 h ≈ 2.2 d, vs ~3.8 d on 1× B300.

Alternative use of the second GPU: run two independent seeds in parallel
(one per GPU) — doubles statistical power for the M5.5 ablation at the
same wall-clock. Pick based on whether you want speed or seeds.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `bootstrap.sh: Conda env 'retriever' missing` | wrong docker image | Confirm `pantomiman/reason-over-search-v1:v2` (or `:v1`) |
| `bootstrap.sh: nvidia-smi not found` | not a GPU instance | Vast template selection issue; pick a GPU machine |
| Retriever times out during rollouts | flat IP fallback by mistake | Confirm `local_retriever/retriever_config.yaml`'s `index_path` is the IVF; restart with `--num_retriever 8` (`bootstrap.sh` does this) |
| `KeyError: 'task_name'` in DataLoader | running pre-fix code | `git pull`; fix is in `training/src/datasets/search_r1.py` |
| `No actor environment registered for ... SearchR1Environment` | running pre-fix registry | `git pull`; fix is in `training/src/registry.py` |
| `unknown arg: --` from launcher | running pre-fix wrapper | `git pull`; fix is in `training/scripts/run_grpo_1xa100.sh` |
| `RuntimeError: No CUDA GPUs are available` (during uv install) | v2 venv being built inside Ray actor | bootstrap.sh handles this from host shell; if you bypass bootstrap, run the v2 build step manually per `training/fix/CHANGES.md` |
| `AttributeError: 'Qwen3_5Model' object has no attribute 'layers'` | `_v2: false` was overridden | Use the YAML default (`_v2: true`); ensure the v2 venv exists |
| `CUDA error: an illegal memory access` in `torch_chunk_gated_delta_rule` | sequence packing enabled with Qwen3.5 | Add `policy.sequence_packing.enabled=false` (always required) |
| Ray cluster startup timeout | stale `/tmp/ray` | `rm -rf /tmp/ray` and relaunch |
| Retriever exits with `FileNotFoundError: ./indexes/...` or `./corpus/...` | retriever-asset download skipped or failed | Re-run `bash training/scripts/bootstrap.sh` (idempotent); check `local_retriever/{corpus,indexes,models}/` |
| `ERROR: model path not found: .../eval/qwen3.5_0.8b...` from sbatch_m4 / run_m4 | M4 model download skipped or `SKIP_M4_MODELS=1` | Run `bash scripts/m4_download_models.sh` |
| `bad ownership or modes for file /root/.ssh/authorized_keys` | boot hook didn't run | See § 2 manual fix |
| `cuda out of memory` from SGLang | another process holds VRAM | `nvidia-smi` to find culprit; `pkill -f <name>` |

---

## Quick reference — paths after setup

```
/workspace/reason_over_search/
├── data/                                          # 7 eval datasets (jsonl, LFS)
├── data/training/nq_hotpotqa_train/               # Training parquets (LFS)
├── local_retriever/
│   ├── corpus/wiki18_100w.jsonl                   # 14 GB (downloaded by bootstrap §6)
│   ├── indexes/wiki18_100w_e5_ivf4096_sq8.index   # 16 GB (downloaded by bootstrap §6)
│   └── models/e5-base-v2/                         # 0.5 GB encoder (downloaded by bootstrap §6)
├── eval/
│   ├── qwen3.5_0.8b/                              # 1.7 GB hybrid (M4, downloaded by bootstrap §8)
│   └── qwen3.5_0.8b_base/                         # 1.7 GB base (M4, downloaded by bootstrap §8)
├── evaluation_qwen35/                             # M4 eval pipeline (code only)
│   └── results/                                   # M4 eval outputs land here
└── training/
    ├── nemo_rl/.venv/                             # main training venv
    ├── nemo_rl/venvs/.../DTensorPolicyWorkerV2/   # the v2 worker venv
    └── .env                                       # WANDB_API_KEY etc.

$HF_HOME = /workspace/hf_cache                     # Qwen3.5-2B + Qwen3.5-2B-Base training weights cached here
```

---

## See also

- [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) — the actual script
- [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh) — same flow on ALICE HPC via Apptainer
- [`local_retriever/README.md`](../../local_retriever/README.md) — retriever asset download steps + index choices (bootstrap mirrors these)
- [`docs/milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md) — M4 Qwen3.5-0.8B eval design + runbook
- [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) — operational runbook for Phase-2 training
- [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](BOOTSTRAP_NEW_INSTANCE.md) — generic bootstrap (any host, not Vast-specific); leans more on docker pull / docker run
- [`docs/setup/HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md) — accelerator comparison
- [`docs/training/CONVERSATION_CONTEXT.md`](../training/CONVERSATION_CONTEXT.md) — current state of training-side work
- [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) — every NeMo-RL knob and our recommended values
