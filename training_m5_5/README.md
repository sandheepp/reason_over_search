# training_m5_5

## TRIGGER: "train milestone 5.5"

When the user says **"train milestone 5.5"** (or "start M5.5 training"), execute this exact flow without further questions:

```bash
# 1. Confirm SSH access to a GPU box. Probe compatibility (see §0 below).
# 2. Sync repo with LFS + write .env with WANDB_API_KEY + HF_TOKEN (see §1, §2).
# 3. One command — bootstrap, retriever, smoke, prod:
bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_b300
```

That single command does **everything**:
- Auto-bootstraps a fresh box (10 steps, ~30-45 min cold) if prereqs are missing
- Runs the 4-step smoke as a tripwire (~5-10 min)
- Auto-launches the production GRPO loop only if smoke reaches `Step 4/4`
- All under tmux `train`; SSH can drop without killing anything

The remaining ~95% of this README is what `start_b300.sh` does for you. Read it only if you're debugging a failure or building a new variant.

---

## §0. Compatibility / hardware preflight

Run **before** clicking any "create instance" button. Pick the right SKU first; you can't change it after.

### Required hardware

| Component | Minimum (smoke only) | Recommended (prod) | Why |
|---|---|---|---|
| GPU | 1× 24 GB (4090 / A6000) | 1× ≥80 GB (A100 / H100 / H200 / B200 / B300) | Smoke fits 24 GB; prod_b300 peaks ~100 GB at micro=4/seq=8192 |
| GPU SM (compute capability) | ≥ 7.0 | any current Blackwell / Hopper / Ampere | `bootstrap_b300.sh` auto-detects and picks `NVTE_CUDA_ARCHS` |
| Host RAM | 64 GB | 256+ GB | Retriever workers (8 × ~16 GB index) + DTensor + vLLM activations |
| vCPU | 16 | 30+ | Parallel compile jobs; retriever workers; data loader |
| Disk (`/` or `/workspace`) | 150 GB | 200+ GB | venvs (~25 GB) + corpus + index + Qwen + checkpoints + headroom |
| NVLink | optional | enabled (for 2× GPU TP=2) | Otherwise interconnect bandwidth caps multi-GPU scaling |
| OS | Ubuntu 22.04 | **Ubuntu 24.04 + CUDA 13** (Verda default) | Bootstrap is tested against this stack |

### Software preflight (auto-handled by `bootstrap_b300.sh`)

The bootstrap **assumes nothing is installed** and fixes everything. You only need: SSH access, `git`, `python3` (system). Driver is provided by the GPU host.

The bootstrap detects and handles:
1. CUDA toolkit mismatch (host has 13.0, torch wheels are cu129 → swaps default `/usr/local/cuda` symlink to 12.9)
2. Missing InfiniBand headers (`deep-ep` won't compile without them)
3. Missing `ninja` (vLLM kernel JIT) and `cmake` 4.x (sm_103 / sm_120 support needs ≥3.31)
4. `uv` not on Ray actor PATH (Ray strips `~/.local/bin`)
5. Missing cuDNN headers (TE pytorch ext needs `<cudnn.h>`)
6. nv-grouped-gemm GPU init at install time (Ray actor has no GPU → bootstrap builds V2 venv from host shell)
7. Wrong arch flag for Blackwell (`NVTE_CUDA_ARCHS="103"` fails cutlass static_assert → `"90;100"` triggers TE's family expansion to sm_100a + sm_103a)
8. Anonymous HF rate limits (uses `HF_TOKEN` from `.env`; falls back to interactive prompt if missing)

Full lessons-learned with root-cause for each: [docs/setup/B300_RUNBOOK.md](../docs/setup/B300_RUNBOOK.md).

---

## §1. Code + data sync (~3-5 min)

```bash
# On the fresh remote box:
apt-get update && apt-get install -y -qq git-lfs
git lfs install --skip-repo
cd /root
git clone -b m5.5_b300 https://github.com/GaurisankarJ/reason_over_search.git
cd reason_over_search
git lfs install
git lfs pull --include "data/**"   # ~500 MB: MuSiQue parquet, dev jsonls, etc.
```

The repo carries LFS-tracked training data (`data/**/*.{jsonl,parquet}`). Without `git lfs pull` the smoke will fail on dataloader.

---

## §2. Secrets (~10 sec)

```bash
cat > training_m5_5/.env <<EOF
WANDB_API_KEY=<your_wandb_key>      # required for logger.wandb.enabled=true
HF_TOKEN=<your_hf_token>            # write-scoped — used for both fetching tarballs AND uploading sm_103 tarball after build
EOF
chmod 600 training_m5_5/.env
```

`.env` is gitignored. `HF_TOKEN` should have **write access** if you want `package_v2_venv.sh` to upload the V2 venv tarball back to your HF account for future reuse (§5).

---

## §3. Bootstrap (auto-invoked by `start_b300.sh`)

`bash training_m5_5/scripts/bootstrap_b300.sh` runs 10 steps end-to-end. Idempotent — re-running skips anything already done. Steps:

| # | Step | Time | What it does |
|---|---|---|---|
| 1 | apt prereqs | ~2 min | `ninja-build tmux cmake build-essential libibverbs-dev libmlx5-1 libnuma-dev rdma-core` |
| 2 | CUDA 12.9 toolkit | ~3 min | Install + swap `/usr/local/cuda` → 12.9 (torch wheels are cu129) |
| 3 | cuDNN dev headers | ~1 min | `cudnn9-cuda-12 libcudnn9-dev-cuda-12` for TE pytorch ext |
| 4 | uv | ~30 sec | Install + symlink to `/usr/local/bin/uv` so Ray actors find it |
| 5 | cmake 4.x | ~30 sec | `uv tool install cmake` + override `/usr/bin/cmake` (sm_103 / sm_120 need 3.31+) |
| 6 | Main NeMo-RL venv | ~3-5 min | `setup.sh` → `uv sync --extra vllm`. Downloads torch 2.10+cu129, vLLM 0.17, deep-ep, etc. |
| 7 | V2 worker venv | **~3 min fast / ~10-15 min source** | Try tarball fast-path: `<your-hf>/reason-over-search-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz` → fall back to `pantomiman/reason-over-search-v1-venvs` (Hopper) → fall back to source compile with `NVTE_CUDA_ARCHS` matched to detected SM. See "Pre-built V2 venv tarballs" below. |
| 8 | Qwen3.5-0.8B HF cache | ~10 sec | Pre-fetch model weights (avoids slow anonymous DL at vLLM warmup); uses `HF_TOKEN` |
| 9 | Retriever venv | ~30 sec | `local_retriever/.venv_cpu` with faiss-cpu |
| 10 | Retriever assets | ~5-10 min | wiki-18 corpus (14 GB), IVF-SQ8 index (15 GB), e5-base-v2 encoder (0.5 GB), MuSiQue parquet + M4 prompt |

**Parallelization opportunities** (not yet wired into bootstrap, but feasible): steps 6 and 10 are independent of each other after step 5. Future enhancement: kick off step 10 (asset downloads, mostly network-bound) in the background once step 5 finishes, run steps 6+7 (compile-bound) in foreground. Would save ~5-8 min on cold boxes.

### Pre-built V2 venv tarballs

The bootstrap auto-discovers per-arch tarballs in your HF account (via the `HF_TOKEN` whoami → `<user>/reason-over-search-venvs`) before falling back to pantomiman's Hopper-only legacy tarball. Currently published tarballs:

| GPU family | SM | Repo | File | torch / TE | Source |
|---|---|---|---|---|---|
| Hopper (H100, H200) | sm_70/80/89/90 | `pantomiman/reason-over-search-v1-venvs` | `dtensor_policy_worker_v2.tar.gz` | torch 2.10+cu129, TE 2.14+71bbefbf | Vast.ai bootstrap (NeMo-RL upstream tarball) |
| Blackwell-Ultra (B300) | **sm_103** | `sandheepp/reason-over-search-venvs` (public) | `dtensor_policy_worker_v2_sm103.tar.gz` | torch 2.10+cu129, TE 2.14+71bbefbf | Built 2026-05-16 on Verda B300 (commit `907af71`) — see [package_v2_venv.sh](scripts/package_v2_venv.sh) |

To bake your own tarball after a successful bootstrap on a new SM:

```bash
bash training_m5_5/scripts/package_v2_venv.sh
# uploads to <your_hf_user>/reason-over-search-venvs:dtensor_policy_worker_v2_sm${CC}.tar.gz
```

**Reuse semantics**: a tarball built on one SM only works on the **same SM** (sm_103 binaries can't run on sm_90, etc.). PTX forward-compat only works within a major architecture family and only sm_X → sm_Y where Y > X. See [docs/setup/B300_RUNBOOK.md](../docs/setup/B300_RUNBOOK.md#what-each-sm_xx-actually-is) for the full SM table.

---

## §4. Smoke → prod chain (~5-10 min smoke, then prod auto-launches)

`bash training_m5_5/scripts/start_b300.sh --smoke-first --mode prod_b300` chains:

1. **Pre-flight** — verify everything from §0-§3 is in place
2. **Retriever bring-up** — `retriever_serving.py` in tmux `retriever`, wait for `/health`
3. **Phase 1 — smoke** (4 steps × seq=4096, 20 traj/step) in tmux `train`
4. **Tripwire** — smoke must reach `Step 4/4` and exit 0; otherwise prod does NOT launch
5. **Ray cleanup** between phases (kill `raylet`/`gcs_server`, wipe `/tmp/ray`)
6. **Phase 2 — prod** (whatever `--mode` you passed) in the same tmux session

For 2× B300 (TP=2): `--mode prod_b300_2xgpu`. The launcher refuses prod_b300* on GPUs <80 GB VRAM.

---

## §5. (Optional) Upload V2 venv tarball after first build

After bootstrap finishes on a new GPU SM, package + upload the V2 venv so future bootstraps on the same SM hit the fast-path:

```bash
bash training_m5_5/scripts/package_v2_venv.sh
# → uploads dtensor_policy_worker_v2_sm${CC}.tar.gz to <your_hf>/reason-over-search-venvs
# → next bootstrap on the same SM: ~3 min skip instead of ~15 min compile
```

The script auto-detects HF username via `hf auth whoami`, sanity-checks the venv imports before uploading (refuses broken venvs), and round-trip-verifies the upload.

---

## §6. Verifying / watching the chain

`start_b300.sh` auto-launches three background log streams. Tail them in three separate panes for a full picture during a multi-day run:

```bash
# 1. Training stdout (W&B URL, step counters, errors)
tail -f /root/logs/m5_5_chain_seed42_*.log

# 2. Resource watcher (RAM/disk/GPU/retriever liveness every 30 s)
tail -f /root/logs/m5_5_resources_seed42_*.log

# 3. Trace digest (auto-runs every 10 steps — see §6.1 below)
tail -f /root/logs/m5_5_traces_seed42_*.log

# Interactive tmux
tmux attach -t train      # Ctrl-b d to detach
tmux attach -t retriever

# Kill if needed
tmux kill-session -t train

# W&B URL surfaces in chain log after vLLM warmup (~2 min into prod phase)
```

### §6.1 Trace digest — auto-check every 10 steps (REQUIRED for unattended runs)

[`watch_resources.sh`](scripts/watch_resources.sh) auto-invokes [`check_trace.py`](scripts/check_trace.py) on every 10th step landed (10, 20, 30 …). It analyzes the most recent rollout JSONL and emits a ~20-line digest:

```
========================================================================
  step 60  n=320  (train_data_step60.jsonl)
========================================================================
  reward         mean=0.2055  std=0.2135
  distribution   broken=  0%  floor= 74%  partial= 22%  perfect=  4%
  tool usage     ≥1_call=  0%  mean_calls/rollout=0.00
  retriever      tool_responses=0  errors=0 (0%)
  generation     mean_tokens=561  generic_country_answers=  3%

  🚩 RED FLAGS:
     • TOOL_COLLAPSE: only 0% of rollouts call search (threshold ≥50%) — model abandoning retrieval
     • FLOOR_DOMINANCE: 74% of rollouts at exact 0.1 floor — GRPO advantage signal compressed
```

If any red flag fires, the watcher also emits a `WARN` line into the resource log so it's visible without scrolling through the trace log.

**Why this is required**: the B300 run on 2026-05-16 silently trained for 107 steps with a dead retriever (`Errno 111 Connection refused` on every `<tool_response>`). The model learned a zero-tool policy and we only caught it in post-mortem ([RESULTS_M5_5_B300_seed42.md](../docs/report/RESULTS_M5_5_B300_seed42.md)). The auto-check would have raised `TOOL_COLLAPSE` + `RETRIEVER_BROKEN` flags by step 20, when intervention is still cheap.

Red-flag thresholds in [`check_trace.py`](scripts/check_trace.py):

| Flag | Threshold | What it means |
|---|---|---|
| `TOOL_COLLAPSE` | <50% rollouts call search | Model unlearned tool use. Most often a broken retriever, occasionally reward hacking. |
| `RETRIEVER_BROKEN` | >10% tool_responses are connection errors | FAISS workers dead. Check `dmesg \| grep -i killed`, drop `num_retriever`. |
| `FLOOR_DOMINANCE` | >90% rollouts at exact 0.1 | GRPO advantage compressed — group-baseline is the floor. Compare with M5.1. |
| `GEN_DEGENERATE` | mean gen <100 tokens | Model outputting padding / collapsing. |
| `GENERIC_ANSWERS` | >20% answers are "United States"-style country guesses | Reward-hacking by short-answer guessing. |

### §6.2 Ad-hoc trace check

You can also run the digest manually at any time:

```bash
# Latest step
python training_m5_5/scripts/check_trace.py

# Specific step
python training_m5_5/scripts/check_trace.py --step 50

# Just the digest (no rollout samples — useful for grep-able cron logs)
python training_m5_5/scripts/check_trace.py --no-samples
```

Exit codes: `0` = healthy, `1` = ≥1 red flag, `2` = couldn't read rollouts.

### §6.3 Checkpoints + crash recovery

All prod configs now save a checkpoint **every 10 steps** (was 50 before 2026-05-16). On crash, the worst-case lost work is 10 steps (~30 min on B300, ~50 min on H200). NeMo-RL's checkpoint dir layout:

```
results/grpo/m5_5_<mode>/seed<N>/
├── step_10/policy/weights/   # consolidated fp32 safetensors (~3.2 GB)
├── step_10/policy/tokenizer/
├── step_10/training_info.json
├── step_20/...
└── step_<latest>/...
```

**Resume** — `run.sh` auto-resumes from the latest `step_M/` in the checkpoint dir when re-launched with the same `--mode` and `--seed`:

```bash
# Same command as the original launch — NeMo-RL picks up from the latest step
bash training_m5_5/scripts/start_b300.sh --mode prod_h200 --seed 42
# (or whatever mode the crashed run was using)
```

For a fully fresh run, delete the seed dir first:

```bash
rm -rf /root/reason_over_search/results/grpo/m5_5_h200/seed42/
```

**Disk math** — 311 steps × 1 save/10 = 31 saves × ~3.2 GB = **~99 GB worst case**. To prevent disk-fill:

- `watch_resources.sh` runs `prune_old_ckpts()` when disk hits WARN (<15 GB free), keeping only the 5 most recent saves per seed (~16 GB).
- Manual prune at any time:
  ```bash
  ls -d /root/reason_over_search/results/grpo/m5_5_*/seed*/step_* | \
      sort -t_ -k2n | head -n -5 | xargs rm -rf
  ```
- Override the auto-prune keep-count via `--keep-ckpts N` to the watcher (e.g., `KEEP_LAST_N_CKPTS=10` for a 10-deep safety net on bigger disks).

`save_optimizer: false` is intentional — checkpoints are ~3.2 GB instead of ~8.9 GB. Optimizer state re-warms fast at the constant LR; minor disturbance on resume.

Expected timings on B300:
- Bootstrap (cold): ~30-45 min
- Smoke (4 steps): ~5-10 min
- Prod first step lands: ~15-20 min after smoke pass
- Full prod (622 steps): ~3.8 d on 1× B300, ~2.2 d on 2× B300 TP=2

---

## Original milestone narrative

GRPO training scaffold for **milestone M5.5 — F1+format reward ablation** on Qwen3.5-0.8B / MuSiQue. This is the **F1 + 0.1 partial-credit floor + format-gate** variant of the reward-shape ablation triad:

- **M5.1** ([`training_m5_1/`](../training_m5_1/)) — F1-only baseline (the existing in-flight run).
- **M5.5** (this directory) — F1 + 0.1 floor when format valid but F1=0 (paper-faithful 3-tier shaping).
- **M5.6** ([`training_m5_6/`](../training_m5_6/)) — EM-only sibling.

Every knob except the reward function is byte-identical to M5.1 — this is a mechanism ablation, not a "which is best" race. Narrative + design decisions: [docs/milestone_5/MILESTONE_5_5.md](../docs/milestone_5/MILESTONE_5_5.md). Paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md). Audit of what differs from M2: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md).

## What this milestone answers

Does the ReSearch paper's 0.1 partial-credit floor (`reward=0.1` when format valid but F1=0) change end-of-training behaviour and held-out EM/F1 relative to F1-only and EM-only, on Qwen3.5-0.8B / MuSiQue? Motivation: [PHASE_1_SALVAGE.md Finding 1](../docs/milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal) observed the floor compresses tool-use signal into a 3-6 pp reward band; the paper never ablates it.

## Reward spec

```
reward(rollout) =
  0.0    if not is_valid_format(rollout)            # format broken
  0.1    if extract_solution(rollout) is None       # format ok, no <answer>
  0.1    if f1(answer, gold) == 0                   # format ok, answer wrong
  f1     if f1(answer, gold) > 0                    # format ok, partial / full
```

The floor is NOT additive (F1=0.5 → reward=0.5, not 0.6); it only applies when F1=0. `is_valid_format` is the ReSearch state-machine walker ported to Qwen3.5 nested-XML schema with `<answer>` (not `\boxed{}`). Full spec + ReSearch source-of-truth mapping in [src/rewards/search_r1.py](src/rewards/search_r1.py).

## What's different from M2

Same as M5.1: byte-aligned with the M4 eval pipeline ([`evaluation_qwen35/`](../evaluation_qwen35/)) so a trained checkpoint is directly evaluable.

| | M2 ([`training/`](../training/)) | M5.5 (this directory) |
|---|---|---|
| Target model | Qwen3.5-2B | **Qwen3.5-0.8B** |
| Training data | NQ + HotpotQA mix | **MuSiQue only** |
| Reward | EM + format shaping (Search-R1 paper-faithful) | **F1 + 0.1 floor (format-gated)** on `<answer>X</answer>`; no `\boxed{}` wrap |
| Parsers / scorer / tool schema | local copies of FlashRAG functions | **re-exported from `evaluation_qwen35`** so train and eval can't drift |
| Prompts | local `.txt` files (M2) | materialised from `evaluation_qwen35.flashrag.search_r1.templates` via `scripts/sync_m4_prompts.py` |
| W&B project | `reason-over-search` (shared) | `reason_over_search_m5_5` (per-experiment isolation) |

## What's different from M5.1

Per [MILESTONE_5_5.md §3](../docs/milestone_5/MILESTONE_5_5.md), every knob except the reward is byte-identical to [`training_m5_1/configs/m5_1_research_paper.yaml`](../training_m5_1/configs/m5_1_research_paper.yaml). The diff is confined to [src/rewards/search_r1.py](src/rewards/search_r1.py) (`FORMAT_BONUS = 0.1` + `is_valid_format` walker) and the W&B project name.

## Environment setup

`nemo_rl/` here is a **symlink** to `../training/nemo_rl/` (disk-constrained; the upstream vendor is 23 GB and `/workspace` only has ~25 GB free). The `setup.sh` flow follows the symlink and creates the venv inside it, which means **M5.5 and M2 share the same venv state**. That's fine while we're on the same NeMo-RL version (v0.6.0); if you ever bump NeMo-RL in one experiment, you must break the symlink first (otherwise the bump leaks to the other experiment).

```bash
cd training_m5_5
bash setup.sh                       # uv sync --extra vllm against ../training/nemo_rl/
source nemo_rl/.venv/bin/activate   # resolves through the symlink
```

The pre-warmed wheel cache at `/.uv/cache/` makes this fast on Vast (`pantomiman/reason-over-search-v1`). For ALICE HPC see [scripts/bootstrap_alice.sh](scripts/bootstrap_alice.sh) (Apptainer SIF flow).

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

## Configure W&B (one-time per instance)

```bash
cp training_m5_5/.env.example training_m5_5/.env
# edit training_m5_5/.env, fill WANDB_API_KEY (+ optional CHECKPOINT_DIR_BASE).
```

`training_m5_5/.env` is gitignored.

## Run

**Pre-flight**: retriever live at `127.0.0.1:3005` (see [`local_retriever/README.md`](../local_retriever/README.md)); `data/training/musique/train.parquet` present; `src/prompts/m5_qwen35_user.txt` present; `nemo_rl/.venv/` materialized.

```bash
# 50-step end-to-end smoke (20 traj/step). Target: <= 25 s/step on 1x A100-80GB.
bash scripts/smoke.sh
bash scripts/smoke.sh --seed 7

# Production run (M5.5 paper-faithful recipe; identical knobs to M5.1 except reward).
# configs/m5_5_research_paper.yaml is committed (paper-faithful + M5.2 system gains
# O1 fused AdamW + R2 vLLM async_engine). Schedule: 622 steps × 320 trajectories.
# Wall-clock: ~10-13 days on 1x A100-80GB per the live M5.1 trajectory
# (RESULTS_m5.md §4.1). Submitted to ALICE `gpu-a100-80g` partition at the 7-day
# cap via scripts/sbatch_m5_5.sh; resume-from-latest-ckpt across sbatch slots.
bash scripts/run.sh --mode prod --seed 42

# ALICE sbatch submission:
sbatch scripts/sbatch_m5_5.sh         # 1x A100-80GB
sbatch scripts/sbatch_m5_5_2xa100.sh  # 2x A100-80GB variant

# Hydra overrides:
bash scripts/smoke.sh -- policy.train_micro_batch_size=8 grpo.max_num_steps=20
```

W&B run name: `qwen3.5-0.8b-musique-m5_5_<mode>-seed<N>-<TS>`. Checkpoints land under `${CHECKPOINT_DIR_BASE:-results/grpo}/m5_5_<mode>/seed<N>/` every 50 steps (not timestamped, so resumes work).

## Folder layout

```
training_m5_5/
├── README.md                       # this file
├── setup.sh                        # uv venv + uv sync against nemo_rl/ symlink
├── .env, .env.example, .gitignore
├── nemo_rl/ -> ../training/nemo_rl/   # SYMLINK; shared with M2
├── src/                            # overlay; see docs/report/CODE_SETUP_m5.md §3.2 for the per-file diff
│   ├── chat_template/tools.py      # re-exports QWEN35_SEARCH_TOOL from evaluation_qwen35
│   ├── datasets/search_r1.py       # dataset-agnostic parquet adapter
│   ├── environments/parsers.py     # qwen_native parse_query delegates to evaluation_qwen35
│   ├── environments/search_r1_env.py # F1 fallback + near_em_rate metric + kwarg-dispatch fix
│   ├── processors/search_r1.py     # docstring updated; dataset-agnostic
│   ├── prompts/m5_qwen35_user.txt  # pre-staged from M4.2 canonical
│   ├── prompts/_archive_m2/        # M2 prompt files preserved for ablation
│   ├── rewards/search_r1.py        # **M5.5: F1 + 0.1 partial-credit floor + is_valid_format walker** (the ablation knob)
│   ├── parallel_plan_qwen35.py     # DTensor parallel plan for Qwen3.5 hybrid arch
│   └── registry.py                 # `training_m5_5.src.*` import paths
├── configs/
│   ├── m5_smoke.yaml               # M5 pipeline-validation smoke (20 traj/step × 50 steps)
│   ├── m5_smoke_2xa100.yaml        # 2x A100 smoke variant
│   ├── m5_5_research_paper.yaml    # M5.5 production config (LIVE; 622 steps, 320 traj/step)
│   ├── m5_5_research_paper_2xa100.yaml # 2x A100 production variant
│   └── _archive_m2/                # M2 2B configs
├── scripts/
│   ├── run.sh                      # --mode smoke|prod --seed N
│   ├── smoke.sh                    # alias for `run.sh --mode smoke`
│   ├── run_grpo.py                 # imports training_m5_5.src.registry, hands off to NeMo-RL examples/run_grpo.py
│   ├── prep_musique.py             # MuSiQue train -> data/training/musique/train.parquet
│   ├── sync_m4_prompts.py          # materialises M4 prompt mode into src/prompts/
│   ├── bootstrap.sh                # Vast bootstrap (M2-shape; rewrite for M5.5 when needed)
│   ├── bootstrap_alice.sh          # ALICE HPC Apptainer SIF bootstrap
│   ├── sbatch_m5_5.sh              # ALICE sbatch wrapper (1x A100-80GB, 7-day cap)
│   ├── sbatch_m5_5_2xa100.sh       # ALICE sbatch wrapper (2x A100-80GB)
│   ├── upload_ckpts_watcher.sh     # polling HF Hub uploader for new step_N/ ckpts
│   ├── extract_smoke_samples.py    # extract (prompt, response, reward) tuples per combo for SMOKE_RESULTS.md
│   └── _archive_m2/                # M2 wrappers preserved
├── tests/                          # pure-Python tests (reward + parser + format-helper)
└── fix/CHANGES.md                  # M2 changelog; M5 changes summarized in docs/report/CODE_SETUP_m5.md
```

`docs/training/CONVERSATION_CONTEXT.md` is the per-strand bootstrap doc for the training side.

## Sibling experiments (the reward-shape ablation triad)

M5.5 is the **F1+format** leg of a three-experiment ablation:

```bash
training_m5_1/   # F1-only baseline (in-flight on B200 Spheron, see experiment_1_b200)
training_m5_5/   # F1 + 0.1 floor + format gate (this directory)
training_m5_6/   # EM-only sibling
```

The W&B project key in each copy's yaml (`logger.wandb.project: reason_over_search_m5_<N>`) auto-isolates the runs. See [docs/milestone_5/MILESTONE_5_5.md §"Parallel experiments"](../docs/milestone_5/MILESTONE_5_5.md) for the design rationale; [docs/milestone_5/MILESTONE_5_6.md](../docs/milestone_5/MILESTONE_5_6.md) for the EM-only sibling.

## Pointers

- M5.5 milestone narrative: [docs/milestone_5/MILESTONE_5_5.md](../docs/milestone_5/MILESTONE_5_5.md)
- M5.6 sibling (EM-only): [docs/milestone_5/MILESTONE_5_6.md](../docs/milestone_5/MILESTONE_5_6.md)
- M5.1 baseline (F1-only, in flight): [docs/milestone_5/MILESTONE_5.md](../docs/milestone_5/MILESTONE_5.md), [training_m5_1/](../training_m5_1/)
- M5 paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md)
- M5 code-setup audit: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md)
- M5 smoke results log: [docs/report/RESULTS_SMOKE_m5.md](../docs/report/RESULTS_SMOKE_m5.md)
- Motivation for the 0.1-floor ablation: [PHASE_1_SALVAGE.md Finding 1](../docs/milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)
- M2 NeMo-RL operational guide (still the foundation): [training/README.md](../training/README.md), [docs/training/README.md](../docs/training/README.md)
- M4 eval pipeline (the rollout shape we're aligned to): [docs/milestone_4/MILESTONE_4.md](../docs/milestone_4/MILESTONE_4.md), [docs/report/CODE_SETUP_m4.md](../docs/report/CODE_SETUP_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470); official code: [Agent-RL/ReSearch](https://github.com/Agent-RL/ReSearch)
