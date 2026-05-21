---
title: ALICE hardware fixes — M5.1/M5.5/M5.6 sbatch hardening session
tags: [milestone-5, alice, sbatch, hardware, postmortem]
source: live working session 2026-05-12 (experiment_1_alice branch)
created: 2026-05-12
updated: 2026-05-12
---

# Hardware fixes 2026-05-12 — getting the M5 sbatches to actually work on ALICE

Single-session log of the bugs caught + fixed before the 6 M5 production
sbatches (M5.1 / M5.5 / M5.6 × {1× A100, 2× A100}) were resubmitted. Sticks
to facts, not aspiration: each fix is paired with the symptom that surfaced
it and the verification status.

The doc lives in [`docs/milestone_5/`](.) alongside the M5 narrative; see
also [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) for the
research_v2 postmortem that seeded this work.

## 1. Context (why 8 bugs in one day)

The M5.1 / M5.5 / M5.6 sbatches were authored on Vast.ai 1× A100 + 24 GB local
NVMe; they passed there. ALICE has a different stack:

- Apptainer SIF instead of Docker (read-only rootfs, different bind semantics)
- `/zfsstore` NFS-style storage (~40 MB/s reads, vs Vast.ai's ~1 GB/s NVMe)
- 252 GB RAM nodes (vs Vast.ai's 64 GB)
- 64 CPUs per node (vs Vast.ai's 16-32)
- SLURM SBATCH headers (Vast.ai used bash directly)

Most fixes are environment-portability issues that the Vast.ai validation
didn't expose. The remaining ones are correctness bugs that production-shape
testing would have hit on either platform.

## 2. Bugs caught + fixes applied (in order discovered)

| # | Bug | Surfaced via | Fix | Commit |
|---|---|---|---|---|
| 1 | Venv pre-check `[ -x .../python ]` follows symlink into `/.uv/python_install/...` (container-only), fails on host | first in-srun smoke (rc=1 in 1 s) | `-d` on dir instead of `-x` on python | `8b8525d` |
| 2 | M5.5/M5.6 2x sbatch pre-checks referenced `m5_1_research_paper_2xa100.yaml` (wrong filename) | grep audit during deep audit | rename refs to `m5_5_…` / `m5_6_…` | `8b8525d` |
| 3 | Ckpt config crash: `metric_name: "train/loss/mean"` doesn't start with `train:`/`val:`; `keep_top_k: 0` would delete every save | already documented in research_v2 (a1 prod crashed at step-50 first save, ~19.5h compute lost) | `metric_name: null`, `keep_top_k: null`, `save_optimizer: false` (3.2 GB/save vs 8.9 GB) | `3aa04ff` |
| 4 | HF watcher had no per-experiment namespace → all 3 experiments would push to the same HF repo names | code review while porting watcher to M5.5/M5.6 | repo name = `${HF_REPO_PREFIX}-m5_X_${MODE}-seed${SEED}-step${N}` per experiment | `3aa04ff` |
| 5 | Smoke yamls had `enabled: false` for ckpt → smoke didn't actually exercise the save round-trip (the load-bearing fix from research_v2) | code review while applying fix #3 | smoke yamls flipped to `enabled: true`, `save_period: 2`, `max_num_steps: 4` | `3aa04ff` |
| 6 | Retriever `/health` timeout hard-coded 1800 s → not enough for cold zfsstore (8 retrievers × ~7 min first-cold-read = >30 min) | in-srun smoke timeout at exactly 30 min | bump default to 5400 s; override via env var | `8363b79`, `456171f` |
| 7 | HF watcher's apptainer exec didn't forward `CHECKPOINT_DIR_BASE` → watcher reads `results/grpo` while trainer writes `results/grpo_m5_5` (for M5.5/M5.6) | deep audit before re-fire | `--env CHECKPOINT_DIR_BASE=…` in watcher apptainer exec | `9296123` |
| 8 | `--mem=160g` (1x) / `200g` (2x) too small: `faiss.read_index()` loads full 16 GB index per retriever → 8 × 17.5 GB = 140 GB + 60 GB trainer + 15 GB overhead = ~215 GB | user caught my wrong "17 GB total" reading; verified at "Initializing retriever 7/8" with RSS=105 GB | bump `--mem=240g` for all 6 sbatches (node has 252 GB total) | `eb2e6b1` |
| 9 | Ray's default `_temp_dir=/scratchdata` is read-only inside the apptainer SIF rootfs → `OSError: Read-only file system: '/scratchdata'` during `ray.init()` | smoke verification rc=1 in <1 min, before training start | bind-mount `${REPO_ROOT}/logs/ray_scratch_${RUN_ID}` → `/scratchdata` in apptainer | `e6ddd9d` |
| 10 | DTensor `tensor_parallel_size: 1` in all 3 2xa100 yamls (DDP across 2 GPUs, not TP=2) → skipped ~70% of the projected 2× A100 speedup | user reading HARDWARE_COMPARISON.md §3, vs current yaml | flip DTensor TP=1 → 2 in all 3 2xa100 prod yamls; kept micro=1 pending 2-GPU smoke | `68f6bd4` |
| 11 | DTensorPolicyWorkerV2 venv empty on disk (skeleton from May 2 only, no payload) → `uv run --locked --extra automodel` fails because the env-builder Ray actor is GPU-less and nv-grouped-gemm's setup.py calls `torch.cuda.init()` | re-fired smoke; "DTensorPolicyWorkerV2/lib/python3.13/..." not found | download `dtensor_policy_worker_v2.tar.gz` (6.5 GB) from `pantomiman/reason-over-search-v1-venvs` and extract (~12 GB) via SLURM `cpu-short` job | runtime fix; no commit |
| 12 | Sbatch arg parser didn't accept `--` for Hydra overrides → couldn't pass `grpo.max_num_steps=1` for a prod-shape OOM check | designing the 2x verification stage 4 | add `--) shift; EXTRA_OVERRIDES=("$@"); break ;;` case in sbatch_m5_1_2xa100, forward to run.sh | `596afe6` |

## 3. The CPU + RAM math (now sized correctly)

A 0.8B model's compute footprint is small; the load-bearing memory is the
retriever's FAISS index and the trainer's DTensor activation peak.

**Per-process RAM**:
- Retriever: ONE Python process with N async workers (semaphore-bounded). `faiss.read_index()` loads the full 16 GB IVF-SQ8 index into RAM **per retriever wrapper**. 8 retrievers = 8 × ~17.5 GB = ~140 GB.
- DTensor trainer: ~60 GB peak (micro=1, seq=8192, 320 traj/step) — measured in research_v2's v7 smoke.
- vLLM + Ray driver + Python overhead: ~10-15 GB.
- **Total peak**: ~215 GB.

**CPU**:
- Retriever: 8 async workers × `OMP_NUM_THREADS=4` = up to 32 concurrent FAISS threads (semaphore-bounded).
- Trainer (vLLM + DTensor + Ray): ~10 CPUs in flight during a step.
- **Total peak**: ~42 cores.

**Node capacity** (gpu-a100-80g):
- 64 CPUs, 252 GB RAM, 2-3 A100s per node.
- 1× A100 allocation: 40 CPUs + 240 GB → tight on CPU at peak, comfortable on RAM (12 GB OS buffer).
- 2× A100 allocation: 48 CPUs + 240 GB → comfortable on both.

## 4. zfsstore is the throughput bottleneck

Measured (in-srun on node875 via `dd` skip-512-count-128 8 MB blocks):
- zfsstore: **~40 MB/s sequential read** (cold cache).
- /tmp on node875: ~200 MB/s sequential write; reads faster from Linux page cache.

Implications:
- First retriever's FAISS load: 16 GB / 40 MB/s = **~7 min** cold.
- Subsequent retrievers benefit from warm page cache: ~1-2 min each.
- 8-retriever boot: **~7 + 7×~2 = ~21 min worst-case** cold; ~5-8 min on a hot node.
- Bumped `RETRIEVER_HEALTH_TIMEOUT_S` to **5400 s (90 min)** as 50 % margin.

Further workarounds NOT applied yet (defer until needed):
- Pre-warm via `dd if=index of=/dev/null` before retriever launch (deterministic ~7 min).
- Copy index to compute-node `/tmp` (199 GB free on node875) → all workers read from local fast storage.
- FAISS mmap mode (`IO_FLAG_MMAP`) — requires editing `retriever_serving.py`, not done.

## 4.5 Plan B results (RECORDED 2026-05-12 ~12:35 CEST)

Plan B (Option B) fired inside srun 2266786 on node875 with `RETRIEVER_NUM_WORKERS=4` (smallmem path to fit the 120 GB srun allocation). All three experiments completed; one false start on M5.6 surfaced one final code bug (env imported `f1_check` instead of `em_check`), fixed and retried.

| Experiment | rc | step 4 reached | ckpt step_2 | ckpt step_4 | WANDB run | HF upload step_2 |
|---|---:|---|---:|---:|---|---|
| M5.1 | 0 | ✓ | 6.36 GB | 6.36 GB | `reason_over_search_m5_1/runs/5n2gy5r7` | `pantomiman/qwen3.5-0.8b-grpo-musique-m5_1_smoke-seed42-step2` ✓ |
| M5.5 | 0 | ✓ | 6.36 GB | 6.36 GB | `reason_over_search_m5_5/runs/<run>` | `pantomiman/qwen3.5-0.8b-grpo-musique-m5_5_smoke-seed42-step2` ✓ |
| M5.6 (first) | 1 | — | — | — | — | — (ImportError before training) |
| M5.6 (retry, after env em_check fix) | 0 | ✓ | 6.36 GB | 6.36 GB | `reason_over_search_m5_6/runs/mndyd280` | `pantomiman/qwen3.5-0.8b-grpo-musique-m5_6_smoke-seed42-step2` ✓ |

What this proves end-to-end for the **1× A100** sbatch path (modulo the 4-vs-8 retriever count, math-verified):

- sbatch syntax + pre-checks
- `/scratchdata` apptainer bind for Ray
- Retriever boot + `/health`
- WANDB credential pickup from `.env`
- Training (4 GRPO steps) completes
- **Checkpoint save** at step 2 + step 4 with `save_consolidated=true`, `save_optimizer=false`, `metric_name: null`, `keep_top_k: null` (the load-bearing prod fix from research_v2)
- HF Hub upload daemon launches, picks up at least step 2 (step 4 missed because squashfuse_ll times out mid-run; tracked as a follow-up patch before May 15)
- cleanup trap kills retriever + HF watcher cleanly

What it does NOT prove: 8-retriever shape under `--mem=240g` (covered by sbatch 2274508 tomorrow on gpu-short) and 2-GPU TP=2 (covered by sbatch 2275683 + srun 2275986 tomorrow).

## 5. Verification matrix (what we know works vs trust)

| Verification | Status | What it proves | What it does NOT prove |
|---|---|---|---|
| **Plan B**: in-srun 1x smoke, num_retriever=4, 120 GB allocation | RUNNING (m5_1 ✅ PASSED rc=0 with ckpt + WANDB + HF upload of step_2 to `pantomiman/qwen3.5-0.8b-grpo-musique-m5_1_smoke-seed42-step2`; m5_5/m5_6 in progress) | sbatch scaffold + ckpt + HF + WANDB + cleanup trap | 8-retriever shape; --mem=240g allocation |
| **sbatch 2274508**: 1x A100 + 240g + 8 retrievers, gpu-short, fires `m5_verify_seq.sh` | QUEUED, StartTime 2026-05-13 08:23 CEST | Full 8-retriever + 240g shape (closes Plan B gap) | 2x shape |
| **sbatch 2275683**: 2x A100 + 240g, gpu-short, fires `m5_verify_seq_2xa100.sh` (3 smoke_2xa100 + stage 4 m5_1 prod_2xa100 1-step OOM check) | QUEUED, StartTime 2026-05-13 13:50 CEST | DTensor TP=2 + full prod batch shape (320 traj, seq=8192, micro=1) doesn't OOM | micro=2 unlock (not flipped yet) |
| **srun 2275986**: 2x A100 + 240g, interactive shell, 4h walltime | QUEUED, StartTime 2026-05-13 17:50 CEST | Hands-on poking after sbatch 2275683 completes | n/a |

## 6. Wall-clock projection for 2× A100 prod (does the run fit a single 7-day sbatch?)

Anchors from HARDWARE_COMPARISON.md §3:
- 1× A100 anchor: ~23.7 min/step (a1 step-38-42, "improve-and-grow" phase)
- 2× A100 TP=2 projection: ~8.8 min/step with micro=2 unlock; ~15-17 min/step at micro=1

| Config | Per-step | 622 steps | Fits 7-d walltime cap (168 h)? |
|---|---:|---:|---|
| 1× A100 (anchor, micro=1) | 23.7 min | ~246 h (10.2 d) | **NO** — needs 2 resumes |
| 2× A100 TP=2, **micro=1** (current 2xa100 prod yaml) | ~15.8 min | ~164 h (6.8 d) | **MARGINAL** — ~4 h buffer |
| 2× A100 TP=2, **micro=2** (if 2-GPU smoke confirms no OOM) | ~8.8 min | ~91 h (3.8 d) | **COMFORTABLE** — ~77 h buffer |

So the 2× A100 prod runs **probably fit one srun**, but only with confidence after either:
- Tomorrow's stage-4 OOM check at micro=1 passes AND step-time stays ≤16 min/step at steady state; OR
- A follow-up smoke at micro=2 demonstrates no OOM, after which we flip the prod yaml.

The 5-7 day firm-StartTime buffer on the 2× A100 prod sbatches gives time
to do the micro=2 smoke and flip if we want the comfortable budget.

## 7. Sensitivity to a3 step-time (rollout-length drift)

a1's anchor was during a phase where rollout length grew toward ~1000 tokens.
If a3's steady state is shorter (more terminal answers, fewer search turns)
the per-step time drops; if longer, it grows. Scaling table (linear in step
time at fixed batch shape):

| 1× A100 anchor | 2× A100 TP=2 wall (micro=2) | 1-srun fit on 2× A100? |
|---|---:|---|
| 15 min/step | ~2.4 d / ~58 h | YES, big buffer |
| 20 min/step | ~3.2 d / ~77 h | YES, comfortable |
| 23.7 min/step (a1 anchor) | ~3.8 d / ~91 h | YES, 77 h buffer |
| 30 min/step | ~4.8 d / ~115 h | YES, 53 h buffer |

So even at the high end (30 min/step), a single 7-day 2× A100 srun completes
the 622-step run with the micro=2 unlock. **Without** micro=2 we lose the
buffer; whether one srun suffices then depends on whether a3 stays under
~23 min/step at steady state.

## 8. Open issues (not blocking today's resubmit)

- **HF watcher container-lifetime**: in Plan B m5_1 smoke, the watcher uploaded step_2 to HF Hub successfully, then squashfuse_ll (apptainer's SIF mount process) timed out before the step_4 upload cycle. In prod (12 saves over 25 days), only the first 1-2 ckpts would upload before the container dies. Fix: run watcher OUTSIDE apptainer with system python + `huggingface_hub`, OR keep the SIF mount alive via heartbeat. Tracking before May 15.
- **micro=2 unlock**: kept conservative micro=1 in 2xa100 prod yaml. A 2-GPU smoke at micro=2 would unlock the comfortable 1-srun budget. Deferrable; the firm StartTimes (May 15+) leave time.

## 9. Files touched today

```
training_m5_1/configs/m5_1_research_paper.yaml         # ckpt fix (already on research_v2, merged in)
training_m5_1/configs/m5_1_research_paper_2xa100.yaml  # ckpt + DTensor TP=2
training_m5_1/configs/m5_smoke.yaml                    # ckpt-enabled, save_period=2, max_num_steps=4
training_m5_1/configs/m5_smoke_2xa100.yaml             # NEW (Option C-1)
training_m5_1/scripts/sbatch_m5_1.sh                   # venv check, timeout, /scratchdata, --mem=240g, HF watcher wired, cleanup trap
training_m5_1/scripts/sbatch_m5_1_2xa100.sh            # all of the above + -- pass-through for Hydra overrides
training_m5_1/scripts/run.sh                           # new smoke_2xa100 mode case
training_m5_1/scripts/upload_ckpts_watcher.sh          # per-experiment repo namespace (-m5_1_ prefix)
training_m5_1/.env                                     # populated WANDB + HF_TOKEN + HF_REPO_PREFIX (gitignored, on ALICE)
training_m5_1/.env.example                             # already had HF doc from research_v2 merge

training_m5_5/configs/m5_5_research_paper.yaml         # ckpt fix
training_m5_5/configs/m5_5_research_paper_2xa100.yaml  # ckpt + DTensor TP=2
training_m5_5/configs/m5_smoke.yaml                    # ckpt-enabled
training_m5_5/configs/m5_smoke_2xa100.yaml             # NEW (C-1)
training_m5_5/scripts/sbatch_m5_5.sh                   # same template as m5_1
training_m5_5/scripts/sbatch_m5_5_2xa100.sh            # same template as m5_1 2xa100
training_m5_5/scripts/run.sh                           # new smoke_2xa100 mode case
training_m5_5/scripts/upload_ckpts_watcher.sh          # NEW (porte from m5_1, -m5_5_ prefix)
training_m5_5/.env                                     # symlinked to training_m5_1/.env

training_m5_6/* (mirror of training_m5_5 with m5_6 substitutions)

docs/milestone_5/HARDWARE_FIXES_2026-05-12.md          # THIS doc
```

Branch: `experiment_1_alice`. Commits 8b8525d → 596afe6 (12 commits today).

## 10. Next steps — runbook after Plan B passes

Execute in order. Each step has a clear PASS gate before proceeding.

### 10.1 — Resubmit the 6 prod sbatches with all today's fixes

PASS gate: Plan B verification rc=0 for all 3 (m5_1, m5_5, m5_6) with ckpts + WANDB + at least step_2 HF upload.

```bash
# On ALICE, from /zfsstore/user/s4374886/omega/reason_over_search-m5:

# 1. Scancel the 6 old prod sbatches (they hold the pre-fix script snapshot)
scancel 2266965 2266966 2266968 2266996 2267007 2267008

# 2. Resubmit all 6 with the fixed scripts (SLURM snapshots the script at submit time)
sbatch training_m5_1/scripts/sbatch_m5_1.sh           # M5.1 1x
sbatch training_m5_1/scripts/sbatch_m5_1_2xa100.sh    # M5.1 2x
sbatch training_m5_5/scripts/sbatch_m5_5.sh           # M5.5 1x
sbatch training_m5_5/scripts/sbatch_m5_5_2xa100.sh    # M5.5 2x
sbatch training_m5_6/scripts/sbatch_m5_6.sh           # M5.6 1x
sbatch training_m5_6/scripts/sbatch_m5_6_2xa100.sh    # M5.6 2x

# 3. Capture the new JobIDs + StartTimes for the prod tracker
squeue -u $USER -o "%.10i %.20j %.2t %.20S %R" > /zfsstore/user/s4374886/omega/post_resubmit_queue.txt
```

### 10.2 — Tomorrow's autonomous verifications (no action needed)

These run themselves while you sleep. Check results when they finish.

| Time (CEST) | Job | Output |
|---|---|---|
| Wed 08:23 | sbatch 2274508 (1x A100 + 240g + 8 retrievers full shape) | `/zfsstore/user/s4374886/omega/smoke_verify/` |
| Wed 13:50 | sbatch 2275683 (2x A100 + 240g, 3 smoke_2xa100 + stage 4 prod 1-step OOM check) | `/zfsstore/user/s4374886/omega/smoke_verify_2xa100/` |
| Wed 17:50 | srun 2275986 (interactive 2x A100, 4h walltime) | manual poke session |

After Wed 13:50 verification completes:
- If ALL pass → 2x prod sbatches good to start (M5.1 2x lands May 15)
- If FAIL → scancel + retune the 2x prod sbatches BEFORE May 15. There's 2 days of buffer.

### 10.3 — Apply HF watcher container-lifetime patch (before May 15)

Symptom (captured in Plan B m5_1): `INFO: Terminating squashfuse_ll after timeout` after the watcher uploads its first checkpoint; subsequent uploads fail with `Transport endpoint is not connected`. In prod (12 saves over 25 days at save_period=50) only step_50 might upload before the container dies.

Two candidate fixes (pick one before May 15):

1. **Run watcher OUTSIDE apptainer**: install `huggingface_hub` into a host-side python venv (eg `/zfsstore/user/s4374886/omega/hf_uploader_venv/`), invoke directly from sbatch instead of via apptainer exec. Decouples the watcher's lifetime from the training apptainer.
2. **Heartbeat the SIF mount**: have the watcher periodically `touch` a file inside `/workspace/reason_over_search/` to keep the mount referenced.

Option (1) is cleaner. Add to the post-resubmit todo.

### 10.4 — Cleanup runbook (after the 6 prod sbatches are resubmitted)

Run after step 10.1 completes successfully (the new prod sbatches are queued with new JobIDs).

```bash
# Optional: scancel the now-redundant smoke sbatches that don't add new info
# (Keep them if you want extra verification data; cancel to free queue slots)
scancel 2273476           # very old gpu-short m5_1 smoke (legacy)
# Keep 2274508 (1x full shape verify, useful tomorrow)
# Keep 2275683 (2x verify with stage 4, useful tomorrow)
# Keep 2275986 (2x interactive srun, useful tomorrow)
# Keep 2266786 (current srun, in use)

# Remove smoke artifacts from /zfsstore working dir
rm -rf /zfsstore/user/s4374886/omega/smoke_inrun
rm -rf /zfsstore/user/s4374886/omega/smoke_verify_smallmem
rm -rf /zfsstore/user/s4374886/omega/smoke_verify         # for sbatch 2274508 (created when it runs)
rm -rf /zfsstore/user/s4374886/omega/smoke_verify_2xa100  # for sbatch 2275683
rm /zfsstore/user/s4374886/omega/m5_smoke_seq.sh
rm /zfsstore/user/s4374886/omega/m5_verify_seq.sh
rm /zfsstore/user/s4374886/omega/m5_verify_seq_smallmem.sh
rm /zfsstore/user/s4374886/omega/m5_6_retry.sh
rm /zfsstore/user/s4374886/omega/*.master.log
rm /zfsstore/user/s4374886/omega/v2_venv_extract_sbatch.sh
rm /zfsstore/user/s4374886/omega/v2_venv_download.log
rm /zfsstore/user/s4374886/omega/v2_venv_extract.log
rm /zfsstore/user/s4374886/omega/v2_venv_extract_*.out
rm /zfsstore/user/s4374886/omega/v2_venv_extract_*.err
# Keep m5_verify_seq_2xa100.sh + m5_2xverify_sbatch.sh because sbatch 2275683 references them

# Remove smoke ckpt dirs (only step_2/step_4 of 4-step smokes; not from prod runs)
rm -rf /zfsstore/user/s4374886/omega/reason_over_search-m5/results/grpo/m5_smoke
rm -rf /zfsstore/user/s4374886/omega/reason_over_search-m5/results/grpo_m5_5/m5_smoke
rm -rf /zfsstore/user/s4374886/omega/reason_over_search-m5/results/grpo_m5_6/m5_smoke
rm -rf /zfsstore/user/s4374886/omega/reason_over_search-m5/results/grpo*/m5_smoke_2xa100

# Smoke training/retriever logs in repo logs/ (gitignored, but cleaner to remove)
rm /zfsstore/user/s4374886/omega/reason_over_search-m5/logs/m5_*_2266786_*

# Optional: remove the smoke-validation HF Hub repos
# (since they're 6.36 GB each, you may want to keep step_2 from each experiment
#  as evidence of the upload working; or delete them. Do it from a python repl
#  with the HF API + your token.)
# Repos created today:
#   pantomiman/qwen3.5-0.8b-grpo-musique-m5_1_smoke-seed42-step2
#   pantomiman/qwen3.5-0.8b-grpo-musique-m5_5_smoke-seed42-step2
#   pantomiman/qwen3.5-0.8b-grpo-musique-m5_6_smoke-seed42-step2
```

### 10.5 — Commit + push the final clean state

```bash
cd /Users/somedude/Documents/Obsidian/code/omega/reason_over_search
git status   # should be clean if no edits since the last commit (cleanup is data-only on ALICE)
# If anything dirty, commit + push
```

## 12. Vast 2x A100 verification session (2026-05-12 evening)

User rented Vast `23.127.144.217:12802` (2× A100-SXM4-80GB NVLink, Docker overlay) to verify the 2xa100 path before ALICE's prod sbatches fire May 14-19. Stayed on `experiment_1_alice` branch throughout. Session summary:

### 12.1 — TP=2 FSDP `reset_sharded_param` crash on vocab-parallel embed/lm_head

Initial 2xa100 prod 1-step OOM check crashed during the first `model.to("cpu")` call in `offload_after_refit`. Traceback:

```
File ".../torch/distributed/fsdp/_fully_shard/_fsdp_param.py:560 in _reset_sharded_param"
RuntimeError: a Tensor with 124160 elements cannot be converted to Scalar
```

Length 248320 vs 124160 → Qwen3.5-0.8B vocab is 248320, split across 2 ranks = 124160 per shard. Same bug class as `pytorch/pytorch#136228`. NeMo-Automodel doesn't have a built-in parallel plan for `Qwen3_5ForConditionalGeneration`, so its heuristic auto-applies vocab-parallel sharding to `embed_tokens` + `lm_head` → triggers the FSDP narrow bug.

### 12.2 — Fix: `custom_parallel_plan` excluding embed/lm_head (Option A, commits ca5386e + 4aa7168)

New file `training_m5_X/src/parallel_plan_qwen35.py` (3 copies, identical 35 lines):

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
def custom_parallel_plan():
    return {
        "model.layers.*.self_attn.q_proj":    ColwiseParallel(),
        "model.layers.*.self_attn.k_proj":    ColwiseParallel(),
        "model.layers.*.self_attn.v_proj":    ColwiseParallel(),
        "model.layers.*.self_attn.qkv_proj":  ColwiseParallel(),
        "model.layers.*.self_attn.o_proj":    RowwiseParallel(),
        "model.layers.*.mlp.gate_up_proj":    ColwiseParallel(),
        "model.layers.*.mlp.up_proj":         ColwiseParallel(),
        "model.layers.*.mlp.gate_proj":       ColwiseParallel(),
        "model.layers.*.mlp.down_proj":       RowwiseParallel(),
        # NO embed_tokens, NO lm_head -> replicated (avoids FSDP narrow bug)
    }
```

Wired into all 6 × 2xa100 yamls:
```yaml
dtensor_cfg:
  tensor_parallel_size: 2
  custom_parallel_plan: training_m5_X.src.parallel_plan_qwen35.custom_parallel_plan
```

Side-effect: embed + lm_head replicated across 2 GPUs costs ~2× 1 GB extra memory, trivial vs the activation savings from TP=2.

### 12.3 — Pre-existing bug: duplicate `cluster.num_nodes` in 3 prod yamls

While porting Option A, surfaced that `m5_1_research_paper_2xa100.yaml`, `m5_5_research_paper_2xa100.yaml`, `m5_6_research_paper_2xa100.yaml` all had `num_nodes: 1` declared TWICE in `cluster:` block. YAML parse silently picked the last; both load successfully but fragile. Fixed via sed; would have crashed any 3 ALICE 2x prod sbatches at config-load if a sibling change had reordered keys.

### 12.4 — `offload_after_refit` gated on `cpu_offload` (commit ca5386e)

Patch in `training/nemo_rl/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`:

```python
def offload_after_refit(self) -> None:
    # TP_OFFLOAD_GUARD_2026_05_12: gate on cpu_offload; FSDP reset_sharded_param
    # crashes on vocab-parallel embed/lm_head at TP>1 when calling model.to("cpu") here.
    if self.cpu_offload:
        self.model = self.move_to_cpu(self.model)
```

Default config has `cpu_offload: false` so this branch is skipped. Defensive belt-and-suspenders alongside §12.2.

### 12.5 — `micro=2` fragmentation OOM at step 3 (UNRESOLVED, micro=1 stays the default)

Three Vast attempts to enable `train_micro_batch_size=2 + logprob_batch_size=2`:

| Attempt | vLLM `gpu_memory_utilization` | `PYTORCH_CUDA_ALLOC_CONF` | Outcome |
|---|---|---|---|
| 1 | 0.7 | (default) | OOM at step 3 generation: tried 14.5 GB, 12-14 GB free |
| 2 | 0.55 | (default) | OOM at step 3 generation: saved 2 GB total, still failed |
| 3 | 0.7 | `max_split_size_mb:64` | No OOM in 1h 06 min; reached step 2 training; **but throughput collapsed** — step 1 generation 7.5 min (normal), step 1 training 32+ min (vs ~2 min historically with micro=1) → ~30× slowdown |

`max_split_size_mb:64` "works" in the sense of past-the-failure-point, but at a wall-clock penalty that makes it strictly worse than micro=1. Killed at 1h 19 min.

**Verdict**: ship **micro=1 + default allocator** to ALICE. The fragmentation pattern at micro=2 is the documented `log_softmax` allocator scar from the M5.2 v7 postmortem (§3 of `RESULTS_SMOKE_m5.md`). The unblocking lever is to patch `nemo_rl/distributed/model_utils.py:1378` to chunk the logits→fp32 cast (deferred to M5.3); allocator tweaks cannot work around it.

### 12.6 — Verification matrix outcome on Vast (SXM4 NVLink topology)

| Test | Outcome |
|---|---|
| 3× smoke_2xa100 (m5_1, m5_5, m5_6) end-to-end with Option A | PASSED, ckpts saved at step_2 + step_4 |
| 2xa100 prod 1-step OOM check at micro=1 | PASSED |
| 2xa100 prod 5-step OOM check at micro=2 | FAILED (§12.5) |

Important caveat: ALICE A100s are **PCIe**, not SXM4. NVLink-vs-PCIe inter-GPU bandwidth differs ~4×. The Option A custom_parallel_plan fix is correct on either topology (it's a code-path fix, not a bandwidth fix), but step-time will be slower on ALICE PCIe. The 2x verify on ALICE (sbatch 2275683 + srun 2275986, 2026-05-13) is the actual production-topology validation.

### 12.7 — Commits landed on `experiment_1_alice`

| Commit | Subject |
|---|---|
| `0041727` | feat(training): Option B fallback (FSDP-only, TP=1) for 2xa100 |
| `590b4bd` | fix(configs): remove duplicate `cluster.num_nodes` in 3 prod 2xa100 yamls |
| `ca5386e` | feat(training): Option A — custom_parallel_plan for Qwen3.5 + offload gating |
| `4aa7168` | docs(m5): record Vast 2xa100 verification + Option A wiring |

All pushed to `origin/experiment_1_alice`. ALICE working dir at `/zfsstore/user/s4374886/omega/reason_over_search-m5` pulled cleanly (was behind 4 commits; fast-forward).

## 13. ALICE 2274508 cold-start surprise + retriever boot bottleneck

### 13.1 — `m5_verify_seq.sh` recreated for 1xa100 path

Found missing on ALICE (referenced by the existing 1xa100 sbatch but had been cleaned in §10.4). Recreated as a 1xa100 mirror of `m5_verify_seq_2xa100.sh`: stages 1-3 run `bash sbatch_m5_X.sh --mode smoke` for m5_1, m5_5, m5_6 (4 steps each); stage 4 runs `bash sbatch_m5_1.sh --mode prod` with `save_period=5 grpo.max_num_steps=5 checkpointing.enabled=false train_micro_batch_size=1` for 5-step prod OOM check at the 1xa100 shape. Bumped stage 4 from 1 → 5 steps mid-flight by editing the on-disk script while sbatch 2274508 was running (the frozen sbatch reads this file fresh each stage).

### 13.2 — Retriever cold-boot measured at 2157s (36 min) on cold zfsstore

`m5_1_2274508_retriever.log` shows 36 min from `apptainer exec ... retriever_serving.py` to first `/health` 200. 8 retriever workers × E5-base-v2 model load (199 weight-shard loop) + 16 GB IVF-SQ8 FAISS index mmap, all cold-read from zfsstore at ~40 MB/s. Consistent with the conservative 5400s timeout in `sbatch_m5_1*.sh`.

This is 15% of every 4h verify-walltime budget burned on retriever boot. For long-haul prod sbatches (7 day walltime) it's a one-time cost, acceptable. For verifies, it's expensive.

### 13.3 — Training Python process stuck in `Dl` for 1h+ after retriever healthy

After retriever `/health` at 22:36 CEST, `run_grpo.py` (PID 1134955) launched. As of 23:43 CEST:

- Wall: 1h 07 min
- CPU: **23 seconds total**
- State: `Dl` (uninterruptible disk-wait, multi-threaded)
- GPU 0: 0% util, 2.5 GB used (retriever residue only)
- Retriever served 0 POST /retrieve requests (training never reached first rollout)

48 min of <0.5% CPU is not normal cold-start I/O. Most likely culprit: apptainer's squashfuse_ll layer reading the SIF rootfs over zfsstore is serialised on a single cold-read path (Python stdlib + torch + nemo_rl + vllm imports all hit it). HF cache + Qwen3.5-0.8B weights compound the I/O wait.

**This is the first time we've actually let an ALICE training Python come up end-to-end today**. The prior "successful" sruns (2266786 m5_smoke_alloc, 2266839 m5_smoke_mig) were `srun` allocations that ran `m5_verify_seq.sh` but were cancelled at 10:46 CEST before the training step started. So the cold-zfsstore + apptainer-FUSE Python startup cost is a previously-unmeasured failure mode that 2274508 surfaced.

### 13.4 — Mitigation plan for 2026-05-13 2x verify + future ALICE runs

User constraint (2026-05-12 23:14 CEST): "we need 8 [retrievers] for training at least, so don't dial it back" → can't reduce `RETRIEVER_NUM_WORKERS` below 8.

Three levers to evaluate during the 2x A100 srun 2275986 (2026-05-13 06:55 CEST):

1. **Warm-cache test**: tonight's 2274508 reads have warmed the zfsstore page cache. If 2275683 (02:51 CEST) and 2275986 (06:55 CEST) land on node870 (or any node sharing zfsstore cache), retriever cold-boot should drop from 36 min to 5-10 min. First diagnostic check.
2. **Pre-warm at sbatch start**: explicit `vmtouch -t <index>` + `cat venv-files > /dev/null` before the retriever launch. Burns ~5 min of compute time to force pages resident; saves ~25 min on first `/health`. Tested via the new sbatch 2293032 (`m5_1_smoke_check_sbatch.sh`, submitted 2026-05-13 00:13 CEST, queues at end of priority).
3. **Stage to node-local scratch** (deferred): copy index + encoder weights + critical Python stdlib paths to `$TMPDIR` (node-local NVMe) before retriever start. Cleanest fix but needs `$TMPDIR` capacity check + script rewiring.

### 13.5 — sbatch 2293032 (diagnostic 1xa100 smoke check)

Submitted at 00:13 CEST (after user request "submit an srun again for 1xa100 check"). Same shape as 2274508 but ONLY runs `bash sbatch_m5_1.sh --mode smoke` (no m5_5, m5_6, no prod stage), and runs a pre-warm block of `vmtouch -t` on the index + `find | cat > /dev/null` on hf_cache and venv before launching. Queued behind 7 priority-blocked jobs; SLURM-estimated start `2026-05-13T10:55`. Will run AFTER the prod 1xa100 sbatch (2276314 on 14 May), so won't pre-validate the prod sbatches; it WILL validate whether pre-warming kills the cold-start problem.

## 14. Tomorrow's plan (2026-05-13)

| CEST time | Job | Goal | What "OK" looks like |
|---|---|---|---|
| ~02:51 | sbatch 2275683 (2x verify, m5_verify_seq_2xa100.sh) | Confirm Option A on ALICE PCIe | 3 smoke_2xa100 ckpts saved + stage 4 5-step prod_2xa100 no-OOM at micro=1 |
| ~06:55 | srun 2275986 (interactive 2xA100, 4h walltime) | Manual poke session inside the allocation | Run `bash sbatch_m5_1_2xa100.sh --mode smoke_2xa100` inside the srun; confirm first training step time-to-completion |
| (later) | sbatch 2293032 (1xa100 smoke check w/ pre-warm) | Diagnose retriever cold-boot solution | Retriever `/health` in <10 min instead of 36; first training step prints within 5 min after that |

**Inside the 06:55 srun**, the operator job is:

1. Verify training step lands cleanly (no FSDP narrow crash, no fragmentation OOM)
2. Capture per-step wall-clock for steady-state projection (current projection: 8.8 min/step at micro=2, 15.8 min/step at micro=1; we'll be at micro=1)
3. Diagnose retriever cold-boot time on warm zfsstore (key data point for whether pre-warm helps)
4. If anything fails, scancel the relevant queued prod sbatches before they fire (M5.1 2x May 15, M5.6 2x May 18, M5.5 2x May 19) — same 1-3 day buffer pattern as §10.2

After 02:51 + 06:55 both pass, the 6 prod sbatches (2276314-2276319) are good to start firing May 14-19.

## 11. References

- [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) — research_v2's a1 / a2 postmortem (ckpt fix originates here, §7)
- [`../setup/HF_CHECKPOINT_UPLOAD.md`](../setup/HF_CHECKPOINT_UPLOAD.md) — HF watcher operator doc (research_v2)
- [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md) §3 — 2× A100 wall-clock projection
- [`MILESTONE_5.md`](MILESTONE_5.md) — M5 narrative + run history


## 12. Vast 2xA100-SXM4 verification + Option A upgrade (2026-05-12 PM)

Rented a Vast.ai 2× A100-SXM4-80GB instance to verify the 2xa100 prod
path in depth before the M5.1 2× sbatch starts 2026-05-15 21:05 CEST.
Vast iteration is minutes vs ALICE's queue-wait days, and the SXM4
hardware (NVLink) is a tighter test of TP=2 memory than ALICE's PCIe.

### 12.1 Bugs caught on Vast (in order)

1. **Missing `bin/python` symlinks in uv-managed venvs.** The Vast
   workspace was rsync'd from a prior instance; rsync didn't preserve
   the hardlinks to `/.uv/python_install/cpython-3.13.13-linux-x86_64-gnu/bin/python3`.
   Fix: re-symlink in each `training/nemo_rl/.venv/bin/`. Vast-specific.

2. **uv `--locked --extra automodel` tries to compile causal-conv1d
   + transformer-engine from git source on every run.** The Vast Docker
   image (`pantomiman/reason-over-search-v1:v2`) doesn't ship these
   pre-compiled like the ALICE SIF does. Fix: NeMo-RL's
   `_env_builder()` at `nemo_rl/utils/venvs.py:110` has an early-return
   path `if not force_rebuild and python_path.exists(): return ...`.
   The prebuilt sub-venv at
   `training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/bin/python` already
   exists; deleting any stale `STARTED_ENV_BUILDER` marker triggers
   the early-return and skips uv. Vast-specific (ALICE's SIF has the
   pre-compiled venvs baked in).

3. **`/tmp` (overlay, 35 GB) fills during HF model download + Ray
   temp.** `HF_HOME` and `RAY_TMPDIR` were unset, so HF downloaded
   Qwen3.5-0.8B (~1.7 GB) to `/root/.cache/huggingface` and Ray
   spilled to `/tmp/ray`. Both on the small overlay. Fix on Vast:
   add `HF_HOME=/workspace/hf_cache`, `RAY_TMPDIR=/workspace/ray_tmp`,
   `TRANSFORMERS_CACHE=/workspace/hf_cache/hub` to the experiment's
   `.env`. Pre-download model to `/workspace/hf_cache` once. Vast-only;
   ALICE's sbatch already binds these via apptainer.

4. **TP=2 FSDP `reset_sharded_param` crash** (`length=248320` vs local
   `124160`) on any `model.to(device)` call after refit (CPU offload
   AND CUDA onload). Triggered in `offload_after_refit()` and
   `prepare_for_lp_inference()` of `dtensor_policy_worker_v2.py`. Root
   cause: `Qwen3_5ForConditionalGeneration` is unknown to NeMo-Automodel,
   so heuristic plan wraps `embed_tokens`/`lm_head` with vocab-parallel
   sharding; FSDP doesn't reconcile the shard dim with full vocab on
   `narrow`. Same family of bug as
   [pytorch/pytorch#136228](https://github.com/pytorch/pytorch/issues/136228);
   no upstream fix in NeMo-RL HEAD.

5. **Pre-existing yaml parse error**: duplicate `cluster.num_nodes: 1`
   in all 3 prod_2xa100 yamls (lines 476–477). OmegaConf rejects with
   `ConstructorError: found duplicate key`. Would have crashed every
   ALICE 2x sbatch at config-load on 2026-05-15. Surfaced only because
   the 2xa100 path had never run end-to-end. Caught while running the
   prod 1-step OOM check on Vast.

### 12.2 Fixes committed to `experiment_1_alice`

| Commit | Scope | Net effect |
|---|---|---|
| **0041727** | Option B (`tensor_parallel_size: 1`) + `offload_after_refit` cpu_offload guard + `force_hf:true` (no-op) | Safe fallback; ~1.3-1.5× speedup vs 1× A100 |
| **590b4bd** | Remove duplicate `cluster.num_nodes` in 3 prod yamls | Unblocks ALICE 2x prod sbatches from immediate yaml-parse crash |
| **ca5386e** | Option A: `tensor_parallel_size: 2` + `custom_parallel_plan` + 3 plan files | Full ~2-2.5× speedup; the FSDP bug workaround |

`force_hf: true` was tried first based on a third-party suggestion and
DOES NOT help (only affects state-dict adapter, not the FSDP path);
kept in yaml as a no-op for traceability.

### 12.3 The custom parallel plan

`training_m5_X/src/parallel_plan_qwen35.py` (identical across 3
experiments per the M5 self-contained convention):

```python
{
    # Transformer-layer projections sharded across TP=2:
    "model.layers.*.self_attn.q_proj":    ColwiseParallel(),
    "model.layers.*.self_attn.k_proj":    ColwiseParallel(),
    "model.layers.*.self_attn.v_proj":    ColwiseParallel(),
    "model.layers.*.self_attn.qkv_proj":  ColwiseParallel(),
    "model.layers.*.self_attn.o_proj":    RowwiseParallel(),
    "model.layers.*.mlp.gate_up_proj":    ColwiseParallel(),
    "model.layers.*.mlp.up_proj":         ColwiseParallel(),
    "model.layers.*.mlp.gate_proj":       ColwiseParallel(),
    "model.layers.*.mlp.down_proj":       RowwiseParallel(),
    # NO embed_tokens, NO lm_head -> replicated (avoids the bug)
}
```

The Qwen3.5 `Qwen3_5GatedDeltaNet` (Mamba-style) layers don't match
any pattern and fall back to FSDP-only handling (no TP). They're a
minority of the 24 layers; the speedup loss is small.

### 12.4 Verification matrix (Vast, 2xA100-SXM4-80GB)

| Run | Mode | Steps | Outcome |
|---|---|---|---|
| Option B smoke (v6) | TP=1 dtensor / TP=2 vllm | 4 / 4 | PASS, ckpts step_2 + step_4 |
| Option A smoke (Hydra override, v7) | TP=2 + custom plan | 4 / 4 | PASS, ckpts saved |
| Option A smoke (yaml-only, v8) | TP=2 + custom plan via yaml | 4 / 4 | PASS, confirms ALICE invocation path |
| Option A prod 1-step micro=1 | TP=2, micro=1, seq=8192, 64 prompts × 5 gens | 1 / 1 | PASS, 52 min wall, peak ~74 GB on GPU 0 |
| Option A prod 5-step micro=2 | TP=2, micro=2, seq=8192 | 2 / 5 | **OOM at step 3** during training pass; fragmentation pattern (needed +14.56 GB, only 12.72 GB free) |

### 12.5 The micro_batch_size=2 question

Tempting throughput lever (~2× train pass), but step 3 OOMs from
fragmentation on a margin that fits steps 1 + 2 cleanly. Step 1 + 2
GPU 0 peaks at ~78 GB (within 80 GB), but PyTorch reserved-but-unallocated
fragments accumulate across the refit/offload/onload cycles. By step 3
the contiguous free is too small for the 14.56 GB activation peak.

`PYTORCH_ALLOC_CONF=expandable_segments:True` would mitigate but
**breaks NeMo-RL's CUDA-IPC weight sharing** between policy and vLLM
worker (`pidfd_getfd: Operation not permitted` — known issue,
documented in `training_m5_X/scripts/run.sh` comment block).

Conclusion: `train_micro_batch_size: 1` stays for now. micro=2 is a
follow-up that may be unlocked with activation-checkpointing or
optimizer CPU offload (separate experiment).

### 12.6 What stays on Vast, doesn't go to ALICE

- `.env` additions (`HF_HOME`, `RAY_TMPDIR`, `TRANSFORMERS_CACHE`):
  ALICE's sbatch already provides these via apptainer bind
  (`HF_HOME_HOST`, `RAY_SCRATCH_HOST`)
- `bin/python` symlink fix: ALICE's SIF has venvs baked in correctly
- `/tmp` cleanup: ALICE uses zfsstore for working dirs

### 12.7 Tomorrow's ALICE verifications

Auto-verifications fire on existing queued sbatches; they pull
`experiment_1_alice` HEAD (now `ca5386e`) at job-start:
- **sbatch 2274508** ~21:32 CEST: 1xA100 full-shape verify (unaffected by Option A)
- **sbatch 2275683** ~23:24 CEST: 2xA100 verify with stage 4 prod_2xa100
  1-step OOM check — this is the ALICE-side validation of Option A
- **srun 2275986** ~Thu 01:35 CEST: interactive 2xA100, 4h walltime

If any of these surface a problem specific to ALICE's PCIe topology
(vs Vast's NVLink) or apptainer environment, fix before May 15 21:05.

### 12.8 Files committed today (Vast tuning)

```
training_m5_1/configs/m5_1_research_paper_2xa100.yaml   # TP 1 -> 2, custom plan ref
training_m5_1/configs/m5_smoke_2xa100.yaml              # same
training_m5_5/configs/m5_5_research_paper_2xa100.yaml   # same + dup num_nodes removed
training_m5_5/configs/m5_smoke_2xa100.yaml              # same
training_m5_6/configs/m5_6_research_paper_2xa100.yaml   # same + dup num_nodes removed
training_m5_6/configs/m5_smoke_2xa100.yaml              # same
training_m5_1/src/parallel_plan_qwen35.py               # NEW (the plan)
training_m5_5/src/parallel_plan_qwen35.py               # NEW
training_m5_6/src/parallel_plan_qwen35.py               # NEW
training/nemo_rl/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py  # cpu_offload guard
```
