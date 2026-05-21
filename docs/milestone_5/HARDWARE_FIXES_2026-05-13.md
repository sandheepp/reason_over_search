---
title: ALICE hardware fixes — 2026-05-13 overnight autonomous verification + bug catches
tags: [milestone-5, alice, sbatch, hardware, postmortem, overnight]
source: autonomous overnight session 2026-05-13 (experiment_1_alice branch)
created: 2026-05-13
updated: 2026-05-13
---

# Hardware fixes 2026-05-13 — overnight verification + bug catches

Continuation of [`HARDWARE_FIXES_2026-05-12.md`](HARDWARE_FIXES_2026-05-12.md). Autonomous session from 23:55 CEST 2026-05-12 through morning 2026-05-13, validating the M5 sbatches end-to-end on ALICE before the 6 queued prod sbatches (2276314-2276319) fire.

## 1. Timeline of the night

| Time (CEST) | Event |
|---|---|
| 22:00 (Wed) | 2274508 fires (1xA100 smoke verifier) on node870 |
| 22:00 (Wed) | Retriever begins cold-boot from zfsstore |
| 22:36 (Wed) | Retriever `/health` passes (waited 2157s = **36 min**) |
| 00:19 (Thu) | `run_grpo.py` Python clears `Dl` after **1h 43m** disk-wait on node870 |
| 00:21 (Thu) | **Ray init crash**: `Failed to start the dashboard` + `node timed out` |
| 00:30 (Thu) | Diagnosis: `include_dashboard=True` hardcoded in `virtual_cluster.py:170` |
| 00:35 (Thu) | Patch applied on ALICE + committed `bc49e0d`; pushed to git |
| 02:00 (Thu) | 2275683 fires (2xA100 autonomous smoke verifier) — fails in 1-5s |
| 02:05 (Thu) | Diagnosis: `SLURM_SUBMIT_DIR` path bug in m5_verify_seq_2xa100.sh; precheck looking for index in wrong directory |
| 02:10 (Thu) | Patch `unset SLURM_SUBMIT_DIR` in m5_verify_seq*.sh applied on ALICE |
| 02:28 (Thu) | m5_1 smoke_2xa100 launched in-srun (2275986 on node870) — retriever cold-boot 32 min, then Ray init crash AGAIN (despite dashboard patch — node870 is genuinely degraded) |
| 02:34 (Thu) | Launched m5_1 smoke 1xA100 in-srun (2293289 on node871) with pre-warm |
| 02:38 (Thu) | Retriever `/health` passes in **5 min 5 s** (warm cache + pre-warm dd) |
| 02:58 (Thu) | First training step lands; step 1 = 135.77s |
| 03:10 (Thu) | 2293289 m5_1 smoke PASS rc=0 — 4 steps + ckpts (135/87/75/88s) |
| 04:00 (Thu) | 2275986 walltime; 2293289 walltime |
| 06:02 (Thu) | 2306117 m5_2x_srun_2 RUNS on **node871** (--exclude=node870) |
| 06:15 (Thu) | m5_1 smoke_2xa100 launched in 2306117 |
| 06:30 (Thu) | m5_1 smoke_2xa100 PASS rc=0 in **14m 45s** (retriever 65s) |
| 06:51 (Thu) | m5_5 smoke_2xa100 PASS rc=0 in **18m 18s** |
| 07:25 (Thu) | m5_6 smoke_2xa100 PASS rc=0 in **12m 14s** |
| 07:32 (Thu) | m5_1 prod_2xa100 5-step OOM check launched |
| 08:00 (Thu) | Step 1 in training pass; GPU 0=43GB / 1=76GB |
| 09:03 (Thu) | Step 1 complete: 2580.96s = 43 min |
| 09:11 (Thu) | Step 2 complete: 2460.83s = 41 min |
| 09:11 (Thu) | **Step 3 generation PASSED — no OOM** (the Vast micro=2 failure point) |

## 2. Bugs caught + fixes applied (overnight)

| # | Bug | Symptom | Fix | Commit |
|---|---|---|---|---|
| 1 | Ray dashboard subprocess silently dies on ALICE apptainer SIF; main `ray.init()` then crashes with "No such file: dashboard.err" + nvidia_cutlass_dsl recursion error | 2274508 failed at 00:21 CEST after 1h 43m disk-wait + 2 min Ray init | `include_dashboard=False` in `training/nemo_rl/nemo_rl/distributed/virtual_cluster.py:170` | `bc49e0d` |
| 2 | `m5_verify_seq_2xa100.sh` inherits `SLURM_SUBMIT_DIR` from the wrapper sbatch; inner `sbatch_m5_X*.sh` resolves `REPO_ROOT` to wrong dir; precheck "Retriever index missing" exits in <1s | 2275683 all 4 stages bailed in 1-5s with "Retriever index missing: local_retriever/indexes/wiki18_..." | `unset SLURM_SUBMIT_DIR` at top of `m5_verify_seq*.sh` (both 1x + 2x versions) | (ALICE-side patch; not yet in git for the seq scripts since they live in `/zfsstore/.../omega/`, not in the repo) |
| 3 | `sbatch_m5_1.sh` (1xA100 entry script) doesn't accept `--` pass-through; can't pass Hydra overrides to it (only `sbatch_m5_1_2xa100.sh` had it from 2026-05-12 batch) | First prod 1xA100 test in 2293289 exited rc=2 with `unknown arg: --` | Added `--` case in arg parser + `EXTRA_OVERRIDES_STR` forwarding to `run.sh` invocation | (ALICE-side patch; needs commit to git) |
| 4 | **node870 specifically is degraded**: Python cold-start takes 1h 43m (vs 10-20 min on node871/872), then Ray init times out even with dashboard fix. Cause: unclear (possibly NIC/IPC issue, possibly leftover RAM occupation from prior failed jobs not cleaned by SLURM) | All 3 attempts on node870 (2274508, 2275986, 2275986-relaunch) hit either Ray timeout or never reached training | `--exclude=node870` in all new sbatches (cold-prod-shape, smoke v2, v2 prod) | (sbatch headers) |

## 3. Pre-warm pattern (36 min → 65s retriever boot)

Cold zfsstore retriever boot was **2157s = 36 min** on 2274508 (no pre-warm). After adding a pre-warm step before the apptainer exec that runs `retriever_serving.py`, boot dropped to **65-76 s** (a **~30× speedup**).

The pre-warm in the new sbatches:
```bash
# Force-load the FAISS IVF-SQ8 index into page cache (~16 GB)
dd if=local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index of=/dev/null bs=4M
# Force-load the E5-base-v2 encoder weights
find /zfsstore/user/s4374886/hf_cache/hub -maxdepth 5 -name '*e5-base*' -type d \
  | xargs -I{} find {} -type f | xargs -P 8 -I{} dd if={} of=/dev/null bs=1M
# Force-load the apptainer venv python binary (touches squashfuse rootfs)
cat training_m5_1/nemo_rl/.venv/bin/python > /dev/null
```

Page cache survives in node RAM for hours of subsequent reads. The pre-warm is only useful if:
- The zfsstore page cache is empty/cold (fresh node, or evicted under memory pressure)
- Node has enough RAM (~32 GB minimum for index + encoder + venv chunks)

Result on node871: retriever boot dropped from cold ~30 min to warm-with-prewarm 65-296 s across sequential runs.

## 4. Verification matrix — what passed where

| Test | Where | Result |
|---|---|---|
| m5_1 smoke_2xa100 | Vast 2x A100 SXM4 NVLink (2026-05-12) | ✅ PASS |
| m5_5 smoke_2xa100 | Vast 2x A100 SXM4 | ✅ PASS |
| m5_6 smoke_2xa100 | Vast 2x A100 SXM4 | ✅ PASS |
| prod_2xa100 1-step OOM | Vast 2x A100 SXM4 micro=1 | ✅ PASS |
| prod_2xa100 5-step OOM | Vast 2x A100 SXM4 **micro=2** | ❌ OOM at step 3 gen (3 attempts) |
| m5_1 smoke | ALICE node871 1xA100 (2293289) | ✅ PASS in 4 steps (135/87/75/88s) |
| m5_1 smoke_2xa100 | ALICE node871 2xA100 (2306117) | ✅ PASS in 14m 45s |
| m5_5 smoke_2xa100 | ALICE node871 2xA100 (2306117) | ✅ PASS in 18m 18s |
| m5_6 smoke_2xa100 | ALICE node871 2xA100 (2306117) | ✅ PASS in 12m 14s |
| prod_2xa100 5-step OOM | ALICE node871 2xA100 (2306117) **micro=1** | ✅ Step 3 generation PASSED (the Vast OOM point); step 1=43m, step 2=41m; step 3 training in progress at doc-write time |

Conclusion: **Option A `custom_parallel_plan` is validated on ALICE PCIe topology**. `micro=1` is OOM-safe at prod shape. `micro=2` cannot be unblocked without patching `nemo_rl/distributed/model_utils.py:1378` to chunk the logits→fp32 cast.

## 5. Safety-net sbatches submitted

Submitted overnight to provide redundant verification before the queued prod sbatches (2276314-2276319) fire:

| Job | Purpose | Partition | Walltime | Start (CEST) | Pre-warm | --exclude |
|---|---|---|---|---|---|---|
| **2312311** | cold prod-shape 2xA100 (3 steps, NO pre-warm) — simulates what queued prod will face | gpu-short | 4h | 2026-05-13 11:20 | no | node870 |
| **2312312** | cold prod-shape 1xA100 (3 steps, NO pre-warm) | gpu-short | 4h | 2026-05-13 19:25 | no | node870 |
| **2312326** | smoke verifier v2 2xA100 (autonomous, all 3 smokes + 1-step prod OOM) | gpu-short | 4h | 2026-05-13 15:25 | yes | node870 |
| **2312327** | smoke verifier v2 1xA100 (autonomous) | gpu-short | 4h | 2026-05-13 23:25 | yes | node870 |
| **2312328-2312333** | 6 backup v2 prod sbatches (m5_1/5/6 × 1x/2x); HF prefix v2 to avoid collision with v1 prods | gpu-a100-80g | 7d | Reserved-for-maintenance (waits for May 18 reservation to pass) | yes | node870 |

Purpose:
- **2312311 + 2312312** test the **cold** path (no pre-warm) — if these pass, the queued prod sbatches (which also lack pre-warm) will work cold
- **2312326 + 2312327** are autonomous validators that re-prove what was manually verified in 2306117 + 2293289
- **2312328-2312333** are safety-net backup prods if any of the original 6 fail; they have pre-warm + node870 exclusion + HF v2 namespace baked in

## 6. Original prod sbatches (2276314-2276319) — status

| JobID | Name | State | Reason | StartTime |
|---|---|---|---|---|
| 2276314 | m5_1_grpo_1xa100 | PENDING | Priority | Unknown |
| 2276315 | m5_1_grpo_2xa100 | PENDING | Priority | Unknown |
| 2276316 | m5_5_grpo | PENDING | Priority | Unknown |
| 2276317 | m5_5_grpo_2xa100 | PENDING | Priority | Unknown |
| 2276318 | m5_6_grpo | PENDING | Priority | Unknown |
| 2276319 | m5_6_grpo_2xa100 | PENDING | Priority | Unknown |

All 6 submitted 2026-05-12 12:37:34 CEST, priority scores 449111 (1x) / 449346 (2x), `StartTime=Unknown`. SLURM has not estimated firm start times. They are waiting for a contiguous 7-day window on `gpu-a100-80g` that doesn't get killed by the May 18 maintenance reservation (`root_22` on node[863-865, 867, 871-876, 885], 10:00-18:00 May 18). Could be days before they start.

**Risks if/when they fire:**

1. **node870 lottery**: original sbatches don't have `--exclude=node870`. If SLURM picks node870, they'll hit the same 1h 43m Python cold-start + Ray timeout we saw on 2274508.
2. **No pre-warm**: cold zfsstore retriever boot will take ~36 min (one-time cost, manageable for 7-day runs).
3. **Ray dashboard fix is in git**, will be picked up via the bind-mounted nemo_rl source.

Mitigations:
- Cold-prod-shape smokes (2312311 + 2312312) will tell us whether the cold path works
- If cold path works, originals are fine
- If cold path fails, scancel + use the v2 backups

## 7. Open items + next steps

1. **Wait for prod 5-step result** (in progress) — currently in step 3 training, walltime ends 10:15
2. **Commit ALICE-side patches to git**:
   - `m5_verify_seq*.sh` `unset SLURM_SUBMIT_DIR` fix (in `/zfsstore/.../omega/`, not in repo — copy to `training_m5_1/scripts/` if appropriate)
   - `sbatch_m5_1.sh` `--` pass-through patch (in `/zfsstore/.../reason_over_search-m5/training_m5_1/scripts/`)
3. **Wait for cold-prod-shape result** (2312311 at 11:20 CEST today)
4. **Document this file in `docs/log.md`**
5. **Wiki lint pass**

## 8. References

- [`HARDWARE_FIXES_2026-05-12.md`](HARDWARE_FIXES_2026-05-12.md) — yesterday's fixes (Option A, 11 bugs, dup yaml, etc)
- [`M5_2X_VERIFY_RUNBOOK_2026-05-13.md`](M5_2X_VERIFY_RUNBOOK_2026-05-13.md) — the runbook this session executed
- [`MILESTONE_5.md`](MILESTONE_5.md) — M5 narrative
- Git commit `bc49e0d` — Ray dashboard disable
