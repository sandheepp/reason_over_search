---
title: M5 2xA100 verify runbook — operator steps inside srun 2275986 (2026-05-13)
tags: [milestone-5, alice, sbatch, runbook, 2xa100]
source: live planning 2026-05-12 23:55 CEST (experiment_1_alice branch)
created: 2026-05-13
updated: 2026-05-13
---

# M5 2xA100 verification runbook — 2026-05-13 inside srun 2275986

Operator checklist for the **interactive 2xA100 srun starting 06:55 CEST 2026-05-13** (jobid 2275986, 4h walltime). The 02:52 CEST sbatch 2275683 runs first and autonomously; this srun is the interactive follow-up + the longer prod-shape test the autonomous sbatch doesn't cover.

## 0. Goal (what "verified" means)

Block the 6 queued production sbatches (2276314-2276319, fire May 14-19) from running if any of the below fails. Pass criteria:

1. **No FSDP `_reset_sharded_param` crash** on any of m5_1 / m5_5 / m5_6 2xa100 smoke at TP=2.
2. **No CUDA OOM** at any step of any smoke or the 5-10 step prod test.
3. **Per-step wall-clock in expected range** (8-18 min/step at 2xa100 micro=1, vs Vast SXM4 NVLink measured 7.5 min for step 1 generation; ALICE PCIe will be slower).
4. **Checkpoint saves succeed** (smoke ckpt at step 2 + step 4; prod ckpt at step 5).
5. **Retriever cold-boot ≤15 min on warm zfsstore** (32+ min would mean cold-start fix didn't help).

Each check below has a "what to do if it fails" branch.

## 1. Pre-srun (06:30-06:55 CEST) — read autonomous 2x verify output first

The autonomous sbatch 2275683 (`m5_verify_seq_2xa100.sh`) finished by 06:55 (started 02:52, 4h walltime). Read its output before entering the srun so we know what was already verified.

```bash
# From your laptop:
ssh alice
ls -lt /zfsstore/user/s4374886/omega/reason_over_search-m5/logs/ | head -10

# Find the verify dir (timestamp 20260513T0052Z-ish)
LOGDIR=$(ls -dt /zfsstore/user/s4374886/omega/reason_over_search-m5/logs/verify_2xa100_* | head -1)
echo "$LOGDIR"

# Smoke logs — look for PASS/FAIL markers
grep -E '^---.*CHECKS|PASS|FAIL' "$LOGDIR"/m5_*_smoke_2xa100.log
grep -E '^---.*CHECKS|PASS|FAIL' "$LOGDIR"/m5_1_prod_2xa100_1step.log

# Master summary
tail -60 /zfsstore/user/s4374886/omega/m5_2xverify_full_*.out
```

**Decision branches:**

| 2275683 outcome | Action in srun |
|---|---|
| All 3 smokes PASS + prod 5-step PASS | Skip §2 (smokes already done); jump to §3 (longer prod test for more data) |
| Smokes PASS, prod fails | Investigate prod failure first (§4); skip smokes in srun |
| Any smoke FAILS | Re-run the failing smoke inside the srun for live debug (§2) |

## 2. Inside the srun (06:55 CEST onward)

### 2.1 — Attach + sanity

```bash
ssh alice
srun --jobid=2275986 --pty bash

# now inside the allocation
hostname              # capture which node (node870? 875? matters for warm-cache)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# expect: 0, NVIDIA A100 80GB PCIe, 81920 MiB
#         1, NVIDIA A100 80GB PCIe, 81920 MiB

# Confirm zfsstore warm cache state (from 2274508 + 2275683)
free -g | head -2
# look at "buff/cache" column — should be tens of GB if cache is warm
```

### 2.2 — m5_1 smoke at 2xa100 (target: ~15 min wall)

```bash
cd /zfsstore/user/s4374886/omega/reason_over_search-m5

# Clean prior ckpt
rm -rf results/grpo_2xa100/m5_smoke_2xa100/seed42

# Launch (the script handles retriever boot + apptainer + training)
TS=$(date -u +%Y%m%dT%H%MZ)
LOGFILE=logs/m5_1_smoke_2xa100_srun_${TS}.log
bash training_m5_1/scripts/sbatch_m5_1_2xa100.sh --mode smoke_2xa100 2>&1 | tee "$LOGFILE"

# WHILE IT RUNS — in a separate shell on the login node:
ssh alice
ssh nodeXXX 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'
# repeat every 30s to track step progression

# AFTER COMPLETION — verify ckpt saves
ls -lh results/grpo_2xa100/m5_smoke_2xa100/seed42/step_2/policy/weights/
ls -lh results/grpo_2xa100/m5_smoke_2xa100/seed42/step_4/policy/weights/
# expect ~6.36 GB each (Qwen3.5-0.8B in bf16)

# Capture metrics
grep -E 'GPU Memory|step |reward|OutOfMemoryError|Traceback|RuntimeError' "$LOGFILE" | tail -40
```

**Pass criteria:**
- rc=0
- `Retriever healthy (waited N s)` — capture N (target ≤900s if warm cache)
- step_2 + step_4 ckpts saved at ~6.36 GB each
- No `RuntimeError: a Tensor with 124160 elements`
- No `OutOfMemoryError`

**If fails:**
- `_reset_sharded_param` crash → Option A custom_parallel_plan didn't load. Check `dtensor_cfg.custom_parallel_plan` in `training_m5_1/configs/m5_smoke_2xa100.yaml`; check `training_m5_1/src/parallel_plan_qwen35.py` exists.
- OOM → drop `vllm_cfg.gpu_memory_utilization` from 0.7 to 0.6 in `m5_smoke_2xa100.yaml`; re-run.
- Retriever timeout → check `m5_1_<JOBID>_retriever.log` for which worker stalled.

### 2.3 — m5_5 smoke at 2xa100 (target: ~15 min)

Same as §2.2 but `training_m5_5/scripts/sbatch_m5_5_2xa100.sh`. M5.5 uses a different system prompt + reward variant; the FSDP/TP-2 surface is identical so it should pass cleanly if m5_1 passed.

```bash
cd /zfsstore/user/s4374886/omega/reason_over_search-m5
rm -rf results/grpo_m5_5_2xa100/m5_smoke_2xa100/seed42
TS=$(date -u +%Y%m%dT%H%MZ)
LOGFILE=logs/m5_5_smoke_2xa100_srun_${TS}.log
bash training_m5_5/scripts/sbatch_m5_5_2xa100.sh --mode smoke_2xa100 2>&1 | tee "$LOGFILE"
```

Pass criteria + failure branches: same as §2.2.

### 2.4 — m5_6 smoke at 2xa100 (target: ~15 min)

```bash
cd /zfsstore/user/s4374886/omega/reason_over_search-m5
rm -rf results/grpo_m5_6_2xa100/m5_smoke_2xa100/seed42
TS=$(date -u +%Y%m%dT%H%MZ)
LOGFILE=logs/m5_6_smoke_2xa100_srun_${TS}.log
bash training_m5_6/scripts/sbatch_m5_6_2xa100.sh --mode smoke_2xa100 2>&1 | tee "$LOGFILE"
```

Pass criteria + failure branches: same as §2.2.

## 3. Prod-shape 5-10 step test (target: 60-100 min)

The smoke shape (20 traj/step, seq=2048) doesn't exercise the prod-shape OOM surface. This stage runs m5_1 prod_2xa100 at micro=1 for 5-10 steps to verify activation memory stays under the 80GB-per-GPU budget at the real workload.

```bash
cd /zfsstore/user/s4374886/omega/reason_over_search-m5

# clean prior partial run
rm -rf results/grpo_2xa100_prod_test/m5_prod_2xa100/seed42

# Launch with overrides: 10 steps, no ckpt save (faster), micro=1
TS=$(date -u +%Y%m%dT%H%MZ)
LOGFILE=logs/m5_1_prod_2xa100_10step_srun_${TS}.log
CHECKPOINT_DIR_BASE=results/grpo_2xa100_prod_test \
bash training_m5_1/scripts/sbatch_m5_1_2xa100.sh --mode prod_2xa100 \
  -- grpo.max_num_steps=10 checkpointing.enabled=false \
  2>&1 | tee "$LOGFILE"
```

**Captures (write these to a follow-up commit):**

```bash
# Per-step timing
grep -E 'step.*completed|total step time' "$LOGFILE" | head -20

# Max GPU memory across the run
grep -E 'GPU Memory after|memory.used' "$LOGFILE" | sort -u | tail -10

# Any near-OOM allocator warnings
grep -E 'CUDA out of memory|memory fragmentation|nearing.*memory' "$LOGFILE"
```

**Pass criteria:**
- rc=0
- 10 steps completed
- No CUDA OOM at any step
- Steady-state step time ≤18 min (per HARDWARE_FIXES §7 micro=1 projection: 15.8 min/step → 7-day walltime fits)

**If fails:**
- OOM at step 3+ generation phase → fragmentation pattern same as Vast micro=2; **DON'T** try `max_split_size_mb:64` (30× slowdown). Lower vllm `gpu_memory_utilization` from 0.7 → 0.55 in `m5_1_research_paper_2xa100.yaml`; re-run.
- OOM at step 1 generation → activation budget too tight; reduce `max_total_sequence_length` from 8192 to 6144 OR reduce `num_prompts_per_step` from 64 to 32 (kills cost projection though).
- Step time >25 min/step → PCIe inter-GPU bandwidth bottleneck killing TP=2 speedup. Decision: ship prod at 1xA100 instead (May 14-15 1xA100 sbatches 2276314 already queued).

## 4. Decision matrix (end-of-srun)

| §2.2-2.4 smokes | §3 prod 10-step | Action on prod sbatches (2276314-2276319) |
|---|---|---|
| ALL PASS | PASS, step time ≤15 min | All 6 prod sbatches good to fire May 14-19 |
| ALL PASS | PASS, step time 15-20 min | 6 prod OK, but flag wall-clock budget in supervisor update |
| ALL PASS | PASS, step time >20 min | scancel 2xa100 prod sbatches (2276315, 2276317, 2276319); keep 1xa100 sbatches (2276314, 2276316, 2276318); 2xa100 doesn't pay off at PCIe |
| ALL PASS | OOM at step 3+ | Apply `gpu_memory_utilization: 0.55` to 3 prod 2xa100 yamls; re-test (use srun 2293289 at 12:55) |
| ALL PASS | OOM at step 1 | scancel 2xa100 prod sbatches; debug seq length or batch size |
| ANY smoke fails | n/a | scancel 2xa100 prod sbatches; debug per §2.2 fail branch |

## 5. Backup window (12:55 CEST srun 2293289, 1xA100)

If anything in §2 / §3 fails on the 2xA100 srun, the 1xA100 interactive srun (2293289, starts 12:55 CEST, 4h walltime) is the fallback debug window. Use it for:

- Iterating fixes to yamls/configs that the 06:55 srun surfaced
- Running 1xA100 smoke to validate the 1xA100 prod path (mirrors what 2293032 sbatch will autonomously verify at 10:55)
- Final retriever cold-boot diagnostics (does pre-warm at sbatch start cut boot time from 36 min to <10 min?)

Attach pattern same as §2.1: `srun --jobid=2293289 --pty bash`.

## 6. Files referenced

- `training_m5_1/scripts/sbatch_m5_1_2xa100.sh` (and m5_5, m5_6 siblings)
- `training_m5_1/configs/m5_smoke_2xa100.yaml` (and m5_5, m5_6 siblings)
- `training_m5_1/configs/m5_1_research_paper_2xa100.yaml` (and m5_5, m5_6 siblings)
- `training_m5_1/src/parallel_plan_qwen35.py` (Option A custom plan — must exist)
- `HARDWARE_FIXES_2026-05-12.md` §12-§14 (background)
- `/zfsstore/user/s4374886/omega/m5_verify_seq_2xa100.sh` (what 2275683 ran autonomously)
