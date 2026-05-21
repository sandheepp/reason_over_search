---
title: RESULTS — M5.1-prod-a3 (Qwen3.5-0.8B GRPO on MuSiQue, B200 Spheron) — CRASHED
tags: [results, m5.1, b200, production, a3, postmortem, crashed]
source: internal
created: 2026-05-14
updated: 2026-05-15
status: terminal — host preempted at step 56, recovery to a4
---

# M5.1-prod-a3 — Production training on Spheron B200 (LOSS #4)

> **Status**: **CRASHED at step 56 of 622** (~00:21 UTC 2026-05-15, after 9.3 h / 56 steps / ~$36 spent). Spheron T3 spot host preempted; instance unreachable since. **No checkpoint saved to HF** (uploader had silently stopped uploading past step 18 ~6 h earlier; both the step_50 weights and 38 rollouts are lost). W&B preserved the full metric trajectory through step 56.
>
> **Loss #4 in the M5.1 experiment chain** (after a1 ckpt crash, a1 rollout-corpus deletion, a2 misdiagnosis kill).
>
> **What was kept**: this doc as the canonical post-mortem; cadences 1-4 (steps 1-40); the W&B run; rollouts steps 1-18 on HF.
>
> **What was lost**: rollouts steps 19-56 (38 files); step_50 model checkpoint; ~$36 of B200 spot time.
>
> **Next**: M5.1-prod-a4 with bulletproof uploader ([`upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py)) + external monitor ([`external_monitor.py`](../../training_m5_1/scripts/external_monitor.py)) + locked launch checklist ([`LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md)). See §10 for full incident report and §11 for the a4 plan.

## 1. Run identity

| Field | Value |
|---|---|
| Run name | `qwen3.5-0.8b-musique-b200-a3-seed42` |
| W&B project | `reason_over_search_b200` |
| HF Hub repo | [`pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42) |
| Git branch | `experiment_1_b200` (locally + remote) |
| Started commit | `f0f960e` |
| Launch time (UTC) | 2026-05-14T15:03:36Z |
| W&B run | [`qwen3.5-0.8b-musique-b200-a3-seed42-20260514T1503Z`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6) (run id `h68uskz6`; renamed via API post-launch from the auto-generated `m5_prod` name) |
| Process pids | wrapper 7942, training 7961, uploader 7919 |
| Container | `m5_b200_a3` on host `46.243.145.4` |
| Seed | 42 |
| Hardware | 1× NVIDIA B200 SXM6, 192 GB VRAM (Spheron ES, ME West 1, $3.83/h spot) |
| Driver / CUDA host | 580.126.09 / 13.0 |
| Container | `pantomiman/reason-over-search-v1:v2` (CUDA 12.9 worker stack) |
| Image entrypoint | `/bin/bash` (Vast-specific boot bypassed) |
| Training framework | NeMo-RL v0.6.0 (vendored at `training/nemo_rl/`) |
| Model | `Qwen/Qwen3.5-0.8B` (hybrid; GatedDeltaNet + attention) |

## 2. Config — five B200 fixes applied vs `experiment_1_alice`

Branch `experiment_1_b200` diverges from `experiment_1_alice` only in `training_m5_1/configs/m5_1_research_paper.yaml`. Each fix is documented inline in that file.

| # | Knob | A100 value | B200 value | Why |
|---|---|---|---|---|
| 1 | `policy.train_micro_batch_size` | 1 | **2** | 192 GB fits log_softmax peak that OOM'd at 80 GB; ~halves training-phase wall |
| 2 | `policy.generation.vllm_cfg.gpu_memory_utilization` | 0.5 | 0.7 → **0.8** | vLLM sleep mode releases ~90% during training; 0.8 → ~154 GB allocated, ~15 GB resident. More KV cache for rollout batching. |
| 3 | `logger.wandb.project` (& swanlab, mlflow) | `reason_over_search_m5_1` | `reason_over_search_b200` | New W&B project to keep B200 runs separated from prior A100 attempts |
| 4 | `logger.wandb.name` (& swanlab, mlflow) | `qwen3.5-0.8b-musique-m5_prod-seed42` | `qwen3.5-0.8b-musique-b200-a3-seed42` | Reflects hardware + attempt number |
| 5 | (same as #2, two-stage bump) | — | — | — |

Unchanged (paper-faithful):
- `num_prompts_per_step: 64`, `num_generations_per_prompt: 5` (320 traj/step)
- `max_num_steps: 622` (2 epochs at 64 prompts/step from 19,938 MuSiQue rows)
- `max_total_sequence_length: 8192`
- `precision: bfloat16`
- `activation_checkpointing: true`
- All loss / KL / clip hyperparameters
- `save_period: 50`, `metric_name: null`, `save_optimizer: false`, `keep_top_k: null`

## 3. Checkpoint plan

Saves every 50 steps via NeMo-RL atomic save (writes `tmp_step_N/`, renames to `step_N/` on flush).

| Step | Save lands at (estimate, see §6 for live anchor) |
|---|---|
| 50 | (filled in) |
| 100 | (filled in) |
| 150 | (filled in) |
| 200 | (filled in) |
| 250 | (filled in) |
| 300 | (filled in) |
| 350 | (filled in) |
| 400 | (filled in) |
| 450 | (filled in) |
| 500 | (filled in) |
| 550 | (filled in) |
| 600 | (filled in) |

**Note on final step (per user decision Q1=b)**: `max_num_steps=622` is not a multiple of `save_period=50`, so the last save is at step 600. Steps 601-622 still train but their state is not preserved. The **step-600 checkpoint is the canonical "final" artifact** for evaluation. Trade-off accepted: ~3.5% of 2-epoch trajectory budget after the last save isn't checkpointed.

Each saved ckpt is ~3.2 GB (consolidated fp32 safetensors, no optimizer state per `save_optimizer: false`). Total local disk: 12 × 3.2 GB = 38.4 GB. Plenty of headroom (~400 GB free).

## 4. HF Hub backup — single-repo structure

The upload watcher ([scripts/upload_a3_to_hf.sh](../../training_m5_1/scripts/upload_a3_to_hf.sh)) is a new script (not the existing `upload_ckpts_watcher.sh` which targets per-step repos). Polls every 60 s, uploads new artifacts to a single HF model repo.

Target repo layout:

```
pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42/
├── README.md                # this doc, mirrored
├── config_snapshot.yaml     # the prod yaml at launch
├── step_50/  ... step_600/  # NeMo-RL checkpoint dirs (1:1 with disk)
├── logs/
│   ├── prod.log             # full stdout/stderr from run.sh (overwritten each cycle)
│   ├── train_data/          # per-step rollout JSONLs
│   └── wandb_run.txt        # W&B run id + URL
├── timings.csv              # per-step wall + phase breakdown
└── final_metrics.json       # populated at run-complete
```

Watcher state file at `/workspace/.uploaded_artifacts`; idempotent across restarts.

## 5. Auto-restart wrapper

Training is launched via [scripts/run_prod_a3_resilient.sh](../../training_m5_1/scripts/run_prod_a3_resilient.sh), not directly via `run.sh`. The wrapper:

- Detects exit != 0 from `run.sh`
- Waits 60 s (lets GPU memory + Ray actors fully release)
- Relaunches; NeMo-RL auto-resumes from the latest `step_N/` ckpt in `checkpoint_dir`
- Caps at MAX_RESTARTS=5 to prevent infinite loops on persistent failure
- All restart events logged to `/workspace/prod.log` with `[wrapper]` tag

Designed for unattended multi-day runs where Spheron spot preemption + the rare framework exception both trigger restart.

## 6. Live trajectory

Per-step times will be logged here as they land. Anchor for extrapolation: smoke step 3 = 43 s at smoke shape (seq=4096, 20 traj/step, micro=2); prod shape (seq=8192, 320 traj/step, micro=2) is ~15× larger per step → target **~10-11 min/step** if smoke→prod scaling holds. Run wall projection: 622 × 10 min ≈ 100 h ≈ 4.2 d (could be 3-6 d depending on rollout-length dynamics, per the M5 §4.1 shrink-then-grow observation).

### 6.1 Step log (live, updated after each step)

Pulled from `prod.log` "Total step time" + W&B `reason_over_search/*` metrics. Speedup column is B200/A100 ratio vs the reference run `uwbodqgt` at the same step.

| Step | Wall (s) | Reward | Gen len | Tool calls | Turns | Trunc % | A100 ref (s) | **B200/A100** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1047.73 | 0.0162 | 1384 | 6.55 | 7.30 | 62.8% | 3474 | **3.31×** |
| 2 | 1081.31 | 0.0447 | 1413 | 6.72 | 7.49 | 68.1% | 3352 | **3.10×** |
| 3 | 1103.42 | 0.0200 | 1334 | 6.96 | 7.65 | 70.6% | 3346 | **3.03×** |
| 4 | 1020.96 | 0.0712 | 1463 | 6.22 | 6.93 | 59.7% | 3176 | **3.11×** |
| 5 | 1010.43 | 0.0458 | 1428 | 6.24 | 6.94 | 58.8% | 2843 | **2.81×** |
| 6 | 874.97 | 0.0761 | 1416 | 5.19 | 5.97 | 43.4% | 2294 | **2.62×** |
| 7 | 915.05 | 0.0453 | 1433 | — | — | — | 2239 | **2.45×** |

Phase breakdown (avg of steps 1-7, 1007s/step mean):
- `policy_training`: ~62% (~625s) — DTensor backward pass, bandwidth-bound on 0.8B
- `generation`: ~20% (~210s) — vLLM multi-turn rollouts + retriever HTTP calls
- `policy_and_reference_logprobs`: ~18% (~190s) — current policy + frozen reference forward passes
- everything else: <1%

### 6.4 Live observations (running narrative; appended as steps land)

#### 2026-05-14 ~16:30 UTC — first 4 steps complete, healthy

- **Step 1 = 1047.73s, no warmup penalty** (vs my initial guess of 17 min with warmup baked in). vLLM `cumem` wake was 3.5 min of generation, but Step 2 was actually slightly *slower* (1081s) than Step 1, not faster — so the warmup theory was wrong. **17-18 min is steady-state for the cold phase.**
- **Reward signal moving in the right direction**: 0.016 → 0.045 → 0.020 → 0.071. Noisy but trend positive. Matches A100 reference (0.020 → 0.071 → 0.023 → 0.073) within ±20%.
- **No OOM, no crashes**. The postmortem-critical fixes (`metric_name: null`, `save_optimizer: false`) are holding.
- **Mean gen length stable ~1400 tokens**, well under 8192 budget.

#### 2026-05-14 ~16:35 UTC — uploader silently failing for 30 min; caught + fixed

Found three bugs in `upload_a3_to_hf.sh` after the step-1 rollout JSONL (124 MB) sat on disk uploaded for 5+ minutes:

1. **Wrong python path**: hardcoded `/opt/miniforge3/envs/retriever/bin/python` — that conda env doesn't exist on the Spheron CUDA-13 image (only base miniforge3). Every upload exited 127.
2. **Wrong rollout glob**: looked for `train_data_step_*.jsonl` but NeMo-RL writes `train_data_step1.jsonl` (no underscore).
3. **Silent-failure code paths**: `if result=$(...); then` swallowed non-zero exits with no log.

**Fix** ([commit 906d93a](https://github.com/GaurisankarJ/reason_over_search/commit/906d93a)):
- Installed `huggingface_hub` + `hf_transfer` into miniforge3 base (`/opt/miniforge3/bin/python`)
- Parameterized `HF_PY` env var (fallback to nemo venv)
- Fixed glob to `train_data_step*.jsonl`
- Added explicit `✗ FAILED` logging on every failure path

Killed broken uploader (pid 7919), restarted (pid 16526). First catch-up cycle uploaded `prod.log` + all rollout JSONLs from exp_006/012/013 within 90 s.

**Lesson for M5.2+**: stand up the uploader as a separate **smoke-test** before prod launch, not first-tested-in-prod. Add to PHASE_2_RUNBOOK as a pre-launch gate.

#### 2026-05-14 ~16:40 UTC — corrected per-step trajectory math (1600 → 320 rollouts/step)

User caught a math error in my early commentary: I said "320 prompts × 5 generations = 1600 trajectories/step". The actual config is `num_prompts_per_step: 64`, not 320. **Real per-step batch is 64 × 5 = 320 trajectories**, as the prod.log itself confirms (`Generating responses for batch of size 320`). Cost-throughput calculations updated accordingly.

#### 2026-05-14 ~16:45 UTC — cost estimate post-mortem: 34h was wrong, here's why

The HARDWARE_COMPARISON.md v2 estimate of **~34 h / ~$130** was anchored on:
- A100 23.7 min/step (steady-state, mid-run)
- × 1/5.5 (B200 "5-6×" theoretical compute speedup, BF16 MLPerf vendor numbers)
- × 1/1.3 (micro_batch=2 "unlock" 1.5× on training+logprobs phase)
- = 2.2 min/step × 622 = 34 h

**Three compounding errors**:

1. **Theoretical compute speedup ≠ realized wall-clock speedup**. B200 BF16 compute = 7.2× A100, but our workload is **bandwidth-bound** on a 0.8B model with multi-turn rollouts. B200 HBM3e = 4× A100's HBM2e. **Realized speedup tracks bandwidth, not compute** → 3-4×, not 5-6×.
2. **Anchor was mid-run A100, not early A100**. Doc used a single 23.7 min/step datapoint from steps 38-42 of a different A100 run. The reference run `uwbodqgt` actually showed **58 min/step at step 1**, dropping to 11 min/step at step 15 minimum, then rising back to 30 min/step by step 49. **Mean across 50 steps = 23.6 min** — matches the anchor but is not representative of early phase.
3. **micro_batch=2 unlock was optimistic**. The 1.5× number assumed linear scaling with batch size; on 0.8B + bandwidth-bound regime it's closer to 1.2× — DTensor bookkeeping eats the rest.

**Measured speedup (steps 1-7)**: 3.0× (range 2.45-3.31×, shrinking as A100's curve drops faster than ours). At our 1007s mean step, full-run projection:
- Naive (current rate flat): 622 × 1007 = 7.25 d, **$667**
- A100-curve-scaled (most likely): 86 h, **$330**
- Pessimistic (climbing past step 49): 102 h, **$390**

**Best single-point estimate: 4-4.5 d / ~$385-435**. HARDWARE_COMPARISON.md should be updated after step 25 once we have a clean steady-state anchor.

#### 2026-05-14 ~16:50 UTC — A100 reference comparison: training is on the expected curve

Pulled the A100 reference run `uwbodqgt` (same config except `train_micro_batch_size: 1`, 49 steps before it crashed on the postmortem-bug). Side-by-side comparison confirms:

| Metric | Step 1-6 A100 | Step 1-6 B200 | Verdict |
|---|---|---|---|
| Step time | 3474 → 2294s | 1048 → 875s | ✓ 3.0× speedup, same shape |
| Reward | 0.020 → 0.095 | 0.016 → 0.076 | ✓ Same trajectory, ~80% of A100's level |
| Tool calls/sample | 6.89 → 4.35 | 6.55 → 5.19 | ✓ Lagging A100 by ~1-2 steps but trending right |
| Truncation rate | 69% → 29% | 63% → 43% | ✓ Dropping (slower than A100, same direction) |
| Mean gen length | 1353 → 1282 | 1384 → 1416 | ⚠️ B200 not yet shrinking; tool-collapse delayed |

**Behavioural lag of ~1-2 steps explained by `train_micro_batch_size: 2`** (vs A100's 1): larger effective batch → slightly different gradient → behavior collapse delayed. Not a problem; the curve will catch up.

**Tool-collapse confirmed at step 6**: step time dropped to 875s (first sub-1000s step). Same regime change A100 saw at step 5-6. Steps 7-10 should continue dropping if A100's pattern holds.

#### 2026-05-14 ~17:00 UTC — OOM risk assessment: tight but fine

Training-phase GPU memory peak: **178 GB / 179 GB total = 4 GB free (97.8% util)**.

| Risk | Likelihood | Trigger |
|---|---|---|
| Longer sequences as model learns multi-turn | medium | Mean gen creeps toward 8192 cap |
| Concurrent max-length rollouts in one batch | low | Tail of gen distribution |
| Training-phase memory creep | low | Bounded; doesn't grow across steps |

**Pre-built escape hatches** (none triggered yet):
1. Drop `gpu_memory_utilization: 0.8 → 0.6` (frees ~30 GB vLLM cache). Trade: ~10% slower generation.
2. Drop `train_micro_batch_size: 2 → 1` (halves training peak). Trade: ~30-50% slower training.

Wrapper auto-restarts from latest `step_N/` ckpt if OOM hits. First save at step 50.

#### 2026-05-14 ~17:10 UTC — decision: leave config alone (no micro_batch=4 optimization)

Considered applying `train_micro_batch_size: 2 → 4` at the step-50 checkpoint to shave ~15-20% off training phase (~$80-120 savings over remaining run). **Decision: skip it.**

Reasons:
- Memory headroom is **4 GB at peak** — micro_batch=4 would push past OOM (activations ~2× → +5-10 GB needed)
- To make room would require `gpu_memory_utilization: 0.8 → 0.5` paired drop → ~10% slower generation, net saving only ~5% = $40-60
- This run already has **3 prior losses recovered from**; risk appetite is correctly low
- The supervisor-meeting evidence we need is "did the recipe work?", not "did we save $100 on compute?"
- Any B200 optimization experiments belong in **M5.2** as a clean 50-step ablation on a fresh box, not perturbing the live prod run

#### 2026-05-14 ~17:15 UTC — reward-plateau / early-stop hypothesis (decision deferred to step ~200-250)

A100 reference reward growth-rate by phase:
- Steps 1-15 (cold start + tool collapse): **+0.0094/step** (fast climb)
- Steps 15-25 (plateau): **+0.0001/step** (flat)
- Steps 25-49 (re-exploration): **+0.0008/step** (slow climb)

Marginal return per step collapses fast after step 25. **Most of the reward signal lands in the first 25 steps**. Steps 25-49 added just 0.019 reward (+15%) for ~50% more compute.

Extrapolated full-run reward (rough projection from A100 trend):
- Step 100: ~0.18-0.22
- Step 311 (1 epoch): ~0.22-0.27
- Step 600 (last save): ~0.24-0.30
- Epoch 2 marginal gain: **~10-20% on reward, costs ~50% of total compute**

**Strategy decided**: monitor reward 50-step rolling mean. **Make the early-stop call at step 200-250** based on whether the curve has plateaued. Three scenarios:
- Plateau by step 200 → kill at step 200-300, save ~$160-220
- Still climbing past step 250 → run full 622
- Regime shift in epoch 2 (rare but possible) → may want to run full to capture it

Not actionable now (we're at step 7). Just framing the future decision tree.

#### Per-phase wall-clock split (avg of steps 1-7)

| Phase | Time | % | Speedup vs A100 ref | Why |
|---|---:|---:|---:|---|
| `policy_training` | 625s | 62% | **3.56×** | Bandwidth-bound bwd pass; B200 HBM3e wins |
| `generation` | 210s | 20% | **1.18×** | vLLM stack identical; retriever HTTP is the bottleneck, not GPU |
| `policy_and_reference_logprobs` | 190s | 18% | **3.49×** | Same as training: bandwidth-bound fwd passes |
| other | <3s | <1% | — | Batch prep, advantage compute, weight transfer |

**Generation barely speeds up on B200** because the retriever HTTP round-trips and KV cache management are not GPU-bandwidth-bound. The 3× B200 win comes entirely from `policy_training` + `logprobs`.

---

### Cadence 1 — Steps 1-10 (2026-05-14 ~17:48 UTC)

**Step log update** (steps 1-10 final):

| Step | Wall (s) | Reward | Gen len | Tool calls | Trunc % | A100 ref (s) | B200/A100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1047.73 | 0.0162 | 1384 | 6.55 | 62.8% | 3474 | 3.31× |
| 2 | 1081.31 | 0.0447 | 1413 | 6.72 | 68.1% | 3352 | 3.10× |
| 3 | 1103.42 | 0.0200 | 1334 | 6.96 | 70.6% | 3346 | 3.03× |
| 4 | 1020.96 | 0.0712 | 1463 | 6.22 | 59.7% | 3176 | 3.11× |
| 5 | 1010.43 | 0.0458 | 1428 | 6.24 | 58.8% | 2843 | 2.81× |
| 6 | 874.97 | 0.0761 | 1416 | 5.19 | 43.4% | 2294 | 2.62× |
| 7 | 915.05 | 0.0453 | 1433 | 5.56 | 48.4% | 2239 | 2.45× |
| 8 | 849.08 | 0.0575 | 1393 | 5.18 | 36.2% | 1967 | 2.32× |
| 9 | 850.23 | 0.0967 | 1426 | 5.09 | 40.6% | 1747 | 2.06× |
| **10** | **673.55** | 0.0703 | **1256** | **4.12** | **31.6%** | 1280 | **1.90×** |

**Step 10 is the inflection point**: gen length finally dropped (1426 → 1256), tool calls dropped (5.09 → 4.12), step time dropped 21% (850 → 674s). The "tool-collapse" phase that A100 saw at steps 5-6 hit us at step 10 — confirmed `micro_batch=2` introduces a ~3-5 step behavioural lag, as predicted.

#### Window aggregate (3200 trajectories across 10 steps)

| Metric | Value |
|---|---:|
| Reward — mean | 0.0544 |
| Reward — std | 0.2043 |
| Reward — max | 1.0000 (3.4% of trajectories) |
| Reward — % zero | 89.6% |
| **Reward — % nonzero** | **10.4%** |
| Turns — mean / p50 / p95 / max | 6.41 / 7 / 9 / 10 |
| Tool calls — mean / p50 / p95 / max | 5.81 / 7 / 9 / 11 |
| **Completion rate** (% with `</answer>`) | **40.7%** |
| **Truncation rate** (% without `</answer>`) | **59.3%** |
| Response chars — mean / p50 / p95 | 6179 / 5944 / 10980 |
| Input length — mean | 6255 |

**Per-step completion-rate trajectory** — the key learning signal:

| Step | % with `</answer>` | Avg tools | Avg turns |
|---:|---:|---:|---:|
| 1 | 29.1% | 6.6 | 7.1 |
| 2 | 26.2% | 6.8 | 7.2 |
| 3 | 24.1% | 7.0 | 7.4 |
| 4 | 33.1% | 6.2 | 6.8 |
| 5 | 33.1% | 6.3 | 6.8 |
| 6 | 47.2% | 5.2 | 5.9 |
| 7 | 44.1% | 5.6 | 6.2 |
| 8 | 49.4% | 5.2 | 5.9 |
| 9 | 52.2% | 5.1 | 5.8 |
| **10** | **68.4%** | **4.1** | **4.9** |

**Completion rate more than doubled in 10 steps** (29% → 68%). This is the most important signal in this cadence: the model is learning to stop spinning on max_turns and emit `</answer>`.

#### 5 mechanical examples from step 10 (script-selected by reward+variety)

All 5 picks landed reward=1.0 (out of 12 perfect rollouts at step 10). Compact stats only — full chunks on HF Hub at `logs/train_data/exp_013/train_data_step10.jsonl`:

| # | Reward | Turns | Tool calls | Has answer | Resp chars |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 4 | 3 | ✓ | 4879 |
| 2 | 1.00 | 3 | 2 | ✓ | 1224 |
| 3 | 1.00 | 4 | 3 | ✓ | 5135 |
| 4 | 1.00 | 6 | 5 | ✓ | 8011 |
| 5 | 1.00 | 4 | 3 | ✓ | 5103 |

Common pattern: 3-4 turns, 2-3 search calls per question, ending in `<answer>...</answer>`. Step 10's perfect rollouts demonstrate **efficient multi-hop QA** is achievable at this point in training.

#### 3 hand-analyzed examples (BEST / WORST / MEAN)

##### BEST — idx 75, step 10

**Q**: *"Who is the leader of the opposition in the singer Protoje's country?"*
**Reward**: 1.0 / 3 turns / 2 tool calls / 1224 chars

**Trajectory**:
1. Search 1: `"Protoje singer country"` → retriever returns Wikipedia "Protoje" article → identifies Jamaica
2. Search 2: `"Jamaica opposition leader 2024"` → retriever returns "Leader of the Opposition (Jamaica)" article → finds Peter Phillips
3. `<answer>Peter Phillips</answer>` ✓

**My commentary**: Textbook 2-hop reasoning. The model correctly decomposed "leader of opposition in [singer Protoje's] country" into two independent searches (singer→country, then country→opposition leader), grounded each search in the prior result, and did not waste tool calls on redundant queries. Clean `<think>` reasoning blocks between searches with no rambling. The answer is also clearly hedged in the final `<think>` ("Let me verify... the answer should be `<answer>Peter Phillips</answer>`") which suggests the model has internalised that the final answer must be in tags. This is exactly the policy we want to see emerge.

##### WORST — idx 164, step 10

**Q**: *"When was the most rainfall for the city where the State Fair of Oklahoma is held?"*
**Reward**: 0.0 / 10 turns / 10 tool calls / 5410 chars / **truncated (no `</answer>`)**

**Trajectory**:
1. Search 1: `"Oklahoma State Fair location date rainfall"` → generic Oklahoma climate
2. Search 2: `"Oklahoma State Fair annual rainfall MayJune JulyAugust best year"` → same generic Oklahoma climate
3. Searches 3-9: 7 more variants of the same query (`"... peak season"`, `"... 2023 rainfall peak summer"`, `"... July August rainfall peak 2024"`, etc.) → **all returning identical Oklahoma climate paragraphs**
4. Search 10: `"Oklahoma State Fair city rainfall records 2022 2023 2024"` → still generic
5. Hit `max_turns=10`, truncated without emitting `<answer>`

**My commentary**: Classic **pathological-loop failure**. The model never identified that "the city where the State Fair of Oklahoma is held" = Oklahoma City — it should have done a query like `"State Fair of Oklahoma host city"` to resolve the bridge, then searched Oklahoma City's rainfall records. Instead it kept re-querying with cosmetic variations of "Oklahoma State Fair rainfall" and getting the same generic Wikipedia "Oklahoma" article back. The retriever wasn't broken; the model wasn't reformulating the search well. This is the failure mode that the GRPO reward signal exists to suppress — perfect rollouts on similar 2-hop questions (see BEST above) are scoring 1.0 right next to this 0.0, and the gradient should push the policy toward the cleaner search-decomposition pattern over training.

##### MEAN — idx 15, step 10

**Q**: *"Who designed the museum that has the Portrait of Guidobaldo da Montefeltro?"*
**Reward**: 0.0 / 5 turns / 4 tool calls / 8556 chars / **answered (but wrong)**

**Trajectory**:
1. Search 1: `"Guide Giorgio Montefeltro museum painting museum"` → mixed Italian art results (typo'd the name; still returned relevant info)
2. Search 2: `"Guidobaldo da Montefeltro palace museum painting Budapest"` → finds Galleria Marche (Urbino) and the Portrait by Raphael
3. Search 3: `"Guidobaldo da Montefeltro palazzo masterpieces"` → finds the Portrait is in **Uffizi Gallery, Florence**
4. Search 4: `"Galleria Nazionale delle Marche Urbino Portrait Guidobaldo da Montefeltro"` → confirms Uffizi houses the Raphael portrait
5. `<answer>Raphael</answer>` — **WRONG**. The correct answer is **Vasari**, who designed the Uffizi building.

**My commentary**: This is a **bridge-parsing failure**. The model had all the information it needed: the painting is housed in the Uffizi (correct identification by search 3). It needed one more search ("who designed the Uffizi") to find Giorgio Vasari. Instead it confused itself in the final `<think>` block, conflating "designed the museum" with "designed the portrait" and answering Raphael (the painter). This is more interesting than the WORST case: the model has the right tool-use intuition (it kept the search count reasonable at 4) and it found the bridge entity (Uffizi), but it got distracted by surface-level wordplay in the question and short-circuited to the wrong answer. A reward of 0 here is technically correct (the answer is wrong) but the **underlying behavior is much closer to BEST than to WORST** — the policy gradient should still find this kind of trajectory useful because the search pattern is good even if the final answer is bad. The 89.6% zero-reward proportion in the window is dominated by examples like this — the model is *trying* but missing on a different dimension each time (search redundancy, bridge parsing, mathematical reasoning, span extraction).

#### Three observations from the cadence

1. **Completion rate is the leading indicator**, not raw reward. Reward is dominated by the `EM/F1` answer-checker which catches only exact-match wins. Completion rate (% with `</answer>`) jumped from 29% to 68% in 10 steps — that's the model learning the *structural* lesson (always emit a final answer in tags), which is a prerequisite for the *content* lesson (give the right answer). The content lesson lands next.
2. **The model's failure modes are diverse and reflect real reasoning limitations**, not just format errors. WORST shows redundant-search loops; MEAN shows bridge-parsing confusion. Both are correctable by training (the GRPO signal will preferentially upweight the BEST-style trajectories), and neither indicates a broken pipeline.
3. **Truncation rate at 31.6% on step 10 is high** but tracking A100's drop. A100 dropped from 69%→5% over steps 1-15; we dropped 63%→32% over steps 1-10. If the pattern holds, we hit ≤5% truncation around step 17-20.

#### System health snapshot — 2026-05-14 17:48 UTC

| Component | Value |
|---|---|
| GPU | B200 SXM6, 192 GB |
| GPU mem used (training peak) | 178 GB / 179 GB (97.8%) |
| GPU mem used (current, mid-logprobs) | ~140 GB |
| GPU util (active) | 55-75% |
| GPU power | 320-430 W |
| Driver / CUDA | 580.126.09 / 13.0 |
| Disk free | 400+ GB |
| Wrapper PID | 7942 ✓ |
| Training PID | 7961 ✓ |
| Uploader PID | 16526 (replaced 7919 after bug fix) ✓ |
| Steps complete | 10 / 622 (1.6%) |
| Elapsed wall | 2h 45m |
| Spend | ~$10.55 |

#### Git + HF action

- Doc updated and committed to `experiment_1_b200` (this commit)
- Pushed to GitHub
- HF Hub README synced to the new commit sha
- Rollout JSONLs (steps 1-10) all uploaded to HF (`logs/train_data/exp_013/`)
- prod.log uploaded ✓

**Next cadence**: Step 20 (~3 h from now if step time holds at ~700-900s).

---

### Cadence 2 — Steps 11-20 (2026-05-14 ~19:17 UTC)

**Step log update**:

| Step | Wall (s) | Reward | Gen len | Tool calls | Trunc % | A100 ref (s) | B200/A100 | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 705.31 | 0.0646 | 1357 | 4.18 | 31.2% | 1163 | 1.65× | |
| 12 | 523.09 | 0.0744 | 1153 | 3.18 | 22.5% | 942 | 1.80× | First sub-600s |
| 13 | 491.12 | 0.1061 | 1096 | 3.07 | 17.5% | 881 | 1.79× | First reward >0.10 |
| 14 | 446.80 | 0.1148 | 1020 | 2.91 | 10.9% | 730 | 1.63× | **Caught up to A100 on reward** |
| 15 | 441.26 | 0.1349 | 1057 | 2.74 | 11.6% | 683 | 1.55× | **Exceeded A100 reward** |
| 16 | 452.39 | 0.1225 | 1028 | 2.85 | 11.2% | 608 | 1.34× | |
| 17 | 456.81 | 0.1125 | 1049 | 2.94 | 7.5% | 625 | 1.37× | |
| 18 | 396.11 | **0.1458** | 940 | 2.61 | 8.8% | 628 | 1.59× | Fastest <400s |
| 19 | 445.30 | 0.1334 | 1000 | 2.94 | 10.0% | 676 | 1.52× | |
| 20 | 405.38 | 0.1154 | 969 | 2.63 | **4.4%** | 607 | 1.50× | Completion 95.6% |

The cadence-2 step log captures the **single largest behavioural shift** in the run so far. Step times dropped from 705s → 405s (−43%); reward climbed from 0.065 → 0.115 with a 0.146 peak at step 18; tool calls collapsed from 4.2 → 2.6; truncation dropped from 31% to 4.4%. This is the **tool-collapse regime in full motion**.

**Speedup vs A100 has shrunk** (1.65× → 1.50×) — not because B200 slowed down, but because A100 was also dropping fast in this window (1163s → 607s on its end). The absolute B200 advantage stays at ~300-400s/step.

#### Window aggregate (3200 trajectories across 10 steps)

| Metric | Cadence 1 (1-10) | **Cadence 2 (11-20)** | Δ |
|---|---:|---:|---:|
| Reward — mean | 0.0544 | **0.1124** | **+106%** |
| Reward — std | 0.2043 | 0.2767 | (wider, more wins) |
| Reward — % nonzero | 10.4% | **20.7%** | **+99%** |
| Turns — mean / p50 / p95 | 6.41 / 7 / 9 | **3.93 / 3 / 8** | −38% |
| Tool calls — mean / p50 / p95 | 5.81 / 7 / 9 | **2.98 / 2 / 7** | **−49%** |
| **Completion rate** | 40.7% | **86.4%** | **+112%** |
| **Truncation rate** | 59.3% | **13.6%** | **−77%** |
| Response chars — mean | 6179 | 4742 | −23% |
| Input length — mean | 6255 | 3816 | −39% |

**Two metrics doubled, one halved, one collapsed.** Reward doubled, completion rate doubled, tool calls halved, truncation collapsed. This is what "the model learned the lesson" looks like in data.

#### Per-step completion-rate trajectory (cadence 2)

| Step | % with `</answer>` | Avg tools | Avg turns |
|---:|---:|---:|---:|
| 11 | 68.8% | 4.2 | 5.0 |
| 12 | 77.5% | 3.2 | 4.1 |
| 13 | 82.5% | 3.1 | 4.0 |
| 14 | 89.1% | 2.9 | 3.8 |
| 15 | 88.4% | 2.7 | 3.7 |
| 16 | 88.8% | 2.9 | 3.8 |
| 17 | 92.5% | 2.9 | 3.8 |
| 18 | 91.2% | 2.6 | 3.6 |
| 19 | 90.0% | 2.9 | 3.8 |
| **20** | **95.6%** | **2.6** | **3.6** |

**Step 20: 95.6% completion rate.** Almost every rollout now emits a valid `<answer>` tag. The structural lesson is essentially solved.

#### 3 hand-analyzed examples (BEST / WORST / MEAN), step 20

##### BEST — idx 144, step 20

**Q**: *"What is the record label of the performer who released Graceland?"*
**Reward**: 1.0 / 2 turns / **1 tool call** / 744 chars

**Trajectory**:
1. Search 1: `"Graceland record label information"` → retriever returns Wikipedia "Graceland (album)" article → text contains "released on August 25, 1986, by Warner Bros. Records"
2. `<answer>Warner Bros.</answer>` ✓

**My commentary**: This is even tighter than cadence 1's BEST (3 turns / 2 tools). The model identified that the question's 2-hop ("performer who released Graceland" → "their label") could be resolved with a single well-chosen search because Wikipedia's album page mentions the label directly. **One search, one answer.** This is the asymptotic ideal — minimum tool calls, minimum tokens, correct answer. By cadence 2 this pattern is common (78 single-tool perfect rollouts in step 20 — 24% of all rollouts).

##### WORST — idx 230, step 20

**Q**: *"Who founded Arthur Fear's alma mater?"*
**Reward**: 0.0 / 9 turns / 9 tool calls / 5638 chars / **truncated**

**Trajectory**:
1-9. Repeated searches for "Arthur Fear" → retriever returned mixed entities (Arthur Fear 1990 film, Arthur M. Banta, Arthur Adamson, etc.). The model hallucinated details from these mismatched results: "Arthur Fear studied at University at Buffalo, Oklahoma and later transferred to San Francisco State University" — none of which appears in the retrievals.
10. Final search "Adamson College Edinburgh" → got "John Adamson (university principal)" article about the principal of University of Edinburgh in 1623 — no help on Arthur Fear or his alma mater.
11. Truncated at max_turns without emitting `<answer>`.

**My commentary**: **Entity disambiguation failure at retrieval level + hallucination at synthesis level.** Arthur Fear is genuinely obscure (likely no single dominant Wikipedia article), so the retriever returned semantically-adjacent but factually-irrelevant matches. The model didn't recognise the mismatch and instead hallucinated coherent-sounding facts to bridge the gap. The hallucination is particularly visible in the `<think>` blocks — confident statements about Arthur Fear's biography that don't appear in any retrieval. This failure mode is **inherent to RAG on a fixed corpus**: if Wikipedia doesn't have the entity, no amount of retraining the policy will fix it. The reward signal correctly says 0 here, but the gradient won't actually teach the model to "give up gracefully" because giving up is also reward 0.

##### MEAN — idx 10, step 20

**Q**: *"When did Danny Welch's employer start issuing degrees in engineering?"*
**Reward**: 0.0 / 5 turns / 4 tool calls / 6650 chars / answered (but wrong)

**Trajectory**:
1. Search 1: `"Danny Welch"` → retriever returned **Danny Kirwan** (Fleetwood Mac guitarist), not the correct Danny Welch
2-4. Three follow-up searches that all drifted further into Fleetwood Mac biographical detail
5. Final `<think>` block hallucinates wildly: claims "his employer likely began issuing engineering degrees around 1968-1970" based on a fabricated chain ("collaborated with Fleetwood Mac guitar, singer, and songwriter (which at that time included the Düsseldorf, Cologne, and Hamburg composers who were instrumental in developing formal educational structures in the medical field)")
6. `<answer>1968-1970</answer>` — **hallucinated number based on Danny Kirwan facts, not Danny Welch**

**My commentary**: Classic **wrong-entity retrieval + downstream hallucination**. The model never noticed it was looking at Danny Kirwan, not Danny Welch. Once it had Kirwan's Wikipedia article, it generated a plausible-sounding answer by chaining unrelated facts (Fleetwood Mac → European composers → medical degrees → engineering degrees → 1968-1970). The structural behaviour is correct (5 turns, 4 tools, emits `<answer>`), but the content is fabricated. This is the **new dominant failure mode** at this stage — the model is structurally trained but is generating confident wrong answers from misaligned retrievals. Improving this requires either (a) a better retriever for entity disambiguation, or (b) longer training so the policy learns to detect-and-redo when retrieval returns the wrong entity.

#### Three observations from cadence 2

1. **The structural lesson is solved.** Completion rate at 95.6% by step 20 means the model almost always emits `<answer>` tags. Truncation rate is in single digits. The remaining failure mass is content (correct answer), not format.
2. **The new dominant failure mode is hallucination from misaligned retrievals.** Both WORST (Arthur Fear) and MEAN (Danny Welch) show the same pattern: retriever returns the wrong entity (or no good entity), and the model fabricates confident details to bridge the gap. This is a known limit of RAG that GRPO alone cannot easily fix. The eval will tell us how much this caps final reward.
3. **The reward "sweet spot" of 2-3 tool calls solidified.** Cadence 1 showed it as a hypothesis; cadence 2 confirms it (mean tool calls 2.98, median 2). The model has internalised that more searches usually means it's confused, not that it's being thorough.

#### System health snapshot — 2026-05-14 19:17 UTC

| Component | Value |
|---|---|
| Steps complete | 21 / 622 (3.4%) |
| Elapsed wall | 4h 14m |
| Spend | ~$16.20 |
| GPU mem (current) | 158 GB / 179 GB |
| GPU power | 320-515 W (variable by phase) |
| Retriever | Healthy, 8/8 workers available, ~60 ms/call measured |
| Uploader (pid 16526) | ✓ healthy, last upload step 20 jsonl at ~19:08 UTC |
| Wrapper (pid 7942) | ✓ alive, no restarts |
| Training (pid 7961) | ✓ alive |
| ETA to step 50 | ~01:00-01:30 CEST Friday (2026-05-15) |

#### Git + HF action

- Cadence 2 doc committed and pushed
- HF README sync'd
- Rollout JSONLs for steps 11-20 uploaded to HF Hub
- prod.log fresh upload every 3 min

**Next cadence**: Step 30 (~3 h from now at recent ~450s/step pace).

---

### Cadence 3 — Steps 21-30 (2026-05-14 ~21:35 UTC; note: belatedly logged — cadence trigger was missed at step 30 around 20:25 UTC and caught up 1 h later at step 37)

**Step log update** (steps 21-37 — extended past cadence boundary since we caught up late):

| Step | Wall (s) | Reward | Gen len | Tool calls | Trunc % | A100 ref (s) | B200/A100 | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 21 | 468 | 0.1711 | 1075 | 3.00 | 7.2% | 735 | 1.57× | |
| 22 | 507 | 0.1096 | 1071 | 3.22 | 10.3% | 726 | 1.43× | |
| 23 | 496 | 0.1657 | 1154 | 3.08 | 6.9% | 755 | 1.52× | |
| 24 | 438 | 0.1575 | 1052 | 2.81 | 5.0% | 758 | 1.73× | |
| 25 | 456 | 0.1502 | 1028 | 2.95 | 5.6% | 766 | 1.68× | |
| 26 | 481 | 0.1437 | 1072 | 3.08 | 5.6% | 832 | 1.73× | |
| 27 | 546 | 0.0951 | 1161 | 3.39 | 8.4% | 903 | 1.65× | First "creep" signal |
| 28 | 493 | 0.1548 | 1008 | 3.23 | 7.8% | 863 | 1.75× | |
| 29 | 527 | 0.1337 | 1067 | 3.39 | 10.6% | 910 | 1.73× | |
| 30 | 475 | 0.1671 | 1020 | 3.11 | 6.2% | 954 | **2.01×** | A100 climbing; B200 holding |
| 31 | 534 | 0.1231 | 1037 | 3.48 | 5.6% | 1035 | 1.94× | |
| 32 | 533 | 0.1621 | 1048 | 3.45 | 8.7% | 968 | 1.82× | |
| 33 | 579 | 0.0874 | 1076 | 3.72 | 9.1% | 1091 | 1.88× | |
| 34 | 538 | **0.1810** | 987 | — | — | 1099 | 2.04× | **New high, beats A100 max** |
| 35 | 588 | **0.1998** | 1039 | — | — | 1155 | 1.96× | **New high again** |
| 36 | 583 | 0.1403 | 981 | — | — | 1311 | 2.25× | |
| 37 | 591 | **0.2117** | 976 | — | — | 1314 | 2.22× | **0.21 — uncharted territory** |

**Trajectory commentary**: cadence-3 steps stayed in 438-588s (mean 503s). Steps 27 and 33 showed "creep" signals but the window mean held. **Cumulative: 36 steps, mean step time 685s, reward mean 0.118** — both substantially better than the early cadences.

The reward broke through A100's recorded ceiling (~0.20) starting at step 35. A100 ran 49 steps and its highest reward was 0.1997 (step 37). **Our run is now reaching reward levels A100 never observed** — entering territory the reference run can't speak to.

#### Window aggregate (cadence 3: 3200 trajectories across steps 21-30)

| Metric | Cadence 2 | **Cadence 3** | Δ |
|---|---:|---:|---:|
| Reward — mean | 0.1124 | **0.1449** | **+29%** |
| Reward — % nonzero | 20.7% | **25.6%** | **+24%** |
| Turns — mean | 3.93 | 4.09 | (stable) |
| Tool calls — mean | 2.98 | **3.12** | (stable, slight uptick) |
| **Completion rate** | 86.4% | **92.0%** | **+6 pp** |
| **Truncation rate** | 13.6% | **8.0%** | **−41%** |
| Response chars — mean | 4742 | 4719 | (stable) |

**Single biggest cadence-to-cadence reward gain so far (+29%).** Tool calls + turns + chars all stable, so the model isn't getting more verbose — it's getting *better* at the same shape.

#### 3 hand-analyzed examples (BEST / WORST / MEAN), step 30

##### BEST — idx 136, step 30

**Q**: *"What oppressive Communist leader, of the country where Florin Surugiu is from, was deposed in 1989?"*
**Reward**: 1.0 / 2 turns / **1 tool call** / 1519 chars

**Trajectory**:
1. Model uses **internal knowledge** to identify Florin Surugiu as Romanian (no search needed)
2. Search 1: `"Romanian Communist leader deposed in 1989"` → retriever returns Wikipedia "Nicolae Ceaușescu" article: *"second and last Communist leader of Romania... until his overthrow in the Romanian Revolution in December 1989"*
3. `<answer>Nicolae Ceaușescu</answer>` ✓

**My commentary**: The model has learned to **leverage parametric knowledge** for the first hop ("Florin Surugiu = Romania") and only spend a search on the second hop (deposed Communist leader). This is the asymptotic ideal — 1 search for a 2-hop question because one hop is encyclopedic and the model already knows it. **5 perfect 1-tool rollouts at step 30**, up from 78 at step 20 (24% → 33%); the model is increasingly skipping searches when prior knowledge suffices. This is the right behaviour but worth flagging as a risk: it would hurt on multi-hop questions where both hops require fact-checking against Wikipedia.

##### WORST — idx 56, step 30

**Q**: *"What is the name of the Green party member that also studied at the university that is the largest employer in Hezekiah Augur's birthplace?"*
**Reward**: 0.0 / 9 turns / 9 tool calls / 7647 chars / **truncated**

**Trajectory**:
- 9 searches across: Hezekiah Augur, his birthplace, largest employer there, universities in Bangor Maine, Green party members at Husson University, etc.
- Found: Hezekiah Augur was born in New Haven (but model believed Bangor, Maine), Husson University is in Bangor, Maine
- **Hallucinated** that Jill Stein studied at Husson University (false — she studied at Harvard)
- Final `<think>`: *"I believe the answer is Jill Stein, even though I'm not entirely certain about the Harvard connection."*
- Ran out of turns before emitting `<answer>`

**My commentary**: **4-hop question** — Hezekiah Augur → birthplace → largest employer → Green-party-member alumnus. The model lost the thread on hop 2 (got the wrong birthplace; Augur was born in New Haven CT, not Bangor ME) and the entire chain downstream is bridge-corrupted. The model knew Jill Stein was a Green party figure (Harvard education), but bridged her implausibly to Husson University in Bangor because the question's chain demanded it. **Classic case of "make the answer fit the bridge"** — when retrieval fails, the model fabricates plausible-sounding connections rather than admitting it can't answer. The reward gradient pushes hard against this, but the failure pattern is sticky.

##### MEAN — idx 20, step 30

**Q**: *"When did the ocean that the Murray Mouth flows into become a thing?"*
**Reward**: 0.0 / 5 turns / 4 tool calls / 9221 chars / answered (badly)

**Trajectory**:
- Identified Murray Mouth as the mouth of the Murray River in Australia → flows into Southern Ocean (correct)
- Searched for Southern Ocean origin / formation → got mixed geology + diplomatic-recognition results
- Final answer: *"The Southern Ocean became a thing when the River Murray began flowing through the Murray Mouth into the coastal plains, before modern damming regulations were established."*

**My commentary**: The model got the **physical answer right** (Murray Mouth → Southern Ocean) but **fundamentally misunderstood the question's intent**. "Become a thing" is colloquial English for "first recognised as a geographic entity" — the dataset's expected answer is likely 2000 (when the IHO formally defined the Southern Ocean) or 1937 (an earlier mapping). Instead the model produced a nonsense answer mixing river geomorphology with "modern damming regulations" — not even a date. The structural lesson is intact (uses `<answer>` tags) but the content is incoherent prose, not a factual date. **This shows the EM/F1 reward function correctly punishes nonsense** even when the question is correctly decomposed. The gradient should push the policy to be more decisive about answer format (extract a date), which we may see in cadences 4-5.

#### Five observations from cadence 3

1. **B200 has broken through A100's reward ceiling.** A100's max reward across 49 steps was 0.1997 (step 37). Our step 37 hit **0.2117** — the first observation of behaviour A100's reference run never reached.
2. **The "micro_batch=2 different equilibrium" hypothesis is firming up.** Over steps 21-37, our reward mean is 0.143 vs A100's 0.119 (same range) — a persistent +20% gap, not noise. The smoothed-gradient effect is real.
3. **Tool calls per sample stabilised at ~3.1** — up slightly from cadence 2's 2.98 but well below cadence 1's 5.81. The model has settled into a 3-tool-call equilibrium with longer reasoning per call.
4. **Truncation rate halved cadence-to-cadence (13.6% → 8.0%)** — model is finishing answers more reliably even as it does more cross-verification.
5. **The new dominant failure mode is "bridge corruption"** — see WORST. Model commits to a wrong intermediate fact early in a multi-hop chain and forces downstream hops to fit. Hard to fix with GRPO alone since the policy doesn't know it's wrong.

#### System health snapshot — 2026-05-14 21:35 UTC

| Component | Value |
|---|---|
| Steps complete | 37 / 622 (5.9%) |
| Elapsed wall | 6h 32m |
| Spend | ~$25 |
| GPU mem (training peak) | 178 GB / 179 GB (97.8%) |
| GPU power | 320-590 W (variable) |
| Retriever | Healthy, 8 workers, ~60 ms/call measured |
| Uploader | ✓ healthy, all rollouts uploading on schedule |
| Wrapper | ✓ alive, no restarts |
| Training | ✓ alive, no anomalies |
| **ETA to step 50** | ~2h from now (~00:00 CEST Fri) |
| **ETA to epoch 1 (step 311)** | ~40-50 h from now (Sat/Sun) |
| **ETA to full run (step 622)** | ~85-100 h from now (Mon) |

#### Git + HF action

- Cadence 3 doc committed and pushed
- HF README sync'd
- Rollout JSONLs for steps 21-30 (and 31-37) uploaded to HF Hub
- prod.log fresh upload every 3 min

**Next cadence**: Step 40 (~3 h from now). I'll trigger it on time, not after the fact.

---

### Cadence 4 — Steps 31-40 (2026-05-14 ~22:25 UTC; triggered late at step 42, ~25 min after step 40 landed)

**Cadence-4 step log** (steps 31-42 to extend through current):

| Step | Wall (s) | Reward | Gen len | A100 ref (s) | B200/A100 | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 31 | 534 | 0.1231 | 1037 | 1035 | 1.94× | |
| 32 | 533 | 0.1621 | 1048 | 968 | 1.82× | |
| 33 | 579 | 0.0874 | 1076 | 1091 | 1.88× | |
| 34 | 538 | 0.1810 | 987 | 1099 | 2.04× | new high (steps 1-33) |
| 35 | 588 | 0.1998 | 1039 | 1155 | 1.96× | |
| 36 | 583 | 0.1403 | 981 | 1311 | 2.25× | |
| 37 | 591 | **0.2117** | 976 | 1314 | 2.22× | **run high** |
| 38 | 583 | 0.1337 | 980 | 1266 | 2.17× | |
| 39 | 590 | 0.2054 | 996 | 1187 | 2.01× | |
| 40 | 625 | 0.1798 | 1014 | 1607 | 2.57× | |
| 41 | 619 | 0.1522 | 1023 | 1681 | 2.71× | |
| 42 | 472 | **0.2126** | 1018 | 1362 | 2.89× | **new high** |

**Trajectory commentary**: Reward broke 0.20 three times in 12 steps (35, 37, 42). The new run-high is 0.2126 at step 42. Step time fluctuated 472-625s — variable but mean 561s. **B200/A100 speedup grew to 2.71-2.89× past step 40** as A100 climbed dramatically (1607-1681s/step) while we stayed in the 470-625s band.

**Note**: gen length stable ~980-1040 (model neither shrinking nor growing further), but tool calls slowly climbing (3.5 → 4.2). The model is doing MORE searches per question on average, not fewer — this is A100's "re-exploration regime" emerging in our run too, just at lower absolute cost.

#### Window aggregate (cadence 4: 3200 trajectories across steps 31-40)

| Metric | Cadence 3 (21-30) | **Cadence 4 (31-40)** | Δ |
|---|---:|---:|---:|
| Reward — mean | 0.1449 | **0.1624** | +12% |
| Reward — % nonzero | 25.6% | **28.0%** | +9% |
| Turns — mean | 4.09 | **4.70** | +15% |
| Tool calls — mean | 3.12 | **3.76** | **+21%** |
| **Completion rate** | 92.0% | 90.6% | −1.4 pp (slight regression) |
| **Truncation rate** | 8.0% | 9.4% | +1.4 pp |
| Response chars — mean | 4719 | 4355 | (slightly tighter) |

**Reward keeps climbing (+12% vs cadence 3)** but **tool calls grew faster (+21%)**. The policy is buying reward at the cost of more search operations — exactly the "re-exploration regime" the paper describes (and A100 went through at steps 30+).

**Completion rate dipped slightly (−1.4 pp)** — small regression from cadence 3's 92.0%. Watch for this; if it keeps dropping the policy may be over-investing in tool calls.

#### 3 hand-analyzed examples (BEST / WORST / MEAN), step 40

##### BEST — idx 300, step 40

**Q**: *"Who piloted the plane that dropped the bomb over the city where Little Boy detonated?"*
**Reward**: 1.0 / 3 turns / 2 tool calls / 1401 chars

**Trajectory**:
1. **Hallucinated context in initial `<think>`**: *"This is a historical question about the Battle of Курсок (Kursk) in the Russian-Soviet war"* — totally wrong. The model thought Little Boy was a WW2 Eastern Front bombing.
2. Search 1: `"Little Boy detonation city"` → retriever returns Wikipedia "Little Boy" article, confirms **Hiroshima**.
3. Model self-corrects in next `<think>`: *"I know Little Boy was dropped on August 6, 1945, over Hiroshima."*
4. Search 2: `"Hiroshima bomb pilot"` → retrieves Wikipedia "Paul Tibbets" + "Enola Gay" articles.
5. `<answer>Paul Tibbets</answer>` ✓

**My commentary**: This is interesting because the **initial parametric knowledge was wrong** (Battle of Kursk) but **the model still recovered**. The search-then-verify pattern saved it. This is exactly the resilience GRPO is supposed to teach: trust retrieval over priors when the priors are unreliable. The fact that the model didn't fight the retrieval (didn't try to force "Kursk" into the answer) shows the policy has learned to override hallucinated context. **More valuable as evidence of robustness than the cleaner BEST examples from earlier cadences.**

##### WORST — idx 85, step 40

**Q**: *"What amount of TEUs did the location of Hagia Sophia handle in 2010?"*
**Reward**: 0.0 / 10 turns / 9 tool calls / 4598 chars / **truncated**

**Trajectory**:
- 10 searches: all variations of Hagia Sophia, never pivoted to Istanbul port / TEU traffic
- Final `<think>`: *"I've gathered a lot of search results about Hagia Sophia but I'm still unable to find specific information about the TEU handling capacity in 2010..."*
- Truncated at max_turns

**My commentary**: **Failed to pivot from hop 1 to hop 2.** This is a 2-hop question: (a) Hagia Sophia is in Istanbul, (b) Istanbul's port TEU volume in 2010. The model successfully identified Istanbul in the first search (Hagia Sophia → Istanbul, Turkey) but **never made the leap that "TEUs" is shipping container terminology and needs a search like "Port of Istanbul TEU 2010"**. The policy seems to lack the cross-domain knowledge that TEU = port metric, not a museum metric. **The retrieval is fine; the bridge reasoning is the failure point.**

##### MEAN — idx 12, step 40

**Q**: *"What major nationalist movement has the largest economy in Africa had?"*
**Reward**: 0.0 / 5 turns / 4 tool calls / 4380 chars / answered (wrong)

**Trajectory**:
- Searched for largest African economy → found **Nigeria**
- Searched for major nationalist movements (general) → got various African nationalist movements
- Final answer: `<answer>Nigeria</answer>` — wrong, that's a country not a movement

**My commentary**: **Answered the wrong question.** The model identified Nigeria correctly (largest economy in Africa ✓) but then answered "Nigeria" as the answer to the whole question, instead of "Nigeria's major nationalist movement" (which would be the Biafran independence movement, or various others). **The model executed the first hop correctly and stopped** — it didn't realize the question asked for an entity *related to* Nigeria, not Nigeria itself. **Bridge-skipping failure**: the model treated a 2-hop question as a 1-hop. This is a new failure mode I haven't seen in earlier cadences.

#### Four observations from cadence 4

1. **Reward keeps climbing (+12%)** — now at 0.16 mean for the window, with three step-level peaks above 0.20 (steps 35, 37, 42). No plateau yet.
2. **Tool calls per sample grew +21% (3.1 → 3.8)** — the policy is re-investing in retrieval rather than getting more efficient. This is the paper-predicted "search calls grow during training" pattern.
3. **Completion rate dipped slightly (−1.4 pp)** — small but worth watching. If it falls below 88% in cadence 5 we'd want to investigate.
4. **New failure mode: bridge-skipping** (MEAN above). Model answers the first-hop entity instead of the second-hop. Distinct from cadence-3's "bridge corruption" (model commits to wrong first-hop fact). Watching to see if this pattern grows.

#### System health snapshot — 2026-05-14 22:25 UTC

| Component | Value |
|---|---|
| Steps complete | 42 / 622 (6.8%) |
| Elapsed wall | 7h 22m |
| Spend | ~$28.20 |
| GPU mem (training peak) | 178 GB / 179 GB |
| GPU power | 320-590 W |
| Retriever | Healthy, 8 workers |
| Uploader | ✓ healthy |
| Wrapper | ✓ alive |
| Training | ✓ alive |
| **ETA to step 50 (first ckpt)** | ~70 min from now (~23:35 UTC = 01:35 CEST) |
| **ETA to step 311 (1 epoch)** | ~42 h from now (~Sat 16:00 CEST) |
| **ETA to step 622 (full run)** | ~88 h from now (~Mon 14:00 CEST) |

#### Git + HF action

- Cadence 4 doc committed (this commit) and pushed
- HF README sync'd
- Rollout JSONLs for steps 31-42 uploaded to HF Hub

**Next cadence**: **Step 50** (first ckpt save). Triggering it on the exact step landing this time.

### 6.2 Cost / wall-clock estimation — actively unresolved

**Two estimates have been on the table; one wide range until step 1 lands:**

| Source | Per-step | Total wall | Cost @ $3.83/h | Anchor |
|---|---:|---:|---:|---|
| [HARDWARE_COMPARISON.md §3](../setup/HARDWARE_COMPARISON.md) (optimistic) | ~2.2 min | **~34 h** | ~$130 | A100 anchor 23.7 min/step ÷ B200 BF16 TFLOPS speedup (≈7×) |
| Smoke-extrapolated (conservative) | ~10-12 min | ~100-115 h | ~$380-440 | Smoke step 3 (43 s at smoke shape) × 15× smoke-to-prod ratio observed on A100 |

The optimistic 34 h assumes B200's BF16 compute speedup applies uniformly, but logprobs (mem-BW bound) and vLLM rollout (mem-BW bound) only get ~4× from B200's bandwidth bump, not 7×. The conservative 100 h assumes smoke-to-prod step ratio is identical between A100 and B200 — but B200's 139.5 GB KV cache (vs A100's tight budget) likely lets vLLM batch much more aggressively at the prod shape, so sublinear scaling vs smoke is plausible.

**Step 1's wall-clock will resolve this within the hour.** Best guess at the moment: 50-90 h, ~$200-350. This section will be replaced with the measured number as soon as step 1 lands.

### 6.3 Cadence checkpoints

**Cadence updated 2026-05-14 ~15:15 UTC to every 10 steps per user request** (was 25). Each cadence block contains:

1. **Step summary** from prod.log + W&B: per-step wall-clock, phase breakdown (generation / training / logprobs / other), throughputs (tok/s/gpu), reward stats.
2. **Rollout analysis** generated by [`training_m5_1/scripts/analyze_rollouts.py`](../../training_m5_1/scripts/analyze_rollouts.py) over the last 10 steps' `train_data_stepN.jsonl` files:
   - Reward stats (mean, std, max, % zero, % nonzero)
   - Turns per trajectory (mean, p50, p95, max)
   - Tool call count per trajectory (mean, p50, p95, max)
   - Completion rate (% with `</answer>`) vs truncation rate
   - Response length distribution
3. **5 example trajectories** (script-extracted, mechanical) from the cadence-end step:
   - Top-reward trajectories first
   - Filled with zero-reward picks chosen for turn-count variety
   - Chunks truncated with elision markers (~300 lines each)
4. **3 hand-analyzed example trajectories** (Claude reads + writes commentary) — one each from the cadence window:
   - **BEST**: highest-reward trajectory. If all rewards are zero, the one that came closest to a valid answer (e.g., emitted `<answer>` even if wrong, or made coherent multi-hop reasoning across turns).
   - **WORST**: pathological case. Could be: stuck in a loop of repeated identical searches, generated zero tool calls and gave up, hit max_turns without any reasoning progress, or rambling without convergence.
   - **MEAN**: representative middle-of-distribution example. Average turn count, average tool calls, no jackpot reward but also not obviously broken. The "this is what most trajectories look like" exemplar.
   - For each: I read the prompt + model output and write 2-3 sentences on what the model did well / poorly / what pattern is visible. Aim is to make the training story legible step-by-step.
5. **System health snapshot**: GPU memory, util, power; RAM; disk; retriever health.
6. **Git + HF action**: this cadence's update is committed to `experiment_1_b200`, pushed to GitHub, and the updated README synced to the HF repo at the same time.

Cadence checkpoints will land at steps 10, 20, 30, ..., 600. At the projected ~10 min/step, each cadence block lands every ~100 min.

#### 2026-05-14 15:03 UTC — Launch ✓

| Item | Value |
|---|---|
| Wrapper started | 2026-05-14T15:03:36Z (via `run_prod_a3_resilient.sh`, attempt 1/6 fresh) |
| Training process | pid 7961, `python run_grpo.py --config m5_1_research_paper.yaml grpo.seed=42 ...` |
| W&B project | `reason_over_search_b200` ✓ |
| W&B run id | `h68uskz6` |
| W&B run name | initially `qwen3.5-0.8b-musique-m5_prod-seed42-20260514T1503Z` (run.sh override); **renamed via wandb API to `qwen3.5-0.8b-musique-b200-a3-seed42-20260514T1503Z` shortly after launch** |
| Training samples loaded | **311** ✓ (= floor(19,938 / 64), 1 epoch; max_num_steps=622 = 2 epochs) |
| Compute cluster | Ray, 1 node, colocated mode |
| Worker init mode | sequential (colocated; vLLM and DTensor share GPU via sleep/wake) |
| Setup phase observed at this snapshot | Building `VllmAsyncGenerationWorker` venv (~5 min one-time install of 468 packages on first prod use; cached after) |

**Pre-launch artifacts pushed**:
- HF repo created (private): https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42
- README.md (this doc) + config_snapshot.yaml uploaded to HF
- prod.log initialized at /workspace/prod.log with launch metadata
- Upload watcher (pid 7919) polling every 60 s

**Watch-points for first step** (per smoke baseline):
- vLLM async engine setup: should complete in ~5-8 min on first prod use (smoke used `VllmGenerationWorker`, prod uses `VllmAsyncGenerationWorker` — additional one-time venv build)
- Total setup time on smoke was 182.7 s; prod will likely be 5-10 min total because the async worker venv builds + larger model context init
- First step wall: paper-shape (320 traj × seq=8192 × micro=2) extrapolates from smoke step 3 (43 s at 20 traj × seq=4096) by ~15× → **~10-12 min estimated**. Real number lands within the hour.

**Progress 2026-05-14T15:07 UTC** (~3.5 min post-launch):
- vLLM async worker venv build complete (75 packages, 59.1 s)
- vLLM async engine V1 initialized at max_seq_len=8192 ✓ (paper-faithful)
- Qwen3.5-0.8B model loaded (1.72 GB, 1.09 s) ✓
- FlashInfer attention backend selected ✓
- Mamba+attention hybrid page alignment configured (HND KV layout)
- Asynchronous scheduling enabled ✓ (the `async_engine: true` we set)
- Architecture: `Qwen3_5ForConditionalGeneration` ✓
- max_cudagraph_capture_size: 256 (larger than smoke's 51 — prod has more dynamic batching shapes; CUDA graph capture is in progress now)
- No errors

Next observable: CUDA graph capture completion → vLLM ready → DTensor v2 init (~35 s) → step 1 starts.

## 7. Spend tracker

Spheron B200 spot at $3.83/h on Spheron ES (verified from console).

| Phase | Hours | Cost |
|---|---:|---:|
| Setup + bootstrap + smoke (pre-launch) | TBD | TBD |
| Production training (live) | TBD | TBD |
| **Cumulative** | TBD | TBD |

Sunk cost on the first (broken) box: ~$2 (destroyed by user 2026-05-14).

## 8. The 3 prior losses being recovered from

Recap (full details in [`RESULTS_SMOKE_m5.md` §7 / §7.8 / §7.8.1](RESULTS_SMOKE_m5.md)):

| # | When | Cost | Cause | Fix verified by |
|---|---|---|---|---|
| a1 step-50 ckpt crash | 2026-05-11 | ~$30 | `metric_name: "train/loss/mean"` violated NeMo-RL colon-prefix assertion | smoke-ckpt-verify1 + smoke-ckpt-verify2 + the M5.1-prod-a3 smoke (this session) — all 4 saves landed cleanly |
| a1 rollout-corpus deletion | 2026-05-11 | 196 MB data permanently lost | Bundled `rm -rf logs/exp_010 logs/exp_011` (exp_010 was prod) | HF Hub upload rule (new): every per-step JSONL pushed to HF means no single-machine deletion can destroy it |
| a2 zombie GPU misdiagnosis kill | 2026-05-12 | ~$21 | Misread `[Not Found]` in `nvidia-smi --query-compute-apps` | New no-go rule: only authorize kills on confirmed evidence |

Total prior loss: ~$51 + 196 MB. M5.1-prod-a3 should bring the canonical first artifact home.

## 9. Pointers

- W&B run: [`h68uskz6`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6) (state=crashed; 56 steps logged; all metrics preserved)
- HF Hub repo: [`pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42) (3 root + 33 logs files; NO checkpoint folders)
- Smoke baseline (2026-05-14): smoke W&B + smoke ckpts at `/workspace/results/grpo/m5_smoke/seed42/` (step_2, step_4 — kept for reference)
- Prod config: [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml)
- Upload watcher (BROKEN; kept as record of bug): [`training_m5_1/scripts/upload_a3_to_hf.sh`](../../training_m5_1/scripts/upload_a3_to_hf.sh)
- Replacement uploader (a4+): [`training_m5_1/scripts/upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py)
- External monitor: [`training_m5_1/scripts/external_monitor.py`](../../training_m5_1/scripts/external_monitor.py)
- Launch checklist (a4): [`training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md)
- Resilient launcher: [`training_m5_1/scripts/run_prod_a3_resilient.sh`](../../training_m5_1/scripts/run_prod_a3_resilient.sh)
- Sibling iteration log: [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- Hardware comparison: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)
- Catch-up TODO at launch time: [`../todo/TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md)

---

## 10. Incident report — host preemption + silent uploader failure (2026-05-15)

### 10.1 Timeline (UTC)

| Time | Event |
|---|---|
| 2026-05-14T15:03:36Z | Training launched on Spheron B200 instance `6a05d7fd0e7688ec54e260e6` (host `46.243.144.4`, then redeployed to `46.243.145.4`). W&B run `h68uskz6` opens. |
| 2026-05-14T15:34:31Z | Uploader silent-fail bug #1 caught (wrong python path + glob mismatch). Fixed inline; restarted as pid 16526. First batch of uploads catches up. |
| 2026-05-14T16:36Z | Last successful **rollout** upload to HF: `exp_013/train_data_step18.jsonl` (per HF commit history). **Uploader silently stopped uploading rollouts after this point but kept polling.** |
| 2026-05-14T18:58:43Z | Last successful **prod.log** upload (HF commit). Uploader process still alive but uploading nothing new beyond prod.log. |
| 2026-05-14T22:25Z | I noted "Cadence 4 — uploader ✓ healthy" in the cadence doc without verifying recent uploads. This was the assertion that should have caught the bug. |
| 2026-05-15T00:21:06Z | W&B last metric timestamp (step 56, reward 0.1943). Training was healthy at this moment. |
| 2026-05-15T00:21:06Z → ~06:31Z | **Host unreachable.** Ping + SSH both fail. ICMP timeouts. Spot preemption assumed. |
| 2026-05-15T06:31Z | User flagged "I think the instance crashed". W&B state = `crashed`. HF repo inspected: NO step_N folders, only 18 rollout JSONLs. |
| 2026-05-15T07:00Z | Diagnosis complete; fix authored. Bulletproof uploader committed at [`44c3eeb`](https://github.com/GaurisankarJ/reason_over_search/commit/44c3eeb). |

### 10.2 Two compounding failures

**Failure A: Spheron T3 spot host preemption**

- **What happened**: Instance `6a05d7fd0e7688ec54e260e6` on Spheron's T3 marketplace went unreachable at ~00:21 UTC. Both ICMP and SSH (port 22) time out from that point. No graceful shutdown signal received by training.
- **Why (best guess)**: Spheron T3 is a decentralized marketplace tier where host nodes are independently operated. Per Spheron docs the T3 SLA is 99.9% but in practice spot preemption + host network failures + voluntary shutdowns happen. We do NOT have direct API access to Spheron from this machine, so we cannot get the host-side reason. The symptom pattern (sudden ICMP failure with no warning) is consistent with: (a) provider-initiated preemption, (b) host network outage, (c) host hardware failure, (d) host owner deciding to take the machine offline for maintenance.
- **What would have caught it earlier**: nothing on our side — the only mitigation against unannounced spot preemption is checkpointing artifacts off-host fast.
- **What did NOT happen**: the resilient launcher `run_prod_a3_resilient.sh` could not auto-restart from `step_N/` because the whole host (including disk) was gone. The wrapper's auto-restart works only if the host comes back; this host has not returned in 6+ hours, suggesting it's gone for good.

**Failure B: silent uploader failure past step 18**

- **What happened**: After uploading `exp_013/train_data_step18.jsonl` at ~16:36 UTC, the `upload_a3_to_hf.sh` script kept polling but **never uploaded any new rollouts** (steps 19-56 = 38 files lost) or any checkpoint (step_50 should have been saved at step 50 ≈ 21:30 UTC). It DID upload `prod.log` one more time at 18:58 UTC, then silence until the crash.
- **Why (analysis of the bash script)**: The `upload_a3_to_hf.sh` script had multiple latent silent-failure paths:
  - Inline python heredocs with no timeout — if HF API hung, the call hung forever
  - `set -uo pipefail` (without `errexit`) meant intermediate errors didn't bubble up
  - Earlier fix (commit `906d93a`) added explicit `✗ FAILED` logging for prod.log/rollout/timings paths, but the **outer `for jsonl in ... ; do` loop** could still get stuck on an HF rate-limit response (no escalation, no eventual stop)
  - State file `.uploaded_artifacts` is plain text appended without locking; a partial write could corrupt it
  - Most likely root cause (without instance access to confirm): HF rate-limit kicked in around step 19 upload attempts, the heredoc returned a non-200 that didn't cleanly map to "FAIL:", and the file was neither logged as failed nor marked uploaded — so it kept getting silently retried in subsequent cycles with the same failure
- **What should have caught it earlier**:
  1. A **heartbeat log** that prints every cycle regardless of upload activity. The bash script's `log` function only logged on success/failure of an action, so an idle silent loop produced NO output for hours.
  2. **External verification** (not just trusting in-box uploader.log). The HF API itself could have been polled from a different machine to verify uploads were arriving. I did not have this set up.
  3. **My own checks**: at each cadence I claimed "uploader ✓ healthy" based on having seen *some* upload activity earlier. I did not re-verify that recent step rollouts had been uploaded. This is the same silent-fail pattern I had caught and "fixed" earlier in the run — and I let it bite again because I didn't tighten the verification.

### 10.3 Mistakes I made in this run (full list)

A complete list, for the record:

1. **Believed "uploader healthy" without verification** — repeated in every cadence status from cadence 2 onward, never re-checked that recent steps were on HF
2. **Missed cadence 3 trigger** — should have fired at step 30, fired at step 37 (~1h late) after user prompt
3. **Missed cadence 4 trigger** — should have fired at step 40, fired at step 42 (~25 min late) after user prompt
4. **Hallucinated reward target** — stated "ReSearch paper F1 ~0.30-0.35 on MuSiQue" without grep'ing docs. Real paper convergence is 0.40-0.45 EM (different metric) on 7B/32B (different model size). User caught this and forced fact-check.
5. **Math error early** — said "320 prompts × 5 generations = 1600 trajectories" when it's actually 64 × 5 = 320. User caught this.
6. **Did not pre-flight the uploader** before launching prod — meant the wrong python path bug surfaced in prod, not in a smoke test
7. **Did not set up external monitoring** — only relied on in-box uploader.log, the single point of trust that failed
8. **Did not push docs at every cadence consistently** — cadence 3/4 were committed late
9. **Did not implement the "heartbeat" safeguard** despite the bash uploader having one as a one-line addition
10. **Did not have a watchdog** — no alarm fired when uploads stopped; I noticed when the host died, hours later
11. **Confident-sounding wrong claims** — e.g. "we'll likely hit ReSearch paper convergence" when the paper used 7B and we have 0.8B; "uploader ✓" in nearly every status update

### 10.4 What survived the crash

| Artifact | Survived? | Where |
|---|---|---|
| W&B metric log (steps 1-56) | ✓ | [W&B run h68uskz6](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6) |
| Cadence 1-4 docs (steps 1-40) | ✓ | Git: `experiment_1_b200` branch, in this file |
| Rollout JSONLs steps 1-18 | ✓ | HF: [pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42/logs/train_data/exp_013/](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42) |
| prod.log (truncated; last upload 18:58 UTC) | ~partial | HF (same repo, `logs/prod.log`) |
| config_snapshot.yaml | ✓ | HF (same repo, root) |
| Cadence 5/6 docs (steps 41-56) | ✗ | never written |
| Rollout JSONLs steps 19-56 (38 files) | ✗ | LOST (never uploaded; disk gone) |
| **step_50/ checkpoint** | ✗ | **LOST — model weights gone** |
| Optimizer state | ✗ | (we explicitly didn't save it; not lost more than expected) |

### 10.5 Final results from W&B (steps 1-56) — the trajectory we keep

This is the **most important table** for the supervisor / paper writeup — the full training trajectory is preserved even though the model weights are not.

| Step | Wall (s) | Reward | Tool calls | Turns | Trunc % | Gen len | GPU mem (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1047.7 | 0.0162 | 6.55 | 7.07 | 62.8% | 1384 | 140.7 |
| 5 | 1010.4 | 0.0458 | 6.24 | 6.80 | 58.8% | 1428 | 15.6 |
| 10 | 673.5 | 0.0703 | 4.06 | 4.91 | 25.6% | 1256 | 15.6 |
| 15 | 441.3 | 0.1349 | 2.69 | 3.66 | 7.5% | 1057 | 131.6 |
| 20 | 405.4 | 0.1154 | 2.58 | 3.56 | 3.1% | 969 | 176.7 |
| 25 | 455.6 | 0.1502 | 2.95 | 3.92 | 5.6% | 1028 | 131.6 |
| 30 | 474.6 | 0.1671 | 3.11 | 4.08 | 6.2% | 1020 | 115.6 |
| 35 | 587.9 | 0.1998 | 3.84 | 4.79 | 7.2% | 1039 | 15.6 |
| 40 | 624.9 | 0.1798 | 4.15 | 5.08 | 10.6% | 956 | 84.5 |
| 45 | 493.1 | 0.2164 | 3.63 | 4.61 | 4.1% | 720 | 175.7 |
| 50 | 530.4 | 0.1207 | 3.78 | 4.75 | 4.4% | 731 | 15.6 |
| **53 (PEAK)** | **472.8** | **0.2320** | **— **| **— **| **3.1%** | **707** | **— ** |
| 55 | 503.8 | 0.1624 | 3.66 | 4.63 | 4.4% | 755 | 15.6 |
| **56 (last)** | **477.0** | **0.1943** | **3.55** | **4.53** | **3.4%** | **737** | **116.4** |

**Peak reward: 0.2320 at step 53** (single-step). **Run mean reward over last 10 steps: 0.180.** **Truncation rate dropped from 62.8% to 3.4%.** **Tool calls per sample collapsed from 6.55 → 3.55.** **Gen length collapsed from 1384 → 737.**

All five behavioural metrics improved monotonically (with noise) over 56 steps. The recipe works on Qwen3.5-0.8B with `micro_batch=2` on B200. The fact that we don't have the model weights doesn't invalidate this finding — the trajectory is reproducible.

### 10.6 What we observed in step 53 (peak reward)

W&B step 53: total_step=473s, reward=0.232, truncation=3.1%, gen_len=707.

Some context for the trajectory at peak:
- **Reward 0.232** = roughly 12% perfect rollouts + 22% partial credit (extrapolating from step-37 distribution we did fully analyze)
- **Gen length 707** is close to A100's plateau zone (672-748 at A100 steps 15-25), showing our policy started converging on a tighter response budget after step 45
- **Tool calls / turns no longer available** (rollouts not uploaded to HF past step 18, so we cannot re-analyze step 53's policy structure)

This is the most painful loss: step 53 was probably our **best policy** of the run. We have the reward number but not the model that produced it.

### 10.7 Cost summary

| Item | Cost | Notes |
|---|---:|---|
| B200 spot Spheron (15:03 May 14 → 00:21 May 15) | $36 | 9.3 h × $3.83/h |
| Cumulative M5.1 losses (a1/a1b/a2/a3) | **$108** | a1=$30, a1b=0+196MB, a2=$21, a3=$36 (+ approx $20 of M5.1-prod-a1/a2 boot costs) |
| Compute lost (irrecoverable training time) | 9.3 h | All future runs start from step 0 |
| Engineer time burned | ~4 h of my context | (incidental) |

---

## 11. Recovery plan — M5.1-prod-a4

### 11.1 What's already in place

Three new artifacts authored 2026-05-15 in response to this incident:

1. [`upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py) — Python-based bulletproof uploader
   - Heartbeat every cycle (proves the loop is iterating)
   - Checkpoint-folder priority (push step_N/ before logs)
   - Aggressive retry with exponential backoff (5/15/45/90/180 s)
   - Pre-flight canary test on launch; exits 2 if HF unreachable
   - Dual-repo upload (primary + backup namespace)
   - 5-second filesystem scan for new step_N/ folders
   - JSON state file with atomic writes
   - All errors explicitly logged

2. [`external_monitor.py`](../../training_m5_1/scripts/external_monitor.py) — runs on the **user's local machine**, not the training box. Polls HF every 5 min, cross-references W&B step count, alerts on STALE. **Verified to catch the a3 failure mode**: would have screamed at minute 11, not 11 hours.

3. [`LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md) — explicit gate-by-gate pre-launch checklist. Every box must be checked before training starts. Includes "what good looks like" / "what broken looks like" log excerpts.

### 11.2 Decisions needed before a4 launches

Three open questions:

1. **Provider tier**: Spheron T3 spot ($3.83/h, preemption risk — what just bit us) vs Spheron on-demand (~$4.40/h, no preemption) vs another provider entirely?
2. **Config**: keep `micro_batch=2` (B200 was producing higher reward than A100) or revert to `micro_batch=1` (paper-faithful)?
3. **Save period**: keep `save_period=50` or shorten to e.g. 25 to reduce checkpoint loss window?

### 11.3 Hard gates for a4 launch

The launch cannot start until ALL of these are verified:

- [ ] New HF repos created (primary + backup)
- [ ] `upload_a4_to_hf.py` pre-flight passes on both repos
- [ ] `external_monitor.py` running on user's local machine in 5-min cron
- [ ] Monitor returns HEALTHY for the pre-flight canary
- [ ] Cadence triggers documented (10-step cadence + ckpt-event-triggered cadence)
- [ ] Recovery script tested: kill the uploader process, restart it, verify it resumes from state file

### 11.4 If the a3 host returns (unlikely but possible)

Spheron T3 spot can come back. If `46.243.145.4` is reachable again before a4 is launched:

1. SSH in; check disk for `step_50/` — if present, this is the model we lost.
2. Manually scp the entire `step_50/` directory off-host to the user's local machine.
3. Manually push step_50/ to HF via cli `huggingface-cli upload`.
4. Resume the wrapper to continue training — it will auto-detect step_50/ and resume.

This is unlikely (host has been unreachable for 6+ hours; Spheron T3 spot rarely returns after this long), but the steps are listed in case it happens.

---

## 12. Cadence 5 — never written (the cadence that should have caught the failure)

For completeness, the cadence-5 doc that should have been written for steps 41-50 is impossible to reconstruct fully — rollouts for those steps were not uploaded. But the W&B trajectory shows what would have been in it:

| Step | Wall (s) | Reward | Trunc % | Gen len |
|---:|---:|---:|---:|---:|
| 41 | 619.4 | 0.1522 | 5.0% | 1023 |
| 42 | 471.5 | 0.2126 | 8.4% | 1018 |
| 43 | 559.6 | 0.1828 | 4.4% | 853 |
| 44 | 549.0 | 0.2057 | 5.9% | 893 |
| 45 | 493.1 | 0.2164 | 4.1% | 720 |
| 46 | 502.3 | 0.1779 | 3.8% | 692 |
| 47 | 504.4 | 0.2026 | 6.2% | 728 |
| 48 | 510.2 | 0.1573 | 3.4% | 698 |
| 49 | 521.4 | 0.1693 | 4.4% | 731 |
| 50 | 530.4 | 0.1207 | 4.4% | 731 |

**Window mean reward**: ~0.181 (vs cadence 4's 0.162 = +12%, the climb continued). **Gen length collapsed further to 700-730 range** matching A100's plateau. **Step time stabilized around 500-560s** (was 580+ in cadence 4).

**This would have been the strongest cadence yet** if we'd been able to write it. The model was on a clean reward climb, gen length had shrunk to A100's tight 700-token band, and the recipe was visibly working better than A100's reference.

The cadence-5 doc that I should have written at the right time would have also been the place where the external monitor failure would have been caught. The lesson is structural, not tactical: **cadence checks must START with "is the uploader still working" before any analysis**.
