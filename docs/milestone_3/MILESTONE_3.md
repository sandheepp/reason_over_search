---
title: MILESTONE 3
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-07
---

# Milestone 3: Evaluate [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3) Hybrid

## Context

During the Qwen3-0.6B GRPO ablation block (W&B project `research`, archived in [docs/report/RESULTS_m0_a.md](../report/RESULTS_m0_a.md)), all training runs used the **hybrid** checkpoint (`Qwen3-0.6B`, the post-trained soft-switch reasoning model). The best-documented run in that block is **`p1_basic_w_ex` (W&B id `z7kcxfof`)** — the one that uniquely converged on heavy-tool behaviour (2 search calls / 4 turns, ~2050 token responses, end-of-run reward 0.190 after 1046 steps).

Two model snapshots are available for evaluation:

| Snapshot | Location | Description |
|---|---|---|
| `qwen_3_0.6b` | [`eval/qwen_3_0.6b/`](../../eval/qwen_3_0.6b/) | Qwen3-0.6B hybrid (pre-GRPO; the frozen post-trained checkpoint Qwen releases) |
| `qwen_3_0.6b_v0` | [`eval/qwen_3_0.6b_v0/`](../../eval/qwen_3_0.6b_v0/) | Qwen3-0.6B after 1046 GRPO steps with the p1_basic_w_ex prompt (HF safetensors converted from the `z7kcxfof` verl-FSDP archive on the user's training machine; the raw archive is not retained in this repo). Also published as [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0). |

The verl-FSDP run archive lives externally on the training machine, not in this repo; only the HF-converted safetensors at `eval/qwen_3_0.6b_v0/` are retained.

## Goal

Run Plan B evaluation on both snapshots using **the prompt from the `p1_basic_w_ex_z7kcxfof` training run**, across all 7 benchmarks, on the ALICE cluster.

This answers two questions:
1. What is the zero-shot baseline of Qwen3-0.6B hybrid on these benchmarks (before any GRPO)?
2. Does 1046 steps of GRPO training with the p1_basic_w_ex prompt lift EM on the same benchmarks?

## The p1_basic_w_ex prompt (system message)

Used verbatim as the system message in the eval pipeline:

```text
You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Answers should be based on the search results.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
For example:
Question: What is the nationality of the author of Hamlet?
<search>Hamlet</search>
<result>The Tragedy of Hamlet was written by William Shakespeare.</result>
<search>William Shakespeare</search>
<result>William Shakespeare was an English playwright.</result>
<answer>The final answer is \[ \boxed{English} \]</answer>
```

## Plan B settings (carry-over from M1)

> **⚠ Superseded by M3 actual run (2026-05-07)**: this section captures the *originally planned* Plan B settings carried over from M1. The actual M3 run diverged on several knobs to match the verl-legacy training rollout byte-for-byte (see [`../report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3 for the 14-fix audit), and ran on **full Plan A test/dev sets** (51,713 items / variant) rather than 1k subsamples because `sample_num` is not respected by the FlashRAG `search_r1` pipeline path. Authoritative actual configuration: [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md) § 4. Kept here for plan-vs-execution traceability.

All settings from [docs/report/CODE_SETUP_m1.md](../report/CODE_SETUP_m1.md) applied unchanged, except:

| Setting | M1 value | M3 planned | M3 actual (2026-05-07) | Reason |
|---|---|---|---|---|
| Model | Search-R1 Qwen2.5-3B checkpoints | `eval/qwen_3_0.6b/` and `eval/qwen_3_0.6b_v0/` | same as planned | Different model |
| Prompt format | `SEARCH_R1_TEMPLATE` (user message) | p1_basic_w_ex (system message) | same as planned | Match training distribution |
| Retrieval response wrapper | `<information>` | `<result>` | same as planned (with leading space, byte-aligned to `vllm_rollout.py:419`) | Match training distribution |
| Hardware | Vast.ai RTX 4090 24GB | ALICE A100 80GB | same as planned | In-house cluster |
| `apply_chat` | `True` for both variants | `True` (Qwen3 hybrid) | same as planned | Same rule |
| Retriever index | Flat IP (`--num_retriever 2`) | IVF-SQ8 × 8 workers | same as planned | Throughput under concurrent eval |
| Eval pipeline source | `evaluation_search_r1/` | `evaluation_research/` | same as planned | M3-adapted fork |
| `max_search_turns` | 4 | 4 | **5** (training observed max) | Alignment with training rollout |
| `step_limit` (per call) | 500 | 500 | **8192** (no cap; bounded by `remain_length=4096`) | Alignment with training rollout (no per-step cap) |
| `max_obs_length` | 500 tokens | 500 tokens | **256 tokens** | Match training `max_tool_response_length` |
| `retrieval_topk` | 3 | 3 | **5** | Match training `top_n=5` |
| `enable_thinking` | n/a | n/a | **True** | Otherwise Qwen3 hybrid auto-injects empty `<think></think>` |
| `generator_max_input_len` | 1024 | 1024 | **4096** | Match training `response_length=4096` |
| Subsampling | 1k stratified for the 5 large datasets | 1k stratified for the 5 large datasets | **none — full test/dev sets** (51,713 items / variant; happy-accident upgrade: `sample_num` not respected by FlashRAG `search_r1` path) | See `RESULTS_m3.md` §10.4 |

## Statistical justification

> **⚠ Superseded (2026-05-07)**: this section justifies the originally planned **1 × Plan B** scope. The actual run executed **full Plan A** (51,713 items / variant; population EMs, not subsample estimates), making the SE / McNemar analysis below moot. Kept for reference. See `RESULTS_m3.md` § 10.4 for the actual scope.

M3 was originally planned as **1 seed × Plan B** (subsampled: 1k each for nq/triviaqa/popqa/hotpotqa/2wiki; full for bamboogle=125 and musique=2417). This section justifies that against the alternative — Plan A (full test sets) with multiple seeds — and quantifies the residual uncertainty.

### Greedy decoding makes multi-seed Plan A redundant

Generation uses `temperature=0.0` (greedy). Output is fully determined by input; seeds in the eval pipeline only control *which* items are subsampled, not the model. Under Plan A (all items, no subsampling):

```
k × Plan A ≡ 1 × Plan A   for any k ≥ 1, under greedy decoding
```

Multi-seed Plan A is k× the compute for the same number. The "k seeds for robustness" framing only adds information when seeds change *something* — sampling temperature, subsample selection, augmentation. None apply to Plan A as defined.

### Plan B as an unbiased estimator of the Plan A truth

If subsampling is uniform-random, Plan B's empirical EM is an unbiased estimator of Plan A's population EM. With finite-population correction:

```
SE(p̂) = sqrt((N − n) / (N − 1)) × sqrt(p̂(1 − p̂) / n)
```

For 1 seed × Plan B at the M1 dataset sizes, with realistic p̂:

| Dataset | N (full) | n (Plan B) | p̂ | SE | 95% CI |
|---|---:|---:|---:|---:|---:|
| NQ | 3,610 | 1,000 | 0.40 | 0.013 | ±0.026 |
| TriviaQA | 11,313 | 1,000 | 0.55 | 0.015 | ±0.030 |
| PopQA | 14,267 | 1,000 | 0.40 | 0.015 | ±0.030 |
| HotpotQA | 7,405 | 1,000 | 0.30 | 0.013 | ±0.026 |
| 2WikiMultiHopQA | 12,576 | 1,000 | 0.30 | 0.014 | ±0.028 |
| MuSiQue | 2,417 | 2,417 | 0.10 | 0 | exact |
| Bamboogle | 125 | 125 | 0.10 | 0 | exact |

Each subsampled dataset's EM is known to within **±0.03 EM at 95% confidence** of what Plan A would report.

### What multiple Plan B seeds would buy

Pooling k seeds of n items each (independent draws), expected unique coverage of the full pool is:

```
n_unique = N × (1 − (1 − n/N)^k)
```

For 5 seeds × 1k on TriviaQA (N=11,313): n_unique ≈ 4,640 (41% of full). For NQ (N=3,610): n_unique ≈ 2,890 (80%). SE shrinks by ≈√k_eff, tightening per-dataset CI to ±0.01–0.013 — close to the Plan A truth at a fraction of the compute.

So **5 × Plan B is a reasonable random-sample approximation of 1 × Plan A** (Plan A 3× being a no-op under greedy):

```
P(|p̂_5×PlanB − p_PlanA| < 0.013) ≈ 0.95   (worst case across datasets)
```

### What matters for the M3 comparison question

The deliverable is a paired model comparison: does `qwen3_0.6b_v0` (post-GRPO) lift EM over `qwen3_0.6b` (pre-GRPO)? The unpaired difference statistic:

```
Δ = EM_v0 − EM_base
SE(Δ) ≤ √2 × SE(EM)        ≈ ±0.04 worst case at 1 × Plan B
```

A tighter, paired analysis on the same subsample uses **McNemar's test** on the discordant pairs (b = base wrong & v0 right; c = base right & v0 wrong):

```
McNemar χ² = (b − c)² / (b + c)    under H₀: no model difference
```

Paired comparison removes item-difficulty variance. At n=1k discordants, McNemar detects Δ ≥ 0.02 EM at p<0.05; lifts in the expected range (0.05–0.10 EM if GRPO worked) are detected with high power.

### Conclusion

| Claim | Verdict |
|---|---|
| 5 × Plan B ≈ Plan A truth, within ±0.01–0.013 EM at 95% | Yes (random-sample estimator) |
| 3 × Plan A is more rigorous than 1 × Plan A under greedy | No (identical results) |
| 1 × Plan B is sufficient to detect a real GRPO effect (Δ ≥ 0.05 EM) | Yes (McNemar p<0.01 at this n) |
| Upgrade to 3–5 seeds × Plan B if Δ lands marginal (≈0.02–0.03) | Recommended fallback |

M3 runs at 1 seed × Plan B for both variants. If a marginal result emerges, the same pipeline produces 3–5 seeds at ~5× the compute — still cheaper than 1 × Plan A.

## Environments

| Env name | Python | Purpose |
|---|---|---|
| `retriever` | 3.10 | FAISS retriever service |
| `evaluation_search_r1` | 3.11 | SGLang + evaluation_research eval pipeline |

Both envs are created and verified on the ALICE login node (2026-05-06). The `flashrag` package editable install in `evaluation_search_r1` now points to `evaluation_research/` (contains M3 code changes).

## IVF-SQ8 × 8 workers: resource analysis

The retriever runs the IVF4096-SQ8 index with 8 parallel workers (matching the training-time v1 default). **This is not suitable for paper-fidelity reproduction** (use `--index .../wiki18_100w_e5_flat_inner.index --num_retriever 2` for that), but is fast enough for the M3 comparative eval.

### Index

`indexes/wiki18_100w_e5_ivf4096_sq8.index` — 16 GB on disk, downloaded 2026-05-06. Lives at the **project root** so the relative paths in `local_retriever/retriever_config.yaml` (`./corpus/`, `./indexes/`, `./models/`) resolve when the retriever is launched from there. See §Setup gotchas for the rationale.

### CPU requirements (exact derivation)

| Thread pool | Count | Notes |
|---|---|---|
| FAISS OMP threads | 8 workers × 4 threads = **32** | Diminishing returns beyond 4/worker at nprobe=64 |
| E5-base-v2 encoder | 8 concurrent | Runs single-threaded per call; absorbed in OMP budget |
| FastAPI/uvicorn | ~4 | Event-loop + worker threads |
| SGLang (tokeniser + scheduler) | ~4 | |
| **Total** | **~40** | |

Request: **`--cpus-per-task 40`**. Setting `OMP_NUM_THREADS=4` before launching the retriever ensures FAISS respects the per-worker thread budget.

### RAM requirements

| Component | RAM |
|---|---|
| 8 × IVF-SQ8 index | 8 × 16 GB = 128 GB (measured: 134 GB RSS at v1 launch) |
| 8 × E5-base-v2 encoder | 8 × 440 MB ≈ 3.5 GB |
| Qwen3-0.6B (SGLang, bf16) | ~1.5 GB (VRAM only; negligible CPU footprint) |
| Corpus jsonl + Python overhead | ~10 GB |
| **Total** | **~150 GB** |

Request: **`--mem=160g`** (comfortable headroom; A100 nodes have ≥ 256 GB host RAM).

## Wall-clock estimate (initial; update after validation)

These are pre-run estimates based on model size, hardware specs, and M1 baseline timing. **Update this table after the bamboogle short-partition validation run.**

| Dataset | Items | Est. time (Qwen3-0.6B, A100) |
|---|---:|---|
| bamboogle | 125 | 3–6 min |
| nq | 1 000 | 10–20 min |
| triviaqa | 1 000 | 10–20 min |
| popqa | 1 000 | 10–20 min |
| hotpotqa | 1 000 | 10–20 min |
| 2wikimultihopqa | 1 000 | 10–20 min |
| musique | 2 417 | 25–45 min |
| **Total per variant** | **7 542** | **~1.5–2.5 h** |
| **Both variants** | 15 084 | **~3–5 h** |

Rationale: Qwen3-0.6B (600 M params, 1.2 GB at bf16) on A100 (2 TB/s MBW) is roughly 10× faster per token than Qwen2.5-3B on RTX 4090. Each item averages ~2 search turns × ~300–500 tokens generated + retrieval latency. At INFERENCE_MAX_WORKERS=16 and IVF latency ~0.3 s/call, throughput is dominated by retrieval until the model becomes the bottleneck at larger batch sizes.

## Step-by-step

### 1. Get a SLURM allocation

**For short-partition validation (bamboogle only, one variant):**

```bash
srun \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 \
  --mem=80g \
  --time=00:30:00 \
  --pty bash
```

16 CPUs and 80 GB are sufficient for IVF × 8 (the index loads but not all 8 workers run at peak simultaneously during bamboogle's 125 items). The 30-minute time limit puts this job in the backfill window.

**For full 7-dataset runs, use sbatch (see §5).**

### 2. Load environment and start retriever

```bash
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0

cd /zfsstore/user/s4374886/omega/reason_over_search

export OMP_NUM_THREADS=4
/home/s4374886/.conda/envs/retriever/bin/python local_retriever/retriever_serving.py \
  --config local_retriever/retriever_config.yaml \
  --num_retriever 8 \
  --port 3005 &

# Wait ~60 s for 8 workers to load index (~134 GB RSS on first launch)
sleep 60
curl -sS http://127.0.0.1:3005/health   # → {"status":"healthy","retrievers":{"total":8,"available":8}}
```

The retriever resolves `./corpus/`, `./indexes/`, `./models/` relative to its launch CWD. Launch from the project root and these point to the symlinks/files placed there (see §Setup gotchas).

For the flat IP fallback (paper-fidelity or if IVF not available):
```bash
/home/s4374886/.conda/envs/retriever/bin/python local_retriever/retriever_serving.py \
  --config local_retriever/retriever_config.yaml \
  --num_retriever 2 \
  --index /zfsstore/user/s4374886/omega/re-search/assets/indexes/wiki18_100w_e5_flat_inner.index \
  --port 3005 &
```

### 3. Start the SGLang server

```bash
# qwen_3_0.6b (pre-GRPO)
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python -m sglang.launch_server \
  --model-path eval/qwen_3_0.6b \
  --host 127.0.0.1 --port 3000 \
  --tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code &

# verify
curl -sS http://127.0.0.1:3000/health
```

**If launching via `ssh <node>` from outside an `srun --pty` shell** (i.e., lmod modules aren't auto-loaded so `nvcc` is not on PATH), append `--attention-backend triton --disable-cuda-graph` to avoid flashinfer's nvcc-dependent JIT compile. See §Setup gotchas.

Swap to `eval/qwen_3_0.6b_v0` for the v0 checkpoint run. Restart SGLang between variants.

### 4. Short-partition validation (bamboogle, one variant)

```bash
bash scripts/run_m3.sh qwen3_0.6b bamboogle 1
```

Check the results JSON: confirm search turns average ~1–2, tokens per response reasonable, EM non-zero. Update the wall-clock table above with the observed time.

### 5. Full run via sbatch

```bash
mkdir -p logs
sbatch scripts/sbatch_m3.sh qwen3_0.6b
sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0
```

`sbatch_m3.sh` starts the retriever and SGLang automatically, runs all 7 datasets, then cleans up. It falls back to the flat IP index if the IVF index is missing. Logs go to `logs/m3_<jobid>_m3_eval.{out,err}`.

Time limit: `--time=04:00:00`. If the estimate proves too tight after validation, bump to `06:00:00`.

### 6. Aggregate results

```bash
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python scripts/aggregate.py \
  --output docs/milestone_3/RESULTS_M3.md
```

## Expected results and success criteria

No prior EM numbers exist for Qwen3-0.6B on these benchmarks. Use M1 instruct results as a rough ceiling and M1 base as a rough floor:

| Dataset | M1 base | M1 instruct | Expected 0.6B range |
|---|---:|---:|---|
| Bamboogle | 0.088 | 0.344 | unknown (multi-hop is hard at 0.6B) |
| NQ-1k | 0.390 | 0.402 | lower than 3B |
| TriviaQA-1k | 0.583 | 0.531 | lower than 3B |
| PopQA-1k | 0.424 | 0.413 | lower than 3B |
| HotpotQA-1k | 0.263 | 0.346 | lower than 3B |
| 2WikiMultiHopQA-1k | 0.239 | 0.350 | lower than 3B |
| MuSiQue (2417) | 0.055 | 0.141 | very low |

**Success criterion**: both variants complete all 7 datasets; results are recorded and reproducible. If `qwen_3_0.6b_v0` exceeds `qwen_3_0.6b` on average EM, training helped. If not, document which datasets regressed.

## Deliverables

1. [x] `eval/qwen_3_0.6b/` and `eval/qwen_3_0.6b_v0/` populated and accessible on ALICE. ✓ (done 2026-05-06)
2. [x] `retriever` and `evaluation_search_r1` conda envs created on ALICE. ✓ (done 2026-05-06)
3. [x] IVF-SQ8 index downloaded to `indexes/wiki18_100w_e5_ivf4096_sq8.index` (16 GB). ✓ (downloaded 2026-05-06; relocated from `local_retriever/indexes/` to project root 2026-05-06 — see §Setup gotchas)
4. [x] `evaluation_research/` pipeline fork created with M3 code changes (QWEN3_0_6B_TEMPLATE, `prompt_mode=qwen3`, `<result>` tag swap); editable install verified in `evaluation_search_r1` env. ✓ (done 2026-05-06)
5. [x] `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` created. ✓ (done 2026-05-06)
6. [x] Short-partition validation: bamboogle (qwen3_0.6b) passes; wall-clock table superseded by §Status 2026-05-07 entry.
7. [x] Both variants evaluated on all 7 benchmarks (full Plan A test/dev sets, 51,713 items / variant; happy-accident upgrade from 1k subsamples — `sample_num` not respected by the FlashRAG `search_r1` pipeline path).
8. [x] ~~Results aggregated to `docs/milestone_3/RESULTS_M3.md`~~ — **superseded by [`docs/report/RESULTS_m3.md`](../report/RESULTS_m3.md)** (full numerical record) and [`docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) §4 (consolidated brief).
9. [x] Summary and findings: see [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md) and [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) §4; this milestone's narrative continues in §Status below.

## Setup gotchas (lessons from the slot 1 attempt, 2026-05-06)

These all bit during the first interactive launch on `node872` (job 2120329). Recording the working fixes so future launches don't repeat them.

### 1. Corpus/index/models layout — must sit at the retriever's launch CWD

`local_retriever/retriever_config.yaml` uses relative paths (`./corpus/wiki18_100w.jsonl`, `./indexes/wiki18_100w_e5_ivf4096_sq8.index`, `./models/e5-base-v2/`). They resolve relative to **wherever the retriever process is launched from**, not relative to the config file.

Initial mistake: launched `python local_retriever/retriever_serving.py --config local_retriever/retriever_config.yaml ...` from the project root with `--index` overriding only the index path. The corpus_path stayed relative and resolved to `./corpus/...` at the project root, which didn't exist (corpus had been placed under `local_retriever/corpus/`). Crash:

```
FileNotFoundError: Unable to find '/zfsstore/.../reason_over_search/./corpus/wiki18_100w.jsonl'
```

**Fix (2026-05-06)**: moved the three subdirs to the project root so the relative paths resolve when launching from there:

| Path | Type | Target |
|---|---|---|
| `corpus/wiki18_100w.jsonl` | symlink | `/zfsstore/user/s4374886/omega/flash-rag/corpus/wiki18_100w.jsonl` |
| `indexes/wiki18_100w_e5_ivf4096_sq8.index` | 16 GB file (renamed from `local_retriever/indexes/`) | — |
| `models/e5-base-v2/` | symlink | `/zfsstore/user/s4374886/omega/flash-rag/models/e5-base-v2` |

`local_retriever/{corpus,indexes,models}/` removed. `scripts/sbatch_m3.sh` updated to point `IVF_INDEX` at the new project-root location.

### 2. SGLang requires `ninja` in the env (one-time pip install)

`evaluation_search_r1` did not have `ninja` installed. Without it, flashinfer's JIT compile of attention kernels fails:

```
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

**Fix**: `pip install ninja` in the `evaluation_search_r1` env. Done 2026-05-06.

### 3. `nvcc` not on PATH in non-interactive ssh — use Triton backend

ALICE uses lmod modules. `module load CUDA/12.4.0` works in `srun --pty bash` and in batch (`sbatch`) jobs, but **does not auto-load when you `ssh <node>`** from outside (e.g., to drive a long-running job from the login node). Without CUDA on PATH, flashinfer's CUDA-graph capture fails:

```
Exception: Capture cuda graph failed: Could not find CUDA installation. Please set CUDA_HOME environment variable.
```

**Fix for ssh-launched runs**: pass `--attention-backend triton --disable-cuda-graph` to `sglang.launch_server`. Triton has its own JIT (LLVM-based, no nvcc). Small perf cost vs. flashinfer + cuda-graph but functional.

For `sbatch`-driven runs (where `module load CUDA/12.4.0` actually applies), default flashinfer + cuda-graph is fine.

### 4. First-time corpus → arrow conversion is slow (~1–2 min)

HuggingFace `datasets` converts the 21M-passage `wiki18_100w.jsonl` to its arrow cache on first load. The first retriever worker is slow (~1–2 min). Workers 2–8 reuse the cache and start in seconds. This is a one-time cost per cache location.

### 5. PATH carry-over across nested bash via setsid + nohup

When detaching a long-lived process from an ssh session, both `nohup` and `setsid` are needed:

```bash
ssh <node> "cd $PROJ && setsid bash -c 'PATH=/path/to/env/bin:\$PATH nohup ... > log 2>&1 < /dev/null & echo \$! > pid'"
```

Without `setsid`, the child remains in the ssh session's process group and dies on disconnect even with `nohup`. Without explicit `PATH=…`, the conda env's binaries (e.g., `ninja`) aren't found by subprocess JIT compilers.

## Status

**2026-05-06 (early)**: Models staged (`eval/qwen_3_0.6b/`, `eval/qwen_3_0.6b_v0/`). Both conda envs created and verified. IVF-SQ8 index downloaded (16 GB, originally `local_retriever/indexes/`). `evaluation_research/` fork created from `evaluation_search_r1/` with three targeted changes: `QWEN3_0_6B_TEMPLATE` added to `templates.py`; `prompt_mode='qwen3'` wired through `active_pipeline.py` and `run_eval.py` (system-message chat format + `<result>` observation wrapper); editable install verified. `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` created.

**2026-05-06 (slot 1, ~22:08–22:37, job 2120329, node872)**: Interactive 4 h srun (40 CPU / 160 GB / 1 × A100). Started retriever + SGLang; hit the four gotchas above. Resolved: corpus path (cd into `local_retriever`), `pip install ninja`, switched SGLang to `--attention-backend triton --disable-cuda-graph`. **Slot lost at 22:37:19 to `NODE_FAIL` on node872** (cluster hardware/communication issue, not OOM, not our processes — `sacct` reason `None`, node now `down*`). Logs preserved at `logs/m3_slot1_2120329/`.

**2026-05-06 (post-mortem)**: Moved corpus/index/models to project root so the retriever's relative paths resolve from `cd $REPO_ROOT` (the standard launch pattern in `sbatch_m3.sh`). Documented the four gotchas above. `ninja` permanently installed in `evaluation_search_r1`.

**Pending slots**:
- Job **2120423**: queued, `Resources` reason, planned start 2026-05-06T23:38:40 (4 h). Plan: smoke `qwen3_0.6b` + `qwen3_0.6b_v0` on bamboogle, then full Plan B for `qwen3_0.6b`.
- Job **2121164**: queued, `Priority` reason, planned start 2026-05-07T03:40:00 (4 h, may backfill earlier). Plan: full Plan B for `qwen3_0.6b_v0`.

Next: when 2120423 transitions RUNNING, retry the slot 1 plan with the gotchas already fixed.

**2026-05-07 (COMPLETED)**: Both variants ran on **full Plan A** (no `sample_num` applied — happy-accident upgrade from the planned 1k subsamples; the FlashRAG `search_r1` pipeline path does not respect `sample_num`, so the comparison is statistically Plan A and the per-dataset EMs are population-true). 14 alignment fixes between clone-and-run and the first clean comparison (full audit: [`../report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3). Pre-GRPO interactive ~115 min (job 2120423, node870; mixed `INFERENCE_MAX_WORKERS=16` for NQ/TriviaQA/PopQA → 32 for HotpotQA/2Wiki/MuSiQue). Post-GRPO sbatch **2h 26m 33s** (job 2125009, node875; 32 workers throughout). Headline: average EM 0.102 → 0.155 (+52 % relative, +0.053 absolute) across 51,713 items / variant; 6 / 7 datasets improved; held-out generalisation rules out memorisation. Job 2121164 (the spare slot) was cancelled — both variants fit in slots 2120423 + 2125009. Full numerical record: [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md). Supervisor write-up: [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) §4. **M3 closed; the eval pipeline is now pinned and reusable for Phase-2.**
