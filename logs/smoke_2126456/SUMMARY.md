# Smoke test report — 2126456 (node870, 1× A100 80GB, 4h walltime)

Started: 2026-05-07 22:15 (effective; user walked away at 19:25 boot, I picked up at 22:15)
SIF: `/zfsstore/user/s4374886/apptainer/reason-over-search-v1.sif` (`pantomiman/reason-over-search-v1:v1`)

## Phase 1 — image bring-up — PASS
- Apptainer SIF exists (~20 GB), GPU visible inside container (1× A100 80GB PCIe).
- Container venvs present: `/venv/main`, `/venv/retriever`, `/venv/evaluation_search_r1`. All import flashrag + faiss cleanly.
- nemo_rl venv at `training/nemo_rl/.venv`, torch+CUDA work (`torch.cuda.is_available() == True`).

## Phase 2 — retriever — PASS (after two fixes)

Two issues had to be patched to make the retriever start:

1. **Missing symlinks under `local_retriever/`**: the config expects `./models/`, `./indexes/`, `./corpus/` relative to that dir, but they live one level up at the repo root. Fixed with three symlinks (`local_retriever/{models,indexes,corpus}` → `../{models,indexes,corpus}`).
2. **Apptainer bind missing `/zfsstore/.../flash-rag`**: the assets at the repo root are themselves symlinks pointing into `/zfsstore/user/s4374886/omega/flash-rag/`. The default Apptainer bind only includes the repo + HF cache, so those symlinks were unresolvable inside the container. Fixed with an extra `--bind $FLASH:$FLASH`.
3. **`flashrag.config.Config` does not handle relative paths well**: it tried to load `./models/e5-base-v2/` as a HF repo id and threw `HFValidationError`. Worked around with a smoke-only config (`local_retriever/retriever_config_smoke.yaml`) that uses absolute paths inside the container (`/workspace/reason_over_search/{models,indexes,corpus}/...`).

After these fixes:
- IVF-SQ8 + CPU FAISS, `--num_retriever 1`, port 3005.
- Cold-start ingest of `wiki18_100w.jsonl` (~21 M rows) into HF datasets cache: ~5 min one-off (cache now warm at `/zfsstore/.../hf_cache/datasets/json/default-a8be0c531ac907dd/`, 13 GB).
- FAISS index load: ~3 min (RSS climbs to ~18 GB).
- `GET /health`: `{"status":"healthy","retrievers":{"total":1,"available":1}}` ✓
- `POST /search` (`"Who wrote The Lord of the Rings?"`, top_n=3): returned three top-1-correct Tolkien snippets ✓

## Phase 3 — eval pipeline — SKIPPED

The Search-R1 GRPO checkpoint (`PeterJinGo/SearchR1-...-grpo`) is **not on disk** anywhere under `/zfsstore/user/s4374886`; only `Qwen3.5-2B` and `Qwen3.5-2B-Base` are in `hf_cache/hub/`. Downloading the ~6 GB checkpoint plus standing up SGLang would have used the rest of the walltime, with no headroom left for training. Eval pipeline is not validated here; it was reproduced offline before (`docs/report/RESULTS_m1.md`).

## Phase 4 — training smoke — PARTIAL (init validated; step 1 hit a config bug, mine)

Smallest viable shape (max_num_steps=1, num_prompts_per_step=2, train_global_batch_size=8, train_micro_batch_size=2, sequence_packing off, dynamic_batching on, val off, ckpt off, wandb off).

Two issues had to be patched to make `run_grpo_1xa100.sh` start under Apptainer:

1. **Ray `_temp_dir` defaulted to `/scratchdata`** (read-only inside container). `OSError: [Errno 30] Read-only file system: '/scratchdata'`. Fixed with `RAY_TMPDIR=/tmp/ray_smoke` and `TMPDIR=/tmp` in the apptainer env.
2. **`grpo.val_period=999` does not disable validation**: NeMo-RL still asserts a validation dataset exists. Fixed by setting `grpo.val_period=0` and `grpo.val_at_start=false` (matches the YAML defaults; my override was the problem).

Validated up to actor load:
- Hydra overrides parsed correctly (model name, seed, arm, batch sizes, packing flags).
- Ray cluster came up (GCS, dashboard, monitor, head actors).
- vLLM rollout worker: model loaded, CUDA-graph captured (35 graphs, 0.46 GiB), engine init **182 s**. Sleep-mode swap working: freed 36.2 GiB on `cumem.sleep()`.
- DTensor V2 policy worker (`IsolatedWorkerInitializer`): initialized in **311 s**.
- vLLM backend confirmed for Qwen/Qwen3.5-2B-Base.

Non-fatal warnings (worth noting, not blocking):
- "FLOPS tracker not supported for Qwen3_5Config" — NeMo-RL FLOPs counter doesn't recognise the new arch class. Cosmetic; metric will be zero but training proceeds.
- "TORCH_CUDA_ARCH_LIST is not set" — DeepEP path, not used in our config.
- "Apex is not installed. Falling back to Torch Norm" — expected under the SIF.

Step 1 began (`========================= Step 1/1 =========================` and `▶ Generating responses for batch of size 10...` logged) but failed in the trainer's data-shard step:

```
AssertionError: Total batch size (10) is not a multiple of batch_size (8)
  in nemo_rl/distributed/batched_data_dict.py:375 shard_by_batch_size
```

Cause: my override `policy.train_global_batch_size=8` does not divide the rollout output `num_prompts_per_step=2 × G=5 = 10`. Divisibility is asserted. **Fix for next run**: set `policy.train_global_batch_size=10` (or 5 or 2 or 1) — must be a divisor of `num_prompts × G`. The default in [`grpo_qwen3.5_2b_1xa100.yaml`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml) is consistent with the YAML's own `num_prompts_per_step` and `policy.generation.num_generations_per_prompt`; the bug is that I mis-overrode without checking.

Init alone consumed ~17 min (vLLM init 3 min + Ray + DTensor 5 min + the rest). Per the 2026-05-06 smoke baseline step time at this shape is ~57 s, so a corrected re-launch with init still warm would complete step 1 in well under 5 min. With a cold container the first launch needs ≥30 min walltime budget for safety.

Tear-down note: `__del__` raised `PythonFinalizationError: preexec_fn not supported at interpreter shutdown` while Ray actor shutdown tried to re-init Ray during finalisation. Cosmetic only (no leaked GPU memory in `nvidia-smi` after exit), but worth filing if it recurs.

## What works end-to-end as of this run

- Apptainer SIF + bind layout: works, with one extra bind (`/zfsstore/.../flash-rag`).
- Retriever: works with absolute-path config + symlinks + flash-rag bind.
- Training pipeline: parses config, brings up Ray + vLLM + DTensor; the Qwen3.5 GDN-kernel + dynamic-batching path that Q1 in `docs/research/QUESTIONS.md` describes is configured and didn't crash on init.

## What I patched (reversible, all under `logs/smoke_2126456/` or as symlinks)

- `local_retriever/retriever_config_smoke.yaml` — new file, absolute paths.
- `local_retriever/{models,indexes,corpus}` — three symlinks → repo-root counterparts.
- `logs/smoke_2126456/run_train_smoke.sh` — pinned-shape launcher with RAY_TMPDIR + flash-rag bind.

## What still needs validating (next session)

- Step 1 actually runs end-to-end (rollout + retriever calls + train + logprob produces real numbers). Re-launch with `policy.train_global_batch_size=10` (or another divisor of `num_prompts_per_step × num_generations_per_prompt`); init is now cached, should hit step 1 in ~10 min on a fresh `gpu-short` slot.
- Eval smoke: needs the SearchR1 GRPO checkpoint download (~6 GB) plus SGLang. Skipped here.
- Validate the GDN packing-off mitigation under actual rollout (will the kernel issue surface mid-step?). Tied to `QUESTIONS.md` Q1 follow-up.
