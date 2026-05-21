---
title: Code Setup M5 — Qwen3.5-0.8B GRPO training on NeMo-RL (M5 + M5.1)
tags: [report, training, m5, m5.1, qwen3.5, nemo-rl]
source: internal
created: 2026-05-09
updated: 2026-05-12
---

# Code Setup M5: Qwen3.5-0.8B GRPO Training Pipeline (M5 + M5.1)

**Status (2026-05-12)**: §1-§4 populated from `configs/m5_1_research_paper.yaml`. **Three production-track losses behind us** (see [`RESULTS_m5.md` status panel](RESULTS_m5.md) and [`RESULTS_SMOKE_m5.md` §7 / §7.8 / §7.8.1](RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11)): a1 crashed at step-50 ckpt save (config bug); a1 rollout corpus deleted during disk cleanup (irrecoverable); a2 killed at step 15 on a zombie-GPU misdiagnosis. **No production run currently active.** Config fixed (`metric_name: null`, `keep_top_k: null`, `save_optimizer: false`); awaiting authorization to launch a3. M5.1's only surviving rollout corpus is in [`logs/exp_011_a2_archive.tar.gz`](../../logs/exp_011_a2_archive.tar.gz) (22 MB, steps 1-15).
**Date**: 2026-05-09 (M5 + M5.1 design); 2026-05-11 (a1 launched + crashed); 2026-05-12 (3-loss status; awaiting a3).
**Scope**: documents what changed from the M2 NeMo-RL training scaffold ([`training/`](../../training/), Qwen3.5-2B target, paper-default reward + tag scheme) to the M5 / M5.1 pipeline ([`training_m5_1/`](../../training_m5_1/)) for **Qwen3.5-0.8B** training with the **ReSearch-paper recipe** modulo two intentional divergences (F1-only reward, no `\boxed{}` answer wrapper). Train rollout is byte-aligned to the [M4 eval pipeline](../milestone_4/MILESTONE_4.md) so the trained checkpoint is directly evaluable without re-aligning.
**Cluster**: 1× A100-80GB on Vast.ai.
**Source paths**: [`training_m5_1/`](../../training_m5_1/) (M5 + M5.1), [`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml) (M5 smoke), [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) (M5.1 production), [`training_m5_1/scripts/`](../../training_m5_1/scripts/), milestone narrative at [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md).

---

## 1. Headline Diff vs the M2 Pipeline (`training/`)

| Dimension | M2 (`training/`, Qwen3.5-2B target, smoke-tested) | M5 / M5.1 (this doc, `training_m5_1/`) |
|---|---|---|
| **Target model** | Qwen3.5-2B | **Qwen3.5-0.8B** |
| **Training data** | Search-R1 mix (NQ + HotpotQA train splits) | **MuSiQue train split** (single-dataset; multi-hop, hardest of the four paper benchmarks) |
| **Reward** | Search-R1 `qa_em.py` EM-only (M2 default per `training/src/reward.py`) | **F1-only on `<answer>…</answer>`** (M5.1 divergence #1; paper format reward dropped) |
| **Answer wrap** | `<answer>X</answer>` (already plain in M2) | identical (M5.1 divergence #2 vs paper, which uses `\boxed{}` inside `<answer>`) |
| **Action format** | Qwen3.5 nested-XML `<tool_call>` (M2 qwen_native arm) | identical (carried from M4) |
| **Tool-response wrap** | turn-bounded `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n…\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` | identical (must match M4 byte-for-byte) |
| **`max_search_turns`** | 5 | 5 |
| **`max_obs_length` (per chunk)** | 256 tokens (M2 default) | **120 tokens** (M4 v3 lock; per-chunk, no total cap) |
| **`retrieval_topk`** | 5 | 5 |
| **`generator_max_input_len`** | 4096 | 4096 (smoke) / **TODO M5.1** (paper-aligned) |
| **`enable_thinking`** | True | True |
| **GRPO group size** | TODO M5.1 (paper: G=5; ours likely match, confirm against `Agent-RL/ReSearch`) | TODO M5.1 |
| **KL coefficient** | TODO M5.1 | TODO M5.1 (paper-aligned) |
| **Schedule** | NeMo-RL `grpo.max_num_steps: 1005` (M2 v0.2-yaml-aligned, paper-extrapolated from 0.6-epoch budget) | TODO M5.1 (paper-faithful for the recipe; smoke first, then schedule) |
| **W&B project** | `reason_over_search_2b_v1` (M2 placeholder) | **`reason_over_search_m5_1`** |

---

## 2. What's Unchanged (M2 + M4 audit holds)

The M2 NeMo-RL setup decisions catalogued in [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) and [`docs/training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md) transfer to M5 unchanged:

- NeMo-RL `v0.6.0` vendored install, `uv sync --extra vllm` from the pre-warmed wheel cache in `pantomiman/reason-over-search-v1:v1`.
- `policy.generation.vllm_cfg.*` shape (`tensor_parallel_size`, `gpu_memory_utilization`, `enforce_eager`, `kv_cache_dtype`) inherits from `training/configs/grpo_qwen3.5_2b_1xa100.yaml` adapted for 0.8B (smaller model → can raise `gpu_memory_utilization` and `max_num_seqs`).
- `kl_type=k3` (Schulman 2020) — NeMo-RL default matches verl `low_var_kl`.
- State masking — NeMo-RL `token_loss_mask` via role-based masking; tool-response role ≠ "assistant" → loss=0 on retrieved docs (paper equivalent of verl `state_masking=true`).
- Retriever HTTP contract: same as M2 / M4 (POST `/batch_search` with `{queries, topk, return_scores}`; CPU FAISS IVF-SQ8 default, GPU FAISS opt-in).
- Tokenizer / chat template invocation — identical to M4 ([`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py)); rendered-prompt byte-check is part of the M5 smoke deliverables.

---

## 3. M5 Critical Changes (smoke pipeline)

### 3.1 Folder layout (committed 2026-05-09)

`training_m5_1/` is a self-contained copy of `training/` (motivation in [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md) §"Folder layout"). Future experiments are sibling dirs `training_m5_2/`, `training_m5_3/`, …; no shared `src/` or `configs/` between experiments.

```text
training_m5_1/
  nemo_rl/                            # SYMLINK -> ../training/nemo_rl/  (disk: 23 G; symlink until disk allows a real copy)
  src/                                # overlay; copied from training/src/, then 14 import sites + several files edited
    rewards/search_r1.py              # CHANGED: F1-only on <answer>...</answer> (re-exports normalize_answer + extract_solution from evaluation_qwen35)
    environments/parsers.py           # CHANGED: qwen_native parse_query delegates to evaluation_qwen35 extract_tool_call_query
    environments/search_r1_env.py     # CHANGED: docstring rewrite, em_check fallback -> f1_check, em_hit_rate -> near_em_rate, max_chars kwarg fix
    datasets/search_r1.py             # unchanged structure; data_path now points at MuSiQue parquet
    processors/search_r1.py           # docstring comment updated for MuSiQue
    chat_template/tools.py            # CHANGED: re-export QWEN35_SEARCH_TOOL from evaluation_qwen35
    prompts/m5_qwen35_user.txt        # NEW: pre-staged from M4.2 canonical (qwen35_minimal); regenerate via sync_m4_prompts.py
    prompts/_archive_m2/              # M2 prompt files archived here
    registry.py                       # CHANGED: all imports renamed training. -> training_m5_1.
  configs/
    m5_smoke.yaml                     # NEW: 20 traj/step, 50 steps, MuSiQue, qwen_native, 1xA100
    m5_1_research_paper.yaml          # TODO (M5.1 step 6): paper-faithful production config
    _archive_m2/                      # grpo_qwen3.5_2b_{1,2}xa100.yaml archived here
  scripts/
    run.sh                            # NEW: --mode smoke|prod, --seed N, [-- extra hydra overrides]
    smoke.sh                          # NEW: thin alias for `run.sh --mode smoke`
    run_grpo.py                       # CHANGED: REPO_ROOT/training_m5_1/nemo_rl path; import training_m5_1.src.registry
    prep_musique.py                   # NEW: downloads RUC-NLPIR/FlashRAG_datasets musique/train.jsonl -> data/training/musique/train.parquet
    sync_m4_prompts.py                # NEW: materialises QWEN35_TEMPLATES[mode] into src/prompts/m5_*.txt
    bootstrap.sh, bootstrap_alice.sh  # still M2-shaped; rewrite when M5.1 launches on a fresh Vast box
    _archive_m2/                      # run_grpo_{1,2}xa100.sh + prepare_dataset.py archived here
  tests/
    test_reward_parity.py             # CHANGED: rewritten for F1 semantics; M2 byte-parity assertion dropped
    test_parser_dispatch.py           # unchanged (still pinning qwen_native + paper arm regex behaviour)
    test_format_helpers.py            # unchanged
    test_env_step.py, test_dataset_adapter.py  # unchanged
  setup.sh                            # mirror of training/setup.sh; comments still mention training/ (cosmetic)
  README.md                           # CHANGED: M5.1-specific narrative + run sequence
```

### 3.2 Overlay deltas vs `training/src/` (M5; status 2026-05-10)

| File (M5 layout) | M2 (`training/src/`) | M5 (`training_m5_1/src/`) | Status | Why |
|---|---|---|---|---|
| `rewards/search_r1.py` | EM-with-shaping (Search-R1 `qa_em.py` port) | F1-only on `<answer>...</answer>`; re-exports `normalize_answer` + `extract_solution` from `flashrag.search_r1.reward`; adds `f1_check` | **done** | M5.1 divergence #1; same scorer code path as M4 eval |
| `environments/parsers.py` | local `_RE_QWEN_QUERY` regex | qwen_native arm delegates to `flashrag.search_r1.parser.extract_tool_call_query`; paper arm regex kept local | **done** | Single source of truth for the action parser (M3 14-fix precedent) |
| `environments/search_r1_env.py` | turn-bounded `<tool_response>` wrap; `em_check` fallback; `em_hit_rate` metric | identical wrap + byte-for-byte M4 alignment; `f1_check` fallback under truncation; `near_em_rate` metric (F1 ≥ 0.8); fixed `max_chars_per_chunk` kwarg dispatch bug (was `max_chars=`, would TypeError on first qwen_native search turn) | **done** | Hard alignment + M5.1 reward consistency + latent bug fix |
| `chat_template/tools.py` | local `SEARCH_TOOL` dict | re-export `QWEN35_SEARCH_TOOL` (aliased as `SEARCH_TOOL`) from `flashrag.search_r1.templates` | **done** | Locks training schema = eval schema |
| `prompts/m5_qwen35_user.txt` | n/a (M2 used `search_r1_qwen_native_user.txt`) | pre-staged from M4.2 canonical (`qwen35_minimal`); written by `scripts/sync_m4_prompts.py --mode <key>` | **done; re-run after M4.4 lock** | Dynamic sync avoids hard-coding the prompt text in training overlay |
| `datasets/search_r1.py` | reads NQ + HotpotQA parquet | unchanged code; new `data_path: data/training/musique/train.parquet` in m5_smoke.yaml | **done** (dataset-agnostic adapter; only the config path changes) | M5.1 dataset choice (MuSiQue only) |
| `processors/search_r1.py` | docstring mentions NQ+HotpotQA only | docstring updated to mention both NQ+HotpotQA (M2) and MuSiQue (M5.1) | **done** (no functional change; `data_source` field optional) | Documentation only |
| `registry.py` | `from training.src...` imports | `from training_m5_1.src...` (14 import sites renamed across src/, tests/, scripts/) | **done** | Package isolation between sibling experiments |
| **Tests** | M2 byte-parity assertions | rewritten `test_reward_parity.py` with re-export identity checks + F1 semantics + 5 hand-picked rollouts (milestone doc M5.1 step 6 requirement) | **done** (23 pure-Python tests pass) | M5.1 invariants |

### 3.3 Smoke config (`configs/m5_smoke.yaml`) vs production (`configs/m5_1_research_paper.yaml`)

Smoke validates the pipeline end-to-end and produces a per-step time number on 1× A100-80GB. Production is paper-faithful + M5.2 system gains.

| Knob | Smoke (v6, locked) | Production (M5.1, live) | Notes |
|---|---|---|---|
| `policy.model_name` | `Qwen/Qwen3.5-0.8B` | same | hybrid; base variant would be a separate sub-run |
| `policy.train_global_batch_size` | 20 | **320** | smoke 4×5; prod 64×5 |
| `policy.train_micro_batch_size` | 2 (v6 lock; v4 OOM'd at 4) | **1** (v7 OOM at 2; seq=8192 doubles log_softmax peak) | NeMo-RL `model_utils.py:1378` doesn't chunk fp32 cast on TP=1 path → deferred patch (M5.3) would unlock micro=2 |
| `policy.max_total_sequence_length` | 4096 | **8192** | paper budget; doubles training memory peak vs smoke |
| `policy.generation.max_new_tokens` | 1024 | 1024 | Group-C lock 2026-05-10 (paper budget is 8192 total ≈ 5 turns × 1024) |
| `policy.generation.vllm_cfg.gpu_memory_utilization` | 0.5 | 0.5 | v5 OOM forced 0.7 → 0.5; v7 stayed 0.5 (no headroom to give back) |
| `policy.generation.vllm_cfg.async_engine` | false | **true** | M5.2 R2 lever (vLLM continuous batching); ~1.3–1.5× on rollout phase |
| `policy.optimizer.kwargs.fused` | false | **true** | M5.2 O1 lever (fused AdamW); ~1.03–1.08× on full step |
| `policy.dynamic_batching.enabled` | true | true | Qwen3.5 hybrid arch can't use sequence_packing |
| `policy.sequence_packing.enabled` | false | false | crashes GatedDeltaNet kernel (training/fix/CHANGES.md §5) |
| `policy.dtensor_cfg._v2` | true | true | v1 hard-codes `model.model.layers`; Qwen3.5 hides them behind nemo_automodel |
| `grpo.num_prompts_per_step` | 4 | **64** | paper bs=256; ours = 256/4 chunked smaller (epoch-count preserved) |
| `grpo.num_generations_per_prompt` | 5 | 5 | paper G=5 |
| `grpo.max_num_steps` | 10 (Group-C lock) | **622** | prod: 2 epochs × ⌊19,938 / 64⌋ = 2 × 311 |
| `grpo.use_leave_one_out_baseline` | false (Group-C) | false (Group-C) | paper uses group-mean, not LOO |
| `grpo.max_rollout_turns` | 10 | 10 | bounded by 8192-token budget anyway |
| `loss_fn.reference_policy_kl_penalty` | 0.001 | **0.001** | verl `kl_loss_coef` (paper) |
| `loss_fn.reference_policy_kl_type` | k3 | **k3** | NeMo-RL `k3` ≡ verl `low_var_kl` |
| `loss_fn.ratio_clip_{min,max}` | 0.2 / 0.2 | **0.2 / 0.2** | PPO clip ε=0.2 (paper) |
| `policy.optimizer.kwargs.lr` | 1.0e-6 | **1.0e-6** | verl `lr=1e-6` (paper) |
| `policy.optimizer.kwargs.weight_decay` | 0.01 | **0.01** | paper |
| LR schedule | constant (Group-C) | **constant** | paper `lr_warmup_steps_ratio=0.0` |
| `policy.generation.temperature` / `top_p` | 1.0 / 1.0 | **1.0 / 1.0** | verl rollout (paper) |
| `data.dataset_name` | `search_r1` (MuSiQue parquet) | same | single-dataset |
| `data.train.data_path` | `data/training/musique/train.parquet` | same | MuSiQue train, 19,938 rows |
| `env.search_r1.top_n` | 5 | 5 | paper `retrieval_topk=5` |
| `env.search_r1.max_obs_chars` | 1024 (Group-C) | 1024 | paper has no cap; we keep safety net |
| `checkpointing.enabled` | false (smoke) | true; `save_period: 50`, `keep_top_k: 0` | first ckpt at step 50 |
| `checkpointing.metric_name` | n/a | `null` (was `train/loss/mean` @ db0852b — crashed a1 step 50) | NeMo-RL requires `train:` or `val:` colon prefix; `null` bypasses the assertion. Fix postmortem [`RESULTS_SMOKE_m5.md` §7](RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11) |
| `checkpointing.keep_top_k` | n/a | `null` (was `0` @ db0852b — would delete all saves) | `0` is slice-from-zero (`checkpoint.py:266` — wipes everything); `null` retains all (`checkpoint.py:46` docstring) |
| `checkpointing.save_optimizer` | n/a | `false` | Per-save 8.9 GB → 3.2 GB (verified by smoke-ckpt-verify2). Loses exact-resume on interruption (AdamW re-warms at constant LR). |
| `grpo.val_period` / `val_at_*` | 0 / false | 0 / false | val disabled (MuSiQue dev parquet not generated; eval out-of-band via `evaluation_qwen35`) |
| `logger.wandb.project` | `reason_over_search_m5_1_smoke` | `reason_over_search_m5_1` | per-experiment isolation |

The smoke result anchor (v6, 10 steps, mean 93.1 s/step ex-warmup) and the v7 production OOM postmortem are in [`RESULTS_SMOKE_m5.md` §2-§3](RESULTS_SMOKE_m5.md#2-v6--m5-smoke-pipeline-validation-smoke-shape--success).

### 3.4 Verification (M5)

Two byte-level checks before declaring M5 smoke green:

1. **Rendered prompt parity**. On 1 hand-picked MuSiQue question, capture `tokenizer.apply_chat_template(...)` output from (a) `training_m5_1/src/retrieval_env.py:render_initial_prompt` and (b) `evaluation_qwen35/flashrag/pipeline/active_pipeline.py` qwen35-branch first-turn prompt. `diff -u` must be empty.
2. **Reward path on 5 hand-picked rollouts**:
   - "Right entity, exact match" → F1 = 1.0
   - "Right entity, extra words" → 0 < F1 < 1
   - "Wrong entity" → F1 = 0
   - "Empty `<answer>`" → F1 = 0
   - "Format-broken (no `<answer>` tag)" → F1 = 0 (no format penalty added — this is the M5.1 divergence #1)

### 3.5 What gets logged (M5 smoke) — Partial (2026-05-10)

**Step 1 ground truth from the 4th smoke attempt** ([wandb run `jeedpsjq` in project `reason_over_search_m5_1`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m5_1/runs/jeedpsjq)):
- Setup wall-clock: **73.0 s** (vLLM init 44.1 s + Policy init 9.0 s + other 15.7 s; both worker venvs reused from disk cache)
- **Step 1 wall-clock: 145.58 s** at 20 trajectories (4 prompts × 5 group)
- Step 1 rollout produced a 5-turn dialogue on most prompts (env.step exercised the full search→retrieve→continue→answer path)

**Step 2 OOM'd**: `log_softmax` over 248,320 vocab needed 15.15 GB, only 14.84 GB free; off by 0.31 GB. Caused by vLLM's sleep-mode resident footprint (6.74 GB) + PyTorch allocator fragmentation. Fixed by:
- `train_micro_batch_size: 4 → 2` (halves the log_softmax tensor per microbatch)
- `gpu_memory_utilization: 0.7 → 0.5` (vLLM gives back more memory on sleep)
- (`PYTORCH_ALLOC_CONF=expandable_segments:True` was tried but breaks NeMo-RL's CUDA IPC weight sharing — rejected)

**Group C resolutions applied to m5_smoke.yaml** (see [`../milestone_5/PAPER_VS_OURS_M5.md §8`](../milestone_5/PAPER_VS_OURS_M5.md) for the locked decisions):
- `use_leave_one_out_baseline: true → false` (paper uses group-mean)
- `max_new_tokens: 500 → 1024` (paper budget is 8192 total)
- `max_obs_chars: 480 → 1024` (paper has no per-obs cap)
- `lr_warmup: 14 steps → 0` (paper uses no warmup)
- `max_num_steps: 50 → 10` (smoke validation only needs ~5 stable steps)

**Remaining smoke deliverables** (from the next launch with the above fixes):
- per-step wall-clock mean / p50 / p95 over steps 1-10
- reward/mean trajectory + `near_em_rate` (F1 ≥ 0.8 rate)
- generation-token-count distribution
- `tool_call_counts/mean` per rollout
- gradient-norm + clip-ratio trajectories

---

## 4. M5.1 Critical Changes (ReSearch-paper-aligned config) — awaiting a3

`configs/m5_1_research_paper.yaml` is **committed; not currently running**. Run history:
- **a1** (exp_010, launched 2026-05-11 01:05 UTC, W&B `uwbodqgt`): ran 49 clean steps, **crashed at first ckpt save** ([`RESULTS_SMOKE_m5.md` §7](RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11)). Rollout corpus subsequently deleted by accident ([§7.8.1](RESULTS_SMOKE_m5.md#781-data-deletion-loss--deleted-a1s-rollout-corpus-during-disk-cleanup-2026-05-11)).
- **a2** (exp_011, launched 2026-05-11 22:23 UTC, W&B `2b95h2fg`): ran 15 clean steps, **killed mid-run on a misdiagnosis** ([`§7.8`](RESULTS_SMOKE_m5.md#78-companion-postmortem--the-zombie-gpu-memory-misdiagnosis-2026-05-12)). Rollout corpus archived in [`logs/exp_011_a2_archive.tar.gz`](../../logs/exp_011_a2_archive.tar.gz).
- **a3** (TBD): awaits user authorization. Same yaml; ckpt fix verified end-to-end by two smokes.

Spine populated from the live config + the paper-vs-ours audit.

### 4.1 Paper-vs-ours mapping (PAPER_VS_OURS_M5.md)

Companion at [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md): clause-by-clause mapping of every concrete number in the [ReSearch paper](https://arxiv.org/abs/2503.19470) and every concrete config value in [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) to our `configs/m5_1_research_paper.yaml`. Group-A/B/C decisions catalogued there; Group-C decisions also surfaced inline in the smoke yaml. **Divergences kept** (flagged in red in PAPER_VS_OURS_M5):

- **Reward**: paper = F1 + 0.1 floor for non-empty + format gate; ours = F1-only on `<answer>…</answer>` content (matches M4 eval, simpler reward surface).
- **Answer wrap**: paper = `<answer>The final answer is \[ \boxed{X} \]</answer>`; ours = `<answer>X</answer>` (M4 already drops `\boxed{}`; eval scorer accepts both).
- **`num_prompts_per_step`**: paper 256; ours 64 (1× A100-80GB OOMs at 256). Same per-epoch trajectory count; just more optimizer steps with smaller advantage chunks.
- **`train_micro_batch_size`**: paper doesn't specify; ours 1 (v7 OOM at 2 — NeMo-RL TP=1 fp32-cast path; deferred patch in M5.3 would unlock 2).
- **`max_obs_chars`**: paper no cap; ours 1024 char safety net (~256 tokens) for retrieval-blowup protection.

### 4.2 Hyperparameters (live `m5_1_research_paper.yaml`)

| Knob | Paper / verl | Ours (yaml) | Match? |
|---|---|---|---|
| Algorithm | GRPO | GRPO | ✓ |
| KL type | verl `low_var_kl` | `k3` (NeMo-RL equivalent) | ✓ |
| KL coefficient | `kl_loss_coef = 0.001` | `reference_policy_kl_penalty: 0.001` | ✓ |
| Group size G | `n_agent = 5` | `num_generations_per_prompt: 5` | ✓ |
| Prompts per step | 256 (paper bs) | **64** | ⚠ hardware divergence |
| Train micro-batch | unspecified | **1** | ⚠ v7 OOM at 2 |
| Max response length / total seq | `max_prompt_length = 8192` | `max_total_sequence_length: 8192` | ✓ |
| Generation max-new-tokens | (per-turn implicit; paper budget 8192 total) | `max_new_tokens: 1024` × ~5 turns | ✓ |
| PPO clip ε | 0.2 | `ratio_clip_{min,max}: 0.2` | ✓ |
| Learning rate | `lr = 1e-6` | `lr: 1.0e-6` | ✓ |
| Weight decay | (verl default 0.01) | `weight_decay: 0.01` | ✓ |
| Optimizer | AdamW | `torch.optim.AdamW` (`fused: true` — M5.2 O1) | ✓ |
| Warmup | `lr_warmup_steps_ratio = 0.0` | LinearLR start=end=1.0 (no-op) + ConstantLR | ✓ |
| Schedule | ~156 steps at bs=256 | **622 steps at bs=64** | ✓ (same trajectory count) |
| Epochs | 2 | `max_num_epochs: 2` | ✓ |
| Generation temperature / top_p | 1.0 / 1.0 | `temperature: 1.0`, `top_p: 1.0` | ✓ |
| Retrieval `top_n` | 5 | 5 | ✓ |
| Dataset | NQ + HotpotQA (paper mix) | **MuSiQue only** | M5.1 dataset divergence (hardest single benchmark) |
| Reward | F1 + format + EM partial | **F1 only on `<answer>...</answer>`** | M5.1 divergence #1 |

### 4.3 Reward implementation

`training_m5_1/src/rewards/search_r1.py` (overlay; replaces `training/src/reward.py`). Re-exports `normalize_answer` + `extract_solution` from `evaluation_qwen35.flashrag.search_r1.{answer_utils,reward}` so train and eval use the **same scorer code path** (M3-style train/eval drift impossible by construction):

```python
def compute_reward(rollout):
    pred = extract_solution(rollout["response"])    # first <answer>…</answer> block
    if pred is None:
        return 0.0                                   # no format penalty (M5.1 divergence #1)
    return f1_score(pred, rollout["gold_answer"])   # token-level F1, lowercase + strip + punct-norm
```

Tests live in `training_m5_1/tests/test_reward_parity.py` (re-export identity checks + F1 semantics + 5 hand-picked rollouts: right-exact / right-extra-words / wrong / empty / format-broken).

### 4.4 Verification (M5.1)

Two byte-level checks (both deliverables of the M5.1 launch); the first is automated in the test suite, the second is the 5-rollout sanity check.

1. **Rendered prompt parity**: `training_m5_1/src/processors/search_r1.py`'s first-turn output must equal `evaluation_qwen35.flashrag.pipeline.active_pipeline` qwen35-branch first-turn prompt byte-for-byte. `diff -u` must be empty.
2. **Reward path on 5 hand-picked rollouts** — see §3.4 categories.

**Wall-clock projection (revised live)**: ~10.6 min/step steady-state × 622 steps = **~4.5 d on 1× A100-80GB** (~$130 on Vast). Per-GPU alternative estimates in [`../setup/HARDWARE_COMPARISON.md` §3](../setup/HARDWARE_COMPARISON.md#3-m51-wall-clock--cost-estimates-by-hardware): 1× H100 SXM ~2 d / $90, 1× B200 ~14–16 h / $90. Per-step trajectory (collapsed 58 → 10 min over steps 1–17) in [`RESULTS_SMOKE_m5.md` §6.2](RESULTS_SMOKE_m5.md#62-per-step-trajectory-live-refresh-as-steps-land).

---

## 5. M5 Run sequence

```bash
# Bootstrap fresh GPU instance per docs/setup/SETUP_INSTANCE.md
cd /workspace/reason_over_search
cd training_m5_1
bash setup.sh                           # uv sync --extra vllm against vendored nemo_rl/

# M5 smoke (pipeline validation; 10 steps, smoke shape)
bash scripts/smoke.sh                   # 93.1 s/step (mean ex-warmup); see RESULTS_SMOKE_m5 §2.

# M5.2 baseline (production shape × 10 steps; no system gains)
bash scripts/run.sh --mode prod -- grpo.max_num_steps=10 grpo.val_period=0 checkpointing.enabled=false

# M5.1 full run (paper-faithful + Phase 2 system gains winner)
bash scripts/run.sh --mode prod
```

Per-experiment W&B project naming: `reason_over_search_m5_<N>` (smoke uses `…_smoke` suffix). Avoids cross-experiment metric pollution.

---

## 6. M5.2 — System-gains iteration plan

After the M5.1 paper-faithful baseline lands (`m5_1_research_paper.yaml` authored 2026-05-10), Phase 2 explores **non-paper-faithful throughput levers** on the same branch (paper-faithful state preserved on `research_v2_a`).

Lever menu, scope, and per-iteration protocol: [`RESULTS_SMOKE_m5.md` §5](RESULTS_SMOKE_m5.md#5-phase-2--m52-system-gains-plan).

User-imposed boundaries (2026-05-10):
- **Paper-faithful values are locked**: G=5, batch shape, KL term, reward, max_total_sequence_length, num_steps.
- Source menu: [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), filtered to levers that change only throughput.
- Skipped: G=5→4 (touches GRPO baseline), seq=4096→3072 (truncation rate), DAPO / drop-KL / LoRA / async-GRPO (training math), EAGLE-3 / 8-bit-AdamW / decolocation (PR-level effort or 2-GPU).

Per-iteration protocol:
1. Single-lever delta against the v7 baseline. No stacking until each lever is individually measured.
2. 10 steps at production shape (`num_prompts_per_step=64`, `max_total_sequence_length=8192`).
3. Record mean s/step (steps 2-10, ex-warmup), peak VRAM, new failure modes.
4. Stack levers in order; record in [`RESULTS_SMOKE_m5.md` §6](RESULTS_SMOKE_m5.md#6-phase-2-results--todo); commit each smoke's config + log under `logs/exp_<N>/`.

---

## 7. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 / M5.1 smoke results: [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- M5.1 paper-vs-ours mapping (companion to this doc): [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) (TODO)
- M2 NeMo-RL paper-to-NeMo mapping (the foundation): [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md), [`../training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md), [`../training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md)
- M2 smoke (the wall-clock anchor): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- M4 eval pipeline (the rollout-shape source of truth): [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md), [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470), notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md)
- ReSearch official codebase: [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch)
- Active recipe-ablation plan: [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)
