---
title: README
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-06
---

# Training docs

This directory documents the training-side reproduction of [Search-R1](https://www.alphaxiv.org/abs/2503.09516) on **Qwen3.5-2B** via [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). Owned by [Milestone 2](../milestone_2/MILESTONE_2.md).

> Code lives at [`training/`](../../training/); these docs explain the *why*.

---

## Index

| File | What it covers | Read when |
|---|---|---|
| [TRAINING_DATA.md](TRAINING_DATA.md) | `PeterJinGo/nq_hotpotqa_train` schema, our reshape (strip prebaked template, `prompt → messages`), NeMo-RL row-schema mapping | working on the dataset prep script or wondering about `messages[0].content` |
| [CHAT_TEMPLATE.md](CHAT_TEMPLATE.md) | Two chat-template arms — Qwen3.5 native `<tool_call>` (default) vs. paper's `<search>` (ablation) — with verbatim Qwen3.5 jinja | wiring the tokenizer / prompt; choosing an arm |
| [PAPER_VS_OURS_TRAINING.md](PAPER_VS_OURS_TRAINING.md) | The canonical hyperparameter audit — every knob in our run vs. the upstream verl yaml that produced the paper's published checkpoints | before launching a training run |
| [VERL_REFERENCE.md](VERL_REFERENCE.md) | verl-side references porting to NeMo-RL: HTTP retriever contract, KL/GRPO mappings, FSDP→DTensor translations, JSONL log shape to mirror | reading verl scripts; resolving a NeMo-RL config knob |
| [VALIDATION.md](VALIDATION.md) | In-loop validation plan — datasets, cadence, metrics, sampling | wiring W&B; deciding val_period |
| [NEMO_RL_KNOBS.md](NEMO_RL_KNOBS.md) | Memory + throughput + algorithm knobs with our recommended starting values for A100 80GB; concrete starting yaml | tuning a config |

---

## End-to-end view

```
                  PeterJinGo/nq_hotpotqa_train (HF)
                          │
                          ▼  prepare_dataset.py
        data/training/nq_hotpotqa_train/{train,test}.parquet  (Git LFS)
                          │
                          ▼  SearchR1Dataset    (training/src/datasets/)
                  task_name = search_r1_{arm}
                          │
                          ▼  search_r1_processor  (training/src/processors/)
            apply_chat_template + extract golden_answers
                          │
                          ▼  GRPO rollout loop  (vendored NeMo-RL)
                  policy generates  ──→  SearchR1Environment
                                          │   (training/src/environments/)
                                          │
                          ┌───────────────┴──────────────┐
                          ▼                              ▼
                  parse_query() per arm             extract_solution()
                          │                              │
                          ▼                              ▼
                  POST /batch_search                EM via reward
                  (local_retriever:3005)            (training/src/rewards/)
                          │                              │
                          ▼                              ▼
                  format_docs_*  ──→  observation     reward + terminated
                          │                              │
                          └───────────────┬──────────────┘
                                          ▼
                             next-turn generation
```

Each layer is decoupled — chat template lives in the processor + env, reward lives in `rewards/`, retrieval contract lives in the env. Swapping arms is a config flip; swapping the reward (M3 ablation) is a one-file change.

---

## Step-5 audit summary

Lock-in mapping between [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) (the EM-only baseline that produced the published GRPO checkpoints we evaluated in M1) and our NeMo-RL setup.

| Knob | verl value (v0.2 yaml) | NeMo-RL key | Our value | Status |
|---|---|---|---|---|
| Optimizer LR | `1e-6` | `policy.optimizer.kwargs.lr` | `1e-6` | match |
| Warmup ratio | `0.285` | `policy.scheduler` LinearLR `total_iters` | `286` (= 0.285 × 1005) | match |
| Total steps | `total_training_steps=1005` | `grpo.max_num_steps` | `1005` | match — paper text says 500; verl yaml + published checkpoints are 1005 |
| Save cadence | `save_freq=100` | `checkpointing.save_period` | `100` (but `checkpointing.enabled: false` first-pass) | **paper-match planned**; first-pass disables checkpointing |
| Val cadence | `test_freq=100` | `grpo.val_period` | **`0` first-pass; `100` planned** | **first-pass disables validation** — re-enable per [VALIDATION.md §7](VALIDATION.md#7-re-enabling-validation-planned-not-active) |
| Val at start | `val_before_train=true` | `grpo.val_at_start` | **`false` first-pass; `true` planned** | first-pass disables validation |
| KL coef (β) | `kl_loss_coef=0.001` | `loss_fn.reference_policy_kl_penalty` | `0.001` | match |
| **KL estimator** | `kl_loss_type=low_var_kl` | `loss_fn.reference_policy_kl_type` | `k3` (NeMo-RL default) | **byte-identical** — both compute Schulman 2020 k3 |
| **State masking** | `state_masking=true` (`<information>` zero-masked) | role-based `token_loss_mask` ([grpo.py:1685-1693](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)) | automatic | **equivalent**, no config knob — env emits `role: tool`, gradient masks it |
| Clip ratio (ε) | `0.2` | `loss_fn.ratio_clip_min` / `ratio_clip_max` | `0.2` | match |
| Group size G | `n_agent=5` | `grpo.num_generations_per_prompt` | `5` | match |
| Trajectories / step | 512 (`train_batch_size=512`) | `num_prompts_per_step × num_generations_per_prompt` | `102 × 5 = 510` | 2-traj rounding; harmless with `force_on_policy_ratio: false` |
| Train global batch | n/a | `policy.train_global_batch_size` | `510` | matches the trajectories-per-step (upstream convention) |
| Train micro batch | `ppo_micro_batch_size=64` (verl-only) | `policy.train_micro_batch_size` | `4` (1× and 2× A100) | conservative for 2B@seq=4096 with `activation_checkpointing: true` |
| Max prompt len | `4096` | `policy.max_total_sequence_length` | `4096` | match |
| Max response len | `500` | `policy.generation.max_new_tokens` | `500` | match |
| Max search turns | `max_turns=4` | `env.search_r1.max_turns` + `grpo.max_rollout_turns` | `4` | match |
| Topk retrieved | `retriever.topk=3` | `env.search_r1.top_n` | `3` | match |
| Rollout temp / top_p | `1.0` / `1.0` | `policy.generation.{temperature, top_p}` | `1.0` / `1.0` | match |
| **Max obs len** | `max_obs_length=500` tokens | `env.search_r1.max_obs_chars` | `2000` chars | **char proxy** (~4 char/token; pure-Python parsers, no tokenizer) |
| Reward function | EM-based (`flashrag/search_r1/reward.py`) | `training/src/rewards/search_r1.py` | byte-identical port | verified by 15 parity tests |

**Knowing divergences from the paper** (recorded in [PAPER_VS_OURS_TRAINING.md §9](PAPER_VS_OURS_TRAINING.md#9-divergence-summary-one-place-to-glance)): model family (Qwen3.5-2B vs Qwen2.5-3B), chat template (qwen_native vs paper), variant naming (hybrid vs instruct), hardware (1×–2× A100 vs 8× H100), framework (NeMo-RL vs verl). Everything else matches.

---

## training/src/ overlay architecture

NeMo-RL is vendored at [`training/nemo_rl/`](../../training/nemo_rl/) at pinned `v0.6.0`. We **never edit it directly** — Search-R1-specific behavior lives as a pure overlay under [`training/src/`](../../training/src/) that registers itself with NeMo-RL's pluggable registries at startup.

| Overlay file | Role | Plugged into |
|---|---|---|
| [`rewards/search_r1.py`](../../training/src/rewards/search_r1.py) | Byte-identical port of M1 EM scorer (`compute_search_r1_reward`, `em_check`, `extract_solution`, ...) | imported by env actor |
| [`prompts/search_r1_paper.txt`](../../training/src/prompts/search_r1_paper.txt) | Paper instruction string (`{}` placeholder) | loaded as `task_data_spec.prompt` for `paper` arm only |
| [`chat_template/tools.py`](../../training/src/chat_template/tools.py) | OpenAI-style `search` tool schema (qwen_native arm) | passed as `tools=[SEARCH_TOOL]` to `tokenizer.apply_chat_template` |
| [`datasets/search_r1.py`](../../training/src/datasets/search_r1.py) | `SearchR1Dataset(RawDataset)` — loads our parquet, sets `task_name = f"search_r1_{arm}"` | `DATASET_REGISTRY["search_r1"]` (monkey-patched — no `register_dataset` upstream) |
| [`processors/search_r1.py`](../../training/src/processors/search_r1.py) | `search_r1_processor` — reads `messages[0].content` + `golden_answers`, dispatches qwen_native vs paper arms | `register_processor("search_r1_processor", ...)` |
| [`environments/parsers.py`](../../training/src/environments/parsers.py) | Pure-Python `parse_query`, `format_docs_*`, `retriever_failed_message` (testable without torch/ray/nemo_rl) | re-exported by env |
| [`environments/search_r1_env.py`](../../training/src/environments/search_r1_env.py) | `SearchR1Env` (testable plain class) + `SearchR1Environment = ray.remote(SearchR1Env)` | `register_env("search_r1", "training.src.environments.search_r1_env.SearchR1Environment")` |
| [`registry.py`](../../training/src/registry.py) | Single import-side-effect module that populates DATASET / PROCESSOR / ENV registries idempotently | imported once by the launch script |

**Wiring contract** — the launch script does:

```python
import training.src.registry  # populates DATASET_REGISTRY, PROCESSOR_REGISTRY, ENV_REGISTRY
from examples.run_grpo import main
main()
```

After that, the training loop sees `dataset_name: search_r1`, `processor: search_r1_processor`, `env_name: search_r1` as if they were built-in.

**Tests** — [`training/tests/`](../../training/tests/) covers reward parity (15 tests, byte-identical to the M1 eval pipeline), parser dispatch (per-arm regex behavior), format helpers (Qwen3.5 chat-template marker construction), env-step against a mocked retriever (search/answer/exhausted paths), and dataset adapter (column preservation, `task_name` correctness, validation split). Pure-Python tests run anywhere; tests needing torch/ray/nemo_rl skip cleanly outside the training venv.

---

## Hardware + runtime expectations

Smoke-anchored (1× A100 80GB SXM, ~57 s/step at 20 traj/step → linearly extrapolated to 510 traj/step × 1005 steps); full derivation in [`PAPER_VS_OURS_TRAINING.md §7`](PAPER_VS_OURS_TRAINING.md#7-compute) and [`SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config):

| Hardware | Wall-clock / run | $ / run (Vast on-demand) |
|---|---|---|
| **1× A100 80 GB SXM** *(current default)* | **11–17 d** (264–408 h) | **$300–490** at ~$1.20/h |
| **1× H100 80 GB SXM** *(best $/run)* | **5–8.5 d** (120–204 h) | **$240–410** at ~$2.00/h |
| **2× A100 80 GB SXM** | **6.5–9.5 d** (156–228 h) | **$370–550** at ~$2.40/h |
| **1× H200 141 GB SXM** | **4–7 d** (96–168 h) | **$270–470** at ~$2.80/h |

The original Phase-2 plan (3 seeds × {base, hybrid} = 6 runs) is **superseded** by the recipe-ablation pivot in [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md): with $1000 budget total, that supports ~2–3 runs. Recommended hardware: 1× H100 if available, 1× A100 otherwise.

**Step-100 health check.** Regardless of hardware, at step 100 (~1 h H100 / ~2.5 h A100) check the W&B `train/reward_mean` curve. If flat at ~0, abort and debug rather than burning the rest of the run.

Retriever: separate process on `127.0.0.1:3005`. **IVF4096-SQ8 index** (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB on disk); 8 workers each load their own copy of the index ⇒ ~128 GB host RAM (≥150 GB instance recommended). The flat IP index (~65 GB single copy) times out under training rollout HTTP load; M1 eval still uses Flat IP for paper-fidelity. See [`local_retriever/README.md`](../../local_retriever/README.md).

## Running training

- **What to run** (commands, args, env vars): [`training/README.md`](../../training/README.md)
- **Vast.ai sequence** (boot, retriever setup, smoke run, monitoring, eval gate): [`../milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md)
- **Milestone scope + status**: [`../milestone_2/MILESTONE_2.md`](../milestone_2/MILESTONE_2.md)

## Going deeper (educational)

For when you want to understand what's actually happening, not just what to set:

- [`../edu/SEED.md`](../edu/SEED.md) — what `--seed` controls, where it propagates, why we run 3 seeds × 2 variants
- [`../edu/GPU_MEMORY.md`](../edu/GPU_MEMORY.md) — what's in VRAM during GRPO (rollout vs training), with concrete numbers for our A100 80GB config and an OOM recovery ladder
