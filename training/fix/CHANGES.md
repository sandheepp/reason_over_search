# Training fixes — May 2026 smoke-run unblock

These are the in-tree changes made to get a fresh `pantomiman/reason-over-search-v1:v1` Vast box from clone → first GRPO step. Each item lists the file, the symptom, and what changed.

## Code changes

### 1. `training/scripts/run_grpo_1xa100.sh`

- **Added `--` passthrough** so callers can append arbitrary Hydra overrides after the wrapper's flags (e.g. `--variant base ... -- grpo.max_num_steps=2 grpo.num_prompts_per_step=4`). Without this, the wrapper exits with `unknown arg: --` for any extra Hydra knobs.
- **Removed `data.validation.arm=${ARM}` from the `OVERRIDES` array.** The default config has `data.validation: null` (validation is gated off in Phase 2). Hydra cannot override a nested key under a `null` parent; the run errored at startup with `ConfigCompositionException: Could not override 'data.validation.arm'`.

### 2. `training/src/registry.py`

- **Patched `ACTOR_ENVIRONMENT_REGISTRY`** to map `training.src.environments.search_r1_env.SearchR1Environment` → `PY_EXECUTABLES.SYSTEM`. NeMo-RL's `create_env` (`environments/utils.py`) looks up `actor_class_fqn` in this dict to decide which uv venv to use for the Ray actor. Without this entry the run dies before training with `ValueError: No actor environment registered for training.src.environments.search_r1_env.SearchR1Environment`.

### 3b. `training/src/processors/search_r1.py`

- **Added `enable_thinking=True` to both arms' `apply_chat_template` calls.** Qwen3.5-2B-Base shares the exact same `chat_template` Jinja as the instruct model (confirmed from `tokenizer_config.json`). Without `enable_thinking=True`, `add_generation_prompt=True` emits `<think>\n\n</think>\n\n` — an already-closed think block — before the model generates a single token. Both the paper prompt and the `qwen_native` system prompt instruct the model to "conduct reasoning inside `<think>` and `</think>` first", but there is no open block left to fill. With `enable_thinking=True` the generation prefix becomes `<|im_start|>assistant\n<think>\n` (open), so the model writes its reasoning, closes `</think>`, then generates `<search>`/`<tool_call>` or `<answer>` — exactly what both prompts intend.

- **Restructured `qwen_native` arm to mirror the paper arm's prompt layout.** Old layout had the protocol (think/search/answer instructions) baked into the system prompt; new layout puts only `"You are a helpful assistant. Answer the user's question by using the search tool when you need external knowledge."` in the system prompt, and moves the protocol into a user-message template (`training/src/prompts/search_r1_qwen_native_user.txt`) that wraps the bare question. This matches the Search-R1 paper's structure (paper arm puts everything in one user message). Smoke-validated 2026-05-02: mean reward jumped from 0.000 (v1) to 0.270 at step 1 (30% EM hit rate vs 0%). The processor now requires `task_data_spec.prompt` for both arms; the YAML `data.default.prompt_file` defaults to the qwen_native user template.

### 3d. `training/nemo_rl/nemo_rl/algorithms/grpo.py` + `experience/rollouts.py`

- **Routed project-specific rollout metrics to their own `reason_over_search/` W&B namespace.** Twelve keys are now under this prefix instead of mixed with upstream `train/`: turn/tool-call shape (`min_turns_per_sample`, `mean_tool_calls_per_sample`, `max_tool_calls_per_sample`, `min_tool_calls_per_sample`), format/abort signal (`aborted_ratio`), response-length stats (`resp_len_mean`, `resp_len_max`, `resp_len_min`, `response_length/clip_ratio`), reward stats (`reward_mean`, `reward_max`, `reward_min`). The sync rollout path (`run_multi_turn_rollout`) previously had no aggregated reward stats at all — added here using `total_rewards`. The async rollout path already had `mean/max/min_total_reward` upstream; the new `reward_*` aliases are computed alongside without removing the upstream keys. Pop happens *before* the `metrics.update(rollout_metrics)` merge in both sync and async paths so the keys don't double-log via the bigger `metrics` dict's later `train/` log call. Constant `_REASON_OVER_SEARCH_METRIC_KEYS` and helper `_pop_reason_over_search_metrics` defined near the top of `grpo.py` for one-place editing if more project metrics are added later.

### 3c. `training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`

- **Set `policy.generation.stop_strings` to `["</tool_call>", "</search>", "</answer>"]`.** The env's `next_stop_strings` only kicks in *after* the first `env.step()` call — turn 1 reads from `current_batch.get("stop_strings", [None] * batch_size)` ([rollouts.py:388](../nemo_rl/nemo_rl/experience/rollouts.py#L388)) which was `None` everywhere. Without a fallback, vLLM had no stop strings on the first turn and the model generated *past* its own `</tool_call>`, hallucinating fake `<tool_response>` blocks with raw JSON contents — confusing the env (which then injects the real `Doc 1: ...` response on top, polluting the trajectory with two consecutive tool responses). vLLM's `_merge_stop_strings` ([vllm_worker.py](../nemo_rl/nemo_rl/models/generation/vllm/vllm_worker.py)) takes the union of config-level and batch-level stop strings, so listing both arms' tags in the YAML is safe — neither arm emits the other arm's closing tag in normal generation. Smoke-validated 2026-05-02: hallucination rate 18/20 → 0/20, mean trajectory length 7567 → 6456 chars (-15%).

### 3. `training/src/datasets/search_r1.py`

- **Added a `task_name` column to the loaded HF Dataset.** `AllTaskProcessedDataset.__getitem__` (`processed_dataset.py:118`) reads `entry["task_name"]` whenever `task_data_processors` is a dict — which is always the case on the multi-task path that NeMo-RL takes by default. The reshaped parquet doesn't carry `task_name`, so every dataloader fetch raised `KeyError: 'task_name'`. Mirrors what every upstream `RawDataset` does (e.g. `clevr.py`, `geometry3k.py`, `daily_omni.py`).

## Infrastructure changes (automated in bootstrap.sh)

These are now fully automated by `training/scripts/bootstrap.sh`. The total GPU-instance setup guide (human + agent) is [`docs/setup/SETUP_INSTANCE.md`](../../docs/setup/SETUP_INSTANCE.md).

### 4. v2 (automodel) venv — downloaded from HF Hub, not compiled on Vast

The default config uses `policy.dtensor_cfg._v2: true` → `DTensorPolicyWorkerV2` → uv extra `automodel`. NeMo-RL builds these venvs lazily via Ray actor `_env_builder`, which has **no GPU allocated**. The `automodel` extra pulls `nv-grouped-gemm` whose `setup.py` calls `torch.cuda.init()` at install-time — which fails inside the GPU-less actor with `RuntimeError: No CUDA GPUs are available`, killing training before the first step.

Fix: `bootstrap.sh` (step 3) downloads a pre-built tarball from `pantomiman/reason-over-search-v1-venvs` on HF Hub (~5 GB, ~3 min) and extracts it to the correct venv path. Falls back to host-shell compile (~25 min) if the HF download fails. Either path produces an identical on-disk venv; NeMo-RL's `create_local_venv` detects the existing python and skips any rebuild.

### 5. Sequence packing must be disabled for Qwen3.5

Qwen3.5 is multimodal with interleaved `linear_attention` (Mamba/GatedDeltaNet) and `full_attention` layers. With `policy.sequence_packing.enabled: true`, the `torch_chunk_gated_delta_rule` kernel raises `CUDA error: an illegal memory access was encountered` during `policy.get_logprobs()`. Disabling sequence packing and using dynamic batching instead works.

**These are now the YAML defaults** in both `grpo_qwen3.5_2b_1xa100.yaml` and `grpo_qwen3.5_2b_2xa100.yaml`:
```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```

### 6. Retriever: IVF-SQ8 index + 8 workers is the default for training

With `num_retriever=2` and the flat IP index, the training rollout (~80 concurrent `/batch_search` queries) timed out on ~80% of requests. Rewards collapse to 0/format-only.

`local_retriever/retriever_config.yaml` now defaults to the IVF-SQ8 index. `bootstrap.sh` (step 4) starts the retriever with `--num_retriever 8`. Latency: `30 s timeout` → `~1.7 s` for a 3-query batch. Recall hit < 1% (acceptable for online RL; use flat IP for offline paper-quality eval).
