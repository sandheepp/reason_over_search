---
title: VERL REFERENCE
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-06
---

# Verl Reference Settings (for porting to NeMo-RL)

Stripped-down configuration reference distilled from the user's verl-tested launch scripts (`z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_{1,2}_a100_80gb.sh` from `~/Documents/Obsidian/code/omega/research/verl_latest/`). Those scripts targeted **Qwen3-0.6B** + ReSearch-style retrieval; we cherry-pick the *transferable* settings (retriever HTTP contract, GRPO knobs, agent-loop pattern, rollout logging shape) and discard the model-size-specific tunings.

This file lives in the repo so the porting work survives context switches and we don't have to re-read the user's local scripts every time.

---

## 1. Retriever HTTP contract (canonical)

The verl `re_search_agent` rollout loop POSTs to a retriever **base URL** (no `/batch_search` suffix in config; the agent appends it). Confirmed contract:

| Item | Value |
|---|---|
| Endpoint | `POST {search_url}/batch_search` |
| Default URL | `http://127.0.0.1:3005` (matches Milestone 1's [`local_retriever/`](../../local_retriever/) service) |
| Request body | `{"query": [...], "top_n": <int>}` |
| Response shape | as parsed in `re_search_agent_loop._batch_search_http` (verl-side) |
| Timeout | 5 attempts, 1s backoff, default 300s per attempt |
| Failure mode | LM receives the literal string `Retriever failed: …` |

The Milestone 1 retriever already implements `/batch_search` — see [`local_retriever/retriever_serving.py`](../../local_retriever/retriever_serving.py). **No changes needed retriever-side.**

## 2. GRPO + KL settings (confirmed against upstream Search-R1 verl yaml)

Cross-checked against `Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh` — the EM-only baseline that produced the published GRPO checkpoints we evaluated in Milestone 1.

| Knob | verl flag | Value | Maps to NeMo-RL | Status |
|---|---|---|---|---|
| Optimizer LR | `actor_rollout_ref.actor.optim.lr` | `1e-6` | `optimizer.lr` | match |
| Warmup ratio | `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.285` | `optimizer.warmup_ratio` | match |
| KL loss coef (β) | `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | `loss.reference_policy_kl_penalty` | match |
| **KL loss type** | `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | `loss.reference_policy_kl_type: k3` | **equivalent** (see below) |
| Use KL loss? | `actor_rollout_ref.actor.use_kl_loss` | `True` | always-on in GRPO | match |
| Group size | `actor_rollout_ref.rollout.n_agent` | `5` | `grpo.num_generations_per_prompt` | match |
| State masking | `actor_rollout_ref.actor.state_masking` | `True` | automatic via role-based `token_loss_mask` | **equivalent, no config needed** (see below) |
| Train batch size (prompts/step) | `data.train_batch_size` | `512` (prompts) | `grpo.num_prompts_per_step` | **divergent** — ours is `102` (5× smaller; rollout fits on 1× A100); see [`PAPER_VS_OURS_TRAINING.md §7`](PAPER_VS_OURS_TRAINING.md#7-compute) |
| Trajectories/step (derived) | n_agent × train_batch_size = `2560` | n/a | `policy.train_global_batch_size` | ours is `510` (= 102 × 5) — **5× fewer** |
| PPO mini batch size | `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | n/a — different abstraction | verl: 10 gradient updates / step (2560 traj / 256 = 10); ours: 1 update / step (gbs == prompts × gen). The cheap fix to close that gap is `policy.train_global_batch_size: 51` (10 updates over 510 traj at no extra rollout cost); see [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md). |
| PPO micro batch size | `actor_rollout_ref.actor.ppo_micro_batch_size` | `64` | `policy.train_micro_batch_size` | ours: **`2`** (Qwen3.5-2B with `sequence_packing: false` required, `4` OOMs in get_logprobs); see [`training/fix/CHANGES.md`](../../training/fix/CHANGES.md) §5 |
| Critic warmup | `trainer.critic_warmup` | `0` (no warmup) | n/a — GRPO has no critic | n/a |

**KL loss type — `low_var_kl` ≡ NeMo-RL `k3`.** Both compute the Schulman 2020 k3 estimator: `exp(ref-log) - (ref-log) - 1` (verl: [`core_algos.py:kl_penalty`](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/trainer/ppo/core_algos.py); NeMo-RL: [`algorithms/utils.py:74-75`](../../training/nemo_rl/nemo_rl/algorithms/utils.py#L74-L75)). NeMo-RL clamps both input log-ratio (±20) and output (±10); verl clamps only output (±10). Functionally identical for normal training values. **NeMo-RL's default `kl_type="k3"` matches verl exactly — no override needed.**

**State masking — equivalent, automatic.** verl's `state_masking=true` opts in to a `loss_mask` that zeroes out gradient on `<information>...</information>` tokens (the retrieved docs the model didn't generate). NeMo-RL achieves the same effect via role-based `token_loss_mask` ([`grpo.py:1685-1693`](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)): assistant-role tokens get loss=1, every other role (tool / environment / user) gets loss=0. Zero-config; we only need to ensure our env emits retrieval responses with role≠"assistant" (we use `"role": "tool"`).

These match what we have in [`PAPER_VS_OURS_TRAINING.md`](PAPER_VS_OURS_TRAINING.md) §6 — **so the verl scripts independently confirm the paper's Appendix B.2 numbers.**

## 3. vLLM rollout knobs (verl-side; map to NeMo-RL `policy.generation.vllm_cfg.*`)

These are starting values from the user's 80GB scripts. They were tuned for **0.6B**; for our **Qwen3.5-2B** target, the actual values shipped in [`training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`](../../training/configs/) are:

| Knob | 1× A100 80GB (0.6B) | 2× A100 80GB (0.6B) | NeMo-RL key |
|---|---|---|---|
| `vllm_gpu_mem_util` | `0.72` | `0.78` | `gpu_memory_utilization` |
| `vllm_max_num_seqs` | `6` | `8` | `max_num_seqs` |
| `vllm_max_model_len` | `9216` | `9216` | (limited by `policy.max_total_sequence_length`) |
| `rollout_max_num_batched_tokens` | `14336` | `24576` | `max_num_batched_tokens` |
| `rollout_agent_num_workers` | `2` | `2` | n/a (NeMo-RL uses Ray actor pool count separately) |
| `enforce_eager` | `False` | `False` | match (CUDA graphs on) |
| `enable_chunked_prefill` | `True` | `True` | match if available |

**Our shipped values for Qwen3.5-2B at paper's `max_response_length=500`** (configs at [`training/configs/`](../../training/configs/)):

| Knob | 1× A100 80GB | 2× A100 80GB | NeMo-RL key |
|---|---|---|---|
| `gpu_memory_utilization` | `0.6` | `0.7` (TP=2 frees per-GPU weight footprint) | `policy.generation.vllm_cfg.gpu_memory_utilization` |
| `max_model_len` | `4096` | `4096` | `policy.max_total_sequence_length` (vLLM inherits) |
| `tensor_parallel_size` | `1` | `2` (rollout TP across both GPUs; training stays DDP at TP=1) | `policy.generation.vllm_cfg.tensor_parallel_size` |
| `max_new_tokens` | `500` | `500` | `policy.generation.max_new_tokens` |

Conservative first-run defaults; raise `gpu_memory_utilization` toward `0.8` after observing W&B `gpu_monitoring`. Revisit in [`NEMO_RL_KNOBS.md §8`](NEMO_RL_KNOBS.md#8-what-we-expect-to-retune-after-the-first-run).

## 4. FSDP / memory knobs (verl-side)

verl uses FSDP; NeMo-RL uses DTensor. Map of intent:

| Concern | verl flag | Value | NeMo-RL equivalent |
|---|---|---|---|
| Gradient checkpointing | `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` (1 GPU) / `False` (2 GPU) | `policy.dtensor_cfg.activation_checkpointing` |
| Actor param offload to CPU | `actor_rollout_ref.actor.fsdp_config.param_offload` | `False` | DTensor doesn't offload by default; flip to `dtensor_cfg.cpu_offloading` if memory-pressured |
| Optimizer offload | `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `False` | same |
| Ref-model param offload | `actor_rollout_ref.ref.fsdp_config.param_offload` | `False` | same |
| `torch.compile` | `actor_rollout_ref.actor.use_torch_compile` | `True` | NeMo-RL sets this internally; check post-clone |
| `remove_padding` | `actor_rollout_ref.model.use_remove_padding` | `True` | NeMo-RL uses sequence packing — same effect |
| Dynamic batching | `actor_rollout_ref.actor.use_dynamic_bsz` | `True` | NeMo-RL uses fixed micro-batches; map by setting `train_micro_batch_size` to a value that maxes out the GPU |
| `attn_implementation` | `+actor_rollout_ref.model.override_config.attn_implementation` | `sdpa` | Hugging Face setting — will pass through if NeMo-RL exposes it |
| Allocator config | env `PYTORCH_ALLOC_CONF=expandable_segments:True` | — | set in launch script (env-level) |

## 5. Rollout JSONL log shape (steal this for our W&B metrics)

The user's verl rollouts dump per-step JSONL with these columns (from script comments):

- `prompt` — the rendered prompt string
- `response` — full response including `<tool_call>` / `<tool_response>`
- `messages` — raw chat-message list (role/content)
- `num_turns` — total assistant turns in the rollout
- `num_search_calls` — count of `<tool_call>` invocations

We mirror this in [`VALIDATION.md`](VALIDATION.md) §4 W&B metrics: `val/mean_search_calls`, `val/mean_response_tokens`. Add `val/num_turns_p50/p95` if useful once we see distributions.

## 6. Things deliberately left out

The user's verl scripts include knobs that **don't transfer** to NeMo-RL or that are 0.6B-specific:

- `update_weights_bucket_megabytes=1024` — verl-internal weight-sync detail
- `dataloader_num_workers=8` — NeMo-RL has its own dataloader config
- `train_batch_size=8` and `ppo_mini_batch_size=8` — too small for our setup; we use `102 prompts × 5 gen = 510` trajectories/step (vs paper's `512 × 5 = 2560`)
- `MAX_RESPONSE_LENGTH=8192` — way above paper's `500`; we use the paper value
- `total_epochs=2` and `test_freq=1000` — verl-specific schedule; we use **`grpo.max_num_steps: 1005`** (matches the published-checkpoint verl run; paper text's "500 steps" is superseded by the v0.2 yaml — see [`PAPER_VS_OURS_TRAINING.md §5`](PAPER_VS_OURS_TRAINING.md#5-validation-set--cadence))

## 7. Local-only files (not copied into the repo)

The original launch scripts at `~/Documents/Obsidian/code/omega/research/verl_latest/` are kept local because they target a different repo (verl 0.6B + ReSearch-style training, not Search-R1 + Qwen3.5). If they're useful for cross-reference during the NeMo-RL port, the path is recorded above; otherwise they need not be tracked here.
