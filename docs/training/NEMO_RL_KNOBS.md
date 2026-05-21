---
title: NEMO RL KNOBS
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-06
---

# NeMo-RL GPU Optimisation Knobs

Reference for the NeMo-RL settings that matter when running GRPO on Qwen3.5-2B on **A100 80GB**. Sources: [NVIDIA-NeMo/RL repo](https://github.com/NVIDIA-NeMo/RL), [GRPO docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/grpo.html), [GRPO on DeepScaler tutorial](https://docs.nvidia.com/nemo/rl/latest/guides/grpo-deepscaler.html), the [`grpo_math_1B.yaml` example](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml).

> **Status**: this is a working reference compiled from the public docs and the 1B/32B example configs. Refine in place once we run our first training and observe actual memory / throughput on A100 80GB.

---

## 1. Memory knobs

| Knob | Where | Default | What it controls |
|---|---|---|---|
| `policy.dtensor_cfg.activation_checkpointing` | DTensor backend | `false` | Gradient (activation) checkpointing — recomputes activations in backward, trades compute for memory. **Turn on for Qwen3.5-2B on A100 80GB at seq_len ≥ 4096.** |
| `policy.dtensor_cfg.sequence_parallel` | DTensor backend | `false` | Splits activations along the sequence dim across TP ranks. Reduces per-GPU activation memory at the cost of extra all-gathers. |
| `policy.dtensor_cfg.tensor_parallel_size` | DTensor backend | `1` | Tensor parallelism degree. For 2B on 1× A100 80GB, `1` is fine. For 2× A100, set `2` if memory pressure shows up; otherwise leave at `1` and use DDP. |
| `policy.train_micro_batch_size` | Policy | `4` (1B example); **ours: `2`** | Per-GPU forward/backward chunk. Qwen3.5-2B with `sequence_packing: false` (required, see below) needs `2` on A100 80GB — `4` OOMs in `get_logprobs`. Drop to `1` if still OOM under group=5. |
| `policy.generation_batch_size` | Policy | `32` (1B example) | Rollout batch size during vLLM generation. |
| `policy.max_total_sequence_length` | Policy | `512` (1B), `16384` (32B) | Hard cap on prompt + response length. **Set to `4096` for Search-R1 parity.** |
| `policy.sequence_packing.enabled` | DTensor | `true` | Concatenates short sequences to fill the max length, raising effective throughput. **Must be `false` for Qwen3.5** — interleaved GatedDeltaNet (Mamba-style) layers crash with `CUDA illegal memory access` in `torch_chunk_gated_delta_rule` during `get_logprobs` when sequences are packed. See [`training/fix/CHANGES.md`](../../training/fix/CHANGES.md) §5. Replacement is `policy.dynamic_batching.enabled: true` (token-budget micro-batches). |
| `policy.precision` | Policy | `"bfloat16"` | Mixed precision; keep at bf16 on A100. |

**Multi-node thresholds** ([NeMo-RL GRPO docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/grpo.html)): on 8× A100 80GB, **1 node for 8K** sequence length, **2 nodes for 16–24K**. Our 4K target fits well within 1× A100, so we do not need multi-node.

---

## 2. Throughput knobs (vLLM rollout)

| Knob | Where | Default | What it controls |
|---|---|---|---|
| `policy.generation.backend` | vLLM | `"vllm"` | Inference backend for rollouts. |
| `policy.generation.vllm_cfg.tensor_parallel_size` | vLLM | `1` | TP for the rollout vLLM workers. Must divide `cluster.gpus_per_node`. |
| `policy.generation.vllm_cfg.gpu_memory_utilization` | vLLM | `0.6` | Fraction of VRAM vLLM reserves for KV cache. Conservative defaults shipped: `0.6` on 1× A100, `0.7` on 2× A100. Sweet spot for a ~3B model on 80GB is ~0.8 ([GRPO+LoRA handbook](https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl)); raise after observing W&B `gpu_monitoring`. |
| `policy.generation.vllm_cfg.enforce_eager` | vLLM | `false` | Disables CUDA graphs. Keep `false` for performance; flip to `true` only if hitting graph-related issues. |
| `policy.generation.max_new_tokens` | vLLM | `512` | Per-rollout response cap. **Set to `500` for Search-R1 parity** (paper's max response length). |
| `policy.generation.temperature` | vLLM | `1.0` | Rollout sampling temperature. **`1.0`** matches Search-R1 training rollouts. (Eval is greedy; that's the eval pipeline, not training.) |
| `policy.generation.top_p` | vLLM | `1.0` | Nucleus sampling. `1.0` (no truncation) per Search-R1. |

---

## 3. GRPO algorithm knobs

| Knob | Where | Default | What it controls |
|---|---|---|---|
| `grpo.num_prompts_per_step` | GRPO | `32` (1B example) | Prompts per training step. With `num_generations_per_prompt=16`, that's 32 × 16 = 512 trajectories per step → matches `train_global_batch_size` (one optimizer.step() per training step, on-policy). See [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md) for what happens when the convention is violated. |
| `grpo.num_generations_per_prompt` | GRPO | `16` (1B example) | Group size G in GRPO. **Search-R1 paper uses `5`** for Qwen2.5-3B. Larger G = lower-variance advantage but more rollouts per step. |
| `grpo.max_num_steps` | GRPO | varies | Total training steps. **Search-R1 uses `1005`** (verl `total_training_steps=1005` in v0.2; supersedes paper text's "500"). |
| `grpo.val_period` | GRPO | varies | Validation cadence in steps. **Search-R1 uses `100`** (verl `test_freq=100`). **Currently `0` (disabled) for first-pass training** — see VALIDATION.md §7. |
| `grpo.val_at_start` | GRPO | varies | Run validation at step 0 as a baseline. **Search-R1 uses `true`** (verl `val_before_train=true`). **Currently `false`** (first-pass). |
| `grpo.max_rollout_turns` | GRPO | `999999` | Hard cap on env-loop iterations per rollout. **Search-R1 uses `4`** (verl `max_turns=4` in v0.2). Our env enforces this internally too. |
| `grpo.normalize_rewards` | GRPO | `true` | Normalize rewards within each group (standard GRPO behaviour). |
| `grpo.use_leave_one_out_baseline` | GRPO | `true` | Use leave-one-out baseline for advantage estimation. Standard. |
| `loss.reference_policy_kl_penalty` (β) | Loss | varies | KL penalty against the reference policy. **Search-R1 uses `0.001`** — very weak KL, lets the policy drift. |
| `loss.reference_policy_kl_type` | Loss | `"k3"` | KL estimator. **Default `k3` matches verl's `low_var_kl` byte-identically** — both compute Schulman 2020 k3 (`exp(ref-log) - (ref-log) - 1`). No override needed. See [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §2. |
| `loss.clip_ratio` (ε) | Loss | varies | PPO-style clip range. **Search-R1 uses `0.2`** (standard). |
| `policy.max_grad_norm` | Policy | `1.0` | Gradient clipping. Keep at default. |

---

## 4. Optimizer

| Knob | Default (1B example) | Search-R1 paper |
|---|---|---|
| `optimizer.optimizer` | `"adam"` | adam (implicit) |
| `optimizer.lr` | `5.0e-6` | **`1e-6`** |
| `optimizer.min_lr` | `5.0e-7` | not specified — default to NeMo-RL's `1e-7` (lr/10) |
| `optimizer.weight_decay` | `0.01` | not specified — keep `0.01` |
| `optimizer.adam_beta1` / `beta2` | `0.9` / `0.999` | standard, keep |
| Warmup ratio | not in 1B example | **`0.285`** (Search-R1 — this is unusually high; 28.5 % of training is warmup) |

---

## 5. Cluster / parallelism

| Knob | Default | A100 80GB starting value |
|---|---|---|
| `cluster.gpus_per_node` | `1` | `1` (1× A100) or `2` (2× A100) |
| `cluster.num_nodes` | `1` | `1` for our scale |
| `policy.dtensor_cfg.enabled` | `true` | `true` (default backend) |
| Backend choice | DTensor | DTensor for our scale; switch to Megatron only if scaling up beyond 8B |

---

## 6. Checkpointing & logging

> ⚠️ **Currently `checkpointing.enabled: false`** in both YAMLs — first-pass training is mechanics-only, no weights saved. Re-enable per [`VALIDATION.md §7`](VALIDATION.md#7-re-enabling-validation-planned-not-active).

| Knob | Default | Notes |
|---|---|---|
| `checkpointing.enabled` | `true` | **Currently `false`** (first-pass). Re-enable when validation comes back on. |
| `checkpointing.checkpoint_dir` | `"results/grpo"` | **Set to a path on Vast.ai persistent storage** (e.g. `/workspace/persistent/checkpoints/qwen3.5-2b-{base,hybrid}/seed_{1,2,3}`). Unused while disabled. |
| `checkpointing.metric_name` | `"val:accuracy"` | Best-checkpoint selection metric. Our `SearchR1Env.global_post_process_and_metrics` emits `accuracy` (mean reward, zero-masked for not-properly-ended rollouts) so this works as-is once validation runs. |
| `checkpointing.keep_top_k` | `3` | Retain best 3 + latest. |
| `checkpointing.save_period` | `10` | Every N training steps. **Search-R1 uses every 100 steps**; we should match initially. |
| `logger.wandb_enabled` | `false` | **Set `true`**; key in `training/.env`. |
| `logger.wandb.name` | — | Pattern: `qwen3.5-2b-{variant}-search_r1-{arm}-{Nx}a100-seed{N}-{UTC-timestamp}` (e.g. `...seed42-20260502T1437Z`). Built by the bash wrappers; the timestamp lets repeated launches of the same `(variant, arm, seed)` get distinct W&B runs while still resuming from the same checkpoint dir. |
| `logger.num_val_samples_to_print` | — | Set to ~5 so we eyeball trace quality during training. |

---

## 7. Recommended starting values for Qwen3.5-2B on A100 80GB

The actual configs live at [`training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`](../../training/configs/) — full standalone YAMLs. Below is just the diff vs upstream's `grpo_math_1B.yaml` defaults so you can see what we tuned and why; for the canonical values open the configs.

| Knob | Upstream `grpo_math_1B.yaml` | Ours, 1× A100 | Ours, 2× A100 | Reason |
|---|---|---|---|---|
| `policy.model_name` | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen3.5-2B-Base` | same | M2 target |
| `policy.max_total_sequence_length` | `512` | `4096` | same | [paper] verl `max_prompt_length=4096` |
| `policy.train_global_batch_size` | `512` | `510` | same | matches `num_prompts_per_step × num_generations_per_prompt` |
| `policy.train_micro_batch_size` | `4` | `2` | `2` | Qwen3.5-2B with `sequence_packing: false` (required, see §1) — `4` OOMs in get_logprobs |
| `policy.dtensor_cfg.activation_checkpointing` | `false` | `true` | `true` | [memory] bf16 + 4k seq + group=5 pressures activations |
| `policy.generation.max_new_tokens` | `512` (≈ max_seq) | `500` | same | [paper] verl `max_response_length=500` |
| `policy.generation.vllm_cfg.tensor_parallel_size` | `1` | `1` | `2` | [memory] split rollout weights on 2× A100 |
| `policy.generation.vllm_cfg.gpu_memory_utilization` | `0.6` | `0.6` | `0.7` | conservative defaults; raise after observing W&B `gpu_monitoring` |
| `grpo.num_prompts_per_step` | `32` | `102` | same | [paper] 102 × 5 = 510 ≈ verl's 512 prompts × 5 = 2560 trajectories — **5× fewer trajectories per step than verl**; see PAPER_VS_OURS_TRAINING.md §7 |
| `grpo.num_generations_per_prompt` | `16` | `5` | same | [paper] verl `n_agent=5` |
| `grpo.max_num_steps` | `1000000` | `1005` | same | [paper] verl `total_training_steps=1005` |
| `grpo.val_period` | `10` | **`0` (first-pass; planned 100)** | same | first-pass disables validation; `100` matches verl `test_freq` once re-enabled |
| `grpo.val_at_start` / `val_at_end` | `false` / `false` | **both `false` (first-pass; planned `true` once val is on)** | same | first-pass disables validation |
| `data.validation` | `null` | **`null` (first-pass)** | same | re-add the `dataset_name: search_r1` block alongside `train` once `val_period > 0` |
| `checkpointing.enabled` | `true` | **`false` (first-pass; planned `true`)** | same | first-pass is mechanics-only; flip back when validation comes on |
| `grpo.max_rollout_turns` | `1` | `4` | same | [paper] verl `max_turns=4` (multi-turn search) |
| `loss_fn.reference_policy_kl_penalty` | `0.01` | `0.001` | same | [paper] verl `kl_loss_coef=0.001` |
| `loss_fn.reference_policy_kl_type` | `"k3"` | `"k3"` | same | upstream default = verl `low_var_kl` (byte-identical) |
| `loss_fn.ratio_clip_min` / `ratio_clip_max` | `0.2` / `0.2` | same | same | [paper] PPO ε=0.2 |
| `policy.optimizer.kwargs.lr` | `5.0e-6` | `1.0e-6` | same | [paper] verl LR |
| `policy.scheduler` LinearLR `total_iters` | `50` | `286` | same | [paper] 0.285 × 1005 |
| `cluster.gpus_per_node` | `1` | `1` | `2` | hardware target |
| `data.train.dataset_name` | `OpenMathInstruct-2` | `search_r1` | same | our overlay |
| `env_name` | `math` | `search_r1` | same | our overlay |
| `checkpointing.metric_name` | `val:accuracy` | same | same | [`SearchR1Env.global_post_process_and_metrics`](../../training/src/environments/search_r1_env.py) emits `accuracy` |

---

## 8. What we expect to retune after the first run

These are the knobs most likely to need adjustment based on observed memory / throughput on A100 80GB:

- `gpu_memory_utilization`: try 0.85 if 0.8 leaves headroom; drop to 0.7 if rollouts OOM under group=5.
- `train_micro_batch_size`: raise if backward pass leaves memory unused.
- `activation_checkpointing`: turn off if memory is comfortable — you get ~30 % speedup back.
- `num_prompts_per_step`: scale up if step time is dominated by rollouts.

---

## 9. Tool integration: how the rollout calls our retriever

This is the load-bearing engineering work for Milestone 2 — Search-R1's training loop is *retrieval-augmented*, so the GRPO rollout has to invoke the local retriever HTTP service ([`local_retriever/`](../../local_retriever/)) every time the policy emits `<tool_call>`. NeMo-RL offers two paths:

### Path A — register a custom Ray-actor environment (recommended starting point)

NeMo-RL ships a [`register_env`](https://docs.nvidia.com/nemo/rl/latest/guides/environments.html) hook that lets you bind an `env_name` to a Ray actor implementing `EnvironmentInterface` (`reset`, `step`):

```python
# In training launch script
from nemo_rl.environments.utils import register_env

register_env(
    env_name="search_r1",
    actor_class_fqn="training.src.environments.search_r1_env.SearchR1EnvironmentActor",
)
```

```yaml
env:
  search_r1:
    num_workers: 2          # parallel env actors
    retriever_url: "http://127.0.0.1:3005"
    top_n: 3                # match Milestone 1 retrieval_topk

data:
  env_name: search_r1
```

Implementation lives in `training/src/environments/search_r1_env.py`. Responsibilities:

- Parse the assistant text for `<tool_call><function=search><parameter=query>...` (Qwen3.5 native format from [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md)).
- POST to `{retriever_url}/batch_search` with `{"query":[q],"top_n":<top_n>}`.
- Wrap the response as a `tool` message that the chat template renders into `<|im_start|>user\n<tool_response>...\n</tool_response><|im_end|>`.
- On `<answer>...</answer>`, emit terminal=True, compute EM reward against `golden_answers`, return.

### Path B — NeMo Gym

NeMo-RL also supports [NeMo Gym](https://docs.nvidia.com/nemo/rl/nightly/design-docs/nemo-gym-integration.html) integration via:

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - resources_servers/search_r1/configs/search_r1.yaml
      - responses_api_agents/search_r1_agent/configs/agent.yaml
```

NeMo Gym uses an HTTP agent ↔ resource server pattern (the resource server provides tools and rewards). Heavier-weight than Path A and adds a CPU-only service layer; recommended only if Path A doesn't expose enough orchestration control.

### Recommendation

Start with **Path A** (custom Ray-actor env). It's the simplest mapping of Search-R1's verl `re_search_agent` (a thin Python loop wrapping HTTP calls) onto NeMo-RL idioms, and the retriever HTTP contract is already fixed by Milestone 1 (`POST /batch_search` with `{"query":[...],"top_n":n}` — see [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §1).

Only fall back to NeMo Gym if Path A hits a wall (e.g. NeMo-RL's Ray-actor envs can't intercept generation mid-stream for multi-turn tool calls — verify by reading `nemo_rl/environments/` examples in the cloned repo).
