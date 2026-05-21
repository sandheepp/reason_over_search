---
title: Code Setup M0 — verl-legacy Qwen3-0.6B port (Phase-1 training)
tags: [report, training, m0, phase1]
source: internal
created: 2026-05-04
updated: 2026-05-08
---

# Code Setup M0: ReSearch Port (Phase-1 verl-legacy Qwen3-0.6B)

**Date**: 2026-05-03 (updated 2026-05-08).  
**Scope**: Experimental changes layered over the original ReSearch paper code (Agent-RL/ReSearch, alphaxiv 2503.19470) to fit a single-A100 / single-GPU regime with reward and prompt ablations. Phase-1 builds on the **ReSearch paper only** (not Search-R1); the `<search>` / `<result>` action-tag scheme is ReSearch's. The reward used in the 14 v0 ablation runs is the **paper-faithful ReSearch reward (format + EM partial credit)**.  
**Cluster**: ALICE HPC (Leiden University), 1× A100-40GB. ALICE retired going forward.  
**Source repo**: sibling `re-search/` (path: `/home/s4374886/omega/re-search/` on ALICE; not in this repo).

> **Before retiring this repo:** rotate the WANDB_API_KEY in `re-search/.env` and `re-search/verl_latest/.env` (the live key string was previously inlined here and has been REDACTED on 2026-05-08; if you have not already rotated it, do so now; git history retains the leaked value, so rotation is the only durable mitigation). Entity: `gaurisankarj1996-leiden-university`.

---

## 1. Headline Diff vs. Paper


| Dimension                         | Original Paper (`Agent-RL/ReSearch`)                  | This Port                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model family**                  | Qwen2.5 (7B-Base, 7B-Instruct)                        | **[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** — one post-trained chat checkpoint that supports **thinking** and **non-thinking** in the same weights (32K context); pretraining-only weights are **[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)**. `**enable_thinking`** in `apply_chat_template` is the hard toggle; when thinking is enabled, `**/think`** ↔ `**/no_think**` in user/system turns is the documented soft toggle ([Qwen3 model card § switching](https://huggingface.co/Qwen/Qwen3-0.6B)). |
| **Model size**                    | **7B** params                                         | **0.6B** params (~12× smaller)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Tokenizer / vocab**             | Qwen2.5 (~152K vocab)                                 | Qwen3 (~152K vocab, but Qwen3 chat templates differ — added manual chat builder)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Variants compared**             | base vs **instruct**                                  | base vs **hybrid** (Qwen3 has no separate instruct release)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **verl version**                  | Pinned commit (`src/verl` in upstream repo, vendored) | **Ported to verl_latest** (`verl_latest/`); upstream pin saved as `src/verl_legacy/` for reference                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **flashrag**                      | Vendored                                              | **Vendored verbatim** (no changes to `src/flashrag/`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Compute (single-node example)** | 4 GPUs, TP=2                                          | **1 GPU**, TP=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Compute (paper full)**          | 8 nodes × 8 GPUs = **64 GPUs**, TP=2                  | n/a                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Rollout class**                 | `vllm_with_search` (sync, baked into rollout worker)  | `re_search_agent` (async agent loop, decoupled from rollout) — verl_latest convention                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Tool tags**                     | `<search>...</search>` + `<result>...</result>`       | `**<tool_call>{"name":"search","arguments":"..."}</tool_call>` + `<tool_response>...</tool_response>`** — JSON tool-call format (forced change to align with verl_latest agent loop standard)                                                                                                                                                                                                                                                                                                                                                      |
| **Reward formatter**              | Validates `<search>/<result>` tag pairing             | Validates `<tool_call>` JSON payload + explicitly rejects legacy `<search>` / `<result>` tags                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Retriever placement**           | **GPU** (faiss-gpu via conda)                         | **CPU** (`faiss_gpu: False`) — moved off GPU to free VRAM under 40GB constraint                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Retriever index**               | Full Wikipedia (`wiki18_100w`)                        | Full Wikipedia (`wiki18_100w`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Retriever serving**             | Same FastAPI / FlashRAG pattern                       | Same FastAPI / FlashRAG pattern (kept as-is)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |


---

## 2. Hyperparameter Matrix — Paper vs My GPU Profiles

Paper baseline columns are pulled from `scripts/train/train_multi_node.sh` (full reproduction recipe) and the single-node example in upstream README.  
My columns are the **defaults** in each `verl_latest/*.sh` (overridable via env vars).

### 2.1 Compute & batching


| Setting                  | Paper (multi-node) | Paper (single-node ex.) | x_min 40GB | x_run 40GB | z_min 80GB | z_run 80GB | zz 2×80GB | y 4×L4-24GB |
| ------------------------ | ------------------ | ----------------------- | ---------- | ---------- | ---------- | ---------- | --------- | ----------- |
| n_nodes                  | 8                  | 1                       | 1          | 1          | 1          | 1          | 1         | 1           |
| n_gpus_per_node          | 8                  | 4                       | **1**      | **1**      | **1**      | **1**      | 2         | 4           |
| tensor_model_parallel    | 2                  | 2                       | **1**      | **1**      | **1**      | **1**      | (1)       | (1)         |
| `train_batch_size`       | 256                | 8 (example)             | **4**      | **4**      | **8**      | **8**      | 8         | 4           |
| `ppo_mini_batch_size`    | 256                | 8                       | **4**      | **4**      | **8**      | **8**      | 8         | 4           |
| `gen_batch_size`         | (=train)           | (=train)                | 4          | 4          | 8          | 8          | 8         | (=train)    |
| `rollout.n` (GRPO width) | 5                  | 5                       | **3**      | 5          | 5          | 5          | 5         | 5           |


### 2.2 Sequence lengths


| Setting                          | Paper               | x_min 40GB | x_run 40GB | z_min 80GB | z_run 80GB | zz 2×80GB | y 4×L4-24GB |
| -------------------------------- | ------------------- | ---------- | ---------- | ---------- | ---------- | --------- | ----------- |
| `max_prompt_length`              | 512                 | 512        | 512        | 512        | 512        | 512       | 512         |
| `max_response_length`            | **8192**            | **4096**   | 8192       | **4096**   | 8192       | 8192      | 8192        |
| `vllm.max_model_len`             | implicit ≥8704      | **4608**   | 9216       | **4608**   | 9216       | 9216      | 8704        |
| `ppo_max_token_len_per_gpu`      | 2×(P+R) = **17408** | 18432      | 18432      | 12288      | **24576**  | 20480     | 17408       |
| `log_prob_max_token_len_per_gpu` | 4×(P+R) = **34816** | 18432      | 18432      | 12288      | 24576      | 20480     | 17408       |


### 2.3 vLLM rollout


| Setting                          | Paper     | x_min 40GB | x_run 40GB | z_min 80GB | z_run 80GB | zz 2×80GB | y 4×L4-24GB |
| -------------------------------- | --------- | ---------- | ---------- | ---------- | ---------- | --------- | ----------- |
| `gpu_memory_utilization`         | 0.6       | **0.60**   | 0.62       | 0.76       | 0.72       | 0.78      | 0.70        |
| `vllm.max_num_seqs`              | (default) | 4          | 4          | 8          | 6          | 8         | 3           |
| `rollout_max_num_batched_tokens` | (default) | 12288      | 12288      | (default)  | 14336      | (default) | (default)   |
| `enforce_eager`                  | False     | **True**   | True       | True       | True       | True      | True        |
| `enable_chunked_prefill`         | (default) | True       | True       | True       | True       | True      | True        |


### 2.4 Optimization


| Setting                     | Paper                       | All my profiles                                                                  |
| --------------------------- | --------------------------- | -------------------------------------------------------------------------------- |
| `algorithm.adv_estimator`   | grpo                        | grpo                                                                             |
| `kl_ctrl.kl_coef`           | 0.001                       | 0.001                                                                            |
| `actor.use_kl_loss`         | True                        | True                                                                             |
| `actor.kl_loss_coef`        | 0.001                       | 0.001                                                                            |
| `actor.kl_loss_type`        | low_var_kl                  | low_var_kl                                                                       |
| `actor.optim.lr`            | 1e-6                        | 1e-6                                                                             |
| `actor.use_dynamic_bsz`     | True                        | True                                                                             |
| `actor.entropy_coeff`       | (not set)                   | **0.001** (added)                                                                |
| `actor.use_torch_compile`   | (not set)                   | **True** (added)                                                                 |
| `model.attn_implementation` | (default flash-attn)        | **sdpa** (`override_config.attn_implementation=sdpa` — added for compile compat) |
| `actor.fsdp.param_offload`  | False (multi-node has VRAM) | **True** (single-GPU needs offload)                                              |
| `ref.fsdp.param_offload`    | True                        | True                                                                             |


### 2.5 What I dropped vs. paper (capacity choices)

- **Tensor parallel** removed (TP=2 → 1) because single GPU.
- `**max_response_length`** halved on the `_min` profiles (8192 → 4096) for shorter smoke runs.
- `**rollout.n`** dropped 5 → 3 only on `x_min` (40GB minimal); kept at 5 elsewhere.
- **train_batch_size** 256 → 4 / 8 (single-GPU constraint; gradient accumulation via `use_dynamic_bsz`).
- All offload toggles (`param_offload=True`) added because the actor + ref + vLLM colocate on one card.

---

## 3. Script Variants and Naming Convention

I created two flavors per hardware profile:

- `*_min_*.sh` — **smoke / short experiments**: smaller `max_response_length` (4096), smaller `max_model_len`, lower `rollout_n` (40GB only). Goal: fast iteration to confirm a config trains at all.
- `*_run_*.sh` — **full / long experiments**: paper-equivalent sequence budgets (`max_response_length=8192`, `rollout_n=5`). Goal: get reportable numbers.

Hardware profiles created (all tested):


| Script                                                       | Hardware     | Use case                              |
| ------------------------------------------------------------ | ------------ | ------------------------------------- |
| `x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh` | 1× A100-40GB | Smoke / iteration                     |
| `x_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh`     | 1× A100-40GB | Full run                              |
| `z_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh` | 1× A100-80GB | Smoke / iteration                     |
| `z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh`     | 1× A100-80GB | Full run                              |
| `zz_run_qwen3_0.6b_grpo_vllm_instruct_gpu_2_a100_80gb.sh`    | 2× A100-80GB | Larger global batch                   |
| `y_run_qwen3_0.6b_grpo_vllm_instruct_gpu_4_l4_24gb.sh`       | 4× L4-24GB   | Tested smaller-GPU multi-card path    |
| `archive_old/run_qwen3_0.6b_grpo_vllm_instruct.sh`           | (legacy)     | Pre-port snapshot, kept for reference |
| `archive_old/run_qwen3_0.6b_grpo_vllm_instruct_gpu80.sh`     | (legacy)     | Pre-port snapshot, kept for reference |


`stable/` — internal scratch; not part of the experimental record.

---

## 4. Prompt Ablations

The paper provides one system prompt (`re_search_template_sys`). I produced three variants:


| Template                                    | Source               | Tags used                                | Style                                                                                 | Use                                  |
| ------------------------------------------- | -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------ |
| `re_search_template`                        | Paper, kept verbatim | `<search>` / `<result>`                  | Free-form, single paragraph                                                           | Legacy / base-model fallback         |
| `re_search_template_sys` (paper)            | Paper                | `<search>` / `<result>`                  | Single paragraph, casual phrasing                                                     | (Replaced — not used in port)        |
| `**re_search_template_sys`** (mine)         | Rewritten            | `<tool_call>` (JSON) / `<tool_response>` | Strict structured: numbered process, `# REQUIRED FORMAT`, `# RULES`, `# TOOLS` schema | Default for instruct/hybrid runs     |
| `**re_search_template_sys_minimal`** (mine) | New                  | `<tool_call>` (JSON) / `<tool_response>` | Middle ground: tool schema + brief rules, less prescriptive                           | Ablation against the strict template |


Additional Hydra flags I added for prompt ablations (none of these exist in the paper):

- `+data.re_search_use_chat_format=True/False` — apply chat template (system + user turn) vs. raw concatenated prompt. Paper used `apply_chat=True/False` differently (per-model); this flag generalizes the toggle.
- `+data.re_search_add_qwen_chat=True/False` — bypass `tokenizer.apply_chat_template` and build the Qwen chat string manually (`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant`). Helpful when the tokenizer's chat template diverges from what Qwen3 actually expects post-rollout.
- `+data.re_search_add_thinking=True/False` — mirrors Qwen’s **hard** chat-template knob (whether to prime the assistant with an **empty** block vs not), not the textual `/think` / `/no_think` switches:
  - `True` → after `<|im_start|>assistant`, only `\n` (matches `enable_thinking=True` — model may emit a filled `<think>…` section).
  - `False` (default) → assistant header followed by `\n<think>\n\n</think>\n\n` (same insertion as `apply_chat_template(..., enable_thinking=False)` in [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)).
  - Treat as the **hybrid-thinking on/off ablation for this port** — Qwen3 has no separate 0.6B “instruct-only” SKU like Qwen2.5-7B-Instruct vs Base.

---

## 5. Reward Function Differences

Paper's `compute_score` (`src/verl_legacy/utils/reward_score/re_search.py`):

```python
def compute_score(tokenizer, solution_str, ground_truth) -> float:
    # 1. split on assistant marker
    # 2. validate_format() — check <think>, <answer>, <search>/<result> nesting + \boxed{}
    # 3. if not response.endswith(tokenizer.eos_token) → return 0, 'over length'
    # 4. extract \boxed{} answer; F1 vs ground truth
    # 5. f1>0 → return f1; else → return 0.1 ('wrong answer but good format')
```

Mine (`verl_latest/verl/utils/reward_score/re_search.py`):


| Aspect                         | Paper                                                                                  | Mine                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Function signature**         | `compute_score(tokenizer, solution_str, ground_truth)`                                 | `compute_score(solution_str, ground_truth, tokenizer=None)` (tokenizer optional; verl_latest `NaiveRewardManager` pattern)                                                                            |
| **Tag validation**             | `<search>` / `<result>` order + nesting                                                | `<tool_call>` / `<tool_response>` order + nesting **+ JSON payload validation** (`{"name":"search","arguments":<str>}`) **+ explicit rejection of legacy `<search>`/`<result>`/`<think>` style tags** |
| **EOS rule**                   | Hard requirement: response must end with `tokenizer.eos_token` else `0, 'over length'` | Same rule, but **only enforced when `tokenizer` is passed**; if not, skip EOS check (verl_latest `NaiveRewardManager` doesn't always supply tokenizer)                                                |
| **Ground-truth normalization** | Pass-through to F1 (str / list assumed)                                                | Added `_normalize_ground_truth()` to handle `dict` with `"target"` key (Search-R1 / MuSiQue parquet labels)                                                                                           |
| **Reward shape**               | `f1` if f1>0; `0.1` if format-OK but wrong; `0` else                                   | Same shape (kept paper's three-tier reward)                                                                                                                                                           |
| **Format reward separation**   | Implicit (single `compute_score` call)                                                 | Same — single function (no separate format reward yet; ablation TBD)                                                                                                                                  |


**What this means for results**: the reward shape is paper-faithful (F1 + 0.1 partial credit). The validator is stricter (tool-call JSON payload check). EOS handling is conditionally relaxed when the reward manager doesn't pass tokenizer.

---

## 6. Tool / Tag Format — Forced Change

This was not optional: verl_latest's `re_search_agent_loop` uses the **OpenAI / HuggingFace tool-call convention** (`<tool_call>` JSON + `<tool_response>`) instead of the paper's free-text `<search>` / `<result>` pair.


| Layer                                | Paper                                                                            | verl_latest (mine)                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Stop tokens during rollout           | `stop=['</search>']`                                                             | `stop=['</tool_call>']`                                                              |
| Injected after retrieval             | `<result>\n{joined_docs}\n</result>` (encoded with default `add_special_tokens`) | `<tool_response>\n{joined_docs}\n</tool_response>` (with `add_special_tokens=False`) |
| Search query parsing                 | Plain text inside `<search>`                                                     | JSON parse: `{"name":"search","arguments":"<query>"}`                                |
| Failure mode if model doesn't comply | Free-text query → still searches                                                 | JSON parse failure → empty / no tool call → episode terminates                       |


The prompt template, reward validator, and agent loop all had to be co-updated to use `<tool_call>`. Templates and reward function are documented above; the agent loop is `verl_latest/verl/experimental/agent_loop/re_search_agent_loop.py` (kept upstream verl_latest semantics).

---

## 7. Retriever — CPU Move

Paper uses `faiss-gpu` (conda-installed) and runs the index on a GPU. With a single A100 budget, FAISS-on-GPU competed with vLLM KV cache and FSDP for VRAM, so I moved it.


| Aspect                        | Paper                                             | Mine                                                                                                                                      |
| ----------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| FAISS backend                 | `faiss-gpu==1.8.0` (conda)                        | `faiss-cpu` (`faiss_gpu: False` in config)                                                                                                |
| Index                         | `wiki18_100w` (full ~21M passages)                | `wiki18_100w` (full ~21M passages)                                                                                                        |
| Embed model                   | `e5-base-v2`                                      | `e5-base-v2` (kept)                                                                                                                       |
| Serving pattern               | FastAPI + `flashrag` `get_retriever()` (vendored) | Same FastAPI + flashrag, `gpu_id: "0"` left in config but ignored when `faiss_gpu: False`                                                 |
| Search timeout (rollout side) | `timeout=120` (verl_latest default)               | Configurable via `SEARCH_HTTP_TIMEOUT_S` env (set to **300s** in `z_run` etc.) — CPU search is slower and rollout was hitting the default |


`scripts/serving/retriever_config_mini.yaml` is what the runs above point at.

---

## 8. Environment Split (Why Three Conda Envs)

Paper's upstream README installs everything into one env (`re-search`). I split into three because pip's resolver was failing when SGLang and vLLM dependencies (`lm-format-enforcer`, `openai`, `outlines-core`) coexisted:


| Env                         | Purpose                                           | Requirements file                                                                                  |
| --------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `research311` (SGLang base) | Evaluation serving (SGLang)                       | `requirements.txt` + `pip install -e ".[sglang]"`                                                  |
| `research311-vllm` (vLLM)   | Compatible with vLLM 0.8.5 (Qwen3 floor)          | `requirements-vllm.txt` + `pip install -e ".[vllm]"`                                               |
| `r_t` (training)            | Full PPO training stack: vLLM + flash-attn + verl | `requirements-training.txt` + `requirements-training-flashattn.txt` (built `--no-build-isolation`) |
| `research-eval`             | Evaluation (FlashRAG runner + SGLang client)      | `requirements-evaluation.txt`                                                                      |


Three `setup.py` entry points were added (paper has one):

- `setup.py` — base shared install
- `setup_training.py` — training profile editable install
- `setup_evaluation.py` — evaluation profile editable install

Qwen3-specific floors (Qwen docs): `transformers>=4.51.0`, `sglang>=0.4.6.post1`, `vllm>=0.8.5`.

---

## 9. What's Vendored vs. Modified


| Path in repo                                 | Source                           | Status                                                                                                                                                                                                                                  |
| -------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/verl_legacy/`                           | Paper's pinned verl              | **Vendored verbatim** — kept for reference / fallback only                                                                                                                                                                              |
| `src/flashrag/`                              | Paper's flashrag bundle          | **Vendored verbatim** — no changes                                                                                                                                                                                                      |
| `verl_latest/`                               | Upstream verl (newer release)    | **My port**: dropped in newer verl; ported ReSearch-specific files (agent loop, reward manager, templates, dataset hooks); rewrote `re_search_template_sys`; rewrote reward function; kept all training-side hyperparams paper-faithful |
| `scripts/serving/retriever_serving.py`       | Paper                            | Kept verbatim                                                                                                                                                                                                                           |
| `scripts/serving/retriever_config*.yaml`     | Paper                            | Modified — switched `faiss_gpu` to False, pointed at mini index                                                                                                                                                                         |
| `scripts/train/train.sh`                     | Paper's single-node script       | Kept; only used as the "legacy" runner against `src/verl_legacy/`                                                                                                                                                                       |
| `scripts/train/train_multi_node.sh`          | Paper's full reproduction script | Kept verbatim (not used — no multi-node access)                                                                                                                                                                                         |
| `verl_latest/*.sh` (the six profile scripts) | New                              | **Authored from scratch** — these are the experimental runners                                                                                                                                                                          |


---

## 10. Open Items / Not Yet Ablated

(Tracked here so the supervisor meeting can scope what's actually feasible.)

- **Reward-shape ablations**: paper's three-tier reward (F1 / 0.1 / 0) is unchanged. Future ablations could try pure EM, pure F1 (no partial credit), or splitting format reward from answer reward.
- **Hybrid thinking on/off (`re_search_add_thinking`)**: flag exists but no paired-run results yet.
- **Full vs minimal prompt template**: both compile and run; no comparative numbers yet.
- **Reward EOS strict vs relaxed**: paper-strict is the default when tokenizer is passed; ablation against the relaxed path is straightforward but not run.

---

## 11. Pointers

- Paper repo (upstream): `Agent-RL/ReSearch` — the original `README-OLD.md` in this repo is its README, kept for reference.
- Paper-vs-port code-level audit: `research/docs/reports/legacy_vs_verl_latest_research.md` (Apr 3, 2026).
- Baseline reproduction (different repo, evaluation-side only): `reason_over_search/docs/report/RESULTS_m1.md`.

---

## 12. Updates Since This Doc Was Written (code as of 2026-05-06)

Reviewed from `/home/s4374886/omega/re-search/` (branch `sbatch`). The §10 "Open Items" have been resolved; three areas changed.

### 12.1 Reward function rewritten — v1 (resolves §10 "reward-shape ablations")

`verl_latest/verl/utils/reward_score/re_search.py` is no longer the 3-tier (F1 / 0.1 / 0) described in §5. It is now the **v1 reward** documented in `REWARD_V1.md` alongside it. Key changes:

| Aspect | v0 doc (original re_search.py) | v1 (current re_search.py) |
|---|---|---|
| Correct answer score | `f1` (0 to 1) | `min(1.0, 0.55 * F1 + format_score)` |
| Wrong but well-formed | flat `0.1` | `min(0.12, 0.01 + 0.25 * format_score)` |
| Format score range | implicit / binary | `[0, 0.45]` decomposed into 4 components (see below) |
| Think tag | `<think>` | `<redacted_thinking>` |
| Answer gate | `\boxed{}` required | `\boxed{}` still required; returns ≤0.02 if absent |
| Original preserved | — | saved as `re_search_original.py` |

Format score components (sum to [0, 0.45]):
- Tag pairing and hygiene: `0.10 × clamp(pair_score)`
- Cycle reward (think → tool_call → tool_response → think window): `0.18 × cycle_score`
- Query quality (valid JSON, non-repeated, non-overlong): `0.07 × query_quality_score`
- Thinking quality (non-empty, non-repeated): `0.05 × reasoning_quality_score`
- Sequence bonus (no transition violations): `+0.03`
- Answer closure: `+0.01` non-empty body, `+0.01` answer is last block

### 12.2 Format reward ablation family added (the actual Phase-1 experiments)

Four new reward files implement the ablation series run in W&B `research` and `research_revamp`:

| File | Description |
|---|---|
| `r1_searcher_format_zero.py` | 5-tier milestone reward: first-think (0.10), tool_call (0.20), post-tool think (0.20), answer block (0.20), boxed content (0.30). Score floor -0.50 (negative if language mixing). |
| `r1_searcher_format_one.py` | Component-weighted with pairing (0.12), transition (0.34), cycle (0.22), tool JSON (0.12), answer (0.10), think quality (0.10). Floor -0.50. |
| `r1_searcher_format_two.py` | Variant of format_one with adjusted weights (not inspected in detail). |
| `r1_searcher_format_three.py` | Variant with different mixing penalty and component balance. |
| `r1_searcher_answer.py` | Answer-only F1 (no format component); pure semantic credit. |
| `search_r1_like_qa_em.py` | EM-only (adapted from Search-R1 `qa_em.py`). **This is what gets ported to M2 NeMo-RL.** |

The "format reward separated from answer reward" and "pure EM" items from §10 are now implemented here.

### 12.3 New base model training script

`verl_latest/x_base_min_run_qwen3_0.6b_grpo_vllm_gpu_1_a100_40gb.sh` — a smoke/iteration script for the **base model** specifically (vs the instruct/hybrid scripts in §3). This corresponds to the 5/5 failed base-model cold-start attempts documented in `RESULTS_m0_b.md` (run id family `base_state_machine_a`).

### 12.4 What §10 items remain open

- **Hybrid thinking on/off**: flag was exercised in v1 runs; results frozen in `RESULTS_m0_b.md`.
- **Full vs minimal prompt template**: comparative numbers are in `RESULTS_m0_a.md`/`RESULTS_m0_b.md`.
- **Reward EOS strict vs relaxed**: still not explicitly ablated as a standalone experiment.

---

## 13. Items Not Previously Documented (retirement audit 2026-05-06)

These exist in the repo but were never written up. Captured here before the codebase is retired.

### 13.1 Experiment folder pattern and submit system

The repo has a self-contained per-ablation config injection system. Each folder under `experiments/` contains:

```
experiments/<ablation_name>/
  config.yaml    — Hydra overrides: experiment_name, sbatch variant (z/x_min), output_root, temperature, top_p
  prompt.txt     — system prompt body (replaces re_search_template_sys for this run)
  reward.py      — Python module with compute_score(); typically thin wrapper around one of the r1_searcher_format* files
  runs/          — dated run subdirs created at submit time (each has inputs/ logs/ checkpoints/ rollouts/ snapshots)
```

Submission via `scripts/submit_experiment_sbatch.py experiments/<name>/config.yaml [--dry-run | --smoke-test]`. The script stages a dated run dir, snapshots config/prompt/reward, and calls sbatch. Supports dry-run (stage only) and smoke-test (mock trainer/retriever to validate orchestration).

Active experiments at retirement:

| Folder | Reward used | Notes |
|---|---|---|
| `example_single_config_ablation/` | (template) | Reference only |
| `ablation_zero/` | `r1_searcher_format.py` (v1) | First live run; 2 dated runs in `runs/` (2026-04-14) |
| `ablation_one/` | `r1_searcher_format.py` (v1) | **Last active config** (pointed to by `verl_latest/.env`) |
| `ablation_one_base/` | same | Base-model variant of ablation_one |
| `ablation_two/` | different format variant | Alternative; no completed runs |
| `ablation_xxx/` | (discarded) | Placeholder pattern for rejected ablations |

The `ablation_one` prompt (`experiments/ablation_one/prompt.txt`) is a structured tool-call system prompt with numbered steps, `<tools>` schema, `\boxed{}` termination rule, and `<answer>` block. This is the prompt that was wired into the last Slurm runs.

### 13.2 Slurm orchestration (sbatch/)

Two `.sbatch` files handle the ALICE two-stage workflow (retriever must be healthy before trainer starts):

| File | Partition | GPU | Notes |
|---|---|---|---|
| `retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch` | gpu-a100-80g | 1 | 8 CPUs, 120 GB RAM, 8-hour limit. Stage 1: bootstrap env + start retriever in `r_e` venv. Stage 2: wait for `/health`, launch trainer via `srun --overlap` in `r_t` venv. |
| `retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sbatch` | gpu-a100-mig | 1 | 40 GB MIG variant. Supports `RETRIEVER_CONFIG=retriever_config_mini.yaml` override for fast iteration. |
| `test_retriever_then_training_mock.sh` | (no Slurm) | — | Local mock: validates the health-poll + handoff logic without real retriever/trainer. Pre-flight check. |

### 13.3 Automated prompt ablation — results

`scripts/evaluation/autoresearch_prompt/` ran a Codex-driven loop that generated and scored candidate prompts against the **base model** on Bamboogle (125 rows). Findings (frozen in `CURRENT_BEST.json`):

| Metric | Value |
|---|---|
| Candidates tried | 4 (base model path) |
| Best exact EM | 0/125 = 0.000 (all candidates) |
| Best hybrid avg | 78.97% (baseline prompt, candidate_0) |
| Dominant failure mode | `not_end_with_answer` (model stops early without outputting `<answer>`) |
| Conclusion | Baseline `re_search_template_sys` was not improved by any of the 4 prompt variants tried |

The instruct/hybrid path (`autoresearch_prompt/`) has a longer run history (139 KB `prompts.jsonl`) but the same conclusion: baseline prompt is best. The failure mode `not_end_with_answer` is a base model cold-start issue, not a prompt issue; consistent with the Phase-1 finding that base model cannot bootstrap format from cold start.

### 13.4 `verl_latest/.env` experiment injection pattern

The `verl_latest/.env` file (not committed to the paper's repo but present here) overrides two keys for experiment-specific prompt and reward:

```
PROMPT_TEMPLATE_PATH=../experiments/ablation_one/prompt.txt
RE_SEARCH_REWARD_FUNCTION_PATH=../experiments/ablation_one/reward.py
```

The training scripts read these and pass them as Hydra overrides. This is the mechanism that lets `experiments/ablation_one/` be "active" without editing the training script. To switch ablation: change both paths in `.env` (or use `--prompt_template_name` / custom reward path flags directly on the script).

### 13.5 Legacy parity audit (key risk: EOS handling)

`docs/reports/legacy_vs_verl_latest_research.md` (2026-04-03) documents the code-level differences between `src/verl_legacy` and `verl_latest`. The one non-trivial behavioral risk:

- **Legacy `compute_score`**: if the response does not end with `tokenizer.eos_token`, returns `0` with reason `"over length"`. Truncated rollouts get zero reward.
- **Latest `compute_score`**: EOS is stripped if present but **not required**. Truncated rollouts can still earn reward.

This was an **intentional relaxation** for single-GPU where max_response_length is sometimes lower. But it means v0 Phase-1 runs (latest) may have rewarded some rollouts that legacy would have scored 0. Noted here as a potential training distribution shift; not re-run to confirm impact.

### 13.6 Training data

`data/` at repo root (747 MB total). Primary training set is **MuSiQue** (multi-hop QA):

| Path | Format | Size | Purpose |
|---|---|---|---|
| `data/musique/train.parquet` | parquet | 1.6 MB | Training data for GRPO (MuSiQue) |
| `data/musique/test.parquet` | parquet | 16 KB | Validation during training |
| `data/musique/train.jsonl` | jsonl | 39 MB | Raw JSONL source |
| `data/musique_sft/` | parquet + jsonl | ~90 MB | **SFT variant** (never used in RL runs; staged for potential SFT warm-start) |
| `data/bamboogle/test.jsonl` | jsonl | 17 KB | Eval dataset (primary for Phase-1 ablations) |
| `data/hotpotqa/train.jsonl` | jsonl | 544 MB | Training-time val (not used; available) |
| `data/data.ipynb` | notebook | 2.9 KB | Dataset exploration / prep notebook |
| `data/prepare_musique.py` | python | 1.7 KB | Preprocesses MuSiQue parquet from raw downloads |
| `data/download_dataset.sh` | bash | 0.9 KB | Downloads official parquet/jsonl files |

`data/musique_sft/` was staged for a potential SFT warm-start for the base model (to address the cold-start format failure). It was never used in any run; the Phase-1 conclusion was to use hybrid model instead.

### 13.7 `verl_latest/stable/` — tested configs

Two scripts marked as "stable" (tested, not just smoke-validated):

- `stable/run_qwen3_0.6b_grpo_vllm_instruct_gpu_4_l4_24gb_0.sh` — 4×L4-24GB tested stable config
- `stable/run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb_0.sh` — 1×A100-40GB tested stable config

These are snapshots of configs that were confirmed to run without OOM errors; the `_0` suffix indicates the first stable version. The production profile scripts (`x_run`, `z_run`, etc.) in the parent dir are the current defaults and supersede these.

### 13.8 `train_base.sh` — canonical production script

`verl_latest/train_base.sh` is the primary parameterized launcher (11 KB). The profile scripts (x_min/x_run/z_min/z_run/zz/y) in §3 of this doc are **thin wrappers** that set hardware-specific defaults and call it. Key CLI flags it exposes:

- `--prompt_template_name` — which template to use (`re_search_template_sys`, `re_search_template`, etc.)
- `--actor_model_path` — override model path
- `--rollout_n` — GRPO group size (default 5)
- `--add_qwen_chat` — use manual `<|im_start|>` string (bypass `apply_chat_template`)
- `--add_thinking` — `enable_thinking=True` path (vs empty closed `<think>` block default)

The base model variant uses `re_search_use_chat_format=False` (no chat wrapper during prompt encoding); the instruct/hybrid variant uses `re_search_use_chat_format=True`. This flag was the equivalent of the `apply_chat` divergence fixed in M1 eval (D1 in REPRODUCIBILITY.md).

