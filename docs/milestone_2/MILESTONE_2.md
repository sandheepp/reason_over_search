---
title: MILESTONE 2
tags: []
source: internal
created: 2026-05-01
updated: 2026-05-07
---

# Milestone 2: Training [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

> **Phase 1 status (2026-05-01):** Steps 1–8 complete. Pipeline built end-to-end, smoke-tested on Vast.ai 1× A100-80GB (~57 s/step at smoke shape; full Search-R1 schedule extrapolates to 11–17 days/run, ~$300–490/run); 15 reward-parity tests pass.
> **Phase 2 status (2026-05-07): plan pivoted.** The originally planned 6-run sweep (3 seeds × 2 variants) was superseded on 2026-05-04 by a recipe-focused ablation plan (E2H curriculum + S-GRPO + MC-GRPO + JustRL plain-GRPO control; 2–3 runs total) once Phase-2 wall-clock made paired sweeps unaffordable. See [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 5–6 and [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md).
> **Phase 2 runbook:** [PHASE_2_RUNBOOK.md](PHASE_2_RUNBOOK.md) — concrete sequence to run training on Vast.ai.
> **Audit landing page:** [docs/training/README.md](../training/README.md).

## Context

Search-R1 trains [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) (and 7B) with GRPO, on both **base** and **instruct** variants. We want to do the same training on **[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)**, but `verl` does not support Qwen3.5, so we port the Search-R1-style training loop to **[NeMo-RL](https://github.com/NVIDIA-NeMo/RL)**, which supports GRPO and RLVR for Qwen3.5.

The Qwen3.5 lineup differs from Qwen2.5: it has **base** and a default model with a **soft-switch reasoning mode** (`enable_thinking=true/false`) — there is no separate "instruct" variant. We will train both:

- **base** → [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- **hybrid** (Qwen's default soft-switch reasoning model) → [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

The **reward function** and **chat template** must be **decoupled from the core training code** so they can be swapped cleanly. In Milestone 2 we use the **Search-R1 baseline reward unchanged** — the goal is to verify training works on Qwen3.5 with the paper's reward before ablating in Milestone 3.

## Goal

Reproduce the Search-R1 training pipeline on Qwen3.5-2B, for both base and hybrid variants.

Use the official Search-R1 training dataset (linked from the paper).

Create a `training/` folder at the repo root containing the NeMo-RL setup with the Search-R1-style modifications. Add a new `training` env to the docker image.

## Step-by-step

1. **Set up the training env in docker (hybrid paradigm).** The image at `[docker/reason-over-search-v1/Dockerfile](../../docker/reason-over-search-v1/Dockerfile)` splits two concerns:
  - **Stable surfaces (`retriever`, `evaluation_search_r1`)**: code + conda env baked in, **as before**. No change.
  - **Active surface (`training` / NeMo-RL)**: env baked, code cloned at runtime. Image ships `uv` + a pre-warmed wheel cache for NeMo-RL. The repo gets `git clone`d on Vast; `uv sync` materializes the venv from the cached wheels in seconds-to-minutes.
   **Source layout**: NeMo-RL @ `v0.6.0` is **committed to this repo** at `[training/nemo_rl/](../../training/nemo_rl/)` with submodules (Megatron-LM, Megatron-Bridge, Automodel, Gym; `.git` stripped so it's not a nested repo). Local edits land in this repo's git history. Bump via `[training/setup.sh](../../training/setup.sh)` with `NEMO_RL_REF=<ref> FORCE_RECLONE=1`, commit the new tree, and rebuild the image with `--build-arg NEMO_RL_REF=<ref>` so the wheel cache matches.
   **Dockerfile** additions for Milestone 2:
  - Installs `[uv](https://docs.astral.sh/uv/)` (NeMo-RL's official package manager).
  - **Pre-warms `/root/.cache/uv/`** by shallow-cloning NeMo-RL @ `${NEMO_RL_REF}` into `/tmp`, running `uv sync --extra ${UV_EXTRAS} --no-install-project` to download all wheels, then deleting the temp clone. The wheels stay in the cache.
  - **Does not COPY `training/`** — the source comes from `git clone` on Vast. `.dockerignore` excludes `training/nemo_rl/` from the build context to keep `docker build` fast.
   Build + push:
   **On Vast** (per fresh instance):
   For local dev *outside* docker, `[training/setup.sh](../../training/setup.sh)` does the same `uv sync` (first run downloads ~5 GB; subsequent runs hit uv's local cache).
2. **Download and prep the Search-R1 training dataset, in NeMo-RL's row schema.** Run `[training/scripts/prepare_dataset.py](../../training/scripts/prepare_dataset.py)` — uv inline-script, idempotent. It pulls `[PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)` (169,615 train + 51,713 test, parquet) and applies two transforms in one pass:
  - **Strip the prebaked Search-R1 template** — `prompt[0].content := question`.
  - **Rename `prompt` → `messages`** — match the column name NeMo-RL's `ResponseDataset` routes on (`[response_dataset.py:64](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L64)`). Length-1 list `[{"role": "user", "content": question}]`; `golden_answers` stays as its own column for the M2-step-4 processor to read.
   Output: `data/training/nq_hotpotqa_train/{train,test}.parquet`, committed via Git LFS (matches the `data/**/*.parquet` rule in `[.gitattributes](../../.gitattributes)`). Full schema + the verified NeMo-RL mapping in `[docs/training/TRAINING_DATA.md](../training/TRAINING_DATA.md)`.
   Stripping keeps the dataset **template-agnostic** — the chat template (Qwen3.5 native `<tool_call>` default vs. the paper's `<search>` ablation arm) is applied at **rollout time** via run config (`prompt_file` + `system_prompt_file`) by `math_hf_data_processor`-style processing in `[processors.py:467-477](../../training/nemo_rl/nemo_rl/data/processors.py#L467-L477)`. Rationale: `[CHAT_TEMPLATE.md](../training/CHAT_TEMPLATE.md)` §5.
3. *(Merged into step 2.)* The schema mapping was originally a separate step gated on inspecting `training/nemo_rl/examples/`. After verifying the routing rule in `[response_dataset.py:64](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L64)`, the reshape was small enough to fold into the prep script. Verified mapping documented in `[TRAINING_DATA.md](../training/TRAINING_DATA.md)` §"Mapping to NeMo-RL's row schema".
4. **Apply Search-R1-style modifications** as an **overlay** under `training/src/`. We never modify the vendored NeMo-RL source at `[training/nemo_rl/](../../training/nemo_rl/)` directly — overlay code self-registers into NeMo-RL's pluggable `DATASET_REGISTRY` / `PROCESSOR_REGISTRY` / `ENV_REGISTRY` at launch.
  - **Wire the retriever as a custom Ray-actor environment** (Path A from `[NEMO_RL_KNOBS.md](../training/NEMO_RL_KNOBS.md)` §9): `[training/src/environments/search_r1_env.py](../../training/src/environments/search_r1_env.py)` — `SearchR1Env` (plain class for tests) + `SearchR1Environment = ray.remote(...)` (registered FQN). The actor parses `<tool_call><function=search><parameter=query>...` (qwen_native arm) or `<search>...</search>` (paper arm) from rollouts, POSTs to `{retriever_url}/batch_search`, wraps the response per arm — Qwen3.5 chat-template markers vs. raw `<information>...</information>`. On `<answer>X</answer>`: compute reward via `[training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py)`.
  - **Chat template, reward function, prompt builder** all live under `training/src/`: `[prompts/search_r1_paper.txt](../../training/src/prompts/search_r1_paper.txt)`, `[chat_template/tools.py](../../training/src/chat_template/tools.py)`, `[processors/search_r1.py](../../training/src/processors/search_r1.py)`, `[datasets/search_r1.py](../../training/src/datasets/search_r1.py)`, `[registry.py](../../training/src/registry.py)`.
  - **If a local edit to the vendored NeMo-RL is unavoidable** (no hook we can register against), edit `training/nemo_rl/` directly and commit — it's a tracked tree in this repo. The overlay design has covered every case so far in M2.
   **Done**: 9 overlay files at `[training/src/](../../training/src/)` + 5 unit-test files at `[training/tests/](../../training/tests/)` (19 tests pass on the local Mac; env/dataset modules cleanly skip without nemo_rl/torch). Architecture documented in `[docs/training/README.md](../training/README.md#trainingsrc-overlay-architecture)`.
5. **Verify the training setup matches the paper.** All paper hyperparameters cross-checked against `[Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh)` — the EM-only baseline that produced the published GRPO checkpoints we evaluated in Milestone 1. Canonical record: `[PAPER_VS_OURS_TRAINING.md](../training/PAPER_VS_OURS_TRAINING.md)` §6; verl-side mapping: `[VERL_REFERENCE.md](../training/VERL_REFERENCE.md)` §2; full audit table: `[docs/training/README.md](../training/README.md#step-5-audit-summary)`.
  **Resolved during step 5:**
  - **Reward function:** byte-identical port at `[training/src/rewards/search_r1.py](../../training/src/rewards/search_r1.py)`; 15 parity tests pass (`[training/tests/test_reward_parity.py](../../training/tests/test_reward_parity.py)`).
  - `**kl_loss_type=low_var_kl` ≡ NeMo-RL `loss.reference_policy_kl_type: k3`** — both compute Schulman 2020 k3 (`exp(ref-log) - (ref-log) - 1`) byte-identically; NeMo-RL's `k3` is the default, no override needed.
  - `**state_masking=true` ≡ automatic** in NeMo-RL via role-based `token_loss_mask` (`[grpo.py:1685-1693](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)`) — assistant tokens get loss=1, env (`role: tool`) tokens get loss=0. No config knob.
  - `**test_freq=100`, `total_training_steps=1005`, `val_before_train=true`, `max_turns=4`** — confirmed in v0.2 verl yaml. Maps to NeMo-RL `grpo.{val_period, max_num_steps, val_at_start, max_rollout_turns}`.
  - `**max_obs_length=500`** (verl per-`<information>` block cap on retrieved docs) — implemented as a 2000-char proxy in `[training/src/environments/parsers.py](../../training/src/environments/parsers.py)` (`DEFAULT_MAX_OBS_CHARS`); pure-Python so parsers stay testable without a tokenizer dep. Wired through `env.search_r1.max_obs_chars` in the GRPO yaml.
6. **Write GRPO training scripts** for 1× A100 80GB and 2× A100 80GB.
  - **Configs**: `[training/configs/grpo_qwen3.5_2b_1xa100.yaml](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml)` and `[training/configs/grpo_qwen3.5_2b_2xa100.yaml](../../training/configs/grpo_qwen3.5_2b_2xa100.yaml)` — full standalone YAMLs (no Hydra defaults composition). Every value is tagged in comments as `[paper]` (Search-R1 hyperparameter — DO NOT change without doc update) or `[memory]` (free to retune after observing VRAM/throughput).
  - **Launcher**: `[training/scripts/run_grpo.py](../../training/scripts/run_grpo.py)` — thin wrapper that imports `[training.src.registry](../../training/src/registry.py)` (populates registries) and hands off to NeMo-RL's `examples.run_grpo.main()` with argv (Hydra overrides) passed through.
  - **Bash wrappers**: `[run_grpo_1xa100.sh](../../training/scripts/run_grpo_1xa100.sh)` and `[run_grpo_2xa100.sh](../../training/scripts/run_grpo_2xa100.sh)` take `--variant {base,hybrid}`, `--seed N`, `[--arm {qwen_native,paper}]`. They source `training/.env` for W&B, set Hybrid's `enable_thinking=true` via Hydra `++` override, point the paper arm at the prompt file, build a unique W&B run name + checkpoint dir, and exec the launcher.
  - **Reference settings** (cherry-picked, transferable bits only) in `[VERL_REFERENCE.md](../training/VERL_REFERENCE.md)`. The original verl scripts at `~/Documents/Obsidian/code/omega/research/verl_latest/` stay local.
  - **Folder layout** documented in `[training/README.md#folder-layout](../../training/README.md#folder-layout)`: `configs/`, `scripts/`, `src/`, `tests/` clearly separated.
7. **Connect to W&B.** API key in `[training/.env](../../training/.env.example)` (gitignored — copy `[training/.env.example](../../training/.env.example)` and fill in). Sourced by both bash wrappers before launching. Metric set documented in `[VALIDATION.md §4](../training/VALIDATION.md#4-metrics-logged-to-wb)`; env-side metrics implemented in `[SearchR1Env.global_post_process_and_metrics](../../training/src/environments/search_r1_env.py)` (`accuracy`, `em_hit_rate`, `fraction_of_samples_properly_ended`, `generation_lengths`, `prompt_lengths`, `num_problems_in_batch`).
  The configs default to `logger.wandb_enabled: true` with project `reason-over-search`; W&B run names follow `qwen3.5-2b-{variant}-search_r1-{arm}-{Nx}a100-seed{N}-{UTC-timestamp}` (e.g. `...seed42-20260502T1437Z`), injected via `logger.wandb.name` override. The UTC timestamp suffix means repeated launches of the same `(variant, arm, seed)` get distinct W&B runs while still resuming from the same `(variant, arm, seed)`-keyed checkpoint dir.
8. **Write a paper-vs-ours training comparison.** `[PAPER_VS_OURS_TRAINING.md](../training/PAPER_VS_OURS_TRAINING.md)` — every paper hyperparameter is now cross-checked against the verl yaml (§6) with a "match / equivalent / divergent" status; §7 has projected wall-clock + GPU-hours + Vast.ai $ for both 1× and 2× A100 layouts; §9 lists the five knowing divergences (Qwen3.5-2B vs Qwen2.5-3B, qwen_native vs paper template, hybrid vs instruct naming, A100 vs H100, NeMo-RL vs verl). Observed wall-clock / GPU-hours / $/run cells are still **TBD — filled post-Phase-2** from W&B run summaries.

## Deliverables — Phase 1 status

1. ✓ **Docker image** at `pantomiman/reason-over-search-v1:v2` ships uv + pre-warmed NeMo-RL wheel cache. Stable surfaces (`local_retriever`, `evaluation_search_r1`) baked unchanged from M1.
2. ✓ **Training dataset** prepped + committed via Git LFS — `data/training/nq_hotpotqa_train/{train,test}.parquet` (169,615 + 51,713 rows). NeMo-RL row schema verified.
3. ✓ **Training code** wired up: 9 overlay files at `[training/src/](../../training/src/)`, 5 test files at `[training/tests/](../../training/tests/)` (19 pass; env/dataset modules cleanly skip without nemo_rl). Two GPU-layout configs at `[training/configs/](../../training/configs/)`. Launch scripts at `[training/scripts/](../../training/scripts/)`. W&B wired via `[training/.env](../../training/.env.example)`.
4. ✓ **Code reviewed**: NeMo-RL master config schema cross-checked field-by-field; reward function 15-test parity vs M1 eval pipeline; verl-vs-NeMo-RL hyperparameter audit complete in `[PAPER_VS_OURS_TRAINING.md §6](../training/PAPER_VS_OURS_TRAINING.md#6-hyperparameters)`.
5. ✓ **Docs**: landing page at `[docs/training/README.md](../training/README.md)` with end-to-end view + step-5 audit summary + overlay architecture; `[PHASE_2_RUNBOOK.md](PHASE_2_RUNBOOK.md)` has the concrete Vast.ai sequence.
6. ✓ **Repo state**: clean, all WIP merged to `training-setup` branch and pushed.

## Phase 2 — run training

> **Concrete runbook**: `[PHASE_2_RUNBOOK.md](PHASE_2_RUNBOOK.md)`.

### Context

Once docker + training code are set up (Phase 1 done), run training on Vast.ai.

### Goal

Train Qwen3.5-2B (base + hybrid) via GRPO with the Search-R1 baseline reward. Verify training is stable and produces a checkpoint that improves on the base model.

### Success criteria

- **3 seeds × {base, hybrid} = 6 runs** per ablation arm.
- Note **wall-clock time** and **compute** (GPU-hours, $) per run; record in `[PAPER_VS_OURS_TRAINING.md §7](../training/PAPER_VS_OURS_TRAINING.md#7-compute)`.
- Compare loss / reward / EM curves against the paper's reported training dynamics.
- **Eval gate**: run the trained checkpoint on **Bamboogle** first (fast, n=125). If results trend up vs. the untrained Qwen3.5-2B-Base, run the full Milestone 1 benchmark suite (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) using the existing `evaluation_search_r1/` pipeline.

### Compute estimate

Projections in `[PAPER_VS_OURS_TRAINING.md §7](../training/PAPER_VS_OURS_TRAINING.md#7-compute)`: **~30–50 h / run on 1× A100 80GB** (~~$30–100), **~~18–28 h on 2× A100** (~~$36–112). Total Phase 2 budget: **~~$180–600** for the 6-run plan on 1× A100.

### Checkpoint persistence

Saved to **Vast.ai persistent storage** (`/workspace/persistent/checkpoints/...`) as a starting point — set `CHECKPOINT_DIR_BASE` in `training/.env`. May move to HF Hub or repo LFS later.

### Step-by-step

The full sequence — pre-flight checklist, retriever setup, smoke run, real runs, monitoring, eval gate, post-Phase-2 — lives in `**[PHASE_2_RUNBOOK.md](PHASE_2_RUNBOOK.md)`**. Brief outline:

1. **Pre-flight** (local): tests pass, LFS objects present, docker image current, W&B key ready.
2. **Boot Vast** with `pantomiman/reason-over-search-v1:v2`, 1×/2× A100 80GB, ≥150 GB persistent storage, ≥100 GB RAM.
3. **Set up + validate retriever** at `127.0.0.1:3005`.
4. **Download Qwen3.5-2B models** (base + hybrid) into `training/models/` via `huggingface-cli download Qwen/Qwen3.5-2B-Base` and `Qwen/Qwen3.5-2B`. ~5 GB each (bf16 weights + tokenizer/config). One-time per Vast instance; lives on persistent storage so it survives across instance restarts. Commands in `[training/README.md#models](../../training/README.md#models)`; runbook step at `[PHASE_2_RUNBOOK.md§4](PHASE_2_RUNBOOK.md)`. The launch scripts default to passing the HF repo IDs (`Qwen/Qwen3.5-2B-Base` / `Qwen/Qwen3.5-2B`) — vLLM resolves them via the HF cache populated by these downloads. Override with `policy.model_name=<absolute-path>` to use a specific local copy.
5. **Smoke run** (5 steps, single seed) to verify all pipes connect.
6. **Real runs** — 3 seeds × {base, hybrid} via `bash training/scripts/run_grpo_{1,2}xa100.sh --variant {base,hybrid} --seed N`.
7. **Smoke-eval on Bamboogle** through the M1 eval pipeline.
8. **Full eval suite** if Bamboogle improves; aggregate across seeds.

