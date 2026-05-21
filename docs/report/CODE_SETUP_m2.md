---
title: Code Setup M2 — NeMo-RL Qwen3.5-2B training pipeline
tags: [report, training, m2]
source: internal
created: 2026-05-06
updated: 2026-05-08
---

# Code Setup M2: NeMo-RL Training Pipeline (Phase-2 setup)

**Date**: 2026-05-06 (updated 2026-05-08).  
**Scope**: What changed from the M0 ALICE / Qwen3-0.6B / verl port ([`CODE_SETUP_m0.md`](CODE_SETUP_m0.md)) to the M2 NeMo-RL training pipeline in this repo. Phase-2 (Qwen3.5-2B-Base + Qwen3.5-2B) targets the same ReSearch-style multi-turn search-tool RL recipe, ported to a framework that supports the model.  
**Cluster**: Vast.ai 1× A100-80GB.  
**Source path in this repo**: `training/` (src, configs, scripts, nemo_rl vendor, tests).

---

## 1. Headline Diff vs v0

| Dimension | v0 (research repo, ALICE) | v1 (this repo, Vast.ai) |
|---|---|---|
| **Model** | Qwen3-0.6B (base + hybrid) | **Qwen3.5-2B-Base + Qwen3.5-2B** (~3x bigger) |
| **Framework** | verl (verl_legacy + verl_latest vendored) | **NeMo-RL v0.6.0** (vendored at `training/nemo_rl/`) |
| **Why framework changed** | verl does not support Qwen3.5 | Forced switch; NeMo-RL DTensor backend supports it |
| **Reward** | F1 + 0.1 partial credit (3-tier) | **EM-only** (no partial credit, no format bonus) |
| **Tool tags** | `<tool_call>` JSON only | Two arms: `paper` (`<search>`/`<information>`) and `qwen_native` (`<tool_call>`/`<tool_response>`) |
| **Config format** | Shell scripts with env var overrides | **YAML (Hydra)** + Python launcher (`run_grpo.py`) |
| **Hardware profiles** | 6 scripts (x_min/x_run/z_min/z_run/zz/y) | **2 YAML configs** (1xa100 / 2xa100) + 2 launch scripts |
| **Retriever (training)** | Flat IP, faiss-cpu | **IVF-SQ8 default**, 8 workers (flat times out under rollout load) |
| **Retriever (eval, M1)** | Flat IP (paper-fidelity) | Flat IP still used for M1 eval; IVF-SQ8 for M2 training |
| **Sequence packing** | On (verl default) | **Off** (GDN kernel crash on Qwen3.5; dynamic batching instead) |
| **Environment** | conda (3 envs: research311, research311-vllm, r_t) | **venv** pre-built tarball from HF Hub via `bootstrap.sh` |
| **W&B project** | `research` / `research_revamp` | `reason-over-search` |
| **Observed smoke** | Not measured in v0 | ~57 s/step at 20 traj/step on 1x A100-80GB |

---

## 2. What's Unchanged

These knobs are paper-faithful in both v0 and v1:

| Knob | Value |
|---|---|
| KL coefficient | 0.001 (k3 = low_var_kl in both verl and NeMo) |
| LR | 1e-6 (AdamW) |
| LR warmup | 28.5% of steps (286 of 1005) |
| GRPO group size (G) | 5 |
| max_turns | 4 |
| retriever topk | 3 |
| max_obs (tokens) | ~500 (2000 chars proxy) |
| total steps | 1005 |
| temperature (rollout) | 1.0 |
| retriever HTTP contract | `/batch_search` on `127.0.0.1:3005` |

---

## 3. Critical Bugs Fixed During M2 Smoke (all in `training/fix/CHANGES.md`)

Six bugs stood between clone and first clean GRPO step. Each blocked training:

| # | File | Symptom | Fix |
|---|---|---|---|
| 1 | `configs/*.yaml` | No stop strings on turn 1: model hallucinated fake `<tool_response>` blocks (18/20 trajectories corrupt) | Added `stop_strings: ["</tool_call>","</search>","</answer>"]` to YAML generation config |
| 2 | `src/processors/search_r1.py` | `qwen_native` arm put protocol in system prompt; reward = 0.000 at step 1 | Moved protocol to user message (mirrors paper arm); reward jumped to 0.270 at step 1 |
| 3 | `src/processors/search_r1.py` | Default `apply_chat_template` inserts closed `<think></think>` before generation; model has no open block to fill | Added `enable_thinking=True` to both arms' `apply_chat_template` calls |
| 4 | `src/registry.py` | `ValueError: No actor environment registered` at training startup | Patched `ACTOR_ENVIRONMENT_REGISTRY` to map `SearchR1Environment` → `PY_EXECUTABLES.SYSTEM` |
| 5 | `src/datasets/search_r1.py` | `KeyError: 'task_name'` in dataloader | Added `task_name` column (mirrors every upstream NeMo-RL dataset) |
| 6 | `scripts/run_grpo_1xa100.sh` | `ConfigCompositionException` on startup (null parent override for validation) | Removed `data.validation.arm` from OVERRIDES; added `--` passthrough for Hydra |

---

## 4. Architecture of `training/src/` (the 9-file overlay)

NeMo-RL is extended via a thin overlay; upstream is unmodified except for the metric routing patch in `training/nemo_rl/`:

```
training/src/
  datasets/search_r1.py      — HF Dataset loader for NQ+HotpotQA parquet; adds task_name
  environments/search_r1_env.py — Ray actor: HTTP retriever calls, turn management
  environments/parsers.py    — <search>, <tool_call>, <answer> parsers; dispatches by arm
  processors/search_r1.py    — applies chat template (both arms), wraps question in prompt
  rewards/search_r1.py       — EM-only scorer (normalize_answer + exact_match_score)
  prompts/search_r1_paper.txt           — paper arm user prompt
  prompts/search_r1_qwen_native_user.txt  — qwen_native arm user prompt
  prompts/search_r1_qwen_native_system.txt — qwen_native arm system prompt
  registry.py                — registers env + processor + reward into NeMo-RL
```

19 unit tests in `training/tests/` cover dataset adapter, env step, format helpers, parser dispatch, and reward parity. All pass.

---

## 5. Two Prompt Arms

| Arm | System prompt | User prompt | Tags | Notes |
|---|---|---|---|---|
| `paper` | None (null) | `search_r1_paper.txt` (Search-R1 verbatim) | `<search>` / `<information>` | Paper-faithful; ablation baseline |
| `qwen_native` | `search_r1_qwen_native_system.txt` (brief tool-role desc) | `search_r1_qwen_native_user.txt` (protocol + question) | `<tool_call>` JSON / `<tool_response>` | Default arm; in-distribution for Qwen3.5 chat template |

The YAML default is `qwen_native`. Override via `data.train.arm=paper data.default.prompt_file=training/src/prompts/search_r1_paper.txt data.default.system_prompt_file=null`.

---

## 6. Reward Function (EM-only, no partial credit)

v0 used the ReSearch 3-tier reward (F1 / 0.1 format bonus / 0). v1 uses Search-R1's `qa_em.py` pattern: `em_check(normalize_answer(prediction), normalize_answer(ground_truth))` returns 1.0 or 0.0. No partial credit, no format bonus. This is the paper-faithful choice for the M2 baseline; partial-credit ablation is deferred.

Search-R1's GitHub ships two reward modules; earlier docs conflated them. Only `qa_em.py` (EM-only) is used here. See `docs/training/SMOKE_RESULTS_2026-05-06.md` for the gotcha note.

---

## 7. Key Qwen3.5 Architecture Constraints

Two hard requirements not present in v0 (Qwen3-0.6B is standard transformer):

1. **Sequence packing must be OFF**: Qwen3.5 interleaves GatedDeltaNet (linear-attention) with full-attention layers. The `torch_chunk_gated_delta_rule` kernel raises `CUDA illegal memory access` during `get_logprobs()` with packed sequences. Use `policy.dynamic_batching.enabled=true` instead (`train_mb_tokens = seq_len * micro_batch`).

2. **`enable_thinking=True` required in `apply_chat_template`**: Qwen3.5-Base and the instruct model share the same chat template Jinja. With the default (`enable_thinking=False`), the template inserts a closed `<think>\n\n</think>\n\n` prefix before the model generates anything; the model then has no open block to fill. With `enable_thinking=True` the prefix is `<think>\n` (open), which is what both prompt arms intend.

---

## 8. Config Format Change (shell → YAML)

v0 used shell scripts with env var overrides (e.g. `MAX_RESPONSE_LENGTH=4096 bash z_min_run.sh`). v1 uses Hydra YAML + a thin Python launcher:

```
training/configs/grpo_qwen3.5_2b_1xa100.yaml   # single-GPU config
training/configs/grpo_qwen3.5_2b_2xa100.yaml   # dual-GPU decolocated config
training/scripts/run_grpo_1xa100.sh             # sets --model, --arm, --seed; passes through Hydra overrides via --
training/scripts/run_grpo.py                    # Hydra entrypoint
```

Every paper-faithful knob is tagged `[paper]` inline in the YAML. Every hardware-tuning knob is tagged `[memory]`. Audit landing page: `docs/training/PAPER_VS_OURS_TRAINING.md`, knob guide: `docs/training/NEMO_RL_KNOBS.md`.

---

## 9. Environment Setup (bootstrap.sh replaces conda)

v0 split into 3-4 conda envs on ALICE. v1 uses a single bootstrap script on Vast.ai:

```bash
training/scripts/bootstrap.sh
```

Steps automated:
1. Clone repo, set up LFS
2. Install NeMo-RL editable install + overlay
3. **Download pre-built venvs tarball** from `pantomiman/reason-over-search-v1-venvs` on HF Hub (~5 GB, ~3 min) — avoids GPU-less Ray actor failing to compile `nv-grouped-gemm` at install time
4. Start retriever with IVF-SQ8 index + 8 workers
5. Start W&B agent

Full runbook: [`docs/setup/SETUP_INSTANCE.md`](../setup/SETUP_INSTANCE.md) (Vast.ai / Verda B300 / RunPod-class); [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../setup/BOOTSTRAP_NEW_INSTANCE.md) (any other docker host).

---

## 10. Smoke Results (state as of 2026-05-06)

All smoke runs are in `logs/` (10 variant directories). Latest clean run: `smoke_v4_*` (2026-05-06).

| Metric | Value |
|---|---|
| Step time (smoke shape, 20 traj/step) | ~57 s/step |
| Full run extrapolation (510 traj/step, 1005 steps) | 11 to 17 days on 1x A100-80GB |
| Estimated cost per full run | ~$300 to $490 at $1.20/h |
| Mean reward at step 1 (qwen_native, hybrid) | 0.270 (after processor fix) |
| Hallucination rate (stop-strings fix) | 18/20 → 0/20 |

Full notes: `docs/training/SMOKE_RESULTS_2026-05-06.md`. Archived earlier smokes: `docs/archive/training/`.

---

## 11. What's Still Off / Disabled in the Config

These are intentionally gated for first-pass training; each is a one-line flip once the loop is verified stable:

| Feature | Current state | Re-enable |
|---|---|---|
| Validation in-loop | `val_period: 0` | Set `val_period: 50`; see `docs/training/VALIDATION.md` |
| Checkpointing | `checkpointing.enabled: false` | Set `enabled: true`; checkpoint_dir is pre-configured |
| Venv pre-build (automodel) | Tarball from HF Hub | Already automated in `bootstrap.sh` |

---

## 12. Pointers

- Full audit of paper vs v1 training knobs: `docs/training/PAPER_VS_OURS_TRAINING.md`
- All 6 bug fixes with full symptom+diagnosis: `training/fix/CHANGES.md`
- NeMo-RL knob guide: `docs/training/NEMO_RL_KNOBS.md`
- verl → NeMo-RL translation table (KL types, batch math): `docs/training/VERL_REFERENCE.md`
- Smoke results with reward curves: `docs/training/SMOKE_RESULTS_2026-05-06.md`
- v0 baseline this document extends: `docs/report/CODE_SETUP_m0.md`
