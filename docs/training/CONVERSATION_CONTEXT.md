---
title: CONVERSATION CONTEXT
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Training: Conversation Context (for Claude Code bootstrap)

**Read this first when a fresh agent session starts on training work.** Then read the file map below in order; you'll have full context in ~30 minutes.

> **Style note (from project CLAUDE.md):** no em-dashes (`—`, `--`) in prose; use `;`, `:`, `()`, or `X to Y`. Honoured throughout this file.

---

## What this directory is

`docs/training/` documents the **Phase-2 (M2) NeMo-RL port** of Search-R1's GRPO training loop, targeting Qwen3.5-2B on 1x A100 80GB. It is the *why* behind the code under [`training/`](../../training/) (the *what / how-to-run*). Owned by [Milestone 2](../milestone_2/MILESTONE_2.md).

Two parallel concerns:

1. **Faithful reproduction** of Search-R1's GRPO training (paper-vs-ours audit, hyperparameter parity, EM reward, retrieval HTTP contract).
2. **Engineering for our hardware** (1x A100 80GB SXM via Vast.ai; ~\$1000 budget; thesis deadline 2026-06-15). Forces departures from the paper: smaller batch (102 prompts/step vs paper's 512), `sequence_packing: false` (Qwen3.5 GatedDeltaNet kernel crashes with packing), `train_micro_batch_size: 2` (without packing), IVF-SQ8 FAISS index (flat IP times out under rollout HTTP load).

The active question (the supervisor-facing reframe of the thesis, see [`docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 5): *"Is it feasible to post-train a small LM to Search-R1-level results under realistic resource constraints, and what is the optimised training recipe?"*. Candidate answer ([`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) § 6): a stack of E2H curriculum + S-GRPO + MC-GRPO on a Search-R1 GRPO baseline, with a JustRL plain-GRPO control alongside.

---

## State of the world (2026-05-06)

**Pipeline status: built end-to-end and smoke-tested.** Four smoke combos (`{base, hybrid} × {qwen_native, paper}`) ran successfully on Vast.ai 1x A100 80GB. ~57 s / step at smoke shape (20 traj/step). Extrapolates linearly to ~24 min / step at the real config (510 traj/step) -> 1005 steps in **11 to 17 days** on 1x A100 (5 to 8.5 d on 1x H100). Cost: **\$300 to \$490 / run** at \$1.20/GPU-h on 1x A100 (this is for the affordable 0.6-epoch budget = 1005 steps × 102 prompts ≈ 102k prompts of the 169k-row corpus; matching paper's 3-epoch schedule at our batch shape would be ~5× → ~55–85 d, ~\$1,600–2,400 / run). **Smoke-anchored, with cited math; full derivation in [`SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config) and [`PAPER_VS_OURS_TRAINING.md §7`](PAPER_VS_OURS_TRAINING.md#7-compute).**

**Plan reframe.** The original 6-run plan (3 seeds x 2 variants) is **superseded** by the recipe-ablation plan in [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md). With ~\$1000 USD budget and 11 to 17 d / run on 1x A100 at the affordable 0.6-epoch budget, that supports ~2 to 3 runs total: the JustRL plain-GRPO control + the optimised stack. Phase-2 will **start with Qwen3.5-0.8B** (cheaper smoke + iteration; the architecture is shared with 2B so the pipeline ports trivially) before extending to 2B if the recipe holds. Reward-ablation sweeps are off the table.

**First-pass training config currently disables both validation and checkpointing** (see [`VALIDATION.md`](VALIDATION.md)). The first long run is mechanics verification only; flip the `[DISABLED for first-pass training]` blocks back on per [`VALIDATION.md §7`](VALIDATION.md#7-re-enabling-validation-planned-not-active) before kicking off ablations.

---

## Reading path (~30 min total)

Order matters. Each file has a clear job; cross-references between them resolve the rest.

| # | File | Why read it |
|---|---|---|
| 0 | (this file) | situational awareness |
| 1 | [`README.md`](README.md) | landing page; index of every other file in this dir; end-to-end pipeline diagram; step-5 audit summary; overlay architecture map |
| 2 | [`SMOKE_RESULTS_2026-05-06.md`](SMOKE_RESULTS_2026-05-06.md) | latest smoke results (4 combos x ~9 samples each) + the canonical timing/cost analysis (lifted from the 2026-05-02 small-shape run, archived as [`docs/archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md`](../archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md)) |
| 3 | [`PAPER_VS_OURS_TRAINING.md`](PAPER_VS_OURS_TRAINING.md) | every knowing departure from Search-R1's training setup, with rationale; §7 has the wall-clock + cost extrapolation table with cited math |
| 4 | [`TRAINING_DATA.md`](TRAINING_DATA.md) | dataset schema; verified 169,615 train + 51,713 test rows; how `prepare_dataset.py` reshapes upstream into NeMo-RL row format |
| 5 | [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md) | qwen_native (default) vs paper (ablation) chat-template arms with verbatim Qwen3.5 jinja and rendered turn-by-turn examples |
| 6 | [`VALIDATION.md`](VALIDATION.md) | in-loop validation plan (NQ + HotpotQA test sets); current first-pass disables it; §7 has the re-enable steps |
| 7 | [`NEMO_RL_KNOBS.md`](NEMO_RL_KNOBS.md) | every NeMo-RL config knob that matters with our shipped values; §7 is the authoritative diff-vs-upstream table |
| 8 | [`VERL_REFERENCE.md`](VERL_REFERENCE.md) | verl-side reference settings (HTTP retriever contract, KL/GRPO mappings, FSDP-to-DTensor translations); useful when reading verl scripts |
| 9 | [`docs/setup/SETUP_INSTANCE.md`](../setup/SETUP_INSTANCE.md) | total GPU-instance setup guide (human or agent), covering Vast.ai, Verda B300, and RunPod-class hosts: bootstrap, retriever, eval, training; replaces the older agent-only `SETUP_CLAUDE.md` |
| op | [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) | operational runbook for Phase-2 training (boot, retriever setup, launch, monitoring, recovery); pairs with `SETUP_INSTANCE.md` |

---

## Source-of-truth numbers (use these; do not re-derive from training-time language)

| Fact | Value | Source |
|---|---|---|
| Train corpus rows | **169,615** | [`data/training/nq_hotpotqa_train/train.parquet`](../../data/training/nq_hotpotqa_train/train.parquet); verified `pyarrow.parquet.read_metadata` |
| Test corpus rows | **51,713** | same |
| Total training steps | 1005 | [`grpo.max_num_steps`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml); matches verl `total_training_steps=1005` (paper text's "500 steps" is superseded) |
| Group size G | 5 | [`grpo.num_generations_per_prompt`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml) = paper `n_agent` |
| Prompts / step (paper / verl) | 512 | upstream `train_batch_size=512` |
| Prompts / step (ours) | **102** | [`grpo.num_prompts_per_step`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml); 5x fewer (rollout fits on 1x A100) |
| Trajectories / step (paper) | 2560 (= 512 x 5) | derived |
| Trajectories / step (ours) | **510** (= 102 x 5) | matches `policy.train_global_batch_size` |
| Total trajectories (paper) | ~2.57M (= 2560 x 1005) | derived |
| Total trajectories (ours) | ~513k (= 510 x 1005 = 512,550) | derived |
| Total prompts seen (paper) | ~514k (= 512 x 1005 = 514,560) | derived |
| Total prompts seen (ours) | ~103k (= 102 x 1005 = 102,510) | derived |
| Epochs over corpus (paper) | ~3.03x (= 514,560 / 169,615) | derived |
| Epochs over corpus (ours) | **~0.604x** (= 102,510 / 169,615) | derived; well under 1 epoch |
| Gradient updates / step (paper) | 10 (verl `ppo_mini_batch_size=256` over 2560 traj) | upstream verl yaml |
| Gradient updates / step (ours) | 1 (gbs=510 == prompts x gen → one optimizer.step()) | NeMo-RL convention; cheap fix to close gap = `train_global_batch_size: 51`; see [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md) |
| Smoke step time | **~57 s / 20 traj** | [`SMOKE_RESULTS_2026-05-06.md` "Per-step wall-time"](SMOKE_RESULTS_2026-05-06.md#per-step-wall-time-smoke-shape-20-trajectoriesstep) (mean across 4 combos x 2 steps) |
| Real per-step (linear, ours) | ~24 min (= 25.5 x 57 s) | smoke x (510/20); 1x A100 |
| Real per-step (sub-linear, ours) | ~15 min | empirical heuristic, sequence-packing-off |
| Wall-clock 1x A100 SXM | **11 to 17 d** (264 to 408 h) | 1005 x {15, 24} min |
| Wall-clock 1x H100 SXM | 5 to 8.5 d (120 to 204 h) | smoke table; H100 ≈ 2x A100 bf16 |
| Wall-clock 2x A100 SXM | 6.5 to 9.5 d (156 to 228 h) | smoke table; 1.7x speedup once decolocated |
| \$/run 1x A100 (Vast median, 0.6-epoch budget) | **\$300 to \$490** at \$1.20/h (264 x \$1.20 = \$317; 408 x \$1.20 = \$490) | [`SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock"](SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config) |
| \$/run 1x H100 (recommended, 0.6-epoch budget) | \$240 to \$410 at \$2.00/h | same |
| \$/run 1x A100 (paper-equivalent 3-epoch) | ~\$1,600 to \$2,400 (~5× the 0.6-epoch numbers) | derived; infeasible on \$1000 thesis budget |
| Retriever index | **IVF4096-SQ8** (~16 GB on disk) | [`local_retriever/retriever_config.yaml`](../../local_retriever/retriever_config.yaml); flat IP times out under training rollout HTTP load |
| Retriever workers (training) | **8** | [`bootstrap.sh`](../../training/scripts/bootstrap.sh); each `flashrag.utils.get_retriever()` loads its own copy of the index |
| Host RAM | **≥150 GB** (8 workers x ~16 GB index ≈ 128 GB resident) | [`bootstrap.sh`](../../training/scripts/bootstrap.sh) sanity check; PHASE_2_RUNBOOK |
| `policy.sequence_packing.enabled` | **`false`** (required) | Qwen3.5 GatedDeltaNet kernel raises CUDA illegal memory access with packed sequences during `get_logprobs`; replacement is `dynamic_batching: true` |
| `policy.train_micro_batch_size` | **`2`** | with packing off; `4` OOMs on 1x A100 80GB |
| Validation status (first-pass) | **DISABLED** (`val_period: 0`, `val_at_start: false`) | [`grpo_qwen3.5_2b_*.yaml`](../../training/configs/) |
| Checkpointing status (first-pass) | **DISABLED** (`enabled: false`) | same |
| LR | `1e-6` | matches paper |
| Warmup | `LinearLR` over 286 iters (= 0.285 x 1005) | matches paper |
| KL coef (β) | `0.001` | matches paper |
| KL estimator | `k3` (NeMo-RL default ≡ verl `low_var_kl`) | byte-identical Schulman 2020 k3; see VERL_REFERENCE.md §2 |
| Clip ratio (ε) | `0.2` | matches paper |
| Reward function | pure EM (no shaping) | matches `qa_em.py` (paper-faithful), NOT `qa_em_format.py`; corrected in early-May 2026 |

---

## Code map (so you can answer "where does X live")

```
training/
├── configs/
│   ├── grpo_qwen3.5_2b_1xa100.yaml       ← canonical config (read this; everything inherits from it)
│   └── grpo_qwen3.5_2b_2xa100.yaml       ← 2x layout; only cluster + vLLM TP differ
├── nemo_rl/                              ← vendored upstream @ v0.6.0; DO NOT EDIT (overlay path is below)
│   └── .venv/                            ← uv-managed, materialized by bootstrap.sh / setup.sh
├── src/                                  ← Search-R1 OVERLAY: pure registration, never patches NeMo-RL
│   ├── chat_template/tools.py            ← OpenAI-style `search` tool schema (qwen_native arm)
│   ├── datasets/search_r1.py             ← SearchR1Dataset(RawDataset) → DATASET_REGISTRY["search_r1"]
│   ├── environments/parsers.py           ← pure-Python parse_query / format_docs_* (testable w/o torch)
│   ├── environments/search_r1_env.py     ← SearchR1Env + ray.remote(...) actor; HTTP to retriever
│   ├── processors/search_r1.py           ← arm dispatch (qwen_native vs paper); apply_chat_template
│   ├── prompts/search_r1_paper.txt       ← paper's instruction string (ablation arm)
│   ├── prompts/search_r1_qwen_native_*   ← system + user templates for the qwen_native arm
│   ├── rewards/search_r1.py              ← BYTE-IDENTICAL port of M1's EM scorer (15 parity tests)
│   └── registry.py                       ← single import-side-effect: populates all registries
├── scripts/
│   ├── bootstrap.sh                      ← idempotent: LFS pull, weights, venv, v2 venv, retriever
│   ├── prepare_dataset.py                ← upstream HF parquet → our reshaped LFS-committed parquet
│   ├── run_grpo.py                       ← thin overlay launcher: import registry, hand off to NeMo-RL
│   ├── run_grpo_1xa100.sh                ← bash wrapper (--variant, --seed, --arm, -- HYDRA_OVR…)
│   ├── run_grpo_2xa100.sh                ← same wrapper for 2x layout
│   └── extract_smoke_samples.py          ← post-run: write SMOKE_RESULTS markdown from logs/exp_*/*.jsonl
└── tests/                                ← pytest; reward parity, parser dispatch, env-step (mocked retriever)

training/fix/CHANGES.md                   ← MUST-READ when something doesn't work; documents every smoke-unblock fix
```

The overlay's wiring contract: at launch, [`run_grpo.py`](../../training/scripts/run_grpo.py) does `import training.src.registry` (populates `DATASET_REGISTRY`, `PROCESSOR_REGISTRY`, `ENV_REGISTRY`, `ACTOR_ENVIRONMENT_REGISTRY`), then calls `examples.run_grpo.main()` from the vendored NeMo-RL. After that, the loop sees `dataset_name: search_r1`, `processor: search_r1_processor`, `env_name: search_r1` as if built-in.

---

## Gotchas (read before changing anything)

1. **`policy.sequence_packing.enabled: false` is mandatory for Qwen3.5.** The model interleaves linear-attention (Mamba/GatedDeltaNet) and full-attention layers; the `torch_chunk_gated_delta_rule` kernel raises `CUDA error: an illegal memory access` during `policy.get_logprobs()` when sequences are packed. Use `dynamic_batching: true` (token-budget micro-batches) as the replacement. Surfaced in [`training/fix/CHANGES.md §5`](../../training/fix/CHANGES.md#5-sequence-packing-must-be-disabled-for-qwen35).
2. **`train_micro_batch_size: 2`, not `4`.** Without packing, micro=4 OOMs on 1x A100 80GB in `get_logprobs`. Always pass `policy.train_micro_batch_size=2` on the CLI (the YAML default is correct, but a smoke override might surface a bigger number; explicitly set it).
3. **IVF-SQ8 index, NOT flat IP.** Flat IP needs ~65 GB resident per worker and times out under training rollout HTTP load (>80% of `/batch_search` calls fail). [`local_retriever/retriever_config.yaml`](../../local_retriever/retriever_config.yaml) defaults to IVF-SQ8 (~16 GB); start the retriever with `--num_retriever 8`. M1 eval still uses flat IP for paper-fidelity.
4. **Each retriever worker loads its own copy.** `init_retriever()` calls `flashrag.utils.get_retriever()` in a loop. With 8 workers x 16 GB ≈ 128 GB host RAM resident; bootstrap.sh warns at <150 GB.
5. **Reward function is pure-EM (no shaping).** Search-R1's repo ships `qa_em.py` (paper-faithful) AND `qa_em_format.py` (6-tier shaped). Earlier project docs conflated them; corrected May 2026. Our [`training/src/rewards/search_r1.py`](../../training/src/rewards/search_r1.py) keeps the multi-tier scaffold but defaults all 3 shaping coefficients to `0.0`, collapsing to pure EM. M3 ablations can re-introduce shaping by passing non-zero coefficients explicitly.
6. **`policy.dtensor_cfg._v2: true` requires the `automodel` extra venv.** That extra includes `nv-grouped-gemm` whose `setup.py` calls `torch.cuda.init()` at install time. NeMo-RL builds extras lazily via Ray actor `_env_builder`, which has no GPU allocated → `RuntimeError: No CUDA GPUs are available`. Workaround: `bootstrap.sh` (step 3) downloads a pre-built tarball from HF Hub (`pantomiman/reason-over-search-v1-venvs`); fallback compiles from host shell (~25 min). See [`training/fix/CHANGES.md §4`](../../training/fix/CHANGES.md#4-v2-automodel-venv--downloaded-from-hf-hub-not-compiled-on-vast).
7. **`data.validation.arm` cannot be overridden when `data.validation: null`.** Hydra errors with `ConfigCompositionException`. The 1xa100 wrapper had the dead override removed in May 2026 ([`fix/CHANGES.md` §1](../../training/fix/CHANGES.md#1-trainingscriptsrun_grpo_1xa100sh)); the 2xa100 wrapper was synced 2026-05-06.
8. **Paper text says "500 steps" but verl yaml + published checkpoints are 1005.** We match 1005. Don't be confused if you see the smaller number quoted.
9. **`sequence_packing` and `dynamic_batching` are mutually exclusive at runtime.** One must be on. Default is `dynamic_batching: true`, `sequence_packing: false` (Qwen3.5 forced).
10. **Two arms, same dataset.** `qwen_native` (default) uses Qwen3.5's native `<tool_call>` template; `paper` uses Search-R1's `<search>` tags. The dataset is template-agnostic (the prep script strips upstream's pre-baked instructions). Switch via `--arm` on the wrapper. See [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md) for verbatim renders + diff.

---

## Active task pointers (2026-05-06)

- **Active recipe-ablation plan**: [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md). Stack candidate: E2H + S-GRPO + MC-GRPO; control: JustRL plain-GRPO. Budget: 2 to 3 runs.
- **Thesis story so far**: [`docs/report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) (two-page brief).
- **Phase-1 (Qwen3-0.6B on ALICE) findings the recipe choices are reacting to**: [`docs/report/RESULTS_m0_a.md`](../report/RESULTS_m0_a.md), [`docs/report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md). Most actionable lever surfaced: paper's partial-credit reward creates a 0.1 floor that masks the tool-use signal.
- **Re-enable validation + checkpointing**: [`VALIDATION.md §7`](VALIDATION.md#7-re-enabling-validation-planned-not-active). Single config flip, no code changes; do this before kicking off long ablations.

---

## Cross-doc style invariants

- **Numbers cite their source.** When you write a wall-clock or $/run figure, link to the table or paragraph that derives it (smoke step time, scaling factor, $/h rate, multiplication). PAPER_VS_OURS_TRAINING.md §7 and SMOKE_RESULTS_2026-05-06.md "Full-training wall-clock + cost" are the two canonical sources for compute math.
- **Paper-vs-ours framing is consistent**: the paper-side row uses `512` prompts/step (`n_agent=5` → 2560 trajectories); our row uses `102` prompts/step (`n_agent=5` → 510 trajectories). Don't conflate "trajectories" and "prompts"; that bug surfaced in May 2026 and the §7 table fix is the lesson.
- **Validation/checkpointing are documented as DISABLED first-pass everywhere they appear** (configs, README, NEMO_RL_KNOBS, VALIDATION). When you re-enable, keep all four files in sync.
- **No em-dashes** in any prose written for the user (project-wide style rule, see [`claude/CLAUDE.md`](../../claude/CLAUDE.md)).

---

## Quick-start commands (Vast.ai bootstrap → first step)

```bash
# 1. Clone + bootstrap (idempotent; ~10 min cold, <1 min warm)
cd /workspace
git clone https://github.com/<user>/reason_over_search.git
cd reason_over_search
bash training/scripts/bootstrap.sh

# 2. Fill W&B key (or run with WANDB_MODE=disabled)
$EDITOR training/.env

# 3. Smoke run (5 min once v2 venv exists)
bash training/scripts/run_grpo_1xa100.sh \
  --variant base --seed 42 --arm qwen_native \
  -- grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20 \
     policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true \
     policy.train_micro_batch_size=2

# 4. Full run (after smoke is clean)
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42 \
  -- policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true \
     policy.train_micro_batch_size=2
```

The Qwen3.5-required overrides (`sequence_packing=false`, `dynamic_batching=true`, `train_micro_batch_size=2`) are also the YAML defaults; passing them on the CLI is belt-and-braces. Check W&B for `train/reward_mean` climbing at step 100; if flat at ~0, abort and debug rather than burn 11 to 17 days of GPU time.
