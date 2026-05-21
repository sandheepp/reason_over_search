---
title: MILESTONE 5 (and M5.1) — Qwen3.5-0.8B GRPO training on NeMo-RL, ReSearch-paper recipe
tags: [milestone, training, qwen3.5, m5, m5.1, nemo-rl, research-paper]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# Milestone 5: Train Qwen3.5-0.8B with GRPO on the ReSearch-paper recipe (NeMo-RL)

> **Status (2026-05-12)**: This doc is the **original 2026-05-09 planning narrative** and is kept frozen for reference. For current execution state, see:
> - [`../report/RESULTS_m5.md`](../report/RESULTS_m5.md) — status banner with 3 losses + final-results placeholder + §4.2 reasoning-trace examples.
> - [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) — full iteration log + crash/misdiagnosis/deletion postmortems (§7 / §7.8 / §7.8.1).
> - [`../todo/TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md) — current-day catch-up for fresh sessions.
> - Surviving rollout corpus: [`logs/exp_011_a2_archive.tar.gz`](../../logs/exp_011_a2_archive.tar.gz) (a2 steps 1-15, 22 MB).

## Context

[M4](../milestone_4/MILESTONE_4.md) closed (status: hybrid + base baselines locked, untrained floor at hybrid mean EM 0.057 / n=100/dataset on the M4.2 minimal mode; full Plan-A sweep pending). The M4 eval pipeline ([`evaluation_qwen35/`](../../evaluation_qwen35/)) is the **fixed eval target** for any M5 GRPO checkpoint we produce: same model family (Qwen3.5-0.8B), same prompt mode (`qwen35_native` for hybrid, `qwen35_minimal_no_system` for base), same retriever (IVF-SQ8 × 8 workers), same per-mode budgets (`max_search_turns=5`, `step_limit=8192`, `max_obs_length=256`, `retrieval_topk=5`, `generator_max_input_len=4096`).

M5 stands up the **training-side counterpart**: GRPO-train a Qwen3.5-0.8B checkpoint on the [ReSearch paper](https://arxiv.org/abs/2503.19470) recipe using the existing [`training/`](../../training/) NeMo-RL scaffold (M2 work, smoke-tested on Qwen3.5-2B in [`SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)). The ReSearch paper's reference implementation is verl-based; verl does not support Qwen3.5 (CLAUDE.md gotcha), so M5 ports the recipe knob-by-knob into NeMo-RL rather than running verl directly.

**Sources of truth** (in this order):
1. ReSearch paper, [arXiv:2503.19470](https://arxiv.org/abs/2503.19470) — algorithm, reward, dataset, hyperparameters.
2. ReSearch official codebase ([`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch)) — concrete config values, retriever HTTP contract, prompt strings, rollout shape.
3. User's GitHub `research` repo (Phase-1 Qwen3-0.6B verl runs) — transferable retriever / rollout / GRPO knob choices already validated against this codebase.
4. [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) and [`docs/training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md) — the paper-to-NeMo-RL mapping work already done for M2.

## Goal

Two phases (M5 and M5.1), both targeting **Qwen3.5-0.8B** on **1× A100-80GB**:

| Phase | Goal | Deliverables |
|---|---|---|
| **M5** | Stand up the NeMo-RL training pipeline for Qwen3.5-0.8B with rollout / prompt / tag config **byte-identical to the M4 eval pipeline** (so the trained checkpoint is directly evaluable without re-aligning). Smoke for per-step time and validate end-to-end on a tiny shape. | `training_m5_1/` scaffold copied + adapted from `training/`; one ≤50-step smoke run; [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) (full audit of M5 deltas vs M2 + M4); [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) (per-step time, generation-length distribution, reward-mean trajectory). |
| **M5.1** | Align all remaining knobs (training data, GRPO hyperparameters, KL, group size, response length, advantage normalisation) to the ReSearch paper recipe. Two intentional divergences from the paper: **no format reward**, **no `\boxed{}` answer wrapper** — final reward is **F1 only**, extracted from `<answer>…</answer>`. Smoke again to estimate full-run wall-clock; reverify against the paper config; document. | `training_m5_1/configs/m5_1_research_paper.yaml`; updated [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) (M5.1 section); updated [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) (M5.1 smoke + projected full-run wall-clock); [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md) (clause-by-clause mapping of every paper setting to our YAML, with the two divergences flagged in red). |

After M5.1, running `training_m5_1/scripts/run.sh` will produce a training run that is **identical to the ReSearch paper recipe** modulo:
- Model: Qwen3.5-0.8B (paper: Qwen2.5-7B / 32B-Instruct / Llama3.1-8B variants).
- Reward: F1-only on `<answer>…</answer>` content (paper: format + EM partial credit).
- No `\boxed{}` answer wrapper (paper: yes).

## Why NeMo-RL (and not verl)

Verl's upstream model registry does not support Qwen3.5 (M2 finding; see CLAUDE.md "Gotchas"). Two options were on the table:

1. **Patch verl ourselves** — substantial; the registry coupling reaches into FSDP wrap policies, attention masking, and the rollout-engine adapter. Patching once for Qwen3.5 leaves us holding the patch through every verl bump.
2. **Port the recipe to NeMo-RL** — NeMo-RL @ v0.6.0 already supports Qwen3.5 natively (smoke-tested on 2B in [`SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)). The recipe-port surface is YAML + a small overlay (reward / parser / dataset adapter); cleaner and reuses the M2 work.

We pick (2). The transferable knobs from the paper / official verl scripts are already catalogued in [`VERL_REFERENCE.md`](../training/VERL_REFERENCE.md); M5 extends that mapping to all the ReSearch-specific settings.

## Reward function (M5.1)

Final-answer F1 only; extracted from the **first** `<answer>…</answer>` block in the rollout response.

```text
reward(rollout) = f1(extract_answer(rollout), gold_answer)
                where f1 is token-level F1 (Search-R1 / SQuAD style; lowercase + whitespace + punctuation normalisation; no stop-word strip)
```

**No format reward** (no penalty for missing `<search>` / `<tool_call>` / `<answer>` tags). Paper has `+0.1` partial credit if EM>0 + format-valid; we drop both.

**No `\boxed{}` wrapper.** Paper's prompt produces `<answer>The final answer is \[ \boxed{X} \]</answer>`; M4 already moved to plain `<answer>X</answer>` and M5 keeps that. The eval-side scorer (`evaluation_qwen35/flashrag/search_r1/answer_utils.py:extract_solution`) already accepts both shapes via optional `\boxed{}` matching, so eval doesn't need to change.

**Why F1 not EM**: F1 is more discriminative than EM at the small-model regime (rollouts that get the right entity but with extra words score >0 instead of being lumped with full failures), and it's what ReSearch / Search-R1 / R1-Searcher all report alongside EM. EM-only would push too much density to the zero-reward floor at 0.8B.

**Implementation site**: `training_m5_1/overlay/reward.py` (overrides `training/src/reward.py`; the only file that needs to differ from M2's reward in the overlay).

## Prompt + tag scheme (carry from M4)

The training-side rollout uses the **same chat-template invocation as M4 eval**:

| Concern | Value | Source |
|---|---|---|
| System message | `QWEN35_NATIVE_TEMPLATE` (3-line role + 3-step protocol) for hybrid; no system block for base | [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) |
| User message | `Question: {question}` | M4.1 |
| Tools schema | `tools=[QWEN35_SEARCH_TOOL]` passed to `apply_chat_template` (auto-injects `# Tools` block + format example + `<IMPORTANT>` reminder) | M4.1 |
| Action format | canonical Qwen3.5 nested-XML `<tool_call><function=search><parameter=query>X</parameter></function></tool_call>` | M4.1 |
| Action stop | `[</tool_call>, </answer>, <\|im_end\|>, <\|endoftext\|>]` | M4 |
| Result wrap | `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n{X}\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` | M4 (turn-bounded) |
| Final-answer wrap | `<answer>X</answer>` | M4 (no `\boxed{}`) |
| `enable_thinking` | True | M4.1 |
| `max_search_turns` | 5 | M4 |
| `max_obs_length` (per-chunk) | 120 tokens (M4 v3) | M4 v3 |
| `retrieval_topk` | 5 | M4 |
| `generator_max_input_len` | 4096 (paper-equivalent) | M4 |

This is a hard alignment requirement: any drift between training rollout and eval rollout shows up as a train/eval mismatch and silently degrades the eval score. The M3 14-fix audit ([`CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3) is the precedent — those 14 fixes were entirely about closing this gap for the Qwen3-0.6B family. M5 inherits the M4 settings, so the audit is short by construction.

## Dataset (M5.1)

**MuSiQue only** (training). Paper reports MuSiQue, HotpotQA, 2WikiMultiHopQA, and Bamboogle as multi-hop benchmarks; their training mix uses HotpotQA + 2Wiki. We pick **MuSiQue alone** for M5.1 because:

1. It's the hardest of the four (mean hops 2-4); training on the hard end keeps the curriculum honest if E2H is added later.
2. M1 results (Qwen2.5-3B GRPO instruct) put MuSiQue at EM 0.124 — the largest single-dataset headroom in the paper Table 3, so improvements are detectable.
3. Single-dataset training is the simplest reproducible recipe; cross-dataset training is a controlled additive change for a later sub-milestone (M5.2 candidate).

Eval continues to run on all 7 paper benchmarks via M4's pipeline.

Train file path: `data/musique/train.jsonl` (existing; same file the M1 / M4 eval pipelines read for the dev/test splits, train split converted via `training/scripts/prep_musique_train.py` to be added).

## Folder layout — `training_m5_1/`, `training_m5_2/` …

Each experiment is a **fully self-contained** copy of the [`training/`](../../training/) scaffold at the repo root, named by experiment ordinal. This was an explicit design choice (2026-05-09) over the alternatives (single shared `training/` + per-experiment YAML; nested `training/m5/exp_*/`) because:

1. **Hard isolation while one experiment full-trains.** A long M5.1 run can take days on 1× A100; we want exp 2 to be free to mutate any code path (parser, reward, env, dataset, NeMo-RL version) without touching the active run's working tree.
2. **No build / venv / wheel-cache contention** between experiments running concurrently on different boxes.
3. **Per-experiment W&B project** falls out naturally — each `training_m5_<N>/` ships its own `configs/*.yaml` with `logger.wandb.project` set to `reason_over_search_m5_<N>`.

Layout:

```text
reason_over_search/
  training/                  # M2 / Phase-2 reference scaffold (frozen; do not edit for M5)
  training_m5_1/             # M5 + M5.1 — ReSearch paper recipe (this milestone's experiment 1)
    nemo_rl/                 # vendored NeMo-RL @ v0.6.0 (copy or symlink from training/)
    src/                     # overlay base (copy from training/src/, drift as needed)
      reward.py              # F1-only scorer (the M5.1 divergence from M2)
      parser.py              # `<answer>X</answer>` extractor + qwen3.5 `<tool_call>` parser
      retrieval_env.py       # turn-bounded `<tool_response>` wrap (from M4)
      dataset_adapter.py     # MuSiQue → NeMo-RL row format
      processor.py
      registry.py
    configs/
      m5_smoke.yaml          # M5: tiny shape, ~50 steps, validates end-to-end
      m5_1_research_paper.yaml  # M5.1: full ReSearch paper recipe (Qwen3.5-0.8B / MuSiQue / F1-only)
    scripts/
      run.sh                 # CONFIG=configs/m5_1_research_paper.yaml bash scripts/run.sh
      prep_musique.py        # train-split conversion (jsonl → NeMo-RL format)
      smoke.sh               # 50-step smoke; emits per-step time histogram
    setup.sh                 # mirror of training/setup.sh
    README.md                # what this experiment is, how to launch
  training_m5_2/             # M5.2 (planned) — variant; populated when exp 1 starts full training
  …
```

Subsequent experiments (`training_m5_2/`, `training_m5_3/`, …) follow the same template. Each gets its own `MILESTONE_5.<N>.md` sub-doc once it's defined.

**What's shared, what's copied**:
- `nemo_rl/` (vendored at `v0.6.0`) — can be a **copy** (safest; bumping the vendor in one experiment doesn't ripple) or a **symlink to `training/nemo_rl/`** (saves ~2 GB disk; safe only if no experiment edits the vendor). Default: copy. Symlink is opt-in via a comment in the experiment's `setup.sh`.
- `src/` — always **copied** at experiment-creation time. Diverges freely from `training/src/`. Initial M5.1 overlay differs from `training/src/` only in `reward.py` (F1-only).
- `configs/`, `scripts/`, `setup.sh`, `README.md` — always copied + edited.
- `data/`, `corpus/`, `indexes/`, `models/`, `eval/` — single shared copy at the repo root (existing convention from M3 / M4); experiments **read-only** from those paths.

## Action format and answer extraction

Action format is the canonical Qwen3.5 nested-XML form (M4.1):

```text
<tool_call>
<function=search>
<parameter=query>QUERY</parameter>
</function>
</tool_call>
```

Answer format is plain `<answer>X</answer>` (no `\boxed{}`). Reward parser uses the **first** `<answer>...</answer>` block; if multiple exist (model emitted a draft answer mid-rollout and then corrected), only the first counts — this matches the M4 `extract_solution` rule and avoids reward leaks where the model continues after the answer.

Parser to import / re-export from M4: `evaluation_qwen35/flashrag/search_r1/parser.py:extract_tool_call_query` (action) and `evaluation_qwen35/flashrag/search_r1/answer_utils.py:extract_solution` (answer). The training-side `training_m5_1/src/parser.py` is a thin re-export so eval and training use the **same code path**, not the same logic re-implemented.

## Per-step budget and target

[M2 smoke results](../training/SMOKE_RESULTS_2026-05-06.md) on Qwen3.5-2B at the smoke shape: **~57 s/step** at 20 trajectories/step (1× A100-80GB). Qwen3.5-0.8B at the same shape should be **2-3× faster** per step (forward+backward and rollout-generate both scale with model size). Working target for M5 smoke: **≤25 s/step** at 20 traj/step on 1× A100-80GB. If M5 lands above 35 s/step, that's a flag to investigate before launching M5.1 full-shape (rollout-engine config, vLLM kv-cache size, sequence-packing settings).

[`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) target for the wider recipe-search work is **≤10 h per ablation run**. At 25 s/step that's 1440 steps; at 35 s/step it's 1030 steps. ReSearch paper schedule (per [`PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) §5) is ~500 steps at the published-checkpoint shape — fits comfortably under either bound, before any of the systems wins from [`RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md).

## Run sequence

### M5

1. **Scaffold copy**: `cp -r training/ training_m5_1/`. Strip M2-specific bits not used in M5: any 2B-specific configs go to `training_m5_1/configs/_archive_m2/`.
2. **Overlay deltas** (relative to `training/src/`):
   - `reward.py` → F1-only on `<answer>…</answer>`.
   - `parser.py` → re-exports from `evaluation_qwen35` so train and eval use the same parsers.
   - `retrieval_env.py` → confirm turn-bounded `<tool_response>` wrap matches M4 byte-for-byte.
   - `dataset_adapter.py` → MuSiQue train split.
3. **Smoke config** (`configs/m5_smoke.yaml`): 20 traj/step, 50 steps total, MuSiQue 200-row subsample, no validation, 8× vLLM dp.
4. **Smoke run**: `bash training_m5_1/scripts/smoke.sh` on 1× A100-80GB. Capture per-step time histogram, reward-mean trajectory, generation-length distribution, gradient-norm trajectory, clip-ratio mean.
5. **Verify**: rendered prompt on 1 example matches the M4 prompt byte-for-byte (`tokenizer.apply_chat_template(...)` output identical between train rollout and `evaluation_qwen35` rollout).
6. **Document**: fill in [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) §1 + §2 + §3 (M5 portion); fill in [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) §M5.

### M5.1

1. **Read paper + codebase**: pull every concrete number from the ReSearch paper (algorithm hyperparams, reward shaping that we'll override, dataset, batch shape, schedule) and from [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) configs (group size, KL coefficient, response-length cap, advantage normalisation, vLLM rollout knobs).
2. **Build the mapping table**: every paper setting → our YAML key in `configs/m5_1_research_paper.yaml`, with cite + a one-line note where we diverge. This becomes [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md).
3. **Two intentional divergences** (called out in the mapping table in red):
   - Reward: paper uses format + EM partial credit; we use F1-only.
   - Answer wrap: paper produces `\boxed{…}` inside `<answer>`; we produce plain `<answer>X</answer>`.
4. **Smoke at full shape, partial steps**: ~100 steps at the M5.1 batch shape on MuSiQue, on 1× A100-80GB. Goal: tight per-step time estimate at the production shape so the wall-clock projection is real, not extrapolated from a 20-traj smoke.
5. **Wall-clock projection**: per-step time × paper schedule (steps) → total hours → cost at \$1.20/h. Compare to the [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) ≤10 h target; if over, identify the cheapest knob to cut (usually `max_response_length` or rollout group size).
6. **Re-verify**: re-run the rendered-prompt byte check, re-confirm reward path on 5 hand-picked rollouts (correct, partial-overlap, wrong, empty, format-broken — F1 should be 1.0 / 0<F1<1 / 0 / 0 / 0).
7. **Document**: fill in [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) M5.1 sections; update [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) with the M5.1 smoke + projection; commit `training_m5_1/configs/m5_1_research_paper.yaml`.
8. **Wrap M5.1**: bash `training_m5_1/scripts/run.sh` is now the canonical "ReSearch paper recipe on Qwen3.5-0.8B (with the two divergences)" entry point. The launch is reproducible from a clean Vast box via `bootstrap.sh` + `cd training_m5_1 && bash setup.sh && bash scripts/run.sh`.

## Parallel experiments (M5.2 onwards)

The motivation for the per-experiment folder is to let a second experiment be configured and smoke-tested while M5.1 full-trains. The intended flow:

1. Once M5.1's full training is launched (background, 1× A100 on Vast), `cp -r training_m5_1/ training_m5_2/`.
2. Edit only the files that change for the new experiment (typically: `configs/m5_2_<variant>.yaml`, possibly `src/reward.py` if reward-ablating, possibly `src/dataset_adapter.py` if dataset-ablating).
3. Smoke `training_m5_2/scripts/smoke.sh` on a separate box (or queue-serially after M5.1 wraps).
4. New milestone sub-doc: `MILESTONE_5.2.md` describes the variant + the rationale; [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) gets a new section.

Candidates for M5.2 (not committed yet, just possibilities so the layout is exercised):
- Reward ablation: F1 → EM, or F1 + small format penalty.
- Dataset: HotpotQA train → MuSiQue (the paper-default mix).
- Algorithm: GRPO → MC-GRPO (item #3 from [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) ablation list).

## What's left

| # | Task | Owner | Blocked on |
|---|---|---|---|
| 1 | Scaffold `training_m5_1/` (copy + overlay) | — | nothing |
| 2 | Smoke config + smoke run on 1× A100-80GB | — | (1) |
| 3 | [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) §1-§3 (M5 part) | — | (2) |
| 4 | [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) M5 section | — | (2) |
| 5 | Read [ReSearch paper](https://arxiv.org/abs/2503.19470) + [Agent-RL/ReSearch](https://github.com/Agent-RL/ReSearch) configs, build paper-vs-ours mapping table | — | nothing (parallel to 1-4) |
| 6 | `configs/m5_1_research_paper.yaml` + [`PAPER_VS_OURS_M5.md`](PAPER_VS_OURS_M5.md) | — | (5) |
| 7 | M5.1 smoke + wall-clock projection | — | (6) |
| 8 | Re-verify (rendered prompt + 5-rollout reward check) | — | (7) |
| 9 | [`CODE_SETUP_m5.md`](../report/CODE_SETUP_m5.md) M5.1 sections + [`RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) M5.1 update | — | (7), (8) |
| 10 | Launch M5.1 full training | — | (9) |
| 11 | Define + scaffold M5.2 (parallel experiment, while M5.1 trains) | — | (10) |

## Pointers

- M2 NeMo-RL training scaffold (the source of truth for what M5 is copying / extending): [`../../training/`](../../training/), narrative at [`../milestone_2/MILESTONE_2.md`](../milestone_2/MILESTONE_2.md), Phase-2 runbook at [`../milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md).
- M2 paper-to-NeMo-RL mapping (the foundation M5 extends): [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md), [`../training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md), [`../training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md), [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md).
- M2 smoke (the wall-clock anchor M5 will compare against): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md).
- M4 eval pipeline (the M5 rollout shape is byte-aligned to this): [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md), [`../report/CODE_SETUP_m4.md`](../report/CODE_SETUP_m4.md), [`../report/RESULTS_m4.md`](../report/RESULTS_m4.md), [`../report/RESULTS_SMOKE_m4.md`](../report/RESULTS_SMOKE_m4.md).
- M3 14-fix audit (the precedent for closing train/eval rollout drift): [`../report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3.
- ReSearch paper (algorithm + recipe source of truth): [arXiv:2503.19470](https://arxiv.org/abs/2503.19470), notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md).
- ReSearch official codebase (concrete configs): [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch).
- Active recipe-ablation plan (the wider story M5 sits inside): [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md).
