---
title: MILESTONE 1.1 QWEN BASELINES
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-02
---

# Milestone 1.1 — Untrained Qwen3.5-2B baselines

> **Owner:** Jose
> **Why now:** Milestone 2 trains Qwen3.5-2B (both base and hybrid variants) with GRPO. To call training a "success" we need a baseline number for the *untrained* model on the same benchmarks — otherwise "improvement from training" is undefined. This sub-milestone fills that gap by running the M1 eval pipeline against the untrained Qwen3.5-2B checkpoints. Slots between M1 (eval-side reproduction of Search-R1's published Qwen2.5-3B checkpoints) and M2 Phase 2 (running our own training).
> **Prereq for:** Milestone 2 Phase 2 first-run gate (see [`PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md)).

## Goal

For each of `Qwen/Qwen3.5-2B-Base` and `Qwen/Qwen3.5-2B`, run the existing [`evaluation_search_r1/`](../../evaluation_search_r1/) pipeline against the same 7 benchmarks M1 used (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle), produce a results table, and commit it as the **untrained baseline** so M2 trained checkpoints can be compared against a meaningful floor.

These numbers will likely be **low** — Qwen3.5-2B-Base has no instruction-following post-training, and even the hybrid variant has only generic tool-use post-training, not Search-R1-specific RL. That's expected. The point is to establish the floor.

## What's different from M1's eval

M1 evaluated the *published* Search-R1 GRPO checkpoints (`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-{,it-}em-grpo`) using the **paper's chat template** (`<search>` / `<information>` / `<answer>`). Those checkpoints were trained on those exact tags.

For M1.1 we're evaluating untrained Qwen3.5-2B. Two things change:

1. **Different model family** — Qwen3.5-2B vs Qwen2.5-3B (smaller, newer, different post-training).
2. **Different chat template (for hybrid only)** — we use the M2-baseline `qwen_native` protocol (system prompt with the `search` tool registered, `<tool_call>` / `<tool_response>` for retrieval). For base, we keep it flat (no chat template, see below).

The benchmarks, retriever, FAISS index, and EM scorer stay identical to M1.

## Prompt strategy per variant

> **Updated 2026-05-02** — The training prompts were split into a brief system prompt + a user-message protocol template (matching the paper arm's structure). For eval-vs-train fairness, M1.1 should use the same split. Both files live under [`training/src/prompts/`](../../training/src/prompts/).

### Qwen3.5-2B-Base (no chat template)

The base model has no instruction-following post-training. Concatenate the user-message template (with the question filled in) — the system role intro is uninformative without a chat template.

```python
with open("training/src/prompts/search_r1_qwen_native_user.txt") as f:
    user_template = f.read()
prompt = user_template.format(question)
# → "You must conduct reasoning inside <think> and </think>... Question: {question}"
```

Feed straight to the model with `apply_chat=False`.

### Qwen3.5-2B (hybrid; default soft-switch reasoning)

The hybrid variant has Qwen3.5's chat template AND post-training on `<tool_call>` / `<tool_response>`. Use full chat-template rendering — same code as the training processor:

```python
with open("training/src/prompts/search_r1_qwen_native_system.txt") as f:
    system_prompt = f.read().strip()
with open("training/src/prompts/search_r1_qwen_native_user.txt") as f:
    user_template = f.read()

tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(question)},
    ],
    tools=[SEARCH_TOOL],         # from training/src/chat_template/tools.py
    tokenize=False,
    add_generation_prompt=True,
    add_special_tokens=False,
    enable_thinking=True,
)
```

Qwen3.5's chat template will auto-render the `search` tool's schema into the system area. See [`docs/training/CHAT_TEMPLATE.md §7a`](../training/CHAT_TEMPLATE.md#7a-qwen_native-arm) for what the model actually sees, copy-pasteable.

## The verbatim prompt files

Source of truth — do not duplicate; load from these paths:
- [`training/src/prompts/search_r1_qwen_native_system.txt`](../../training/src/prompts/search_r1_qwen_native_system.txt) — one-line role intro
- [`training/src/prompts/search_r1_qwen_native_user.txt`](../../training/src/prompts/search_r1_qwen_native_user.txt) — protocol + `Question: {}` placeholder

## Pre-flight — what needs to be in place

The current eval pipeline (M1) parses `<search>...</search>` and injects `<information>...</information>`. For the **hybrid** variant under qwen_native it'll emit `<tool_call><function=search><parameter=query>...</parameter></function></tool_call>` and expect `<tool_response>...</tool_response>` back. **The pipeline needs a qwen_native arm.**

The training-side env already has the parsers + formatters:

- [`training/src/environments/parsers.py:parse_query`](../../training/src/environments/parsers.py) — regex extracts the `<parameter=query>` value from `<tool_call>...</tool_call>`.
- [`training/src/environments/parsers.py:format_docs_qwen_native`](../../training/src/environments/parsers.py) — wraps retrieved docs in the Qwen3.5 chat-template-correct `<|im_end|>\n<|im_start|>user\n<tool_response>...\n</tool_response><|im_end|>\n<|im_start|>assistant\n` markers.

These are pure-Python and have no torch/ray deps — they can be imported directly into the eval pipeline.

### Code changes (eval pipeline side)

Roughly two files to touch in [`evaluation_search_r1/flashrag/search_r1/`](../../evaluation_search_r1/flashrag/search_r1/):

1. **`templates.py`** — add a function (or class) that builds the qwen_native prompt. For base: flat-text concat. For hybrid: invoke `tokenizer.apply_chat_template(..., tools=[SEARCH_TOOL])`. Re-export `SEARCH_TOOL` from the training overlay or re-define it here.
2. **`parser.py`** — extend `extract_search_tag_query` (or add a sibling `extract_tool_call_query`) that regex-matches the qwen_native `<tool_call>...<parameter=query>...` block. Mirror [`training/src/environments/parsers.py:_RE_QWEN_QUERY`](../../training/src/environments/parsers.py).

Then thread an `arm: {paper, qwen_native}` config option through `flashrag/pipeline/active_pipeline.py:SearchR1Pipeline` so its `run_item` method dispatches the right parser, observation-formatter, and stop strings (`</tool_call>` / `</answer>` for qwen_native, `</search>` / `</answer>` for paper).

The training-side env already does exactly this dispatch for GRPO rollouts — port the same logic.

### Sanity check before launching the full sweep

Run a 10-question Bamboogle subset on `Qwen3.5-2B-Base` first. Inspect 3 random rollouts in the JSONL output:
- Did the model emit `<answer>...</answer>` at all?
- Did it correctly invoke search (hybrid) or attempt to (base)?
- Are observation tokens being injected back into the prompt cleanly?

If something looks malformed, fix it before spending hours on the full sweep.

## Run instructions

Assumes you're on a Vast.ai instance running `pantomiman/reason-over-search-v1` (same image as M1; the training-side venv at `training/nemo_rl/.venv/` is *not* needed for this — eval uses `/venv/evaluation_search_r1`).

### 1. Bring services up

```bash
# Retriever (CPU FAISS-flat, ~65 GB host RAM) — see local_retriever/README.md
tmux new -s retriever
/venv/retriever/bin/python local_retriever/retriever_serving.py \
    --config local_retriever/retriever_config.yaml \
    --num_retriever 1 --port 3005
# Ctrl-b d
curl -sS http://127.0.0.1:3005/health   # → "healthy"

# SGLang serving the model under test (one variant at a time on a single GPU)
scripts/manage_sglang.sh switch qwen3.5-2b-base \
    --model-path Qwen/Qwen3.5-2B-Base
# Verify on port 3000:
curl -sS http://127.0.0.1:3000/get_model_info | grep model_path
```

(`manage_sglang.sh switch` may need to learn the new variant name — see [`scripts/`](../../scripts/) for how `base` and `instruct` are wired today; add `qwen3.5-2b-base` and `qwen3.5-2b-hybrid` cases.)

### 2. Sweep one variant × 7 datasets × 5 seeds

For Qwen3.5-2B-Base:

```bash
for ds in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
    for seed in 1 2 3 4 5; do
        scripts/run_one.sh qwen3.5-2b-base $ds $seed --arm qwen_native \
            > logs/qwen3.5-2b-base_${ds}_seed${seed}.log 2>&1
    done
done
```

(The `--arm qwen_native` flag is one of the things the eval-side code changes need to add.)

Repeat for `qwen3.5-2b-hybrid` after `manage_sglang.sh switch qwen3.5-2b-hybrid --model-path Qwen/Qwen3.5-2B`.

### 3. Aggregate

```bash
python scripts/aggregate.py --pattern 'evaluation_search_r1/results/*/qwen3.5-2b-*-seed*' \
    > docs/milestone_1/RESULTS_QWEN3_BASELINE.md
```

(Mirror the format of [`docs/report/RESULTS_m1.md`](../report/RESULTS_m1.md): per-benchmark mean ± std across seeds, side-by-side base vs hybrid.)

## Compute estimate

From M1: a single Bamboogle/instruct run on a 4090 takes ~6 min. Scaling rough rules:

- 7 datasets × 5 seeds × 2 variants = **70 runs**.
- Per-dataset wall-clock varies (Bamboogle 125 q in 6 min; NQ 3.6k q in ~2 h on 4090).
- On 4090 single-GPU sequential: **~30–50 hours** of wall-clock.
- On Vast.ai 4× 4090 fleet (per [`docs/setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md)): **~12–20 hours**, **~$30–60 total**.

Same fleet pattern Jose used for M1 Plan A. Recycle.

## Where to put the results

1. **Raw run outputs**: `evaluation_search_r1/results/<benchmark>/<benchmark>_..._qwen3.5-2b-{base,hybrid}_seed{1..5}/metric_score.txt` (the existing path convention).
2. **Aggregated table**: new file `docs/milestone_1/RESULTS_QWEN3_BASELINE.md` — same columns as `RESULTS_m1.md` (NQ / TriviaQA / PopQA / HotpotQA / 2Wiki / MuSiQue / Bamboogle / Avg), one row per variant × arm.
3. **Cost summary**: append actual wall-clock + $ to the bottom of this doc once the sweep completes.

## Decision criteria

This sub-milestone is **done** when:

- [ ] Eval pipeline gains a `qwen_native` arm (both base flat-text and hybrid chat-template paths) — code merged.
- [ ] All 70 runs (2 variants × 7 benchmarks × 5 seeds) complete with EM > 0 (sanity — non-zero output reaches the scorer).
- [ ] [`docs/milestone_1/RESULTS_QWEN3_BASELINE.md`](RESULTS_QWEN3_BASELINE.md) committed with per-benchmark mean ± std and an averages row.
- [ ] Format-validity (`</answer>` close-rate) and length-truncation surfaced per (variant, benchmark) in the same table — same metrics M1 reports.

## Why these numbers matter for M2

The Phase 2 first-run gate ([`PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md#smoke-eval-on-bamboogle)) compares the trained checkpoint's Bamboogle EM against the **untrained** Qwen3.5-2B-Base. If our 1.1 baseline shows base at, say, 4% Bamboogle EM, then the trained model needs to land north of ~10% for us to call training "working". Without 1.1, that gate has no threshold.

These same numbers also feed [`docs/training/PAPER_VS_OURS_TRAINING.md §7`](../training/PAPER_VS_OURS_TRAINING.md) as the "untrained ours" comparison row.

## Open questions for Jose

- Do we want to evaluate on temperature=0 (greedy, M1's choice — see [archive/TEMPERATURE_HYPOTHESIS_WRONG.md](../archive/TEMPERATURE_HYPOTHESIS_WRONG.md)) or temperature=1 (matching training rollouts)? Recommend greedy for parity with M1.
- Single-pass (one rollout per question) or multi-pass (5 generations averaged)? Recommend single greedy pass; matches M1.
- Do we also want a "no-search" baseline (model answers from parametric knowledge alone, no retriever)? Useful for showing "search adds X pp" — but adds another sweep. Skip unless cheap to bolt on.

## See also

- [`MILESTONE_1.md`](MILESTONE_1.md) — original M1 (Search-R1 published checkpoints, Qwen2.5-3B)
- [`CODE_SETUP_m1.md`](../report/CODE_SETUP_m1.md): single source of truth for the eval pipeline knobs (sampling, retrieval_topk, max_search_turns, etc.); re-use for 1.1
- [`docs/eval/REPRODUCIBILITY.md`](../eval/REPRODUCIBILITY.md) — the 10 paper-vs-ours divergences that were fixed in M1; same fixes apply here
- [`docs/training/CHAT_TEMPLATE.md §7`](../training/CHAT_TEMPLATE.md#7-rendered-examples--what-the-model-actually-sees) — exact rendered prompts for both arms (copy-pasteable)
- [`docs/setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md) — fleet config that Jose used for M1 Plan A; recycle
