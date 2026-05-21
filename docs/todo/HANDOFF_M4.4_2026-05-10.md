---
title: Handoff — M4.4 prompt search (new machine)
tags: [todo, handoff, m4, m4.4]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# Handoff: M4.4 prompt search on a fresh machine

You're picking this up on a new box. The repo state at handoff is HEAD = the commit landed alongside this file. Read this end-to-end before doing anything; the M4.2 baseline is locked and you do NOT need to re-run it.

## What M4.4 is

Additive prompt-search sub-phase under M4. The M4.2 baseline (hybrid `qwen35_minimal` mean EM **0.060**, base `qwen35_minimal_no_system` mean EM **0.010**, full Plan A 51,713 items / variant) and the M3-vs-M4 cross-family comparison (avg Δ **−0.042 EM** in favour of the smaller, older-family Qwen3-0.6B from M3) are **closed and locked** in [`docs/report/RESULTS_m4.md`](../report/RESULTS_m4.md). Don't re-run them.

The 1.7× cross-family gap implicates the prompt, not parameters: Qwen3.5 was post-trained on different tool-use shapes than Qwen3, and we have not yet tested the prompt shapes that Search-R1 / ReSearch / ReCall and our own M3 / M3.1 ablations flagged as load-bearing. M4.4 tests four such shapes against the M4.2 control at n=300/dataset.

If no candidate clears the acceptance bar, **ship M4.2 unchanged and move to M5**. This is the 4th prompt iteration on M4 (v1 → v2 → v3 → minimal → M4.4); the higher-leverage next step is training, not more prompt golf.

**Base variant stays at `qwen35_minimal_no_system` per the M4.3 lock — do not re-prompt-search the base. Base lacks the tool-use prior; M4.3 already established that.**

## The plan (verbatim source)

[`docs/milestone_4/MILESTONE_4.md` §M4.4](../milestone_4/MILESTONE_4.md) is the canonical plan. Read it. The Phase-1 candidates A–E are tabulated there with locus / tag scheme / `tools=[]` auto-inject / `enable_thinking` / decision-rule / example axes. The companion scratch file [`milestone_4/M4_PROMPTS_SCRATCH.md`](../milestone_4/M4_PROMPTS_SCRATCH.md) shows the M4.2 / M4.3 locked prompts both as template constants AND as post-`apply_chat_template` rendered output (so you can see the auto-injected `# Tools` + `<IMPORTANT>` block hybrid actually receives).

Acceptance bars:
- Phase 1: candidate replaces A only if mean EM ≥ A + **0.025** at n=300 (Wilson 95 % CI half-width at p=0.06, n=300 ≈ 2.7 pp).
- Phase 2: Phase-1 winner ablation needs +**0.04** over A to commit to Phase 3.
- Phase 3: full sweep at n=51,713 with the locked prompt; replace [`RESULTS_m4.md`](../report/RESULTS_m4.md) §4 hybrid row.

## What's already wired vs what you have to build

### Already in HEAD (don't redo)
- [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) — registry holds `qwen35_minimal` (= A control), `qwen35_minimal_no_system` (M4.3 base lock), and a new `qwen35_recall_port` (verbatim Agent-RL/ReCall `re_call_template_sys` adapted to nested-XML + `<answer>` wrap; **NOT in the 5-candidate Phase-1 plan — standing by as a Phase-2 backup if candidate C wins**).
- [`scripts/run_m4.sh`](../../scripts/run_m4.sh) — accepts `PROMPT_MODE` env var; the wrapper used by all M4 runs.
- [`scripts/orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh) — copy this as the structural template for `orchestrate_m4_4.sh` (skip-aware resumption, phase env vars, retriever pre-flight). Don't re-author from scratch.
- [`evaluation_qwen35/flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) — `_is_qwen35` branch wraps Qwen3.5 nested-XML; the qwen3 branch (M3.1 path) wraps `<search>` / `<result>`; the search_r1 branch (M1 path) wraps `<search>` / `<information>`. **Re-use these branches by mode** — don't add a fourth.

### What you have to build
1. **Phase-1 templates B / C / D / E** in [`templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py):
   - `B = qwen35_searchr1_pure` — verbatim port of [`SEARCH_R1_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L175-L186), empty system, NO `tools=[]` arg, `<search>` / `<information>` tags. Routes through the search_r1 pipeline branch.
   - `C = qwen35_p3_decide_no_ex` — verbatim port of [`P3_DECIDE_NO_EX_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L32-L42) into the system role; user message = `Question: {q}`; NO `tools=[]`; `<search>` / `<result>` wrap (qwen3 branch). **Drop the `\boxed{}` wrapper** — replace with plain `<answer>X</answer>` to match the M4 lock and what M5 training will emit.
   - `D = qwen35_minimal_decide` — A's user message + appended M3.1 decision sentences (the two from `P3_DECIDE_NO_EX_TEMPLATE` lines starting "Use the information…" and "After each search result, decide…"). Same routing as A.
   - `E = qwen35_minimal_nothink` — A unchanged structurally; the difference is `enable_thinking=False` plumbed per-mode through `apply_chat_template`. Closes `<think>\n\n</think>` and removes the in-think-hallucination failure path flagged in [`RESULTS_SMOKE_m4.md` §6.5](../report/RESULTS_SMOKE_m4.md).
2. **Pipeline plumbing for `enable_thinking` per-mode.** Currently `enable_thinking=True` is hardcoded for both Qwen3.5 variants ([MILESTONE_4.md "Variant dispatch"](../milestone_4/MILESTONE_4.md)). E needs it parameterised. Smallest change: add a per-mode override dict in `_is_qwen35` / templates registry.
3. **`scripts/orchestrate_m4_4.sh`** — sequential 5-config × 7-dataset run with skip-aware resumption (re-use the orchestrate_C_then_A.sh skeleton). `START_PHASE` env var so a crashed run resumes mid-sweep without re-running finished cells.
4. **n=100 sanity-check rendering** for each new template before launching the n=300 screen — print `tokenizer.apply_chat_template(messages, tools=…, add_generation_prompt=True, enable_thinking=…)` on Bamboogle item 0 and eyeball the result. Catches bugs cheap. Reference snippet: [`MILESTONE_4.md` §"Sanity check the rendered prompt"](../milestone_4/MILESTONE_4.md#L177).

Estimated wall: **3–4 h coding + ~2 h Phase-1 sweep** on the optimised stack.

## Box bring-up

You'll most likely be on a fresh Vast.ai box. Bootstrap path:

1. [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../setup/BOOTSTRAP_NEW_INSTANCE.md) — `docker pull pantomiman/reason-over-search-v1:v2` (NOT `:v1` — `:v1` rebuild is broken upstream; `:v2` ships transformers 5.7.0 needed for Qwen3.5 `model_type=qwen3_5` AutoConfig). The `:v2` bootstrap.sh idempotently upgrades transformers in the eval venv, so `:v1` boxes also self-heal but skip the friction by pulling `:v2` directly.
2. [`docs/setup/SETUP_INSTANCE.md`](../setup/SETUP_INSTANCE.md) — canonical GPU-instance guide (covers Vast.ai, Verda B300, RunPod-class hosts); `training/scripts/bootstrap.sh` provisions corpus + IVF-SQ8 index + e5-base-v2 encoder + Qwen3.5-0.8B hybrid + base for M4 eval in one command.
3. Disk budget per scenario is in SETUP_INSTANCE.md. M4.4 needs the same M4 footprint (corpus + index + e5 + Qwen3.5-0.8B × 2 variants).
4. Optimised stack settings (re-use these for M4.4): multi-block `<tool_response>` per chunk, per-chunk cap 120 tok, `generator_max_input_len=8192`, retriever IVF-SQ8 × 16 workers + `asyncio.to_thread`, `INFERENCE_MAX_WORKERS=128`. All of these are already defaults in [`scripts/run_m4.sh`](../../scripts/run_m4.sh) — don't change them.

## Files to read first (in order)

1. **This file** — you're reading it.
2. [`docs/milestone_4/MILESTONE_4.md` §M4.4](../milestone_4/MILESTONE_4.md) — the plan, candidate table, acceptance bars, risks.
3. [`docs/milestone_4/M4_PROMPTS_SCRATCH.md`](../milestone_4/M4_PROMPTS_SCRATCH.md) — the M4.2 / M4.3 locked prompts as template constants AND post-render. Reference for what the model currently sees vs. what each candidate changes.
4. [`docs/report/RESULTS_SMOKE_m4.md`](../report/RESULTS_SMOKE_m4.md) — the v1 → M4.2 → M4.3 iteration log. §6 hybrid M4.2 lock, §7 base M4.3 lock + per-variant asymmetry mechanism, §6.5 the `<think>` hallucination finding that motivates candidate E, §10 full-sweep numbers. Do NOT skip §6.5 — it's the empirical hook for E.
5. [`docs/report/RESULTS_m4.md`](../report/RESULTS_m4.md) — the locked M4.2 baseline + cross-family table you're trying to beat.
6. [`docs/training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) — what the Qwen3.5 chat template auto-injects when `tools=[…]` is passed. §7a shows the verbatim render. Reference for B/C (which drop `tools=[]`) and the in-/out-of-distribution argument.
7. [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) — registry + existing constants. Add B/C/D/E here.
8. [`scripts/orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh) — structural template for `orchestrate_m4_4.sh`.
9. [`docs/todo/TODO.md`](TODO.md) — evergreen pending list; M4.4 entry is the live task.
10. [`docs/log.md`](../log.md) — the 2026-05-10 entry has the M4.4 plan-landing record.

## Hard rules (don't violate)

- **Do not re-run the M4.2 baseline.** It is locked. The M4.4 candidates measure delta against it via the n=300 screen; the full sweep only runs in Phase 3 if Phase 1+2 produce a winner over the bar.
- **Do not prompt-search the base variant.** M4.3 locked `qwen35_minimal_no_system` for base. M4.4 sweeps hybrid only.
- **Do not expand Phase 1 beyond the 5 candidates.** If 5 isn't enough, the gap isn't closeable with prompting — that's an evaluative finding, not a "try more" signal. Plan call-out at [`MILESTONE_4.md` §M4.4 "Risks and stop conditions"](../milestone_4/MILESTONE_4.md).
- **Drop the `\boxed{}` wrapper everywhere.** All M4.4 templates must emit `<answer>X</answer>` plain (matches the M4 lock and the M5 training shape).
- **Single seed (`seed=1`) greedy decode for the screen.** Greedy is seed-invariant past `random_sample` subsampling. Don't add seed sweeps; they belong in the deferred backlog.
- **Update [`docs/log.md`](../log.md) as you go.** One bullet per decision / artifact. The repo convention (set 2026-05-10) is "every artifact added in a commit must have a corresponding log.md bullet in the same commit."

## Definition of done

A successor commit lands one of:

- **(success)** A new `qwen35_*` template is locked as the hybrid M4 default; full sweep at n=51,713 re-run with that lock; [`RESULTS_m4.md`](../report/RESULTS_m4.md) §4 hybrid row replaced + §5 cross-family delta refreshed; M5 unblocked with the new prompt as the byte-aligned eval / train shape.
- **(null result)** Phase 1 produces no candidate over the +0.025 bar; M4.2 stays locked; [`RESULTS_m4.md`](../report/RESULTS_m4.md) appended with an "M4.4 null result" §6 row tabulating the 5 candidate means at n=300; M5 proceeds on the M4.2 lock.

In either case: close M4.4 in [`MILESTONE_4.md`](../milestone_4/MILESTONE_4.md) "What's left" and the [`TODO.md`](TODO.md) M4 section, and log it in [`log.md`](../log.md).
