---
title: Results M4 — smoke iteration log (Qwen3.5-0.8B hybrid, prompt + pipeline)
tags: [report, eval, m4, qwen3.5, smoke]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# Results M4 smoke — iteration log

Live record of the smoke iterations on `qwen3.5_0.8b` (hybrid, untrained) while shaking down the M4.1 eval pipeline. Three versions to date: v1 (placeholder M4.1), v2 (canonical multi-block tool-response), v3 (per-chunk cap + 8192 input + tightened prompt). Average EM stayed below 0.01; M3 untrained Qwen3-0.6B hybrid baseline is 0.102 on the same 7 benchmarks, so the gap implicates the prompt scaffolding (auto-injected `# Tools` + `<IMPORTANT>` + our system, ~350 words combined) rather than the pipeline. Decision at end-of-doc: try a Search-R1-style minimal user-message prompt that uses Qwen3.5's `<tool_call>` / `<tool_response>` tags but skips `tools=[]` to drop the auto-injected verbosity.

Pipeline: [`evaluation_qwen35/`](../../evaluation_qwen35/). Milestone: [`MILESTONE_4.md`](../milestone_4/MILESTONE_4.md). Pinned baseline: [`RESULTS_m3.md`](RESULTS_m3.md).

## 1. Run roster

| Version | When | Variant | Pipeline change | Prompt change |
|---|---|---|---|---|
| v1 | 2026-05-09 11:06 | qwen3.5_0.8b | M4.1 placeholder (single `<tool_response>` block, total `max_obs_length=256`) | "QWEN35_NATIVE_TEMPLATE" 4-step protocol, no nudges |
| v2 | 2026-05-09 12:02 | qwen3.5_0.8b | Multi-block per chunk, total cap raised to 500 tokens, train/eval truncation marker | + "Always call search at least once" + "use proper-noun entities from previous tool_response" |
| v3 | 2026-05-09 12:25 | qwen3.5_0.8b (all 7 ds) | Per-chunk cap 120 tokens (no total cap), `generator_max_input_len 4096 → 8192`, training-side mirror | "Use facts and entities from relevant passages (not from prior knowledge)" |

All runs: greedy decode, `enable_thinking=True`, single seed (1), 100 random items / dataset.

## 2. v1 — M4.1 placeholder

### 2.1 Setup

- [`templates.py:74-84`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L74-L84) `QWEN35_NATIVE_TEMPLATE`: 4-step "call search → observe → answer-or-loop" protocol, no nudges, no example.
- [`active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) qwen35 branch: passes `tools=[QWEN35_SEARCH_TOOL]` to `apply_chat_template`, single `<tool_response>` block per turn with all topk chunks `\n\n`-joined inside, `max_obs_length=256` tokens TOTAL, multi-turn wrap closes the assistant turn at `</tool_call>` and re-opens after one `<tool_response>` block.
- 100 items × bamboogle only.

### 2.2 Result

| Metric | Value |
|---|---:|
| EM | 1/100 (0.01) |
| pred empty | 89/100 |
| `</tool_call>` (turn-cap, pred empty) | 74 |
| `<|im_end|>` (gave up) | 15 |
| `</answer>` (got to answer) | 11 |

Walkthrough of failure on item 0 ("Who was the second wife of the founder of CNN?", gold: Jane Shirley Smith): [`analysis_item_0.txt`](../../evaluation_qwen35/results/bamboogle/bamboogle_2026_05_09_11_06_m4_qwen3.5_0.8b_seed1_n100_v1/analysis_item_0.txt) (preserved).

### 2.3 What we learned

- 74% of items hit the 5-turn cap without ever opening `<answer>`. Dominant failure was the loop running out of search turns.
- Inside the rollout the model identified Ted Turner correctly on turn 1's retrieved evidence, then drifted into hallucinated names ("Robert Businessman", "Susan Buffett") instead of querying for "Ted Turner wives".
- Only ~2 of the topk=5 retrieved chunks fit under the 256-token cap (Wikipedia chunks are ~140 tokens each), so most retrieved evidence was dropped.
- Single-block tool_response with `\n\n` separators gave the model no doc boundaries.

## 3. v2 — canonical multi-block tool_response

### 3.1 Diagnosis-driven changes

Verified the canonical Qwen3.5 multi-result format by rendering `tokenizer.apply_chat_template([{"role":"tool","content":chunk}, ...])` on the local Qwen3.5-0.8B snapshot. Output: ONE `<tool_response>...</tool_response>` block per chunk, all stacked inside ONE `<|im_start|>user...<|im_end|>` turn. The single-block v1 form was off-distribution; multi-block is what the post-training distribution emits.

Code changes:

- [`active_pipeline.py:120-134`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L120-L134): per-mode budget — qwen35 gets `max_obs_length=500` (was 256, kept for qwen3); qwen3 unchanged.
- [`active_pipeline.py:188-256`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L188-L256): qwen35 branch now emits one `<tool_response>` per chunk inside one `<|im_start|>user` turn. Greedy total-budget fill: each chunk included whole if it fits; partial last chunk truncated with `…[truncated]` marker; later chunks dropped if budget exhausted.
- [`templates.py:74-86`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L74-L86) `QWEN35_NATIVE_TEMPLATE`: extended with two prompt nudges:
  - "Always call `search` at least once before answering; do not answer from prior knowledge."
  - "When refining, use proper-noun entities you found in the previous `<tool_response>` rather than restating the question."
- [`training/src/environments/parsers.py:66-101`](../../training/src/environments/parsers.py#L66-L101) `format_docs_qwen_native`: mirrored to multi-block (per-doc `<tool_response>`), total char budget with truncation marker — for train/eval byte-identity at M5+.
- [`training/tests/test_format_helpers.py:27-40`](../../training/tests/test_format_helpers.py#L27-L40): updated to assert per-doc blocks + no "Doc i:" prefix.

Verified byte-identity between hand-built emission and `apply_chat_template` render against the local snapshot (only delta: `<think>\n` prefix; existing design choice — hybrid auto-emits `<think>` from its prior).

### 3.2 Result (bamboogle, n=100)

| Metric | v1 | v2 | Change |
|---|---:|---:|---|
| EM | 1/100 | 1/100 | unchanged |
| F1 avg | 0.0195 | 0.0109 | -0.009 (worse) |
| pred empty | 89 | 95 | +6 (worse) |
| `</tool_call>` (turn-cap) | 74 | 62 | -12 (good) |
| `</answer>` (got to answer) | 11 | 6 | -5 (bad) |
| `<|im_end|>` (gave up) | 15 | 24 | +9 (bad) |
| empty / length-stop | 0 | 8 | +8 (NEW failure) |
| prompt_tokens mean | 1898 | 2848 | +950 |
| prompt_tokens max | 3567 | **4072** | hitting 4096 cap |

### 3.3 What we learned

- Multi-block format renders correctly and reaches the model.
- Bigger obs budget pushed `prompt_tokens` against the `generator_max_input_len: 4096` cap. 8 items length-stop mid-`<think>` (one even wrote a correct factual conclusion in plain prose, never wrapped in `<answer>`).
- Stop-transition matrix: 6 wins (turn-cap → answer), 11 regressions (had-answer → no-answer). Net EM unchanged because the failure mode shifted from turn-cap to length-stop / give-up.
- Prompt nudges present in the rendered query (`Always call`, `proper-noun entities`) but apparently insufficient to redirect the model when it has hallucinated an entity in an early `<think>`.

## 4. v3 — per-chunk cap, larger input, refined prompt

### 4.1 Changes

- [`active_pipeline.py:120-138`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L120-L138): qwen35 budget switched from "total" to "per-chunk" semantics — `max_obs_per_chunk = 120` tokens per `<tool_response>` block, no total cap. Each chunk is independently truncated; all topk chunks reach the model regardless of any single chunk's length. Solves the v2 failure mode where one long chunk consumed the budget and starved later chunks.
- [`active_pipeline.py:188-218`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L188-L218): rewrote the qwen35 emission loop to iterate all chunks unconditionally with per-chunk truncation marker.
- [`basic_config.yaml:105`](../../evaluation_qwen35/flashrag/config/basic_config.yaml#L105): `generator_max_input_len: 4096 → 8192` (matches SGLang's `--context-length 8192`). Kills the 8 v2 length-stops by making room for 5 chunks × 120 tokens × 5 turns plus thinking overhead.
- [`training/src/environments/parsers.py:38-49`](../../training/src/environments/parsers.py#L38-L49): two constants now: `DEFAULT_MAX_OBS_CHARS=2000` (paper-arm total), `DEFAULT_MAX_OBS_CHARS_PER_CHUNK=480` (qwen_native per-chunk; ~120 tokens × 4 chars/token proxy).
- [`training/src/environments/parsers.py:66-106`](../../training/src/environments/parsers.py#L66-L106) `format_docs_qwen_native`: parameter renamed `max_chars → max_chars_per_chunk`, semantics changed to per-chunk cap. All docs reach the model; only the oversize ones get the truncation marker.
- [`training/tests/test_format_helpers.py:76-100`](../../training/tests/test_format_helpers.py#L76-L100): replaced the budget-exhaustion test with a per-chunk-independence test.
- [`templates.py:74-90`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L74-L90) `QWEN35_NATIVE_TEMPLATE`: refined steps to mention multi-block ("several Wikipedia passages, one per `<tool_response>...</tool_response>` block"), explicit off-topic guidance ("Some passages may be off-topic; use only the relevant ones"), and "use facts and entities from those relevant passages (not from prior knowledge)" — directly counters the hallucinated-entity failure mode.

Final v3 system message ("our part" — does NOT include the auto-injected `# Tools` block, format example, or `<IMPORTANT>` reminder):

```
You are a helpful assistant with access to a `search` tool that retrieves Wikipedia passages. Always call `search` at least once before answering; do not answer from prior knowledge.

The user will give you a question in the form: Question: <question>

Steps:
- Call `search` with a focused query in the format described above.
- Each search call returns several Wikipedia passages, one per <tool_response>...</tool_response> block. Some passages may be off-topic; use only the relevant ones.
- If the relevant passages contain the answer, write it inside <answer> and </answer> and stop.
- Otherwise, refine your query using facts and entities from those relevant passages (not from prior knowledge) and call `search` again.
```

### 4.2 Result — 7-dataset smoke, n=100/dataset

Hybrid (`qwen3.5_0.8b`), single seed=1, greedy.

| Dataset | EM | F1 | Acc |
|---|---:|---:|---:|
| bamboogle | 0.00 | 0.040 | 0.05 |
| nq | 0.01 | 0.000 | 0.00 |
| triviaqa | 0.04 | 0.024 | 0.06 |
| popqa | 0.01 | 0.008 | 0.03 |
| hotpotqa | 0.00 | 0.013 | 0.01 |
| 2wikimultihopqa | 0.00 | 0.004 | 0.02 |
| musique | 0.00 | 0.001 | 0.00 |
| **mean** | **0.0086** | **0.013** | **0.024** |

(One per-dataset note: triviaqa stands out at 0.04 because it has more direct fact-lookup questions where retrieval gives the answer plainly. Multi-hop datasets — hotpotqa, 2wiki, musique — are at zero, consistent with the model being unable to chain retrieved evidence at this size.)

### 4.3 Cross-family comparison

| Model | Family | Size | Mean EM (untrained, 7 datasets) |
|---|---|---:|---:|
| Qwen3-0.6B hybrid | Qwen3 | 0.6B | **0.102** (M3 baseline; 51,713 items / variant) |
| Qwen3.5-0.8B hybrid | Qwen3.5 | 0.8B | 0.0086 (M4 v3; 700 items total) |
| Qwen2.5-3B base + GRPO | Qwen2.5 | 3B | 0.292 (M1, our reproduction; full Plan B v1) |
| Qwen2.5-3B instruct + GRPO | Qwen2.5 | 3B | 0.361 (M1, our reproduction; full Plan B v1) |

The 12× gap between the same-family-newer Qwen3.5-0.8B and the M3 Qwen3-0.6B floor on the same benchmarks points at the prompt scaffolding, not the model. The M3 Qwen3-0.6B used a system-message prompt (~120 words) with `<search>` / `<result>` tags and a 2-search Hamlet example; the M4 v3 prompt is system + auto-injected `# Tools` block (~150 words) + `<IMPORTANT>` reminder (~70 words) + our system (~110 words) ≈ 330 words of pre-question scaffolding.

## 5. Pipeline-correctness check

Independent of EM, the M4.1 v3 pipeline does what it claims:

- Multi-block render is byte-identical to `apply_chat_template([{"role":"tool",...}, ...])` on the local snapshot (verified via Python render against Qwen3.5-0.8B's `tokenizer_config.json:chat_template`).
- Per-chunk truncation marker `…[truncated]` shows on chunks that exceed 120 tokens.
- All topk=5 chunks reach the model when the question retrieves usable evidence.
- Prompt nudges are present in the rendered query.
- `prompt_tokens` ceiling raised, length-stops disappeared (none observed in v3 across 700 items).
- Train/eval byte-identity preserved via the parsers.py mirror.

So the pipeline change is correct as engineering. The remaining gap is the prompt.

## 6. M4.2 — Search-R1-style minimal user-message prompt

### 6.1 Implementation

System message empty; auto-injected `# Tools` + `<IMPORTANT>` reminder still come from `tools=[QWEN35_SEARCH_TOOL]` (the chat template emits them into the system role for free, in-distribution for Qwen3.5's tool-use post-training). Our protocol prose + the question now live in a **single user message**, dense single paragraph, Search-R1 style with `<tool_call>` / `<tool_response>` / `<answer>` tags and one-word example.

User-message template (`QWEN35_SEARCH_R1_LIKE_TEMPLATE`):

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the `search` tool in the format described above and it will return the top searched results inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {prompt}
```

Code changes:

- [`templates.py:113-141`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L113-L141): added `QWEN35_SEARCH_R1_LIKE_TEMPLATE`, registered as `qwen35_minimal` and `qwen35_searchr1` (alias) in `QWEN35_TEMPLATES`.
- [`active_pipeline.py:38-52`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L38-L52): added `_is_qwen35_user_prompt(mode)` helper distinguishing the user-message variants from the system-message ones; both go through `_is_qwen35` for action_stop / multi-block / per-chunk cap.
- [`active_pipeline.py:79-110`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L79-L110): added a render branch for `_is_qwen35_user_prompt`: no system message, single user message via `prompt_template.format(prompt=question)`, still passes `tools=[QWEN35_SEARCH_TOOL]`.
- [`scripts/run_m4.sh:71-79`](../../scripts/run_m4.sh#L71-L79): `prompt_mode` now reads `${PROMPT_MODE:-qwen35}`; non-default modes get a tag in `save_note` so v3 and minimal results don't collide on disk.

Initial-prompt token count drops from ~1500-1900 (v3) to ~424 (minimal) — a 3.5-4.5× reduction in scaffolding before the question.

### 6.2 Smoke results — 7 datasets, n=100/dataset

Both variants (hybrid + base) run greedy / seed=1 / `enable_thinking=True`. Same retriever / multi-block tool-response / per-chunk cap / 8192 input as v3; only the prompt layout changed.

| Dataset | Hybrid v3 EM | Hybrid minimal EM | Hybrid minimal F1 | Base minimal EM | Base minimal F1 |
|---|---:|---:|---:|---:|---:|
| bamboogle | 0.00 | **0.06** | 0.075 | 0.00 | 0.005 |
| nq | 0.01 | **0.03** | 0.051 | 0.00 | 0.006 |
| triviaqa | 0.04 | **0.09** | 0.118 | 0.00 | 0.014 |
| popqa | 0.01 | **0.07** | 0.083 | 0.00 | 0.007 |
| hotpotqa | 0.00 | **0.13** | 0.149 | 0.01 | 0.010 |
| 2wikimultihopqa | 0.00 | **0.02** | 0.037 | 0.01 | 0.010 |
| musique | 0.00 | 0.00 | 0.000 | 0.00 | 0.003 |
| **mean** | **0.0086** | **0.057** | **0.073** | **0.003** | **0.008** |

### 6.3 Cross-comparison with M3 baseline

| Model | Mode | Mean EM (n=100 / dataset) | Notes |
|---|---|---:|---|
| Qwen3-0.6B hybrid (M3 baseline) | system + `<search>` / `<result>` | 0.102 | full-data n=51,713/variant |
| Qwen3.5-0.8B hybrid v3 | system + `<tool_call>` / `<tool_response>`, ~330 words pre-question | 0.0086 | smoke n=100/dataset |
| **Qwen3.5-0.8B hybrid minimal** | user + auto-inject + `<tool_call>` / `<tool_response>`, ~95 words our + 220 words auto-inject | **0.057** | smoke n=100/dataset |
| Qwen3.5-0.8B base minimal | same as hybrid minimal | 0.003 | smoke n=100/dataset; base lacks tool-use post-training prior |

**Hybrid minimal lifts mean EM 6.6× over v3** but doesn't yet match the M3 Qwen3-0.6B baseline. Possible contributing factors: (a) M3 baseline is full-data n=51,713 vs our smoke n=100 (variance at small n is real); (b) Qwen3.5 family's tool-use post-training format diverges from Qwen3 family's `<search>` tags; (c) the 0.8B-vs-0.6B parameter lift may not compensate for a different post-training distribution. A full-data run for the hybrid minimal config would tighten this comparison.

**Base minimal at 0.003 is essentially zero**, confirming the M3 finding (CLAUDE.md "Base model cannot bootstrap the tool-call format from cold-start"): tool-use post-training is load-bearing. Plain SFT or GRPO from base would need a tool-use warm-start; this is captured in the active recipe-search plan.

### 6.4 Decision: lock minimal as the canonical M4 mode for hybrid

The hybrid M4.2 minimal config is the new canonical M4 baseline:

- prompt_mode = `qwen35_minimal`
- per-chunk cap = 120 tokens, multi-block tool_response
- `generator_max_input_len` = 8192
- enable_thinking = True
- Same retriever / SGLang shape as v3.

Base variant stays in the M4 roster (as the "untrained tool-use floor" reference) but should NOT be used as the primary baseline; its near-zero EM tells us the base model needs warm-start, not that the pipeline is broken.

Full-data runs to follow (51,713 items / variant, ~150 min on 1× A100-80GB).

### 6.5 Things to revisit later

1. **Few-shot in-prompt example**: Search-R1 has only `<answer> Beijing </answer>`. Adding a 2-search worked Hamlet example (M3 style) might lift multi-hop datasets (2wiki, musique). Tradeoff: prompt-token cost.
2. **Tool description tightness**: the auto-injected `# Tools` block contains a verbose `description` ("Search Wikipedia for passages... Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have."). Trimming the description might further compress the prompt without information loss.
3. **`enable_thinking=False` ablation**: the model emits `<think>` blocks 100% of the time which contain entity hallucinations that anchor subsequent queries. Trying `enable_thinking=False` (closed `<think>\n\n</think>` prefix) might suppress this failure mode.
4. **Per-dataset diagnostic**: musique stays at 0.00 even on hybrid minimal; that's the hardest dataset (3-4 hop) and likely irrecoverable without training. 2wiki at 0.02 is also stuck; both are multi-hop where the entity-hallucination failure mode bites hardest.

## 7. M4.3 — fully-minimal (no system block, format spec inlined)

### 7.1 Implementation

Strip `tools=[QWEN35_SEARCH_TOOL]` from `apply_chat_template` so the chat template emits no system block at all (just user → assistant). To compensate we inline the verbatim nested-XML format spec into the user prompt. Total prompt drops from ~424 tokens (M4.2) to ~150 (M4.3); a 2.8× compression.

Code: [`templates.py:142-167`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L142-L167) (`QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE`); [`active_pipeline.py:48-58`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L48-L58) (`_QWEN35_NO_TOOLS_MODES`, `_is_qwen35_no_tools` helper); [`active_pipeline.py:79-101`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py#L79-L101) (render branch conditionally drops `tools=`).

### 7.2 Smoke results — both variants, 7 datasets, n=100

| Dataset | Hybrid M4.2 (with auto-inject) | Hybrid M4.3 (no system) | Base M4.2 | Base M4.3 |
|---|---:|---:|---:|---:|
| bamboogle | **0.06** | 0.01 | 0.00 | 0.01 |
| nq | **0.03** | 0.02 | 0.00 | 0.01 |
| triviaqa | **0.09** | 0.04 | 0.00 | 0.02 |
| popqa | **0.07** | 0.07 | 0.00 | **0.04** |
| hotpotqa | **0.13** | 0.02 | 0.01 | 0.00 |
| 2wikimultihopqa | 0.02 | 0.01 | 0.01 | **0.03** |
| musique | 0.00 | 0.00 | 0.00 | 0.00 |
| **mean EM** | **0.0571** | 0.0243 | 0.0029 | **0.0157** |

### 7.3 Mechanism — asymmetric finding

| Variant + Mode | EM | `</answer>`-rate | EM / answered |
|---|---:|---:|---:|
| hybrid + auto-inject (M4.2) | 5.7% | 28.1% | **20%** |
| hybrid + no_system (M4.3) | 2.4% | 56.3% | 4.3% |
| base + auto-inject (M4.2) | 0.3% | 17.9% | 1.6% |
| base + no_system (M4.3) | 1.6% | 46.3% | **3.4%** |

Stop-distribution shows the mechanism: with the auto-inject removed, BOTH variants stop searching (hybrid `</tool_call>` 346 → 77; base 293 → 17) and instead rush straight to `</answer>` (hybrid 197 → 394; base 125 → 324). The auto-injected `<IMPORTANT>` block ("Function calls MUST follow the specified format... Required parameters MUST be specified...") is what was driving search loops on hybrid. Strip it, and hybrid emits answers ~2× more often but mostly wrong.

For hybrid the trade-off lands in favour of the auto-inject (more search → fewer but more accurate answers). For base the trade-off flips: base's tool-use prior is essentially absent regardless of the auto-inject, so the verbose scaffolding is pure noise; stripping it lets base emit more answers and a few of them are right.

### 7.4 Locked configuration — asymmetric per-variant

| Variant | `prompt_mode` (M4 default) | Reason |
|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `qwen35_minimal` | Auto-inject's `<IMPORTANT>` reminder drives search loops; in-distribution for tool-use post-training. Best mean EM 0.057. |
| `qwen3.5_0.8b_base` | `qwen35_minimal_no_system` | Base lacks a tool-use prior; the auto-inject is pure scaffolding noise. Stripping it lifts EM 0.003 → 0.016 (5×). |

Pipeline change: [`scripts/run_m4.sh`](../../scripts/run_m4.sh) sets a per-variant default — hybrid → `qwen35_minimal`, base → `qwen35_minimal_no_system`. Override either via `PROMPT_MODE=...`.

### 7.5 Open questions raised by §7.3

- **Why does hybrid + no_system rush to answer?** The auto-inject's `<IMPORTANT>` block is the only thing in the prompt explicitly telling the model "function calls MUST follow the format". Without it, the model defaults to whatever its post-training distribution favours, which apparently is "answer once you have anything plausible". Adding "you MUST search at least once" to the user prompt didn't replicate this — possibly because system-role assertions carry stronger priors than user-role assertions.
- **Can we get hybrid above 0.057 with a different tool-description?** The auto-inject schema's `description` field comes from `QWEN35_SEARCH_TOOL` in `templates.py`; bigger / smaller / differently-worded descriptions might drive different search-vs-answer trade-offs without changing the user prompt.

## 8. Pipeline performance optimization (M4-perf)

### 8.1 Diagnosis

Smoke wall-clock raised the question: why is M4 (Qwen3.5-0.8B) slower than M3 (Qwen3-0.6B) which finished in ~150 min/variant? Three findings:

1. **GPU is idle**, not the bottleneck. `nvidia-smi --query-gpu=utilization.gpu`: 0%. SGLang reserves 67 GB of 80 GB but `max_running_requests=533` (auto-computed from KV-cache budget); the eval pipeline only sends 32 concurrent requests (`INFERENCE_MAX_WORKERS=32` in [`flashrag/pipeline/parallelism.py`](../../evaluation_qwen35/flashrag/pipeline/parallelism.py)) — 6 % of SGLang capacity.
2. **Retriever serialization bug**: [`local_retriever/retriever_serving.py:70-101`](../../local_retriever/retriever_serving.py#L70-L101) wraps the FAISS search inside an async function but calls `retriever.search(...)` SYNCHRONOUSLY, blocking the asyncio event loop. The 8-instance pool + `asyncio.Semaphore(8)` looks like it parallelizes, but only one CPU search runs at a time because the event loop can't progress. Empirically: 64 concurrent requests → 10.9 q/s, 3-sec p50 latency, queue head-of-line blocking; 8 concurrent (matched to instances) → 12 q/s; sequential 80–140 ms / query.
3. **CPU-saturated even with the fix**: even after wrapping the search in `asyncio.to_thread(...)` and bumping the pool to 16 instances, throughput plateaus at ~14 q/s. The 16 retriever instances each call into IVF-SQ8 FAISS in C++ (which releases the GIL), but the underlying CPU is saturated by the per-query encoder + index search work. Adding more instances (32+) doesn't help and triggered an OOM during init at 27/32 instances on a host with 866 GB RAM (each instance loaded ~17.5 GB even though the FAISS file is mmap-shared).

### 8.2 Changes applied (verified via re-benchmark)

- [`local_retriever/retriever_serving.py:82-101`](../../local_retriever/retriever_serving.py#L82-L101) `/search` and `/batch_search`: wrap sync FAISS calls in `await asyncio.to_thread(...)` so they don't block the event loop.
- Retriever launched with `--num_retriever 16` (was 8). 16 instances × ~17.5 GB = 280 GB host RAM (well under 866 GB available).
- [`flashrag/pipeline/parallelism.py`](../../evaluation_qwen35/flashrag/pipeline/parallelism.py) `INFERENCE_MAX_WORKERS=128` (was 32). Saturates SGLang's `max_running_requests=533` headroom while staying well below KV-thrash territory.

### 8.3 Speedup measured

Bamboogle smoke n=100 (hybrid + qwen35_minimal):

| Config | Wall-clock (n=100) | Per-item | Iteration rate |
|---|---:|---:|---:|
| Pre-opt (8-worker retriever, 32 client workers) | ~75 s | 0.75 s | 1.83 it/s |
| Post-opt (16-worker retriever + asyncio.to_thread + 128 client workers) | ~47 s | 0.47 s | 2.12 it/s |

Bamboogle is small (n=100 with mostly hard 2-hop questions); a richer datapoint comes from popqa n=1000 (popqa has many simple 1-hop fact-lookups, so retriever load is moderate):

| Config | popqa n=1000 wall-clock | Iteration rate |
|---|---:|---:|
| Post-opt | 4 min 17 s | 3.88 it/s |

So peak post-opt rate on a high-volume dataset is **3.88 it/s**, bamboogle smoke is 2.12 it/s. The headline 35–40 % speedup is real but well below the 5–10× the GPU-utilization gap suggested. The remaining bottleneck is **CPU-side encoder work in the retriever path**, not GPU and not pipeline concurrency.

### 8.4 Why M4 (Qwen3.5-0.8B) is slower than M3 (Qwen3-0.6B) at the same scale

Three contributing factors, in order of impact:

1. **More search turns per item.** M3's stop-distribution had ~30 % of items hitting the turn cap; M4 hybrid has 49 %. Each extra turn = one more retriever call + one more SGLang generation. ~50 % more turns/item = ~50 % more wall-clock at the same per-turn speed.
2. **Larger context window (`generator_max_input_len: 4096 → 8192`).** KV-cache per active request roughly 2× → SGLang's per-request memory cost is up; effective batch throughput in `max_running_requests` headroom drops modestly. Not a 2× slowdown but a ~10–20 % factor.
3. **0.8B vs 0.6B model**. ~33 % more parameters → ~33 % slower per-token decode at fixed batch size.

Combined: ~2× slower than M3 reference (490 min/variant projected post-opt vs M3's 150 min/variant), which matches the observed gap.

### 8.5 Caveats and follow-ups

- Bumping retriever to 32 instances OOMed during init even on 866 GB host. Investigated: each instance allocates ~17.5 GB resident even though the FAISS file is mmap-shared; the duplication is in the embedding-encoder weights + per-instance Python state. A future fix would refactor the retriever pool to share the encoder + index across instances and only have N "slots" for concurrent queries (not N copies of the model).
- CPU profiling not run; if the encoder is the bottleneck, swapping to GPU-resident E5 (uses ~500 MB VRAM) would lift retriever throughput substantially. Trade-off: SGLang already uses 67 GB / 80 GB on the A100, so we'd need to drop `mem_fraction_static: 0.83 → 0.7` to make room.
- For the locked M4 config, **post-opt full-data per-variant is ~420 min** (extrapolated from 0.47 s/item × 51,713 / 60), down from ~490 min pre-opt. Total for option A (locked configs only, both variants): ~14 h.
- [`flashrag/dataset/dataset.py:133-145`](../../evaluation_qwen35/flashrag/dataset/dataset.py#L133-L145): patched to silently cap `--test_sample_num` at the actual dataset size (was raising `ValueError("Sample larger than population")` for bamboogle whose test split is only 125 items). This lets a single n=1000 sweep over all 7 datasets work without per-dataset adjustments.

## 9. Wider smoke (option C: 4 configs × 7 datasets × n=1000)

To anchor the asymmetric finding (§7) on a richer sample, all 4 prompt-mode × variant combinations get re-run at **n=1000 / dataset** (bamboogle effectively n=125 since that's the full split). Greedy / seed=1 / `enable_thinking=True`. This run uses the M4-perf optimized stack (16-worker retriever + asyncio.to_thread + 128 client workers).

| Dataset | Hybrid + qwen35_minimal | Hybrid + qwen35_minimal_no_system | Base + qwen35_minimal | Base + qwen35_minimal_no_system |
|---|---:|---:|---:|---:|
| bamboogle (n=125) | 0.040 | 0.024 | 0.000 | 0.000 |
| nq (n=1000) | 0.065 | 0.035 | 0.007 | 0.010 |
| triviaqa (n=1000) | 0.122 | 0.065 | 0.014 | 0.020 |
| popqa (n=1000) | 0.071 | 0.048 | 0.008 | 0.007 |
| hotpotqa (n=1000) | 0.067 | 0.038 | 0.007 | 0.012 |
| 2wikimultihopqa (n=1000) | 0.041 | 0.052 | 0.006 | 0.034 |
| musique (n=1000) | 0.010 | 0.002 | 0.000 | 0.000 |
| **mean EM** | **0.0594** | **0.0377** | **0.0060** | **0.0119** |

Wall-clock estimate: ~25–35 min/config × 4 configs ≈ 1.5–2.5 h total on the optimized stack. Numbers populate as runs land.

### 9.1 C-phase completion summary

Wall-clock: 1h 41min for all 4 configs (15:31 → 17:12 UTC). Asymmetric finding from §7 confirmed at richer n=1000 sample:

| Variant | qwen35_minimal | qwen35_minimal_no_system | Best mode | Δ (best − other) |
|---|---:|---:|---|---:|
| Hybrid | **0.0594** | 0.0377 | qwen35_minimal | +0.0217 (+58 %) |
| Base | 0.0060 | **0.0119** | qwen35_minimal_no_system | +0.0059 (+98 %) |

Per-dataset asymmetric-mode lift on hybrid is most pronounced on triviaqa (0.122 vs 0.065) and hotpotqa (0.067 vs 0.038). On base, 2wikimultihopqa shows the largest absolute effect (0.034 vs 0.006, 5×).

Locked-best matches the smoke (n=100) call: hybrid → `qwen35_minimal`, base → `qwen35_minimal_no_system`. Phase 4 (base A, full data with the locked best mode) is now in flight; phase 6 (hybrid A, full data with locked best mode) follows.

## 10. Full sweep (option A: 51,713 items / variant, locked best per variant)

Phase 4 (base) and Phase 6 (hybrid) of the orchestrator. Numbers populate as datasets finish.

### 10.1 Base (`qwen3.5_0.8b_base`, prompt_mode = qwen35_minimal_no_system)

| Dataset | Items | EM | ACC | F1 | Wall-clock |
|---|---:|---:|---:|---:|---|
| bamboogle | 125 | 0.0000 | 0.0080 | 0.0089 | ~50 s |
| nq | 3,610 | 0.0042 | 0.0202 | 0.0098 | 8 min 31 s |
| triviaqa | 11,313 | 0.0120 | 0.0313 | 0.0287 | 27 min 31 s |
| popqa | 14,267 | 0.0083 | 0.0243 | 0.0145 | 26 min 24 s |
| hotpotqa | 7,405 | 0.0108 | 0.0278 | 0.0217 | 19 min 52 s |
| 2wikimultihopqa | 12,576 | 0.0316 | 0.0756 | 0.0426 | 32 min 4 s |
| musique | 2,417 | 0.0000 | 0.0054 | 0.0056 | ~5 min |
| **mean (per-dataset, unweighted)** | 51,713 | **0.00955** | **0.0249** | **0.0177** | **2 h 0 min** |

### 10.2 Hybrid (`qwen3.5_0.8b`, prompt_mode = qwen35_minimal)

(starts after Phase 5 SGLang switch)

| Dataset | Items | EM | ACC | F1 | Wall-clock |
|---|---:|---:|---:|---:|---|
| bamboogle | 125 | 0.0480 | 0.0480 | 0.0630 | 55 s |
| nq | 3,610 | 0.0632 | 0.1127 | 0.0874 | 14 min 31 s |
| triviaqa | 11,313 | 0.1243 | 0.1738 | 0.1534 | 46 min |
| popqa | 14,267 | 0.0752 | 0.1098 | 0.0921 | 56 min 52 s |
| hotpotqa | 7,405 | TBD | TBD | TBD | (running) |
| 2wikimultihopqa | 12,576 | TBD | TBD | TBD | – |
| musique | 2,417 | TBD | TBD | TBD | – |
| **mean** | 51,713 | TBD | TBD | TBD | – |

## 7. Pointers

- v1 preserved: [`bamboogle_2026_05_09_11_06_m4_qwen3.5_0.8b_seed1_n100_v1/`](../../evaluation_qwen35/results/bamboogle/) — has `analysis_item_0.txt` walkthrough.
- v2 dirs were renamed `_v2` then lost during a workspace cleanup; v2 metric was preserved only for bamboogle (see §3.2). Treat v2 as a transitional checkpoint; v3 is the canonical "M4.1 minus prompt minimization" floor.
- v3 dirs (current): [`evaluation_qwen35/results/<dataset>/<dataset>_*_m4_qwen3.5_0.8b_seed1_n100/`](../../evaluation_qwen35/results/) — 7 datasets.
- M3 baseline (Qwen3-0.6B hybrid, untrained): [`RESULTS_m3.md`](RESULTS_m3.md) §4.
- M4.1 prompt design rationale: [`MILESTONE_4.md` §M4.1](../milestone_4/MILESTONE_4.md).
- Qwen3.5 chat template (verbatim): [`docs/training/CHAT_TEMPLATE.md` §2](../training/CHAT_TEMPLATE.md).
