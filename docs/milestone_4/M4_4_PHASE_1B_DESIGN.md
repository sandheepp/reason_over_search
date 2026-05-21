---
title: M4.4 Phase 1b — in-distribution prompt ablation (10 candidates, Hamlet 1-shot runs first)
tags: [milestone, eval, m4, m4.4, phase-1b, design]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# M4.4 Phase 1b — 10-candidate in-distribution prompt ablation

> **Status: DESIGN — pending approval before implementation + launch.**
> Supersedes the B / C / D / E plan in [`MILESTONE_4.md` §M4.4](MILESTONE_4.md#L210) after Phase 1a results closed two of the five original slots (recall_port FAILED Δ −0.0084; E FAILED Δ −0.0278). D from the original plan is preserved here as candidate **#2**; the original B / C are dropped and replaced by in-distribution analogues that hold the action / response tags at Hermes nested-XML.

## 1. Why Phase 1b

Phase 1a executed two of five original candidates (recall_port and E). Both failed the +0.025 bar; mechanism in both cases pointed at the **load-bearing role of Qwen3.5's auto-injected `# Tools` + `<IMPORTANT>` scaffold and the open `<think>` channel**. We have not yet tested any of:

- length compression (is A bloated?)
- retrieval-first guards (anti-bypass)
- role-priming (system-role research-assistant frame)
- few-shot demonstration of multi-call behaviour
- explicit 3-stage decomposition
- source-grounding + uncertainty escape hatch
- self-verification before answer
- refine-and-retry encouragement

The original B (Search-R1 verbatim) and C (M3.1 verbatim) put the model **off-distribution on the action tokens** (`<search>` instead of `<tool_call>`). Qwen3.5 was post-trained on Hermes nested-XML; the M3 finding "tag schema is interchangeable" was within-family for Qwen3 and there is no evidence it transfers to Qwen3.5. Phase 1b refactors all candidates to use `<tool_call>` / `<tool_response>` via `tools=[QWEN35_SEARCH_TOOL]` auto-inject, so the prose, locus, decision-rule, and example hypotheses can be tested **without confounding from off-distribution tags**.

## 2. Web-research highlights (May 2026)

Sources at end of doc.

- **Hallucination prior**: Qwen3.5 pattern-matches against training data without retrieval; explicit "verify with `search` first" instructions reduce this.
- **Schema discipline**: enforcing `<answer>…</answer>` reduces format drift by ~60 %.
- **Role priming** helps for specialised tasks (research assistant framing).
- **Self-check at end** catches ~1 in 3 hallucinated specifics on factual QA.
- **Hermes-style tool use** (`<tool_call>` nested-XML) is the recommended scheme for Qwen3 family — which is exactly what `tools=[…]` auto-inject emits.
- **Anti-pattern**: ReAct / stopword-based tool calling on reasoning models — model leaks stopwords into the think section.
- **Anti-pattern**: changing multiple variables simultaneously — confirms our single-knob constraint.
- **Anti-pattern**: blocking the model from expressing uncertainty.

## 3. Held constant (all 10 candidates)

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` hybrid (eval/qwen3.5_0.8b) |
| Action tag | `<tool_call>…</tool_call>` (Hermes nested-XML) |
| Result tag | `<tool_response>…</tool_response>` (Hermes nested-XML) |
| `tools=[QWEN35_SEARCH_TOOL]` auto-inject | YES (chat template injects `# Tools` + format spec + `<IMPORTANT>`) |
| `enable_thinking` | **True** (E confirmed False is catastrophic — 57.7 % empty-pred collapse) |
| Final wrap | `<answer>X</answer>` plain (no `\boxed{}`) |
| Pipeline branch | `qwen35` (existing `_is_qwen35` path) |
| Sampling | Greedy, `temperature=0.0`, `seed=1` |
| n / dataset | 300 (bamboogle full split = 125) |
| Retrieval stack | IVF-SQ8 × 16 workers + `asyncio.to_thread` + 128 client workers |
| Per-chunk cap | 120 tokens |
| `generator_max_input_len` | 8192 |

**The only varying axis is the prompt text (and locus: user / system / both).**

## 4. The 10 candidates

Naming convention: `qwen35_<axis-tag>`. All are added to `QWEN35_TEMPLATES` in [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py).

### #1 — `qwen35_terse` (length / brevity)

**Axis**: drop A's cargo-culted Search-R1 clauses (`<think>` instructions, "after reasoning…", `<tool_response>` reminder which auto-inject covers). Keep only action + format anchor + question.

**Rationale**: A's user message is ~95 words / ~80 user tokens; ~424 total prompt tokens including auto-inject. Most of those clauses are redundant with the auto-injected `# Tools` + `<IMPORTANT>` block. Test: does compressing the user message change anything?

**Locus**: user.

```python
QWEN35_TERSE_TEMPLATE = (
    "Use the `search` tool to look up facts as needed. "
    "When you have the answer, write it inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~30. Total prompt: ~370.

### #2 — `qwen35_decide` (+ decision rules)

**Axis**: append the two M3.1 winner decision sentences to A's user message. This is the original M4.4 candidate D, preserved verbatim.

**Rationale**: M3.1's `p3_decide_no_ex` won within the Qwen3 family with decision-rule scaffolding. Hypothesis: explicit "after each result, decide whether to search again or answer" sentences are the load-bearing piece, separable from the system-role locus that recall_port already failed with.

**Locus**: user.

```python
QWEN35_DECIDE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "Use the information in the search results to determine the final answer. "
    "After each search result, decide whether another search is needed or whether you can provide the final answer. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~100. Total prompt: ~445.

### #3 — `qwen35_p3_decide_xml` (M3.1 prose, system role, in-dist tags)

**Axis**: system-role locus + decision rules + no example — the M3.1 winner shape but using `<tool_call>` tags (in-dist) instead of `<search>` (off-dist).

**Rationale**: tests whether the prose + system-role structure of `P3_DECIDE_NO_EX_TEMPLATE` transfers to Qwen3.5 once the tag confound is removed. Replaces the original C.

**Locus**: **system** + user. User message = `Question: {q}` (hardcoded by the qwen35 pipeline branch, like recall_port).

```python
QWEN35_P3_DECIDE_XML_TEMPLATE = (
    "You are a helpful assistant who can answer questions using the `search` tool.\n"
    "You can call the search tool using the format described above; results will be returned inside <tool_response> and </tool_response>.\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Use the information in the search results to determine the final answer.\n"
    "After each search result, decide whether another search is needed or whether you can provide the final answer.\n"
    "If a search result is incomplete, search again for the missing information.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>X</answer>."
)
```

Estimated total prompt (system + user `Question: {q}` + auto-inject): ~475.

### #4 — `qwen35_search_first` (+ retrieval-first guard)

**Axis**: prepend an explicit "you must search before answering; never answer from prior knowledge" sentence to A's user message.

**Rationale**: E's bypass-and-fabricate failure mode (57.7 % empty preds, confident hallucinations on completed items, 2wiki coin-flip on binary comparisons) confirms that Qwen3.5-0.8B has a strong "answer from prior knowledge" prior. Web research flagged this as Qwen3.5's #1 anti-pattern. Test the most explicit anti-bypass intervention.

**Locus**: user.

```python
QWEN35_SEARCH_FIRST_TEMPLATE = (
    "You must call the `search` tool to verify facts before answering; "
    "never answer from prior knowledge alone.\n"
    "\n"
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, call the `search` tool in the format described above; "
    "results will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "When the search results contain the answer, provide it inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~110. Total prompt: ~455.

### #5 — `qwen35_research_role` (system role-prime only)

**Axis**: short, role-only system message (web research: role-priming helps for specialised tasks). User message is bare `Question: {q}`, matching recall_port / `qwen35` routing.

**Rationale**: tests whether a minimal system-role research-assistant frame moves the loop. Distinguishes from recall_port (verbose system message — failed) by being short and role-only; distinguishes from A (no system message — auto-inject only) by adding a small role prime on top.

**Locus**: **system** only (user is hardcoded `Question: {q}` by the qwen35 pipeline branch — same routing as #3 / recall_port / `qwen35_native`).

```python
QWEN35_RESEARCH_ROLE_TEMPLATE = (
    "You are a research assistant. "
    "For every factual question, verify the answer using the `search` tool before responding. "
    "Provide the final answer inside <answer> and </answer>."
)
```

Estimated total prompt (system + `Question: {q}` + auto-inject): ~365.

### #6 — `qwen35_hamlet_1shot` (full agentic-loop demonstration — **RUNS FIRST**)

**Axis**: append a 2-search Hamlet example to A's user message **with `<think>` blocks inside the example**, demonstrating the full agentic loop (`<think>` plan → `<tool_call>` → `<tool_response>` → `<think>` re-plan → `<tool_call>` → `<tool_response>` → `<think>` synthesise → `<answer>`). Most concrete in-distribution demonstration the model can see.

**Rationale**: M3's `p1_basic_w_ex` used the same Hamlet 2-search example (without `<think>` blocks) and anchored multi-call behaviour in Qwen3-0.6B (end-of-run reward 0.190). Qwen3.5-0.8B has a strong 1-search bias and a bypass-and-fabricate tendency (E confirmed both). Including `<think>` blocks throughout the example matches `enable_thinking=True` rollout shape exactly, so the demo shows the model every piece of the loop it needs to execute. This is the closest the prompt can get to "show, don't tell."

**Locus**: user.

```python
QWEN35_HAMLET_1SHOT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations.\n"
    "\n"
    "Example:\n"
    "Question: What is the nationality of the author of Hamlet?\n"
    "<think>\n"
    "I need to find the author of Hamlet, then look up that author's nationality. "
    "Let me search for the author first.\n"
    "</think>\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nauthor of Hamlet\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_response>The Tragedy of Hamlet was written by William Shakespeare around 1600.</tool_response>\n"
    "<think>\n"
    "The author is William Shakespeare. Now I need to find his nationality.\n"
    "</think>\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nWilliam Shakespeare nationality\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_response>William Shakespeare was an English playwright, widely regarded as the greatest writer in the English language.</tool_response>\n"
    "<think>\n"
    "The search results confirm William Shakespeare was English.\n"
    "</think>\n"
    "<answer> English </answer>\n"
    "\n"
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~280. Total prompt: ~610.

### #7 — `qwen35_decompose` (explicit 3-stage decomposition)

**Axis**: replace A's prose with an explicit 3-stage decomposition pattern from the web research ("list sub-questions → answer using only sources → identify gaps").

**Rationale**: tests whether explicit decomposition instructions move the model from 1-search bias toward multi-step planning. Distinct from #2 (decision rules) which is reactive ("after each search, decide…"); decompose is upfront ("first identify what you need, then search").

**Locus**: user.

```python
QWEN35_DECOMPOSE_TEMPLATE = (
    "Answer the given question by following these steps:\n"
    "1. Identify the facts you would need to know to answer the question.\n"
    "2. For each fact, call the `search` tool to look it up "
    "(format described above; results come inside <tool_response> and </tool_response>).\n"
    "3. Combine the search results and write the answer inside <answer> and </answer>.\n"
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~70. Total prompt: ~415.

### #8 — `qwen35_source_only` (+ source-grounding + uncertainty escape)

**Axis**: append "use ONLY information from `search` results; if not in results, answer `unknown`" to A's user message.

**Rationale**: Web research's strongest single signal ("answer using ONLY information from the provided documents"). The uncertainty escape hatch (`<answer>unknown</answer>`) is essential — without it, the model fabricates instead of declining. Targets bypass-and-fabricate from a different angle than #4: #4 forces retrieval upfront, #8 grounds the answer post-retrieval.

**Locus**: user.

```python
QWEN35_SOURCE_ONLY_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want.\n"
    "\n"
    "Use ONLY information returned by the `search` tool. "
    "If the search results do not contain the answer, write <answer>unknown</answer>.\n"
    "\n"
    "Otherwise, provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~120. Total prompt: ~465.

### #9 — `qwen35_self_check` (+ verification before answer)

**Axis**: append "before writing the final answer, briefly verify it is supported by the search results; if not, search again".

**Rationale**: Web research: "self-check at end catches ~1 in 3 hallucinated specifics on factual QA." Cheapest intervention — adds one short clause; risk is the model thrashes between verify and retry.

**Locus**: user.

```python
QWEN35_SELF_CHECK_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want.\n"
    "\n"
    "Before writing the final answer, briefly verify that it is supported by the search results. "
    "If it is not, refine your query and search again.\n"
    "\n"
    "Then provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~115. Total prompt: ~460.

### #10 — `qwen35_multi_search` (+ refine-and-retry encouragement)

**Axis**: append "if the first result doesn't fully answer, refine your query and search again; continue until you have enough".

**Rationale**: Qwen3.5-0.8B has a 1-search bias (empirical from M4.2 trace inspection). M3 broke this by including a 2-search example. This candidate tests whether explicit instruction (without example) achieves the same. Distinct from #6 (Hamlet example) which is demonstration-based; #10 is instruction-based.

**Locus**: user.

```python
QWEN35_MULTI_SEARCH_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>.\n"
    "\n"
    "If the first search result does not fully answer the question, refine your query and search again. "
    "Continue until you have enough information.\n"
    "\n"
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

Estimated user tokens: ~120. Total prompt: ~465.

## 5. Single-knob coverage matrix

Each candidate ablates exactly one knob relative to A (M4.2 lock). #3 and #5 introduce a system message, which the existing recall_port path already supports without pipeline changes.

| # | Mode | Locus | Tools auto-inject | Decision rules | Example | Role-prime | Retrieval-first | Source-grounding | Self-verify | Multi-search push | Length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | `qwen35_minimal` (control) | user | yes | – | – | – | – | – | – | – | std |
| 1 | `qwen35_terse` | user | yes | – | – | – | – | – | – | – | **terse** |
| 2 | `qwen35_decide` | user | yes | **yes (2 sent.)** | – | – | – | – | – | – | std |
| 3 | `qwen35_p3_decide_xml` | **system** | yes | **yes (M3.1 prose)** | – | implicit | – | – | – | – | system |
| 4 | `qwen35_search_first` | user | yes | – | – | – | **yes** | – | – | – | std |
| 5 | `qwen35_research_role` | **system + user** | yes | – | – | **yes** | – | – | – | – | sys+std |
| 6 | `qwen35_hamlet_1shot` | user | yes | – | **yes** (2-search Hamlet) | – | – | – | – | implicit | long |
| 7 | `qwen35_decompose` | user | yes | – | – | – | – | – | – | – | **3-stage replace** |
| 8 | `qwen35_source_only` | user | yes | – | – | – | – | **yes** + uncertainty escape | – | – | std+ |
| 9 | `qwen35_self_check` | user | yes | – | – | – | – | – | **yes** | – | std+ |
| 10 | `qwen35_multi_search` | user | yes | – | – | – | – | – | – | **yes** | std+ |

## 6. Acceptance bar

Same as Phase 1a: **mean EM ≥ A + 0.025** at n=300/dataset (Wilson 95 % CI half-width at p=0.06, n=300 ≈ 2.7 pp). Below the bar → fail; phase-2 ablation only triggers on a candidate above the bar.

## 6.5 Run order

Per user direction (2026-05-10), **#6 `qwen35_hamlet_1shot` runs first** as the most concrete example-driven candidate. Remaining order: #1 → #5, #7 → #10 (sequential). The sweep is skip-aware so a crash mid-run resumes without re-running completed cells.

I will also upgrade #6 in-place to include `<think>` reasoning inside the example (matching `enable_thinking=True` rollout shape), so the model sees the full agentic loop (`<think>` plan → `<tool_call>` → `<tool_response>` → `<think>` re-plan → `<tool_call>` → `<tool_response>` → `<think>` synthesise → `<answer>`). The original Hamlet-without-think variant is reduced to a Phase-2 ablation if #6 wins.

## 7. Wall-clock estimate

- Per-candidate wall ranges from 6 min (E, terse output) to 17 min (recall_port, verbose). Take median 10 min.
- 10 candidates × 10 min ≈ **1 h 40 min** sequential.
- Services (retriever + SGLang hybrid) currently up; no bootstrap cost.

## 8. Implementation steps (post-approval)

1. **Templates**: add 10 constants + registry entries to [`templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py). ~120 LoC.
2. **Pipeline check**: render each new template via `tokenizer.apply_chat_template` on bamboogle item 0 + 1 multi-hop item; verify no chat-template surprises (especially for #3 and #5 system-role variants).
3. **Pipeline branching**: confirm `_is_qwen35` in [`active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) handles both user-only (#1, #2, #4, #6, #7, #8, #9, #10) and system+user (#3, #5) shapes — recall_port already proved system+user works.
4. **Orchestrator**: extend the `run_E_sweep.sh` pattern: outer loop over modes, inner loop over 7 datasets, n=300, skip-aware (`metric_score.txt` glob).
5. **Run**: launch in background with monitor; report each candidate's mean EM as it lands.
6. **Aggregate**: produce a single comparison table vs A baseline + delta + bar status; identify any candidate ≥ +0.025 lift for phase-2 ablation.

## 9. What I'm explicitly NOT testing

- `enable_thinking=False` — Phase 1a confirmed catastrophic (Δ −0.0278).
- `<search>` / `<information>` or `<search>` / `<result>` tags — off-distribution for Qwen3.5.
- Combined knobs (e.g., decompose + retrieval-first) — single-knob ablation per user instruction. Combinations can be Phase-2 ablations of any Phase-1b winner.
- ReAct / stopword-based prompting — web research flagged as anti-pattern for reasoning models.
- Variants that block uncertainty expression — anti-pattern (only #8 exposes the explicit escape).

## 10. Open questions for review

1. **Token-budget concern on #6 (Hamlet 1-shot)**: ~545 prompt tokens leaves ~7650 for the loop. Should be fine, but worth flagging — multi-hop datasets with 4–5 search rounds × 6 chunks × 120 tokens = ~3600 retrieval tokens, well within budget.
2. **#7 (decompose) is the most aggressive rewrite** — it replaces A's whole prose. Closer to a "different design" than a "single-knob ablation". Keep, drop, or rework?
3. **#3 and #5 system-role variants** add a second template-type to the qwen35 branch. Pipeline already handles this (recall_port did the same), but worth sanity-rendering before launch.
4. **Failure-mode probes**: should I also dump the **empty-pred rate** per candidate alongside EM? Phase 1a showed empty-pred rate is a cleaner failure-mode signal than EM for these prompts (E had 57.7 % empty preds; if a new candidate has 5 % empty preds at lower EM, that's qualitatively different from one with 50 % empty preds at the same EM).

## 11. Phase-2 follow-up (deferred — next milestone TBD)

Phase 1b ran and produced a clear winner `qwen35_terse` (Δ +0.0436) and a marginal secondary `qwen35_research_role` (Δ +0.0257). Three Phase-1b near-misses (`qwen35_decide` Δ +0.0201, `qwen35_source_only` Δ +0.0172, `qwen35_self_check` Δ +0.0133) suggest specific axes (decision rules, source-grounding, self-verify) that ALONE underperform but COULD lift terse further. Phase 2 tests three combination candidates:

| # | mode | composition | rationale |
|---|---|---|---|
| 2A | `qwen35_terse_decide` | terse user message + the two M3.1 decision sentences appended | Phase 1b's narrowest miss (decide) added on top of the winner. If decision rules are additive, this lifts further. |
| 2B | `qwen35_terse_source_only` | terse user message + "use ONLY search results / answer `unknown` if not found" appended | Phase 1b's source_only had 5/7 positives; combining with terse's empty-pred reduction could push it over. |
| 2C | `qwen35_terse_research_role` | research_role system message + terse user message | The two Phase 1b winners stacked. Tests whether the system-role-prime and user-message-compression effects are independent. |

**Acceptance bar (Phase 2)**: mean EM ≥ A + **0.04** = **0.0994** at n=300. Stricter than Phase 1b's +0.025 to control for multiple-testing inflation (we've now screened 12 candidates; need stronger signal to commit to Phase 3).

**If no Phase-2 candidate clears 0.0994**: lock `qwen35_terse` as the M4.4 winner and proceed directly to Phase 3 (full n=51,713 sweep with terse).

**Wall**: 3 candidates × ~10 min = ~30 min sequential. Templates would be ~30 LoC additions to [`templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py); mode names registered in [`_QWEN35_USER_PROMPT_MODES`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) (2A and 2B; 2C has both system + user — uses the system+user routing pattern from `qwen35_research_role` but with terse user). The user message for 2C would need a small pipeline tweak (or the simpler workaround: prepend the role-prime to the terse user message and use the user-only routing).

**Phase 3** (after Phase 2 lock, ~7 h wall, full n=51,713 sweep with the final locked prompt): re-runs all 7 hybrid datasets at full n with the locked prompt; replaces [`RESULTS_m4.md`](../report/RESULTS_m4.md) §4 hybrid row + §5 cross-family delta.

## 12. Phase-4 follow-up — base variant prompt search (deferred — next milestone TBD)

Phase 1b only searched **hybrid**. The base variant is currently locked at `qwen35_minimal_no_system` per [M4.3 §"Variant dispatch"](MILESTONE_4.md#L15) (mean EM 0.010, n=51,713). That lock was driven by the auto-inject-hurts-base finding: `qwen35_minimal_no_system` (no system, no `tools=[]`) beat `qwen35_minimal` (with auto-inject) **5×**, attributed to base lacking the tool-use post-training prior.

Phase 1b changed the picture for hybrid: the M4.2 user prose was bloated, and a 3-sentence user message (terse) cleanly beats it. **The base-variant analog is untested**: does a terse user message + no-auto-inject improve over the M4.3 lock? Possible mechanisms:
1. Base benefits even more from a clean user message than hybrid, because there's no auto-inject to fall back on for protocol semantics.
2. Base might now benefit from auto-inject *if* the user message is clean enough — M4.3 tested auto-inject WITH a bloated user message (`qwen35_minimal`), so the test wasn't clean.
3. Some intermediate variant (e.g., short role-prime in user role since base has no system) might lift further.

### Phase 4 candidates (n=300/dataset, greedy seed=1, base variant)

| # | mode | system | tools=[] | user message | rationale |
|---|---|---|---|---|---|
| 4A | `qwen35_minimal_no_system` (CONTROL) | none | no | M4.3 verbose user msg w/ inline format spec | Re-anchor M4.3 lock at n=300 (current value 0.010 was at n=51,713). |
| 4B | `qwen35_terse_no_system` | none | **no** | Phase-1b terse user msg + inlined `<tool_call>` format spec | Cleanest base candidate: terse prose, no auto-inject, format spec stays inlined per M4.3 lock structure. |
| 4C | `qwen35_terse` (auto-inject) | none | **yes** | Phase-1b terse user msg, format spec auto-injected | Re-test "auto-inject is bad for base" with a clean user message. Cheapest test of mechanism 2. |
| 4D | `qwen35_terse_role_no_system` | none | no | Short role-prime ("You are a research assistant. ...") prepended to terse user msg | No system role available without `tools=[]`, so the role-prime goes in user. Tests Phase-1b winner #2 (research_role) at base. |

**Acceptance bar (Phase 4)**: **mean EM ≥ M4.3 lock + 0.025 = 0.035** at n=300. Same +0.025 threshold as Phase 1b; baseline anchored to the existing locked value.

If 4A re-anchors below 0.010 (n=300 variance) and a candidate just clears the bar relative to 4A's measured value, that's still a valid lift signal (intra-sweep comparison). If 4A re-anchors above 0.020, treat the bar as +0.025 over the re-anchored 4A.

**Implementation cost**:
- ~30 LoC: 3 new template constants (`qwen35_terse_no_system`, `qwen35_terse_role_no_system`) — `qwen35_terse` already exists from Phase 1b; pipeline routing for the `_no_system` variants is already present (M4.3 path via `_QWEN35_NO_TOOLS_MODES`).
- ~30 LoC: extend the orchestrator pattern in `/tmp/run_phase1b_sweep.sh` to a `run_phase4_base_sweep.sh` that boots SGLang with the base model path (`eval/qwen3.5_0.8b_base`) instead of hybrid, then sweeps the 4 modes × 7 datasets.

**Wall**: 4 candidates × ~10 min ≈ **~40 min**. Plus model swap on SGLang: ~3 min.

### Phase 5 — base full benchmark

After Phase 4 locks a base winner (or confirms M4.3 lock if no candidate clears bar): re-run full n=51,713 × 7 datasets on base with the locked prompt. Wall ~7 h on the optimised stack. Replaces [`RESULTS_m4.md`](../report/RESULTS_m4.md) §4 base row and triggers §5 cross-family table refresh (will pair with hybrid Phase 3 results).

If Phase 4 produces NO candidate above bar, M4.3 lock stays and Phase 5 is skipped (base row in [`RESULTS_m4.md`](../report/RESULTS_m4.md) §4 already reflects M4.3 lock at full n).

### Open design questions for Phase 4

1. **`enable_thinking` for base?** Currently `True` per `scripts/run_m4.sh` (mildly off-distribution for base). Worth re-testing False for base specifically — the no-think failure mode that broke hybrid (empty-pred collapse) may be lower-cost for base since base wasn't post-trained on hybrid soft-switch anyway. Phase 4 could add a `qwen35_terse_no_system_nothink` variant. Decide based on Phase 4 wall-budget tolerance.
2. **Larger candidate set?** Phase 4 currently has 4 candidates. If results are interesting (e.g., 4C `qwen35_terse` with auto-inject lifts base), consider expanding with more single-knob ablations modelled on Phase 1b. Defer until first 4 land.
3. **Combined hybrid+base lock**: should M4.4 require both variants to lock before declaring "M4.4 done"? Currently the cross-family table in [`RESULTS_m4.md`](../report/RESULTS_m4.md) §5 uses both variants. Pragmatic answer: yes — schedule Phase 5 to follow Phase 4 directly so the §5 table refreshes once with both new locks.

## Sources (web research, May 2026)

- [How to Use Qwen 3.5 Effectively: Prompt Patterns for Writing, Coding & Research — Macaron](https://macaron.im/blog/how-to-use-qwen-3-5-effectively) — three-stage research chain, constraints block, source-only synthesis, anti-pattern catalogue.
- [Function Calling — Qwen documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — Hermes-style tool use recommended for Qwen3; ReAct anti-pattern.
- [Qwen-Agent — GitHub](https://github.com/QwenLM/Qwen-Agent) — reference agent framework.
- [Function Calling — Qwen-Agent examples](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/function_calling.py) — concrete tool-calling templates.
