---

## title: M4 prompts — scratch file for comments
tags: [milestone, eval, m4, scratch]
source: internal
created: 2026-05-10
updated: 2026-05-10

# M4 prompts — scratch file for comments

> Temporary scratch — feel free to scribble inline. Source-of-truth lives in
> `[evaluation_qwen35/flashrag/search_r1/templates.py](../../evaluation_qwen35/flashrag/search_r1/templates.py)`
> and the variant lock is in `[MILESTONE_4.md](MILESTONE_4.md)` §"Variant dispatch".

## Variant dispatch (M4.2 / M4.3 lock-in, 2026-05-09)


| Variant                 | `prompt_mode`              | System message | `tools=[…]` auto-inject                           | Rationale                                                                                                                                     |
| ----------------------- | -------------------------- | -------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `qwen3.5_0.8b` (hybrid) | `qwen35_minimal`           | empty          | **yes** (`# Tools` + format spec + `<IMPORTANT>`) | hybrid was post-trained on tool-use; auto-inject's `<IMPORTANT>` reminder drives search loops (in-distribution). Mean EM 0.057, 6.6× over v3. |
| `qwen3.5_0.8b_base`     | `qwen35_minimal_no_system` | none at all    | **no**                                            | base lacks tool-use prior; auto-inject is scaffolding noise that crowds out the answer. Mean EM 0.016 vs 0.003 (5×).                          |


Both variants use `enable_thinking=True` (open `<think>\n` generation prefix) and greedy decode (`temperature=0.0`).



---

## Variant 1 — hybrid (`qwen35_minimal`)

**Locus**: protocol prose in user message; system is empty; `tools=[QWEN35_SEARCH_TOOL]` triggers Qwen3.5 chat-template auto-inject.

### Template constant (templates.py:134)

```python
QWEN35_SEARCH_R1_LIKE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```



### What the model actually sees after `apply_chat_template`

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the `search` tool in the format described above and it will return the top searched results inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

The system block (everything between `<|im_start|>system` and `<|im_im_end|>`) is **entirely auto-generated** by Qwen3.5's chat template from `tools=[QWEN35_SEARCH_TOOL]` — we do not write it. Our `QWEN35_SEARCH_R1_LIKE_TEMPLATE` only fills the user turn.



---

## Variant 2 — base (`qwen35_minimal_no_system`)

**Locus**: everything in the user message. **No** system block. **No** `tools=[…]` arg → no auto-inject. Format spec is inlined verbatim into the user message as a fallback anchor.

### Template constant (templates.py:159)

```python
QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```



### What the model actually sees after `apply_chat_template`

```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the search tool by writing:
<tool_call>
<function=search>
<parameter=query>
your query
</parameter>
</function>
</tool_call>
The result will be returned inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

Total prompt token count: ~150 vs ~424 for hybrid (the `# Tools` + `<IMPORTANT>` block is ~220 words alone).



---

## Side-by-side diff


| Aspect                               | Hybrid `qwen35_minimal`                                 | Base `qwen35_minimal_no_system`          |
| ------------------------------------ | ------------------------------------------------------- | ---------------------------------------- |
| System block                         | auto-injected (~220 words)                              | **absent**                               |
| `tools=[QWEN35_SEARCH_TOOL]` passed? | yes                                                     | **no**                                   |
| User-message format spec             | "in the format described above" (refers to auto-inject) | inlined `<tool_call>…</tool_call>` block |
| `<IMPORTANT>` reminder               | yes (auto-injected)                                     | **no**                                   |
| Few-shot example                     | `<answer> Beijing </answer>` only                       | `<answer> Beijing </answer>` only        |
| Token count                          | ~424                                                    | ~150                                     |
| `enable_thinking`                    | True                                                    | True                                     |
| Greedy decode                        | True                                                    | True                                     |




---

## Open question — M4.4 prompt search (status: pending)

M4.2 hybrid lands at mean EM 0.0594; M3 untrained Qwen3-0.6B (smaller, older family) reaches 0.102 on the same 7 benchmarks. 1.7× cross-family gap → prompt headroom is the prime suspect. Phase-1 plan in `[MILESTONE_4.md` §M4.4](MILESTONE_4.md#L195) tests 5 candidates at n=300/dataset; if no candidate beats M4.2 by ≥ +1 pt mean EM, ship M4.2 unchanged.



---

## M4.4 Phase-1 candidates (A–E)

> Phase-1 sweep: 5 prompt_modes × 7 datasets × n=300, greedy / `seed=1`, plain `<answer>X</answer>` (no `\boxed{}`).
> Acceptance bar: mean EM ≥ A + 0.025. Source-of-truth in
> `[MILESTONE_4.md` §M4.4](MILESTONE_4.md#L210); A is locked in
> `[templates.py](../../evaluation_qwen35/flashrag/search_r1/templates.py)`; B/C/D/E to be added.

### A — `qwen35_minimal` (control, M4.2 lock)

**Routing**: empty system, `tools=[QWEN35_SEARCH_TOOL]` → auto-inject `# Tools` + `<IMPORTANT>`,
user-message protocol, `<tool_call>` / `<tool_response>` tags, `enable_thinking=True`.
Baseline mean EM **0.0594**. Template constant + full post-render shown in [Variant 1 above (lines 28–93)](#variant-1--hybrid-qwen35_minimal); restated below in compact form for side-by-side.

```python
# templates.py:134
QWEN35_SEARCH_R1_LIKE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```



---

### B — `qwen35_searchr1_pure` (Search-R1 paper verbatim)

**Routing**: empty system, **no** `tools=[]` arg → **no** auto-inject, user-message protocol,
`<search>` / `<information>` tags (search_r1 pipeline branch), `enable_thinking=True`.
Verbatim port of `[SEARCH_R1_TEMPLATE](../../evaluation_qwen35/flashrag/search_r1/templates.py#L201-L212)`
(byte-identical to the Qwen2.5-3B M1 reproduction that hit EM 0.292).

#### Tool-call I/O pattern (B is OFF-distribution for Qwen3.5)

| Phase | Token pattern |
|---|---|
| Model emits (action) | `<search>query</search>` |
| Pipeline wraps result | `\n\n<information>{passages}</information>\n\n` |
| Pipeline branch | `search_r1` (re-uses [`extract_search_tag_query`](../../evaluation_qwen35/flashrag/search_r1/parser.py)) |
| Stop tokens | `</search>` |

**Why this is off-distribution**: Qwen3.5's tool-use post-training emits `<tool_call><function=NAME>…</function></tool_call>` (nested-XML), not `<search>…</search>` (Search-R1 / Qwen3-family flat-tag). B deliberately tests whether the **prose** of the Search-R1 paper drives the loop even when the **tag tokens** are off-distribution (M3 finding: "tag schema is interchangeable"; the question is whether that holds for Qwen3.5 too).

#### Template constant (to be added as `QWEN35_SEARCH_R1_PURE_TEMPLATE`)

```python
QWEN35_SEARCH_R1_PURE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

#### What the model actually sees after `apply_chat_template`

```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

Total prompt token count: ~150 (no `# Tools` / `<IMPORTANT>` block).



---

### C — `qwen35_p3_decide_no_ex` (M3.1 winner shape, ported)

**Routing**: system role carries protocol + decision rules; user message = `Question: {q}`;
**no** `tools=[]` → **no** auto-inject; `<search>` / `<result>` tags (qwen3 pipeline branch);
`enable_thinking=True`. Verbatim port of
`[P3_DECIDE_NO_EX_TEMPLATE](../../evaluation_qwen35/flashrag/search_r1/templates.py#L32-L42)`
**with `\boxed{}` dropped → plain `<answer>X</answer>`** (the only diff vs M3.1).

#### Tool-call I/O pattern (C is OFF-distribution for Qwen3.5)

| Phase | Token pattern |
|---|---|
| Model emits (action) | `<search>query</search>` |
| Pipeline wraps result | ` <result>\n{passages}\n</result>` |
| Pipeline branch | `qwen3` (M3.1 path; re-uses `extract_search_tag_query`) |
| Stop tokens | `</search>` |

**Why this is off-distribution**: same off-distribution argument as B — Qwen3.5's tool-use post-training emits `<tool_call>…</tool_call>`, not `<search>…</search>`. C additionally moves the protocol prose into the **system role** (vs B/A/D/E where protocol prose lives in the **user role**). Both axes (tag scheme + locus) are simultaneously off-distribution from Qwen3.5's post-training, but **in-distribution for the Qwen3 family** — C is the same shape that drove the M3.1 winner. Test: does that shape's effectiveness transfer across families.

#### Template constant (to be added as `QWEN35_P3_DECIDE_NO_EX_TEMPLATE`)

```python
QWEN35_P3_DECIDE_NO_EX_TEMPLATE = (
    "You are a helpful assistant who can answer questions using a Wikipedia search tool.\n"
    "You can call the search tool by writing: <search> your query </search>\n"
    "You will receive the result in: <result> your search result </result>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Use the information in the search results to determine the final answer.\n"
    "After each search result, decide whether another search is needed or whether you can provide the final answer.\n"
    "If a search result is incomplete, search again for the missing information.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>X</answer>."
)
```

(Diff vs M3.1: last line changed from `<answer>The final answer is \[ \boxed{answer here} \]</answer>` to `<answer>X</answer>`. M4 EM scorer normalises and the plain form matches what M5 training will emit.)

#### What the model actually sees after `apply_chat_template`

```
<|im_start|>system
You are a helpful assistant who can answer questions using a Wikipedia search tool.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Use the information in the search results to determine the final answer.
After each search result, decide whether another search is needed or whether you can provide the final answer.
If a search result is incomplete, search again for the missing information.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>X</answer>.<|im_end|>
<|im_start|>user
Question: <QUESTION GOES HERE><|im_end|>
<|im_start|>assistant
<think>
```

Total prompt token count: ~180.



---

### D — `qwen35_minimal_decide` (A's user message + M3.1 decision rules)

**Routing**: identical to A — empty system, `tools=[QWEN35_SEARCH_TOOL]` auto-inject,
`<tool_call>` / `<tool_response>` tags (qwen35 pipeline branch), `enable_thinking=True`.
Differs from A only in the user message: appends the two M3.1 decision sentences
(verbatim from `P3_DECIDE_NO_EX_TEMPLATE`) before the "If you find no further…" branch.

#### Tool-call I/O pattern (D is IN-distribution for Qwen3.5)

| Phase | Token pattern |
|---|---|
| Model emits (action) | `<tool_call>\n<function=search>\n<parameter=query>\n…\n</parameter>\n</function>\n</tool_call>` (nested-XML, auto-inject-aligned) |
| Pipeline wraps result | `\n<tool_response>{passage}</tool_response>` per chunk (multi-block) |
| Pipeline branch | `qwen35` (same as A; `_is_qwen35` in `active_pipeline.py`) |
| Stop tokens | `</tool_call>` |

D's user message says "call the `search` tool in the format described above" — "above" refers to the auto-injected `# Tools` + `<IMPORTANT>` block in the system message, which spells out the verbatim nested-XML format. Because D inherits A's `tools=[QWEN35_SEARCH_TOOL]` arg, the action + response tags are the exact shape Qwen3.5 was post-trained on. D therefore tests *only* the +decision-rules axis with everything else held in-distribution.

#### Template constant (to be added as `QWEN35_MINIMAL_DECIDE_TEMPLATE`)

```python
QWEN35_MINIMAL_DECIDE_TEMPLATE = (
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

(Diff vs A: two sentences inserted between "You can search as many times as you want." and "If you find no further external knowledge needed,…".)

#### What the model actually sees after `apply_chat_template`

System block is **identical to A's** auto-injected `# Tools` + `<IMPORTANT>` (lines 52–83 above). Only the user message changes:

```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the `search` tool in the format described above and it will return the top searched results inside <tool_response> and </tool_response>. You can search as many times as you want. Use the information in the search results to determine the final answer. After each search result, decide whether another search is needed or whether you can provide the final answer. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

Total prompt token count: ~~445 (~~424 from A + ~21 for the two extra sentences).



---

### E — `qwen35_minimal_nothink` (A with `enable_thinking=False`)

**Routing**: identical to A in everything except `enable_thinking=False` plumbed
through `apply_chat_template`. **Template constant reuses A's `QWEN35_SEARCH_R1_LIKE_TEMPLATE`
byte-for-byte** — no new constant required. The pipeline change is per-mode
`enable_thinking` plumbing (currently hardcoded `True` for both qwen35 modes;
see [HANDOFF §"What you have to build" #2](../todo/HANDOFF_M4.4_2026-05-10.md)).

#### Template constant

Same as A: `QWEN35_SEARCH_R1_LIKE_TEMPLATE` (templates.py:134). Registry-only addition:

```python
QWEN35_TEMPLATES["qwen35_minimal_nothink"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE
```

#### What the model actually sees after `apply_chat_template(..., enable_thinking=False)`

System block is identical to A's auto-injected block (lines 52–83 above). User message is identical to A's. The only diff is the assistant generation prefix — Qwen3.5's chat template emits an empty closed `<think>` block instead of the open-thinking opener `<think>\n`:

```
<|im_start|>user
Answer the given question. ... {full A user message} ... Question: <QUESTION GOES HERE><|im_end|>
<|im_start|>assistant
<think>

</think>
```

Verified empirically against `Qwen/Qwen3.5-0.8B` chat template on bamboogle item 0 (2026-05-10): **A renders to 430 tokens, E to 432 tokens — only +2 tokens** for the closing `</think>` tag, and the assistant prefix lands the model immediately *after* an empty closed think block (no in-think generation possible).

Closing `<think>…</think>` upfront (a) eliminates the in-think entity-hallucination failure path flagged in `[RESULTS_SMOKE_m4.md` §6.5](../report/RESULTS_SMOKE_m4.md), and (b) trims output tokens.



---

### Side-by-side (A–E)


| #     | `prompt_mode`              | Locus  | Tags                              | `tools=[]` | `enable_thinking` | Decision rules | Tokens (~)              | Pipeline branch | Qwen3.5 post-training dist. |
| ----- | -------------------------- | ------ | --------------------------------- | ---------- | ----------------- | -------------- | ----------------------- | --------------- | --------------------------- |
| **A** | `qwen35_minimal` (control) | user   | `<tool_call>` / `<tool_response>` | yes        | True              | no             | ~424                    | qwen35          | **in-dist.**                |
| **B** | `qwen35_searchr1_pure`     | user   | `<search>` / `<information>`      | no         | True              | no             | ~150                    | search_r1       | off-dist. (tag scheme)      |
| **C** | `qwen35_p3_decide_no_ex`   | system | `<search>` / `<result>`           | no         | True              | yes            | ~180                    | qwen3           | off-dist. (tags + locus)    |
| **D** | `qwen35_minimal_decide`    | user   | `<tool_call>` / `<tool_response>` | yes        | True              | yes (2 sent.)  | ~445                    | qwen35          | **in-dist.**                |
| **E** | `qwen35_minimal_nothink`   | user   | `<tool_call>` / `<tool_response>` | yes        | **False**         | no             | ~424 + closed `<think>` | qwen35          | in-dist. (tags); thinking-suppressed mode is documented in chat template but rarely-used on hybrid |


