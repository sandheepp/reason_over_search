# Qwen3 hybrid system-message templates (training-time prompts).
# Each constant is byte-for-byte the system-message used at training time;
# evaluating with a different prompt than training produces off-distribution
# behaviour.

# `p1_basic_w_ex`: basic rules + 2-search Hamlet example.
# Anchors the model on heavy-tool 2-call / 4-turn behaviour (response length
# ~2050 tokens).
QWEN3_0_6B_TEMPLATE = (
    "You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.\n"
    "You can call the search tool by writing: <search> your query </search>\n"
    "You will receive the result in: <result> your search result </result>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Answers should be based on the search results.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>The final answer is \\[ \\boxed{answer here} \\]</answer>.\n"
    "For example:\n"
    "Question: What is the nationality of the author of Hamlet?\n"
    "<search>Hamlet</search>\n"
    "<result>The Tragedy of Hamlet was written by William Shakespeare.</result>\n"
    "<search>William Shakespeare</search>\n"
    "<result>William Shakespeare was an English playwright.</result>\n"
    "<answer>The final answer is \\[ \\boxed{English} \\]</answer>"
)

# `p3_decide_no_ex`: decision-rule scaffolding, NO example.
# Tool-use survives example removal here because the rules section has explicit
# per-step decision guidance; the only prompt of this set for which removing the
# example was harmless.
P3_DECIDE_NO_EX_TEMPLATE = (
    "You are a helpful assistant who can answer questions using a Wikipedia search tool.\n"
    "You can call the search tool by writing: <search> your query </search>\n"
    "You will receive the result in: <result> your search result </result>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Use the information in the search results to determine the final answer.\n"
    "After each search result, decide whether another search is needed or whether you can provide the final answer.\n"
    "If a search result is incomplete, search again for the missing information.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>The final answer is \\[ \\boxed{answer here} \\]</answer>."
)

# Registry: prompt_mode -> system-message template. All `qwen3*` modes share
# the same retrieval-text format / budgets / enable_thinking; only the system
# message differs.
QWEN3_TEMPLATES = {
    "qwen3": QWEN3_0_6B_TEMPLATE,
    "qwen3_p1_basic_w_ex": QWEN3_0_6B_TEMPLATE,  # explicit alias
    "qwen3_p3_decide_no_ex": P3_DECIDE_NO_EX_TEMPLATE,
}

# Qwen3.5 system-message template.
# Canonical Qwen3.5 nested-XML tool use, minimal system message,
# `Question: {q}` user message, plain `<answer>X</answer>` (no `\boxed{}`).
#
# We pass `tools=[QWEN35_SEARCH_TOOL]` to `tokenizer.apply_chat_template`, so
# Qwen3.5's chat template auto-injects (a) the `# Tools` block with the search
# function signature, (b) the verbatim format spec
# (`<tool_call>\n<function=NAME>\n<parameter=ARG>\nVAL\n</parameter>\n</function>\n</tool_call>`),
# and (c) the `<IMPORTANT>` reminder block: verbatim text from the model's
# `tokenizer_config.json:chat_template` field on HF.
# Auto-injection is in Qwen3.5's post-training distribution, so leveraging it
# costs the model zero adaptation vs. the off-distribution flat form.
#
# This template (system role) carries only what the auto-injected block does
# NOT cover: the loop semantics (the user gives a question, the model decides
# whether to search, observes the result, decides whether to answer or search
# again). The format spec itself is auto-injected, so we don't repeat it. No
# few-shot example: we want to see whether the post-training prior alone is
# enough to drive the loop on this 0.8B model.
QWEN35_NATIVE_TEMPLATE = (
    "You are a helpful assistant with access to a `search` tool that retrieves Wikipedia passages. "
    "Always call `search` at least once before answering; do not answer from prior knowledge.\n"
    "\n"
    "The user will give you a question in the form: Question: <question>\n"
    "\n"
    "Steps:\n"
    "- Call `search` with a focused query in the format described above.\n"
    "- Each search call returns several Wikipedia passages, one per "
    "<tool_response>...</tool_response> block. Some passages may be off-topic; "
    "use only the relevant ones.\n"
    "- If the relevant passages contain the answer, write it inside <answer> and </answer> and stop.\n"
    "- Otherwise, refine your query using facts and entities from those relevant passages "
    "(not from prior knowledge) and call `search` again."
)

# OpenAI-style tool schema rendered into the system area by Qwen3.5's chat
# template. Mirror of training/src/chat_template/tools.py:SEARCH_TOOL so the
# eval prompt is byte-identical to the training rollout's qwen_native arm.
# Keep schema small: one `query: str` is all retrieval
# needs, and shallow schemas reduce the `# Tools` token overhead.
QWEN35_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search Wikipedia for passages relevant to the query. Returns the "
            "top-K most relevant chunks. Call this whenever the question "
            "requires factual knowledge you do not already have."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific.",
                },
            },
            "required": ["query"],
        },
    },
}

QWEN35_TEMPLATES = {
    "qwen35": QWEN35_NATIVE_TEMPLATE,
    "qwen35_native": QWEN35_NATIVE_TEMPLATE,  # explicit alias
}

# Search-R1-style minimal user-message prompt for Qwen3.5.
# This template ports Search-R1's compact paper prompt to Qwen3.5
# tags: protocol prose lives in the USER message (Search-R1 style: dense,
# single-paragraph, one-word `<answer> Beijing </answer>` example), and we
# still pass `tools=[QWEN35_SEARCH_TOOL]` so the chat template auto-injects
# the `# Tools` format spec into the system role (in-distribution for
# Qwen3.5's tool-use post-training). System message
# itself is empty: model sees just the auto-injected block, then our user
# message containing the protocol + question.
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

QWEN35_TEMPLATES["qwen35_minimal"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE
QWEN35_TEMPLATES["qwen35_searchr1"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE  # alias

# Variant: same template byte-for-byte as the minimal one; only diff is enable_thinking=False.
# Reuses QWEN35_SEARCH_R1_LIKE_TEMPLATE; run_m4.sh sets enable_thinking=False when
# prompt_mode=qwen35_minimal_nothink, which makes Qwen3.5's chat template emit a
# closed empty `<think>\n\n</think>\n\n` block instead of the open `<think>\n`
# generation prefix. Removes the in-think entity-hallucination failure path.
QWEN35_TEMPLATES["qwen35_minimal_nothink"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE


# Exact mirror of the GRPO training prompt (training_m5_1/src/prompts/m5_qwen35_user.txt).
# Two-line user-message prompt used during GRPO training. Routes through
# the same user-locus branch as qwen35_minimal (tools=[SEARCH_TOOL] auto-inject,
# no system message, enable_thinking=True). The only diff vs qwen35_minimal is
# the user-prompt content: training used this terse prompt, not the Search-R1
# paper-style longer prompt qwen35_minimal carries. Used for in-distribution
# evaluation of trained checkpoints: the policy was trained against this exact
# user message, so eval must match it byte-for-byte to avoid an OOD eval.
M5_QWEN35_TRAIN_TEMPLATE = (
    "Use the `search` tool to look up facts as needed. "
    "When you have the answer, write it inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)

QWEN35_TEMPLATES["qwen35_m5_train"] = M5_QWEN35_TRAIN_TEMPLATE

# Fully-minimal: no system message, no `tools=[]` auto-inject.
# Strips the `tools=[QWEN35_SEARCH_TOOL]` auto-injected `# Tools` schema +
# `<IMPORTANT>` reminder block (~220 words) by NOT passing tools= to
# apply_chat_template. The model now sees ONLY the user message, no system
# block at all. To compensate we inline the verbatim nested-XML format spec
# (lifted from the chat template's auto-inject) so the model still has a
# format anchor; this is technically off-distribution for Qwen3.5's tool-use
# post-training (the auto-inject is what the model was trained on), but it
# tests whether further scaffolding reduction lifts EM. Total prompt
# token count: ~150 (vs ~424 for qwen35_minimal which keeps the auto-inject).
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

QWEN35_TEMPLATES["qwen35_minimal_no_system"] = QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE

# ReCall-port system message (Agent-RL/ReCall prose).
# Verbatim ReCall `re_call_template_sys` (Agent-RL/ReCall, src/verl/utils/dataset/template.py)
# adapted to this prompt lock:
#   - keep `<tool_call>` / `<tool_response>` tags via Qwen3.5 nested-XML (auto-injected by
#     `tools=[QWEN35_SEARCH_TOOL]`); drop ReCall's JSON-args format reminder (auto-inject covers it)
#   - swap `\boxed{}` for `<answer>X</answer>`
#   - drop the `{func_schemas}` placeholder block (the auto-inject already emits the function signature)
#   - replace ReCall's weather/Beijing example with a search-style example (Ted Turner / CNN)
#   - say `search` not "tools" (we have one tool; less abstract for the model)
# System-role: in-distribution for Qwen3.5's hybrid post-training.
# Pipeline routes this through the system + tools=[] branch (active_pipeline._is_qwen35).
QWEN35_RECALL_PORT_TEMPLATE = (
    "In this environment you have access to a `search` tool that retrieves Wikipedia passages. "
    "You may perform multiple rounds of search calls.\n"
    "\n"
    "In your response, first think about the reasoning process inside <think> </think>, "
    "then conduct function calling inside <tool_call> </tool_call> in the format described above. "
    "The results of each search call will be returned inside <tool_response> </tool_response>. "
    "You can continue to call `search` until you have enough information to answer the user's question. "
    "When you have the answer, enclose it inside <answer> </answer> and stop calling functions. "
    "For example: <think> Based on the response from the search call, the founder of CNN is Ted Turner. </think> "
    "<answer> Ted Turner </answer>."
)

QWEN35_TEMPLATES["qwen35_recall_port"] = QWEN35_RECALL_PORT_TEMPLATE

# 10-candidate in-distribution prompt ablation.
# All 10 candidates use `<tool_call>` / `<tool_response>` via tools=[QWEN35_SEARCH_TOOL]
# auto-inject (in-distribution for Qwen3.5 post-training); enable_thinking=True;
# plain <answer>X</answer> (no \boxed{}). The only varying axis is the prompt
# text (and locus: user / system).

# #1 qwen35_terse: length compression. Drop the base template's cargo-culted
# Search-R1 clauses (<think> instructions, "after reasoning...", redundant
# <tool_response> reminder) since auto-inject covers all of those.
QWEN35_TERSE_TEMPLATE = (
    "Use the `search` tool to look up facts as needed. "
    "When you have the answer, write it inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)

# #2 qwen35_decide: base template + the two decision sentences (verbatim from
# P3_DECIDE_NO_EX_TEMPLATE). Same auto-inject routing; user locus preserved.
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

# #3 qwen35_p3_decide_xml: decision-rule shape (system role + decision rules,
# no example) ported to <tool_call>/<tool_response> via auto-inject. Tests
# whether the prose+locus transferred once the off-distribution tag confound
# is removed. Routes through the qwen35 system+`Question: {q}` branch (same as
# recall_port).
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

# #4 qwen35_search_first: base template + retrieval-first guard prepended.
# Targets the bypass-and-fabricate failure mode (empty preds + confident
# hallucinations on completed items).
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

# #5 qwen35_research_role: short system-role role-prime, user is hardcoded
# Question: {q}. Distinguished from recall_port (verbose system) and from the
# base template (no system) by being short-and-role-only.
QWEN35_RESEARCH_ROLE_TEMPLATE = (
    "You are a research assistant. "
    "For every factual question, verify the answer using the `search` tool before responding. "
    "Provide the final answer inside <answer> and </answer>."
)

# #6 qwen35_hamlet_1shot: base template body
# + 2-search Hamlet example with full agentic-loop demonstration (<think>
# planning, <tool_call>, <tool_response>, <think> re-planning, <tool_call>,
# <tool_response>, <think> synthesise, <answer>). Most concrete
# in-distribution demonstration the model can see.
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

# #7 qwen35_decompose: replace the base template prose with explicit 3-stage
# decomposition (identify facts, search each, combine and answer). Explicit
# decomposition is a common recommendation for retrieval tasks.
QWEN35_DECOMPOSE_TEMPLATE = (
    "Answer the given question by following these steps:\n"
    "1. Identify the facts you would need to know to answer the question.\n"
    "2. For each fact, call the `search` tool to look it up "
    "(format described above; results come inside <tool_response> and </tool_response>).\n"
    "3. Combine the search results and write the answer inside <answer> and </answer>.\n"
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)

# #8 qwen35_source_only: base template + source-grounding constraint + uncertainty
# escape (<answer>unknown</answer> if results don't contain the answer). Permits the
# model to express uncertainty.
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

# #9 qwen35_self_check: base template + self-verification step before final answer.
# A self-check at the end can catch hallucinated specifics.
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

# #10 qwen35_multi_search: base template + refine-and-retry encouragement.
# Targets Qwen3.5-0.8B's 1-search bias via an instruction-only intervention.
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

# Registry entries: user-locus templates routed via _QWEN35_USER_PROMPT_MODES
# (see active_pipeline.py); system-locus templates (#3, #5) routed via the
# default _is_qwen35 branch (system + user-Question:{q}, same as recall_port).
QWEN35_TEMPLATES["qwen35_terse"] = QWEN35_TERSE_TEMPLATE
QWEN35_TEMPLATES["qwen35_decide"] = QWEN35_DECIDE_TEMPLATE
QWEN35_TEMPLATES["qwen35_p3_decide_xml"] = QWEN35_P3_DECIDE_XML_TEMPLATE
QWEN35_TEMPLATES["qwen35_search_first"] = QWEN35_SEARCH_FIRST_TEMPLATE
QWEN35_TEMPLATES["qwen35_research_role"] = QWEN35_RESEARCH_ROLE_TEMPLATE
QWEN35_TEMPLATES["qwen35_hamlet_1shot"] = QWEN35_HAMLET_1SHOT_TEMPLATE
QWEN35_TEMPLATES["qwen35_decompose"] = QWEN35_DECOMPOSE_TEMPLATE
QWEN35_TEMPLATES["qwen35_source_only"] = QWEN35_SOURCE_ONLY_TEMPLATE
QWEN35_TEMPLATES["qwen35_self_check"] = QWEN35_SELF_CHECK_TEMPLATE
QWEN35_TEMPLATES["qwen35_multi_search"] = QWEN35_MULTI_SEARCH_TEMPLATE

# Base-variant no-system candidates.
# Both candidates parallel the no-system lock structure (no `tools=[]` auto-inject,
# no system block; format spec inlined into the user message). Routed via
# `_QWEN35_USER_PROMPT_MODES` + `_QWEN35_NO_TOOLS_MODES`.

# Candidate (ii): qwen35_terse_no_system: terse prose with the no-system
# lock structure (no auto-inject, no system). Tests whether the
# intersection "terse user message + no auto-inject" lifts base.
QWEN35_TERSE_NO_SYSTEM_TEMPLATE = (
    "Use the `search` tool to look up facts as needed. "
    "Call the tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "When you have the answer, write it inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_terse_no_system"] = QWEN35_TERSE_NO_SYSTEM_TEMPLATE

# Candidate (iii): qwen35_research_role_no_system: research_role role-prime
# ported to the no-system lock structure. Role-prime
# goes in user role (no system available without `tools=[]`).
QWEN35_RESEARCH_ROLE_NO_SYSTEM_TEMPLATE = (
    "You are a research assistant. "
    "For every factual question, verify the answer using the `search` tool before responding. "
    "Call the tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "Provide the final answer inside <answer> and </answer>. "
    "For example, <answer> Beijing </answer>.\n"
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_research_role_no_system"] = QWEN35_RESEARCH_ROLE_NO_SYSTEM_TEMPLATE

# Fallback near-miss prose variants ported to the no-system structure (no
# auto-inject; format spec inlined). Tests whether the prose interventions
# (decision rules / source grounding / self-verify) lift base when the
# auto-inject scaffolding is gone.

# (B-1) qwen35_decide_no_system: decision rules in user role, no auto-inject.
QWEN35_DECIDE_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "Use the information in the search results to determine the final answer. "
    "After each search result, decide whether another search is needed or whether you can provide the final answer. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_decide_no_system"] = QWEN35_DECIDE_NO_SYSTEM_TEMPLATE

# (B-2) qwen35_source_only_no_system: source-grounding + uncertainty escape, no auto-inject.
QWEN35_SOURCE_ONLY_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want.\n"
    "\n"
    "Use ONLY information returned by the search tool. "
    "If the search results do not contain the answer, write <answer>unknown</answer>.\n"
    "\n"
    "Otherwise, provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_source_only_no_system"] = QWEN35_SOURCE_ONLY_NO_SYSTEM_TEMPLATE

# (B-3) qwen35_self_check_no_system: self-verify before answering, no auto-inject.
QWEN35_SELF_CHECK_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want.\n"
    "\n"
    "Before writing the final answer, briefly verify that it is supported by the search results. "
    "If it is not, refine your query and search again.\n"
    "\n"
    "Then provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_self_check_no_system"] = QWEN35_SELF_CHECK_NO_SYSTEM_TEMPLATE

# Hard imperative scaffold for tool-use survival.
# Tests whether a strong "you MUST call search" opener prevents the GRPO
# tool-call collapse observed under an F1-only reward (tool-call emission
# decaying toward 0% over training). Same body as the no-system base lock but with:
#   - prepended imperative ("You MUST ... Always ... even if you think")
#   - removed "if you find you lack some knowledge" escape clause
#   - removed "If you find no further external knowledge needed" escape clause
#   - "After gathering enough information from search" links the answer to search
# The question: does the prompt shift the initial tool-call propensity high
# enough that GRPO can't easily train it out?
# Companion training file: training_m7_1/src/prompts/m7_3_qwen35_base_user.txt
# Render must be byte-equal between training + eval.
QWEN35_M7_3_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You MUST call the search tool at least once before providing the answer. "
    "Always call search first, even if you think you know the answer.\n"
    "\n"
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "After gathering enough information from search, provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
QWEN35_TEMPLATES["qwen35_m7_3_no_system"] = QWEN35_M7_3_NO_SYSTEM_TEMPLATE

# Search-R1 paper user-message template (baseline reproduction).
SEARCH_R1_TEMPLATE = (
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
