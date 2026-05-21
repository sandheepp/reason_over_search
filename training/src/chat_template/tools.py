"""Tool schemas for the qwen_native chat-template arm.

Qwen3.5's chat template walks an OpenAI-style `tools` list and renders each
tool call as `<tool_call><function=NAME><parameter=ARG>VAL</parameter>
</function></tool_call>` (verbatim jinja in docs/training/CHAT_TEMPLATE.md §2).

This module exposes `SEARCH_TOOL` — the single tool the rollout registers when
`chat_template.arm == "qwen_native"`. Pass it as `tools=[SEARCH_TOOL]` to
`tokenizer.apply_chat_template(...)`. The paper arm doesn't register tools —
it just bakes the instructions into the prompt template
(prompts/search_r1_paper.txt).

Keep the schema small. Qwen3.5 was post-trained on Hermes-XML tool use, so
deeply-nested parameter schemas are in-distribution, but for retrieval one
`query: str` is all we need.
"""

SEARCH_TOOL = {
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
