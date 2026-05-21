import re
from typing import Tuple


def extract_search_tag_query(text: str) -> Tuple[str, str]:
    # Match Search-R1 generation.py:postprocess_predictions — first match, not last.
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if not match:
        return "", "no_search_tag"
    query = match.group(1).strip()
    if not query:
        return "", "empty_search_query"
    return query, "valid_search_tag"


def extract_tool_call_query(text: str) -> Tuple[str, str]:
    """Pull the `query` arg from a Qwen3.5 nested-XML `<tool_call>` block (M4.1).

    Matches the canonical Qwen3.5 tool-call format that the chat template
    auto-injects when `tools=[QWEN35_SEARCH_TOOL]` is passed to
    `apply_chat_template` (see CHAT_TEMPLATE.md §2):

        <tool_call>
        <function=search>
        <parameter=query>
        QUERY TEXT
        </parameter>
        </function>
        </tool_call>

    Mirror of `training/src/environments/parsers.py:_RE_QWEN_QUERY` so the
    eval-side parser is byte-identical to the training rollout's qwen_native
    arm. Permissive: only the `<parameter=query>` body matters; we do not
    enforce the function name (the SEARCH_TOOL schema only registers `search`,
    so any other function name would be a model error and we'd fall through to
    the corrective branch anyway).
    """
    pattern = re.compile(
        r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return "", "no_tool_call_tag"
    query = match.group(1).strip()
    if not query:
        return "", "empty_tool_call_query"
    return query, "valid_tool_call_tag"
