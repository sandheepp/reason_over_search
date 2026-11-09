"""Parser dispatch — `parse_query` must extract the right query per arm.

Imports from `training.src.environments.parsers` (pure-Python, no torch/ray
deps), so this test runs in any venv.
"""
import pytest

from training_m5_1.src.environments.parsers import parse_query


# ---------- qwen_native arm ----------

QWEN_VALID = """<think>I should search.</think>
<tool_call>
<function=search>
<parameter=query>
who founded SpaceX
</parameter>
</function>
</tool_call>"""

QWEN_NO_TOOL_CALL = "<think>I know this.</think><answer>Elon Musk</answer>"

QWEN_EMPTY_QUERY = """<tool_call>
<function=search>
<parameter=query>

</parameter>
</function>
</tool_call>"""

QWEN_TWO_TOOL_CALLS = """<tool_call>
<function=search>
<parameter=query>
first query
</parameter>
</function>
</tool_call>
<tool_call>
<function=search>
<parameter=query>
second query
</parameter>
</function>
</tool_call>"""


def test_qwen_native_extracts_query():
    assert parse_query("qwen_native", QWEN_VALID) == "who founded SpaceX"


def test_qwen_native_returns_none_when_absent():
    assert parse_query("qwen_native", QWEN_NO_TOOL_CALL) is None


def test_qwen_native_treats_blank_query_as_none():
    assert parse_query("qwen_native", QWEN_EMPTY_QUERY) is None


def test_qwen_native_takes_first_when_multiple():
    # vLLM should stop at the first `</tool_call>` so this is unlikely in
    # practice, but if it does happen we lock in "first match" behavior.
    assert parse_query("qwen_native", QWEN_TWO_TOOL_CALLS) == "first query"


# ---------- paper arm ----------

PAPER_VALID = "<think>Need a fact.</think><search>SpaceX founder</search>"
PAPER_NO_SEARCH = "<think>I know.</think><answer>Elon Musk</answer>"
PAPER_EMPTY = "<search></search>"
PAPER_TWO_SEARCHES = "<search>first</search><search>second</search>"


def test_paper_extracts_query():
    assert parse_query("paper", PAPER_VALID) == "SpaceX founder"


def test_paper_returns_none_when_absent():
    assert parse_query("paper", PAPER_NO_SEARCH) is None


def test_paper_treats_blank_query_as_none():
    assert parse_query("paper", PAPER_EMPTY) is None


def test_paper_takes_first_when_multiple():
    assert parse_query("paper", PAPER_TWO_SEARCHES) == "first"


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        parse_query("nonexistent", "anything")
