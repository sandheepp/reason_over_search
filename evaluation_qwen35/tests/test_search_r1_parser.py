from flashrag.search_r1.parser import extract_search_tag_query, extract_tool_call_query


def test_extract_search_tag_query_success():
    text = "<think>Need info</think><search>capital of france</search>"
    query, status = extract_search_tag_query(text)
    assert status == "valid_search_tag"
    assert query == "capital of france"


def test_extract_search_tag_query_missing_tag():
    query, status = extract_search_tag_query("<think>no search</think>")
    assert status == "no_search_tag"
    assert query == ""


def test_extract_search_tag_query_empty_query():
    query, status = extract_search_tag_query("<search>   </search>")
    assert status == "empty_search_query"
    assert query == ""


def test_extract_tool_call_query_success():
    text = "<think>Need info</think><tool_call>capital of france</tool_call>"
    query, status = extract_tool_call_query(text)
    assert status == "valid_tool_call_tag"
    assert query == "capital of france"


def test_extract_tool_call_query_missing_tag():
    query, status = extract_tool_call_query("<think>no tool</think>")
    assert status == "no_tool_call_tag"
    assert query == ""


def test_extract_tool_call_query_empty_query():
    query, status = extract_tool_call_query("<tool_call>   </tool_call>")
    assert status == "empty_tool_call_query"
    assert query == ""


def test_extract_tool_call_query_with_newlines():
    # Model often puts the query on its own line because the prompt example does.
    text = "<tool_call>\nWilliam Shakespeare nationality\n</tool_call>"
    query, status = extract_tool_call_query(text)
    assert status == "valid_tool_call_tag"
    assert query == "William Shakespeare nationality"


def test_extract_tool_call_query_only_first_match():
    # Multi-turn rollout: parser returns the first match, mirroring the search-tag parser.
    text = "<tool_call>q1</tool_call> <tool_response>r1</tool_response> <tool_call>q2</tool_call>"
    query, status = extract_tool_call_query(text)
    assert status == "valid_tool_call_tag"
    assert query == "q1"

