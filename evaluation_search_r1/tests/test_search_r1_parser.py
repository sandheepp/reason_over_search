from flashrag.search_r1.parser import extract_search_tag_query


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

