from flashrag.search_r1.reward import compute_search_r1_reward, extract_solution


def test_search_r1_reward_valid_and_correct():
    text = (
        "<think>Need info</think>"
        "<search>capital of france</search>"
        "<information>Paris is the capital of France.</information>"
        "<think>Done</think>"
        "<answer>France</answer>"
    )
    result = compute_search_r1_reward(text, ["France"])
    assert result["reward"] == 1.0
    assert result["format_valid"] is True
    assert result["retrieval_hit"] is True
    assert result["extracted_answer"] == "France"


def test_search_r1_reward_format_fallback_score():
    # Mismatched <think> tags → invalid format → wrong EM → 0.0 (pure-EM mode;
    # final_format_score defaults to 0).
    text = "<think>foo<answer>wrong</answer>"
    result = compute_search_r1_reward(text, ["France"])
    assert result["format_valid"] is False
    assert result["reward"] == 0.0


def test_extract_solution_returns_last_answer():
    text = "<answer>first</answer> filler <answer> Paris </answer>"
    assert extract_solution(text) == "Paris"


def test_extract_solution_none_when_missing():
    assert extract_solution("no tags here") is None
