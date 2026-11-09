"""Env step tests against a mocked retriever.

Skips gracefully when nemo_rl / torch / requests aren't installed (i.e. when
running outside the training venv). On Vast / inside the NeMo-RL venv these
will execute against the real types.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_rl")
pytest.importorskip("requests")

# Import after skip-guards so test collection doesn't blow up in plain venvs.
from training_m5_6.src.environments.search_r1_env import SearchR1Env  # noqa: E402


# ---------- helpers ----------

def _make_env(arm: str = "qwen_native", **overrides):
    cfg = {
        "arm": arm,
        "retriever_url": "http://test-retriever:1234",
        "top_n": 2,
        "max_turns": 4,
        "request_timeout_s": 5.0,
    }
    cfg.update(overrides)
    return SearchR1Env(cfg)


def _mock_retriever(docs_per_query: list[list[dict]]):
    """Build a context manager that patches requests.post on the env module.

    `docs_per_query` is the response body — one list[Document] per submitted query,
    where Document is `{"id": str, "contents": str}`.
    """

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return docs_per_query

    return patch(
        "training_m5_6.src.environments.search_r1_env.requests.post",
        return_value=_Resp(),
    )


# ---------- search path ----------

def test_qwen_native_search_emits_tool_response_and_continues():
    env = _make_env("qwen_native")
    log = [
        {"role": "user", "content": "<initial prompt>"},
        {"role": "assistant", "content": (
            "<think>Search.</think>"
            "<tool_call><function=search>"
            "<parameter=query>SpaceX founder</parameter>"
            "</function></tool_call>"
        )},
    ]
    metadata = {"ground_truth": ["Elon Musk"], "turn_count": 0}

    fake_docs = [[
        {"id": "1", "contents": "Elon Musk founded SpaceX in 2002."},
        {"id": "2", "contents": "SpaceX is a private aerospace company."},
    ]]
    with _mock_retriever(fake_docs):
        out = env.step([log], [metadata])

    assert out.terminateds[0].item() is False
    assert out.rewards[0].item() == 0.0
    assert "<tool_response>" in out.observations[0]["content"]
    assert "Elon Musk founded SpaceX in 2002." in out.observations[0]["content"]
    assert "<|im_start|>assistant" in out.observations[0]["content"]
    assert out.next_stop_strings[0] == ["</tool_call>", "</answer>"]
    assert out.metadata[0]["turn_count"] == 1


def test_paper_search_emits_information_block():
    env = _make_env("paper")
    log = [
        {"role": "user", "content": "<paper-format prompt>"},
        {"role": "assistant", "content": "<think>Search.</think><search>SpaceX founder</search>"},
    ]
    metadata = {"ground_truth": ["Elon Musk"], "turn_count": 0}

    fake_docs = [[{"id": "1", "contents": "Elon Musk founded SpaceX."}]]
    with _mock_retriever(fake_docs):
        out = env.step([log], [metadata])

    content = out.observations[0]["content"]
    assert content.startswith("\n<information>\n")
    assert content.endswith("</information>\n")
    assert "Elon Musk founded SpaceX." in content
    assert out.next_stop_strings[0] == ["</search>", "</answer>"]
    assert out.terminateds[0].item() is False


# ---------- answer path (terminal) ----------

def test_answer_terminates_with_correct_em_paper():
    env = _make_env("paper")
    log = [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<think>I know.</think><answer>Elon Musk</answer>"},
    ]
    metadata = {"ground_truth": ["Elon Musk"], "turn_count": 0}

    out = env.step([log], [metadata])
    assert out.terminateds[0].item() is True
    # Pure-EM mode (shaping coefs = 0): correct EM → reward 1.0 regardless of
    # whether the format walker accepts this exact shape.
    assert out.rewards[0].item() == 1.0
    assert out.answers[0] == "Elon Musk"


def test_answer_terminates_with_zero_em_when_wrong():
    env = _make_env("qwen_native")
    log = [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<answer>Steve Jobs</answer>"},
    ]
    metadata = {"ground_truth": ["Elon Musk"], "turn_count": 0}

    out = env.step([log], [metadata])
    assert out.terminateds[0].item() is True
    # Wrong EM + format-invalid → 0.0 in pure-EM mode (final_format_score=0).
    assert out.rewards[0].item() == 0.0
    assert out.answers[0] == "Steve Jobs"


# ---------- max-turn enforcement ----------

def test_search_at_max_turns_minus_one_is_exhausted():
    env = _make_env("qwen_native", max_turns=4)
    log = [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": (
            "<tool_call><function=search><parameter=query>q</parameter></function></tool_call>"
        )},
    ]
    # turn_count=3 means the next step would be turn 4 == max_turns; classify exhausted.
    metadata = {"ground_truth": ["X"], "turn_count": 3}

    # No retriever call should happen — assert the mock would never fire.
    with _mock_retriever([]):
        out = env.step([log], [metadata])

    assert out.terminateds[0].item() is True
    assert out.rewards[0].item() == 0.0
    assert out.next_stop_strings[0] is None


# ---------- batched retrieval ----------

def test_batched_retrieval_one_post_for_n_queries():
    env = _make_env("qwen_native")
    logs = [
        [
            {"role": "user", "content": "<p>"},
            {"role": "assistant", "content": (
                "<tool_call><function=search><parameter=query>q1</parameter></function></tool_call>"
            )},
        ],
        [
            {"role": "user", "content": "<p>"},
            {"role": "assistant", "content": "<answer>final</answer>"},  # terminal — no retrieval
        ],
        [
            {"role": "user", "content": "<p>"},
            {"role": "assistant", "content": (
                "<tool_call><function=search><parameter=query>q3</parameter></function></tool_call>"
            )},
        ],
    ]
    metadata = [
        {"ground_truth": ["a"], "turn_count": 0},
        {"ground_truth": ["b"], "turn_count": 0},
        {"ground_truth": ["c"], "turn_count": 0},
    ]
    fake_docs = [
        [{"id": "1", "contents": "doc-for-q1"}],
        [{"id": "2", "contents": "doc-for-q3"}],
    ]
    with patch(
        "training_m5_6.src.environments.search_r1_env.requests.post",
        return_value=type("R", (), {"raise_for_status": lambda self: None,
                                     "json": lambda self: fake_docs})(),
    ) as mocked:
        out = env.step(logs, metadata)
        assert mocked.call_count == 1
        # Only the two non-terminal samples sent queries.
        call_kwargs = mocked.call_args.kwargs
        assert call_kwargs["json"]["query"] == ["q1", "q3"]

    assert out.terminateds.tolist() == [False, True, False]
    assert "doc-for-q1" in out.observations[0]["content"]
    assert "doc-for-q3" in out.observations[2]["content"]
