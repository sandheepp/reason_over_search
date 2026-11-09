"""Format helpers — observation strings the env emits per arm.

Locked-in formats: paper arm wraps in `<information>...</information>` raw,
matching the eval pipeline's expectation (so `compute_search_r1_reward`'s
`is_retrieval_correct` can scan retrieval blocks). Qwen-native arm hand-
constructs Qwen3.5 chat-template markers because the rollout loop
(rollouts.py:498-501) does NOT re-run the chat template on env observations.
"""
from training_m5_6.src.environments.parsers import (
    DEFAULT_MAX_OBS_CHARS,
    DEFAULT_MAX_OBS_CHARS_PER_CHUNK,
    format_docs_paper,
    format_docs_qwen_native,
    retriever_failed_message,
)

DOCS = ["First doc.", "Second doc."]


def test_paper_format_wraps_in_information_tag():
    out = format_docs_paper(DOCS)
    assert out.startswith("\n<information>\n")
    assert out.endswith("</information>\n")
    assert "Doc 1: First doc." in out
    assert "Doc 2: Second doc." in out


def test_qwen_native_format_has_full_chat_markers():
    out = format_docs_qwen_native(DOCS)
    # Closes the assistant turn (which ended at </tool_call>) and opens the
    # synthetic user turn. First chunk gets its own <tool_response> block.
    assert out.startswith("<|im_end|>\n<|im_start|>user\n<tool_response>\n")
    # Closes the synthetic user turn AND re-opens assistant for next-turn gen.
    assert out.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n")
    # Each doc is inside its OWN <tool_response> block (M4.1 v2 multi-block
    # canonical Qwen3.5 shape; no "Doc i:" prefix).
    assert out.count("<tool_response>") == len(DOCS)
    assert out.count("</tool_response>") == len(DOCS)
    assert "<tool_response>\nFirst doc.\n</tool_response>" in out
    assert "<tool_response>\nSecond doc.\n</tool_response>" in out
    # No legacy "Doc i:" numbering inside qwen_native blocks.
    assert "Doc 1:" not in out
    assert "Doc 2:" not in out


def test_retriever_failed_message_routes_per_arm():
    paper_msg = retriever_failed_message("paper", "timeout")
    assert "<information>" in paper_msg
    assert "Retriever failed: timeout" in paper_msg

    qwen_msg = retriever_failed_message("qwen_native", "timeout")
    assert "<tool_response>" in qwen_msg
    assert "Retriever failed: timeout" in qwen_msg


# ---------- max_obs_chars cap (verl's max_obs_length=500 tokens, char proxy) ----------

def test_short_docs_pass_through_uncapped():
    out = format_docs_paper(DOCS, max_chars=DEFAULT_MAX_OBS_CHARS)
    # No truncation marker for tiny docs.
    assert "[truncated]" not in out


def test_paper_format_truncates_oversize_body():
    huge = ["X" * 5000]  # one massive doc
    out = format_docs_paper(huge, max_chars=200)
    # Wrapper tags fully present...
    assert out.startswith("\n<information>\n")
    assert out.endswith("</information>\n")
    # ...but the body inside was capped.
    inner = out.split("<information>\n", 1)[1].split("\n</information>", 1)[0]
    assert len(inner) <= 200
    assert inner.endswith("…[truncated]")


def test_qwen_native_format_truncates_oversize_body():
    huge = ["X" * 5000]
    out = format_docs_qwen_native(huge, max_chars_per_chunk=200)
    # Markers fully present.
    assert out.startswith("<|im_end|>\n<|im_start|>user\n<tool_response>\n")
    assert out.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n")
    # Body capped.
    inner = out.split("<tool_response>\n", 1)[1].split("\n</tool_response>", 1)[0]
    assert len(inner) <= 200
    assert inner.endswith("…[truncated]")


def test_qwen_native_per_chunk_cap_applies_to_each_chunk_independently():
    """M4.1 v3 semantics: PER-CHUNK cap, not total. Every doc reaches the
    model regardless of any single doc's length; only the oversize ones get
    the truncation marker."""
    docs = ["short doc", "X" * 5000, "another short doc"]
    out = format_docs_qwen_native(docs, max_chars_per_chunk=100)
    # All three docs produced their own block.
    assert out.count("<tool_response>") == 3
    assert out.count("</tool_response>") == 3
    # Short docs pass through whole.
    assert "<tool_response>\nshort doc\n</tool_response>" in out
    assert "<tool_response>\nanother short doc\n</tool_response>" in out
    # Oversize doc truncated with marker (not dropped).
    assert "…[truncated]" in out


def test_max_chars_none_disables_cap():
    huge = ["X" * 5000]
    out = format_docs_paper(huge, max_chars=None)
    assert "[truncated]" not in out
    assert "X" * 5000 in out
