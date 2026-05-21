"""Pure-Python parsing & formatting helpers for the Search-R1 env.

Split out so unit tests can import these without dragging in `torch`, `ray`, or
`nemo_rl.*`. The env actor (search_r1_env.py) re-exports and uses these.
"""
from __future__ import annotations

import re
from typing import Optional

VALID_ARMS = ("qwen_native", "paper")

# DOTALL so newlines inside content blocks match `.`.
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RE_PAPER_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL)
# Permissive parser for Qwen3.5 tool calls — only `<parameter=query>` matters.
_RE_QWEN_QUERY = re.compile(
    r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
    re.DOTALL,
)


def parse_query(arm: str, assistant_text: str) -> Optional[str]:
    """Pull the latest search query out of an assistant message; None if absent."""
    if arm == "qwen_native":
        m = _RE_QWEN_QUERY.search(assistant_text)
    elif arm == "paper":
        m = _RE_PAPER_SEARCH.search(assistant_text)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if not m:
        return None
    q = m.group(1).strip()
    return q or None


# Per-observation char cap. Two semantics depending on the format:
#   - format_docs_paper: TOTAL cap across the whole `<information>` body (matches
#     verl's `max_obs_length=500` token cap on the joined Search-R1 paper body;
#     ~500 tokens × 4 chars/token ≈ 2000 chars).
#   - format_docs_qwen_native: PER-CHUNK cap inside each `<tool_response>` block
#     (M4.1 v3 multi-block format; ~120 tokens × 4 ≈ 480 chars/chunk).
# We use char proxies (~4 chars/token for English Wiki text) instead of a real
# tokenizer so parsers.py stays pure-Python and unit-testable without dragging in
# transformers.
DEFAULT_MAX_OBS_CHARS = 2000           # paper-arm total
DEFAULT_MAX_OBS_CHARS_PER_CHUNK = 480  # qwen_native per-chunk (≈120 tokens)


def _truncate_body(body: str, max_chars: int | None) -> str:
    if max_chars is None or len(body) <= max_chars:
        return body
    # Truncate hard, append a marker so the LM knows it's been cut. The marker
    # itself fits inside the cap (subtract its length).
    marker = "\n…[truncated]"
    keep = max(0, max_chars - len(marker))
    return body[:keep] + marker


def format_docs_paper(docs: list[str], max_chars: int | None = DEFAULT_MAX_OBS_CHARS) -> str:
    """Wrap retrieved docs in the paper's `<information>` envelope.

    `max_chars` caps the docs body (numbered Doc lines) before wrapping —
    matches the *intent* of verl's `max_obs_length=500` token cap, using a
    char proxy. Set to None to disable.
    """
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    body = _truncate_body(body, max_chars)
    return f"\n<information>\n{body}\n</information>\n"


def format_docs_qwen_native(
    docs: list[str], max_chars_per_chunk: int | None = DEFAULT_MAX_OBS_CHARS_PER_CHUNK
) -> str:
    """Hand-craft the Qwen3.5 chat-template markers for a tool response.

    The previous assistant turn ended at `</tool_call>` (vLLM stop string),
    no `<|im_end|>` was emitted. We close the assistant turn, open a synthetic
    user turn carrying one `<tool_response>...</tool_response>` block PER doc,
    close the synthetic user turn, and re-open assistant so the next
    generation continues in-distribution. This per-doc multi-block shape is
    what `tokenizer.apply_chat_template([{"role":"tool","content":d}, ...])`
    produces against Qwen3.5's chat template, so the rollout stays canonical
    (verified on Qwen3.5-0.8B; see eval-side mirror in
    evaluation_qwen35/flashrag/pipeline/active_pipeline.py qwen35 branch).

    `max_chars_per_chunk` caps EACH `<tool_response>` block independently
    (M4.1 v3 semantics): every chunk reaches the model regardless of any
    single chunk's length, and oversize chunks get a `…[truncated]` marker.
    This avoids the v2 failure mode where a long chunk consumed the whole
    budget and starved later chunks.
    """
    blocks: list[str] = []
    for d in docs:
        if max_chars_per_chunk is None or len(d) <= max_chars_per_chunk:
            blocks.append(f"<tool_response>\n{d}\n</tool_response>")
        else:
            truncated = _truncate_body(d, max_chars_per_chunk)
            blocks.append(f"<tool_response>\n{truncated}\n</tool_response>")
    body = "\n".join(blocks)
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def retriever_failed_message(
    arm: str, reason: str, max_chars: int | None = DEFAULT_MAX_OBS_CHARS
) -> str:
    """Wrap a retrieval failure as a tool/info response so the LM can recover."""
    msg = f"Retriever failed: {reason}"
    if arm == "qwen_native":
        # Single-chunk failure msg; pass the (total) max_chars as the per-chunk cap —
        # equivalent for n=1 doc and keeps the API caller-stable.
        return format_docs_qwen_native([msg], max_chars_per_chunk=max_chars)
    return format_docs_paper([msg], max_chars=max_chars)
