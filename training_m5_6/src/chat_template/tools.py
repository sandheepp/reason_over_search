"""Tool schemas for the qwen_native chat-template arm (M5.1 overlay).

Re-exports `QWEN35_SEARCH_TOOL` from the M4 eval pipeline
(`evaluation_qwen35/flashrag/search_r1/templates.py`) as `SEARCH_TOOL`, so
the training rollout's tool schema is **the same Python object** that the
eval pipeline renders into prompts. This closes the same train/eval drift
gap as the parser/reward re-exports in `parsers.py` / `rewards/search_r1.py`;
the M3 14-fix audit (CODE_SETUP_m3.md §3) is the precedent.

The schema content: one tool, `search(query: str)`. Pass it as
`tools=[SEARCH_TOOL]` to `tokenizer.apply_chat_template(...)`. The paper
arm doesn't register tools (it bakes the protocol into the prompt text
instead).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Mirror of the sys.path trick used in src/rewards/search_r1.py and
# src/environments/parsers.py — see those for the full rationale.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_QWEN35_ROOT = _REPO_ROOT / "evaluation_qwen35"
if str(_EVAL_QWEN35_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_QWEN35_ROOT))

from flashrag.search_r1.templates import QWEN35_SEARCH_TOOL  # noqa: E402

# Public alias kept stable for the processor's import; the existing M2
# processor expected `SEARCH_TOOL`, not `QWEN35_SEARCH_TOOL`.
SEARCH_TOOL = QWEN35_SEARCH_TOOL
