"""M5.5 F1+format reward — ReSearch's 3-tier shaping, Qwen3.5 nested-XML schema.

Reward formula:

    reward(rollout) =
        0.0            if not is_valid_format(rollout)               # format broken
        FORMAT_BONUS   if extract_solution(rollout) is None          # format ok, no <answer>
        FORMAT_BONUS   if f1(answer, gold) == 0                      # format ok, answer wrong
        f1             if f1(answer, gold) > 0                       # format ok, partial / full

`FORMAT_BONUS = 0.1` matches the ReSearch paper's 3-tier choice and the
Phase-1 v0 observation that this floor masks the tool-use signal. The floor
is NOT additive (F1=0.5 -> reward=0.5, not 0.6); it only applies when F1==0.

The format validator ports ReSearch's `validate_template_format`
(Agent-RL/ReSearch:src/verl/utils/reward_score/re_call.py:101) to the schema
M5 actually emits: Qwen3.5 nested-XML tool-calls (M4.1 canonical form) +
`<answer>...</answer>` instead of `\\boxed{...}`.

ReSearch source-of-truth -> M5.5 mapping:

    ReSearch (math, ReCall arm)         | M5.5 (search-tool, Qwen3.5-0.8B)
    ------------------------------------|-----------------------------------------------
    rollout ends with EOS               | rollout ends with </answer>  (vLLM stop string)
    paired <think> + at least one       | same
    paired <tool_call>                  | same
    <tool_call> content = JSON          | <tool_call> content = nested XML:
      with "name" + "arguments" keys    |   <function=NAME>...<parameter=ARG>VAL</parameter>...</function>
    last response contains \\boxed       | last assistant turn ends with </answer>
    answer = inside \\boxed{...}         | answer = inside the LAST <answer>...</answer>

The Qwen3.5 nested-XML form is what `tools=[QWEN35_SEARCH_TOOL]` auto-injects
into the chat template (the format the model was post-trained on). JSON-in-
<tool_call> would be off-distribution. Cross-reference:
- evaluation_qwen35/flashrag/search_r1/templates.py:54-61 (auto-inject form)
- evaluation_qwen35/flashrag/search_r1/parser.py:39 (extractor regex)

The walker treats the joined assistant+tool content as one document (the M5
env passes `solution_str = "".join(m["content"] for m in log if m["role"] != "user")`,
so chat-template role markers are stripped before reaching us). That means
ReSearch's per-turn checks become total-count checks across the joined text;
the structural invariants we care about (balance, terminal answer, JSON-vs-XML
validity per tool-call) are preserved.

Answer scaffold: `<answer>...</answer>`, NOT `\\boxed{...}`. This is an
intentional carry from M4 (CLAUDE.md "two intentional divergences"). Keeping
it avoids an M3-style train/eval scoring drift with the M4 eval pipeline,
which extracts `<answer>` content via the same `extract_solution` re-export.

Companion doc: docs/milestone_5/MILESTONE_5_5.md.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

# Re-use the M4 eval's extract_solution + normalize_answer. Same precedent as
# M5.1: train and eval code paths must share string normalisation to avoid
# M3 14-fix-style train/eval drift.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_QWEN35_ROOT = _REPO_ROOT / "evaluation_qwen35"
if str(_EVAL_QWEN35_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_QWEN35_ROOT))

from flashrag.search_r1.reward import (  # noqa: E402
    extract_solution,
    normalize_answer,
)

# 0.1 partial-credit floor from the ReSearch paper / Phase-1 v0 observation.
# This is the load-bearing knob the ablation targets.
FORMAT_BONUS: float = 0.1

__all__ = [
    "compute_search_r1_reward",
    "em_check",
    "extract_solution",
    "f1_check",
    "is_valid_format",
    "normalize_answer",
    "FORMAT_BONUS",
]


# Pattern for one Qwen3.5 nested-XML tool-call block content. Inside a
# <tool_call>...</tool_call> wrap, the model emits:
#     <function=NAME>
#     <parameter=ARG_NAME>
#     ARG_VALUE
#     </parameter>
#     ...optional more <parameter=...>...
#     </function>
# Optional surrounding whitespace; one or more <parameter=...> required.
_RE_FUNCTION_BLOCK = re.compile(
    r"<function=[A-Za-z_][A-Za-z0-9_]*>"        # opening <function=name>
    r"(?:\s*<parameter=[A-Za-z_][A-Za-z0-9_]*>.*?</parameter>\s*)+"  # >=1 parameters
    r"</function>",                              # closing </function>
    re.DOTALL,
)

_RE_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def is_valid_format(solution_str: str) -> Tuple[bool, str]:
    """Return `(is_valid, reason)`. Qwen3.5 nested-XML schema with <answer>.

    Checks (mirrors ReSearch's `validate_template_format`, ported to our schema):
      1. <think> open-count == close-count AND count >= 1.
      2. <tool_call> open-count == close-count.
      3. Each <tool_call>...</tool_call> block content matches the Qwen3.5
         nested-XML function-call shape (at least one <function=...> with at
         least one <parameter=...>).
      4. <answer> open-count == close-count AND >= 1 closed block.
      5. The rollout, stripped of trailing whitespace, ends with </answer>
         (terminal answer; corresponds to ReSearch's "last response contains
         \\boxed" + EOS gate, adapted to our `</answer>`-as-vLLM-stop-string
         convention).
    """
    text = solution_str
    stripped = text.rstrip()

    # 1. <think> pairing + presence.
    # Qwen3.5 chat template (enable_thinking=True) auto-prepends `<think>\n`
    # at the start of each assistant turn. The model's generated content (the
    # `solution_str` we receive) only includes content AFTER that prefix, so
    # the opening `<think>` tag is NOT in the rollout text — only the closing
    # `</think>` and any additional explicit opens the model emits mid-content.
    # Effective open count = explicit_opens + template_injected_opens (one per
    # assistant turn = one per closing `</think>`, since the model must close
    # thinking before any tool_call or answer). Validation: model emitted at
    # least one close, and any explicit opens are balanced (each followed by
    # its own close). Single-prepend would only handle single-turn rollouts;
    # multi-turn (K tool calls → K+1 assistant turns → K+1 closes) needs this
    # template-injection-aware accounting.
    think_open = text.count("<think>")
    think_close = text.count("</think>")
    if think_close == 0:
        return False, "no </think> close present (model never closed thinking)"
    if think_open > think_close:
        return False, f"think tags unbalanced: explicit opens={think_open} > closes={think_close}"

    # 2. <tool_call> pairing
    tc_open = text.count("<tool_call>")
    tc_close = text.count("</tool_call>")
    if tc_open != tc_close:
        return False, f"tool_call tags unbalanced: open={tc_open} close={tc_close}"

    # 3. Each <tool_call> block content matches nested-XML function-call shape
    for idx, m in enumerate(_RE_TOOL_CALL_BLOCK.finditer(text)):
        body = m.group(1).strip()
        if not body:
            return False, f"tool_call[{idx}] body is empty"
        if not _RE_FUNCTION_BLOCK.search(body):
            return False, f"tool_call[{idx}] does not contain a valid <function=NAME>...<parameter=...>...</function> block"

    # 4. <answer> pairing + at least one closed block
    a_open = text.count("<answer>")
    a_close = text.count("</answer>")
    if a_open != a_close:
        return False, f"answer tags unbalanced: open={a_open} close={a_close}"
    if a_close == 0:
        return False, "no closed <answer> block"

    # 5. terminal-answer check: stripped rollout ends with </answer>
    if not stripped.endswith("</answer>"):
        return False, "rollout does not terminate on </answer>"

    return True, "ok"


def em_check(prediction: str, golden_answers) -> float:
    """SQuAD-style EM. Kept for telemetry parity with M5.1."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def f1_check(prediction: str, golden_answers) -> float:
    """SQuAD-style token-overlap F1; best over the gold list (multi-answer).

    Same implementation as M5.1; duplicated here so this module is
    self-contained.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for golden in golden_answers:
        gold_tokens = normalize_answer(golden).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def compute_search_r1_reward(solution_str: str, golden_answers) -> dict:
    """M5.5 F1+format reward. Return shape matches M5.1's contract."""
    is_valid, reason = is_valid_format(solution_str)
    if not is_valid:
        return {
            "reward": 0.0,
            "extracted_answer": None,
            "f1": 0.0,
            "em": 0.0,
            "format_valid": False,
            "format_reason": reason,
        }
    answer = extract_solution(solution_str)
    if answer is None:
        return {
            "reward": FORMAT_BONUS,
            "extracted_answer": None,
            "f1": 0.0,
            "em": 0.0,
            "format_valid": True,
            "format_reason": reason,
        }
    f1 = f1_check(answer, golden_answers)
    em = em_check(answer, golden_answers)
    reward = f1 if f1 > 0.0 else FORMAT_BONUS
    return {
        "reward": float(reward),
        "extracted_answer": answer,
        "f1": float(f1),
        "em": float(em),
        "format_valid": True,
        "format_reason": reason,
    }
