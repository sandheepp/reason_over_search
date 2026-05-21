"""M5.1 F1-only reward for the Qwen3.5-0.8B GRPO run on the ReSearch recipe.

Replaces the M2 paper-faithful EM-with-shaping reward (still preserved at
`training/src/rewards/search_r1.py`) with token-level F1 against the gold
answer extracted from the **first** `<answer>...</answer>` block in the
rollout. No format reward, no `\\boxed{}` wrap; see
`docs/milestone_5/MILESTONE_5.md` §"Reward function (M5.1)" for the two
intentional divergences from the ReSearch paper recipe.

    reward(rollout) = f1(extract_solution(rollout), gold_answer)

`normalize_answer` and `extract_solution` are re-exported from
`evaluation_qwen35.flashrag.search_r1.reward` so that the M4 eval-time
scorer and the M5 train-time reward share the **same code path**, not the
same logic re-implemented. The M3 14-fix audit (CODE_SETUP_m3.md §3) is the
precedent for why train/eval drift is dangerous; re-exporting closes that
gap by construction.

API kept stable for `training_m5_1/src/environments/search_r1_env.py`:
`compute_search_r1_reward(solution_str, gold)`, `em_check`, and
`extract_solution` are exported under the same names as the M2 reward.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# The M4 eval pipeline lives at `<repo-root>/evaluation_qwen35/`, with the
# `flashrag` package as its child. Its inner `__init__.py` files use
# non-relative imports (`from flashrag.search_r1...`), so the directory that
# must be on sys.path is `evaluation_qwen35/`, not the repo root. Inject it
# here so this overlay is importable from any entry point (launcher, pytest,
# REPL). If another `flashrag` is also installed in the venv, the first import
# wins — we deliberately put evaluation_qwen35's flashrag in front.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_QWEN35_ROOT = _REPO_ROOT / "evaluation_qwen35"
if str(_EVAL_QWEN35_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_QWEN35_ROOT))

from flashrag.search_r1.reward import (  # noqa: E402
    extract_solution,
    normalize_answer,
)

__all__ = [
    "compute_search_r1_reward",
    "em_check",
    "extract_solution",
    "f1_check",
    "normalize_answer",
]


def em_check(prediction: str, golden_answers) -> float:
    """SQuAD-style EM. Kept for the env's truncation fallback path."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def f1_check(prediction: str, golden_answers) -> float:
    """SQuAD-style token-overlap F1; best over the gold list (multi-answer).

    Token = whitespace-split, after `normalize_answer` (lowercase, strip
    punctuation, strip articles a/an/the, collapse whitespace). No stop-word
    strip, matching Search-R1 / ReSearch convention. Pred and gold both empty
    yields F1=1.0 (degenerate but well-defined); one empty yields 0.0.
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
    """M5.1 F1-only reward; drop-in for the M2 reward signature.

    Returns the same dict shape as the M2 version (the env reads `.reward`),
    minus the format-validity diagnostics which are not part of M5.1's
    contract. `em` is reported for telemetry only; the policy is optimised
    against `reward` (= `f1`).
    """
    answer = extract_solution(solution_str)
    if answer is None:
        return {
            "reward": 0.0,
            "extracted_answer": None,
            "f1": 0.0,
            "em": 0.0,
        }
    f1 = f1_check(answer, golden_answers)
    em = em_check(answer, golden_answers)
    return {
        "reward": float(f1),
        "extracted_answer": answer,
        "f1": float(f1),
        "em": float(em),
    }
