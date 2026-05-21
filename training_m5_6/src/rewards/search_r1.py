"""M5.6 EM-only reward — Search-R1 paper-faithful 0/1 reward on Qwen3.5-0.8B.

Reward formula (Search-R1 §3.4):

    reward(rollout) =
        1.0   if extract_solution(rollout) is not None and em_check(answer, gold) == 1
        0.0   otherwise (no <answer>, format broken, or wrong answer)

No format bonus, no F1 partial credit, no retrieval-hit shaping. Implements
exactly the Search-R1 paper's `r_phi(x, y) = EM(a_pred, a_gold)` choice with
all shaping coefficients defaulted to 0 (which is what the M2 module's
zero-coefficient configuration already does).

This module re-exports `compute_search_r1_reward` from
`training/src/rewards/search_r1.py` (the M2 paper-faithful EM-with-shaping
implementation) with shaping coefficients fixed at 0. The thin re-export is
deliberate: it keeps the M5.6 and M2 code paths byte-identical, removing one
class of train/eval drift bugs.

API kept stable for `training_m5_6/src/environments/search_r1_env.py`:
`compute_search_r1_reward(solution_str, gold)`, `em_check`, `extract_solution`
exported under the same names as M5.1's F1-only reward.

Companion doc: docs/milestone_5/MILESTONE_5_6.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_m2_reward_module():
    """Same circular-import workaround as M5.5 — see notes there."""
    repo_root = Path(__file__).resolve().parents[3]
    m2_path = repo_root / "training" / "src" / "rewards" / "search_r1.py"
    spec = importlib.util.spec_from_file_location("_m2_search_r1", str(m2_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load M2 reward module from {m2_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m2 = _load_m2_reward_module()
em_check = _m2.em_check
extract_solution = _m2.extract_solution
is_valid_sequence = _m2.is_valid_sequence
normalize_answer = _m2.normalize_answer
_m2_compute_reward = _m2.compute_search_r1_reward

__all__ = [
    "compute_search_r1_reward",
    "em_check",
    "extract_solution",
    "is_valid_sequence",
    "normalize_answer",
]


def compute_search_r1_reward(solution_str: str, golden_answers) -> dict:
    """M5.6 EM-only reward — calls the M2 module with all shaping at 0.

    The M2 module's `compute_search_r1_reward` with shaping coefficients at
    0 implements exactly Search-R1's `r(x,y) = EM(a_pred, a_gold)`:
      - format broken: 0.0
      - no <answer>: 0.0
      - format ok, EM=0: 0.0
      - format ok, EM=1: 1.0

    Return dict shape matches M5.1's contract; `f1` is populated for telemetry
    parity (always 1.0 when EM=1 since exact match implies F1=1) by an extra
    quick check downstream if needed. Here we keep f1 implicit (not in the
    dict) since the gradient is on `reward` only.
    """
    out = _m2_compute_reward(
        solution_str=solution_str,
        golden_answers=golden_answers,
        structure_format_score=0.0,
        final_format_score=0.0,
        retrieval_score=0.0,
        score=1.0,
    )
    # Surface a uniform telemetry dict to match the M5.1 / M5.5 shape.
    return {
        "reward": float(out["reward"]),
        "extracted_answer": out["extracted_answer"],
        "f1": 1.0 if out["reward"] == 1.0 else 0.0,
        "em": float(out["reward"]),
        "format_valid": bool(out["format_valid"]),
    }
