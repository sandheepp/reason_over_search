"""M5.1 reward semantics + train/eval shared-path parity.

Two flavors of assertion:

1. Shared-path parity: `normalize_answer` / `extract_solution` / `em_check`
   are re-exported from `evaluation_qwen35/flashrag/search_r1/reward.py`, so
   train- and eval-time scoring share one code path. The tests below pin
   that down so a future refactor that forgets the re-export trips here
   rather than at smoke time.

2. F1-only semantics: the milestone (docs/milestone_5/MILESTONE_5.md
   §"Run sequence" M5.1 step 6) requires re-confirming the reward path on
   5 hand-picked rollouts (correct, partial-overlap, wrong, empty,
   format-broken; F1 should be 1.0 / 0<F1<1 / 0 / 0 / 0). Those live as the
   `CASES_F1` table below.
"""
from training_m5_1.src.rewards import search_r1 as ours
from flashrag.search_r1 import reward as eval_side  # type: ignore[import-not-found]

GOLD = ["Wilhelm Conrad Röntgen", "Röntgen"]


# ----- 1. Shared-path parity (re-exports must not silently diverge) -----

def test_normalize_answer_is_reexported():
    """ours.normalize_answer is eval_side.normalize_answer (same callable)."""
    assert ours.normalize_answer is eval_side.normalize_answer


def test_extract_solution_is_reexported():
    """ours.extract_solution is eval_side.extract_solution (same callable)."""
    assert ours.extract_solution is eval_side.extract_solution


def test_em_check_byte_parity():
    """ours.em_check is a local re-implementation but produces identical output."""
    for pred in ("Wilhelm Conrad Röntgen", "wilhelm conrad röntgen", "Marie Curie", ""):
        assert ours.em_check(pred, GOLD) == eval_side.em_check(pred, GOLD)


# ----- 2. F1-only semantics (the milestone's 5 hand-picked rollouts) -----

# (case_name, rollout_text, gold, expected_f1_assertion)
CASES_F1 = [
    # 1. correct: exact match -> F1 = 1.0
    ("correct", "<answer>Wilhelm Conrad Röntgen</answer>", GOLD, "eq_1"),
    # 2. partial-overlap: shared tokens but not exact -> 0 < F1 < 1
    ("partial", "<answer>Wilhelm Röntgen Wilhelm</answer>", GOLD, "between"),
    # 3. wrong: zero token overlap -> F1 = 0
    ("wrong", "<answer>Marie Curie</answer>", GOLD, "eq_0"),
    # 4. empty: <answer></answer> -> F1 = 0
    ("empty", "<answer></answer>", GOLD, "eq_0"),
    # 5. format-broken: no <answer> tag at all -> F1 = 0
    ("format_broken", "I think the answer is Wilhelm Conrad Röntgen", GOLD, "eq_0"),
]


def test_f1_reward_semantics():
    for name, text, gold, kind in CASES_F1:
        out = ours.compute_search_r1_reward(text, gold)
        f1 = out["reward"]
        if kind == "eq_1":
            assert f1 == 1.0, f"{name}: expected F1=1.0, got {f1}"
        elif kind == "eq_0":
            assert f1 == 0.0, f"{name}: expected F1=0.0, got {f1}"
        elif kind == "between":
            assert 0.0 < f1 < 1.0, f"{name}: expected 0<F1<1, got {f1}"
        else:
            raise AssertionError(f"unknown kind {kind!r}")


def test_f1_reward_dict_keys():
    """Stable return shape; the env reads `reward`, telemetry reads `f1` / `em`."""
    out = ours.compute_search_r1_reward("<answer>Röntgen</answer>", GOLD)
    assert set(out.keys()) == {"reward", "extracted_answer", "f1", "em"}
    # Reward IS F1 under M5.1 (no shaping).
    assert out["reward"] == out["f1"]


def test_f1_picks_best_over_multi_gold():
    """Multi-answer gold: F1 = max over the list."""
    multi_gold = ["Wilhelm Conrad Röntgen", "Röntgen"]
    out = ours.compute_search_r1_reward("<answer>Röntgen</answer>", multi_gold)
    assert out["reward"] == 1.0  # matches "Röntgen" exactly
