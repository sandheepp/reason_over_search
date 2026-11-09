"""M5.6 EM-only reward — Search-R1 paper-faithful 0/1 reward.

The module re-exports the M2 paper-faithful reward
(`training/src/rewards/search_r1.py`) with all shaping coefficients defaulted
to 0; that reduces the multi-tier reward to pure EM. This file pins:

  1. Reward semantics on Qwen3.5 nested-XML rollouts:
       EM match -> 1.0
       wrong / partial / empty / missing answer -> 0.0
  2. `normalize_answer` behaviour parity with the M4 eval pipeline (so the
     M5 train-time scoring and M4 eval-time scoring agree on string
     equivalence even though M5.6 re-exports the M2 callable, not the M4 one).
  3. Multi-gold extraction (any match -> 1.0).
  4. Telemetry shape (env reads `reward`; pin the others).
"""
from training_m5_6.src.rewards import search_r1 as ours

TC = (
    "<tool_call>\n"
    "<function=search>\n"
    "<parameter=query>\n"
    "author of Hamlet\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
TR = "<tool_response>\nShakespeare wrote Hamlet.\n</tool_response>"


def _full_rollout(answer: str) -> str:
    return (
        f"<think>plan</think>\n"
        f"{TC}\n{TR}\n"
        f"<think>got it</think>\n"
        f"<answer>{answer}</answer>"
    )


# ----- 1. Reward semantics (pure 0/1 EM) ---------------------------------

def test_reward_em_match_returns_one():
    out = ours.compute_search_r1_reward(_full_rollout("Shakespeare"), ["Shakespeare"])
    assert out["reward"] == 1.0
    assert out["em"] == 1.0


def test_reward_em_match_after_normalize():
    """EM is normalised (lowercase + strip articles + strip punct + collapse whitespace)."""
    out = ours.compute_search_r1_reward(
        _full_rollout("wilhelm conrad röntgen"), ["Wilhelm Conrad Röntgen"]
    )
    assert out["reward"] == 1.0


def test_reward_wrong_answer_returns_zero():
    out = ours.compute_search_r1_reward(_full_rollout("Bacon"), ["Shakespeare"])
    assert out["reward"] == 0.0
    assert out["em"] == 0.0


def test_reward_partial_match_returns_zero():
    """Strict EM rejects partial matches — even if token-overlap F1 would be > 0."""
    out = ours.compute_search_r1_reward(
        _full_rollout("William Shakespeare"), ["Shakespeare"]
    )
    assert out["reward"] == 0.0


def test_reward_empty_answer_returns_zero():
    """`<answer></answer>` extracts to empty string; em vs non-empty gold = 0."""
    out = ours.compute_search_r1_reward(_full_rollout(""), ["Shakespeare"])
    assert out["reward"] == 0.0


def test_reward_missing_answer_returns_zero():
    """No <answer> block in the rollout -> no extracted answer -> 0."""
    text = f"<think>x</think>{TC}{TR}"
    out = ours.compute_search_r1_reward(text, ["Shakespeare"])
    assert out["reward"] == 0.0
    assert out["extracted_answer"] is None


# ----- 2. EM is format-walker-INDEPENDENT for qwen-native ----------------

def test_reward_em_ok_even_when_m2_walker_rejects_qwen_native_tags():
    """The inherited M2 walker rejects qwen-native rollouts (state machine
    only knows <search>/<information>, not <tool_call>/<tool_response>), but
    EM-only ignores format validity. So a qwen-native rollout with the right
    answer still scores 1.0 — and the telemetry `format_valid=False` reflects
    the walker mismatch but doesn't affect the gradient."""
    out = ours.compute_search_r1_reward(_full_rollout("Shakespeare"), ["Shakespeare"])
    assert out["reward"] == 1.0
    # `format_valid` is dead telemetry for M5.6; assert its current behaviour
    # for documentation, not as a contract:
    assert out["format_valid"] in (True, False)


# ----- 3. Multi-gold ------------------------------------------------------

def test_reward_picks_any_match_over_multi_gold():
    out = ours.compute_search_r1_reward(
        _full_rollout("Röntgen"), ["Wilhelm Conrad Röntgen", "Röntgen"]
    )
    assert out["reward"] == 1.0


def test_reward_multi_gold_no_match_returns_zero():
    out = ours.compute_search_r1_reward(
        _full_rollout("Marie Curie"), ["Wilhelm Conrad Röntgen", "Röntgen"]
    )
    assert out["reward"] == 0.0


# ----- 4. normalize_answer parity with M4 eval ---------------------------

def test_normalize_answer_byte_parity_with_eval_path():
    """M5.6 re-exports normalize_answer from the M2 module (NOT from
    evaluation_qwen35), so identity check is moot. Verify behavioural parity
    instead: outputs must agree on a representative input set."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "evaluation_qwen35" / "flashrag" / "search_r1" / "reward.py"
    spec = importlib.util.spec_from_file_location("_eval_q35_reward", str(p))
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    cases = [
        "Wilhelm Conrad Röntgen",
        "The Beatles",  # article in front
        "  whitespace  collapse  ",
        "Punc?tu!ation.",
        "MIXED case String",
        "",
    ]
    for s in cases:
        assert ours.normalize_answer(s) == eval_mod.normalize_answer(s), s


# ----- 5. Telemetry shape -------------------------------------------------

def test_reward_dict_keys_stable():
    out = ours.compute_search_r1_reward(_full_rollout("Shakespeare"), ["Shakespeare"])
    expected = {"reward", "extracted_answer", "f1", "em", "format_valid"}
    assert set(out.keys()) == expected


def test_reward_dict_keys_stable_on_failure():
    out = ours.compute_search_r1_reward("garbage", ["X"])
    expected = {"reward", "extracted_answer", "f1", "em", "format_valid"}
    assert set(out.keys()) == expected
