"""M5.5 F1+format reward — format-walker + reward semantics.

The walker ports ReSearch's `validate_template_format` to our schema (Qwen3.5
nested-XML + `<answer>...</answer>` answer scaffold; see the module docstring
of `training_m5_5/src/rewards/search_r1.py` for the source-of-truth mapping).
This file pins:

  1. Shared-path: `extract_solution` / `normalize_answer` re-exported from the
     M4 eval pipeline (so train-time + eval-time scoring share the SAME
     callable). Identity-tested where possible; behaviour-tested otherwise.
  2. Format-walker: 5 green cases (incl. zero-tool-call answer, multi-hop
     rollout, partial F1) and 6 red cases (one per failure mode).
  3. Reward semantics: 0.0 / 0.1-floor / F1 / 1.0 transitions.
  4. Telemetry shape: stable dict keys for the env.
"""
from training_m5_5.src.rewards import search_r1 as ours

# Canonical Qwen3.5 nested-XML tool-call (M4.1 form, what
# `tools=[QWEN35_SEARCH_TOOL]` auto-injects).
TC = (
    "<tool_call>\n"
    "<function=search>\n"
    "<parameter=query>\n"
    "author of Hamlet\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
TC2 = (
    "<tool_call>\n"
    "<function=search>\n"
    "<parameter=query>\n"
    "Shakespeare nationality\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
TR = "<tool_response>\nShakespeare wrote Hamlet.\n</tool_response>"
TR2 = "<tool_response>\nShakespeare was English.\n</tool_response>"


# ----- 1. Shared-path with M4 eval pipeline ------------------------------
#
# Identity testing (`is`) is unreliable here: the M5.5 reward module inserts
# `evaluation_qwen35/` into sys.path AT IMPORT TIME, then imports
# `flashrag.search_r1.reward.{extract_solution,normalize_answer}`. Loading the
# eval module a second time via importlib produces a DIFFERENT module object
# (CPython caches by sys.modules key, not by file path; our absolute-path load
# uses a fresh module name). So we test behavioural parity instead.

def _load_eval_q35_reward():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "evaluation_qwen35" / "flashrag" / "search_r1" / "reward.py"
    spec = importlib.util.spec_from_file_location("_eval_q35_reward", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_solution_behaviour_parity():
    eval_mod = _load_eval_q35_reward()
    cases = [
        "<answer>X</answer>",
        "<answer>X</answer> trailing junk",
        "<answer>X</answer><answer>Y</answer>",  # last-wins
        "no answer here",
        "<answer></answer>",
        "<answer>multi\nline</answer>",
    ]
    for s in cases:
        assert ours.extract_solution(s) == eval_mod.extract_solution(s), s


def test_normalize_answer_behaviour_parity():
    eval_mod = _load_eval_q35_reward()
    cases = [
        "Wilhelm Conrad Röntgen",
        "The Beatles",
        "  whitespace  collapse  ",
        "Punc?tu!ation.",
        "MIXED case String",
        "",
    ]
    for s in cases:
        assert ours.normalize_answer(s) == eval_mod.normalize_answer(s), s


# ----- 2. Format walker: green cases -------------------------------------

def _full_rollout(answer: str) -> str:
    """A two-hop qwen-native rollout, parameterised by final <answer>."""
    return (
        f"<think>plan: search for author</think>\n"
        f"{TC}\n{TR}\n"
        f"<think>need nationality</think>\n"
        f"{TC2}\n{TR2}\n"
        f"<think>got it</think>\n"
        f"<answer>{answer}</answer>"
    )


def test_format_green_full_rollout():
    is_valid, reason = ours.is_valid_format(_full_rollout("Shakespeare"))
    assert is_valid, f"expected valid; got reason={reason!r}"


def test_format_green_single_hop():
    text = (
        "<think>plan</think>"
        f"{TC}{TR}"
        "<think>got it</think>"
        "<answer>Shakespeare</answer>"
    )
    assert ours.is_valid_format(text)[0]


def test_format_green_zero_tool_calls():
    """Model answered without searching — still valid (e.g., known fact)."""
    text = "<think>I know this from training data</think><answer>Shakespeare</answer>"
    assert ours.is_valid_format(text)[0]


def test_format_green_multi_tool_call_in_one_block():
    """Two <function=...> blocks inside one <tool_call> — atypical but valid."""
    body = (
        "<function=search>\n"
        "<parameter=query>x</parameter>\n"
        "</function>\n"
        "<function=search>\n"
        "<parameter=query>y</parameter>\n"
        "</function>"
    )
    text = (
        "<think>plan</think>"
        f"<tool_call>{body}</tool_call>"
        f"{TR}"
        "<think>got it</think>"
        "<answer>X</answer>"
    )
    assert ours.is_valid_format(text)[0]


def test_format_green_trailing_whitespace_after_answer():
    """Vllm's stop string strips trailing whitespace; tolerate it anyway."""
    text = _full_rollout("Shakespeare") + "\n  \n"
    assert ours.is_valid_format(text)[0]


def test_format_green_template_injected_open_single_turn():
    """Qwen3.5 chat template auto-prepends `<think>\\n` at each assistant turn
    start; that prefix is NOT in `solution_str` (which is built from
    `m["content"]` only). So a well-formed single-turn rollout has 0 explicit
    `<think>` opens but 1 `</think>` close. Must still validate."""
    text = "plan: I know this</think><answer>Shakespeare</answer>"
    is_valid, reason = ours.is_valid_format(text)
    assert is_valid, f"expected valid; got reason={reason!r}"


def test_format_green_template_injected_open_multi_turn():
    """Multi-turn rollout with K=2 tool calls → 3 assistant turns → 3
    template-injected opens + 3 closes. The concatenated `solution_str` will
    show 0 explicit opens and 3 closes."""
    text = (
        "plan: search for author</think>"
        f"{TC}{TR}"
        "need nationality</think>"
        f"{TC2}{TR2}"
        "got it</think>"
        "<answer>Shakespeare</answer>"
    )
    is_valid, reason = ours.is_valid_format(text)
    assert is_valid, f"expected valid; got reason={reason!r}"


# ----- 3. Format walker: red cases (one per failure mode) ----------------

def test_format_red_missing_think():
    """No `</think>` close anywhere — model never closed its (template-injected
    or explicit) thinking block. Invalid."""
    text = f"{TC}{TR}<answer>X</answer>"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "think" in reason


def test_format_red_think_unbalanced():
    text = "<think>x<think>y</think><answer>X</answer>"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "think" in reason and "unbalanced" in reason


def test_format_red_tool_call_unbalanced():
    text = f"<think>x</think><tool_call>raw<answer>X</answer>"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "tool_call" in reason and "unbalanced" in reason


def test_format_red_tool_call_no_function():
    text = (
        "<think>x</think>"
        "<tool_call>raw search text</tool_call>"
        f"{TR}<think>y</think><answer>X</answer>"
    )
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "tool_call" in reason and "function" in reason


def test_format_red_function_without_parameter():
    text = (
        "<think>x</think>"
        "<tool_call><function=search></function></tool_call>"
        f"{TR}<think>y</think><answer>X</answer>"
    )
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "function" in reason or "parameter" in reason


def test_format_red_tool_call_with_json_body():
    """ReSearch reward uses JSON; ours rejects JSON (Qwen3.5 is XML-trained)."""
    text = (
        "<think>x</think>"
        '<tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'
        f"{TR}<think>y</think><answer>X</answer>"
    )
    assert not ours.is_valid_format(text)[0]


def test_format_red_no_answer():
    text = f"<think>x</think>{TC}{TR}"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "answer" in reason


def test_format_red_answer_not_terminal():
    """Answer block emitted but not at the end — rollout didn't terminate cleanly."""
    text = f"<think>x</think><answer>early guess</answer>{TC}{TR}"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "terminate" in reason


def test_format_red_answer_tags_unbalanced():
    text = f"<think>x</think>{TC}{TR}<answer>X</answer><answer>Y"
    is_valid, reason = ours.is_valid_format(text)
    assert not is_valid
    assert "answer" in reason


# ----- 4. Reward semantics (0.0 / 0.1 / partial / 1.0) -------------------

def test_reward_format_invalid_zero():
    """Missing <think> -> format invalid -> reward = 0."""
    out = ours.compute_search_r1_reward(f"{TC}{TR}<answer>Shakespeare</answer>", ["Shakespeare"])
    assert out["reward"] == 0.0
    assert out["format_valid"] is False


def test_reward_format_valid_em_match_returns_one():
    out = ours.compute_search_r1_reward(_full_rollout("Shakespeare"), ["Shakespeare"])
    assert out["reward"] == 1.0
    assert out["em"] == 1.0
    assert out["f1"] == 1.0
    assert out["format_valid"] is True


def test_reward_format_valid_partial_f1_returns_f1():
    out = ours.compute_search_r1_reward(
        _full_rollout("the author Shakespeare"), ["Shakespeare"]
    )
    # F1: pred tokens={the, author, shakespeare} (the->space after article-strip)
    # gold tokens={shakespeare}. Common=1. P=1/2 (after article strip 'the'->''),
    # R=1. F1=2*0.5*1/1.5=0.667.
    assert 0.0 < out["reward"] < 1.0
    assert out["reward"] == out["f1"]
    assert out["em"] == 0.0
    assert out["format_valid"] is True


def test_reward_format_valid_f1_zero_returns_floor():
    out = ours.compute_search_r1_reward(_full_rollout("Bacon"), ["Shakespeare"])
    assert out["reward"] == ours.FORMAT_BONUS
    assert out["f1"] == 0.0
    assert out["em"] == 0.0
    assert out["format_valid"] is True


def test_reward_format_valid_empty_answer_returns_floor():
    """`<answer></answer>` extracts to empty string; f1=0 -> floor."""
    text = _full_rollout("")
    out = ours.compute_search_r1_reward(text, ["Shakespeare"])
    assert out["reward"] == ours.FORMAT_BONUS
    assert out["format_valid"] is True


def test_reward_floor_value_is_0_1():
    """The 0.1 floor is the load-bearing knob — pin its value."""
    assert ours.FORMAT_BONUS == 0.1


# ----- 5. Multi-gold and answer-extraction semantics ---------------------

def test_reward_picks_best_f1_over_multi_gold():
    """Multi-answer gold: F1 = max over the list."""
    text = _full_rollout("Röntgen")
    out = ours.compute_search_r1_reward(text, ["Wilhelm Conrad Röntgen", "Röntgen"])
    assert out["reward"] == 1.0  # exact match on "Röntgen"


def test_reward_uses_last_answer_block():
    """Two <answer> blocks (model self-corrects) — score the LAST."""
    text = (
        "<think>plan</think>"
        f"{TC}{TR}"
        "<think>wait</think>"
        "<answer>Bacon</answer>"  # earlier (wrong) attempt
        "<think>actually</think>"
        f"{TC2}{TR2}"
        "<think>now sure</think>"
        "<answer>Shakespeare</answer>"  # final (correct)
    )
    out = ours.compute_search_r1_reward(text, ["Shakespeare"])
    assert out["reward"] == 1.0
    assert out["extracted_answer"] == "Shakespeare"


# ----- 6. Telemetry shape ------------------------------------------------

def test_reward_dict_keys_stable():
    """Env reads `reward`; W&B logs read everything else. Pin the shape."""
    out = ours.compute_search_r1_reward(_full_rollout("Shakespeare"), ["Shakespeare"])
    expected = {"reward", "extracted_answer", "f1", "em", "format_valid", "format_reason"}
    assert set(out.keys()) == expected


def test_reward_dict_keys_stable_on_failure():
    """Same dict shape on format-invalid rollouts (no surprise None / missing keys)."""
    out = ours.compute_search_r1_reward("garbage", ["X"])
    expected = {"reward", "extracted_answer", "f1", "em", "format_valid", "format_reason"}
    assert set(out.keys()) == expected
