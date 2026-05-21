"""Reward parity vs M1 eval pipeline.

Both modules should be byte-identical past the docstring (verified by diff in
the 4a/4b commit). This test asserts they produce identical outputs on a
representative set of solution strings — guards against any future drift.
"""
from training.src.rewards import search_r1 as ours
from flashrag.search_r1 import reward as eval_side  # type: ignore[import-not-found]

GOLD = ["Wilhelm Conrad Röntgen", "Röntgen"]

# A few rollout-shape strings spanning the reward function's branches.
CASES = [
    # 1. Perfect: valid format + correct EM + retrieval hit
    (
        "<think>Need to look this up.</think>"
        "<search>first nobel physics prize winner</search>"
        "<information>Wilhelm Conrad Röntgen received the first Nobel Prize in Physics</information>"
        "<answer>Wilhelm Conrad Röntgen</answer>",
        GOLD,
    ),
    # 2. Valid format + correct EM but no retrieval hit
    (
        "<think>I know this.</think>"
        "<answer>Wilhelm Conrad Röntgen</answer>",
        GOLD,
    ),
    # 3. Valid format + wrong EM
    (
        "<think>Hmm.</think>"
        "<search>nobel physics</search>"
        "<information>Marie Curie won the second Nobel</information>"
        "<answer>Marie Curie</answer>",
        GOLD,
    ),
    # 4. Invalid format + correct EM
    (
        "I think the answer is <answer>Wilhelm Conrad Röntgen</answer>",
        GOLD,
    ),
    # 5. Invalid format + no answer
    (
        "<search>foo</search><search>bar</search>",  # double search, no info, no answer
        GOLD,
    ),
    # 6. Empty assistant text
    ("", GOLD),
    # 7. Answer with surrounding whitespace (must normalize equal)
    (
        "<answer>  Wilhelm  Conrad  Röntgen  </answer>",
        GOLD,
    ),
    # 8. Multiple <answer> blocks — extract_solution takes the LAST
    (
        "<answer>Marie Curie</answer><think>wait</think><answer>Wilhelm Conrad Röntgen</answer>",
        GOLD,
    ),
]


def test_reward_byte_parity():
    for solution, gold in CASES:
        ours_out = ours.compute_search_r1_reward(solution, gold)
        their_out = eval_side.compute_search_r1_reward(solution, gold)
        assert ours_out == their_out, (
            f"divergence on case: {solution[:60]!r}\nours={ours_out}\ntheirs={their_out}"
        )


def test_em_check_byte_parity():
    for pred in ("Wilhelm Conrad Röntgen", "wilhelm conrad röntgen", "Marie Curie", ""):
        assert ours.em_check(pred, GOLD) == eval_side.em_check(pred, GOLD)


def test_normalize_answer_byte_parity():
    samples = ["The Quick Brown Fox.", "  hello,  world!  ", "An apple A day", ""]
    for s in samples:
        assert ours.normalize_answer(s) == eval_side.normalize_answer(s)
