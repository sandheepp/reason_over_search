import re
import string
from typing import List, Tuple


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def is_valid_sequence(text: str) -> Tuple[bool, str]:
    # Tag count check excludes <information> because retrieved Wikipedia
    # passages occasionally contain the literal string "</information>".
    for tag in ("think", "search", "answer"):
        opening_count = len(re.findall(f"<{tag}>", text))
        closing_count = len(re.findall(f"</{tag}>", text))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, text)
    state = "start"

    for part in parts:
        if not part.strip():
            continue
        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            # Released GRPO checkpoint often skips <think> and opens <search>
            # directly from start or after a previous <information> turn.
            elif part == "<search>" and state in ["start", "after_think", "information"]:
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            # Same: model can answer directly from start, after a search round, or post-think.
            elif part == "<answer>" and state in ["start", "after_think", "after_search", "information"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # Inside a tag body: any content is fine.
            # Outside (start / after_X / information): the trained model
            # interleaves plain-text planning prose between tag blocks, so
            # accept it instead of rejecting the whole rollout.
            continue
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
    return True, "Valid sequence format"


def extract_solution(solution_str: str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) <= 0:
        return None
    answer_text = matches[-1].group(1).strip()
    # qwen3 mode (p1_basic_w_ex prompt) wraps the answer as `\boxed{X}`. Unwrap so EM/F1
    # compare against X, not the whole "The final answer is \[ \boxed{X} \]" wrapper.
    # `\{+ ... \}+` tolerates accidental `{{X}}` from a stray Python format-string escape.
    boxed = re.search(r"\\boxed\{+\s*(.+?)\s*\}+", answer_text, re.DOTALL)
    if boxed:
        return boxed.group(1).strip()
    return answer_text


def extract_information_blocks(text: str) -> List[str]:
    return [match.strip() for match in re.findall(r"<information>(.*?)</information>", text, re.DOTALL)]


def is_retrieval_correct(text: str, golden_answers: List[str]) -> bool:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_search_r1_reward(
    solution_str: str,
    golden_answers,
    structure_format_score: float = 0.0,
    final_format_score: float = 0.0,
    retrieval_score: float = 0.0,
    score: float = 1.0,
):
    is_valid_format, reason = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, golden_answers)
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        if is_valid_format:
            reward = structure_format_score + retrieval_score if retrieval_correct else structure_format_score
        else:
            reward = 0.0
    else:
        if em_check(answer, golden_answers):
            reward = score if is_valid_format else (score - structure_format_score)
        elif is_valid_format:
            reward = structure_format_score + retrieval_score if retrieval_correct else structure_format_score
        else:
            reward = final_format_score

    return {
        "reward": float(reward),
        "format_valid": bool(is_valid_format),
        "retrieval_hit": bool(retrieval_correct),
        "extracted_answer": answer,
        "format_reason": reason,
    }

