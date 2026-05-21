"""Search-R1 data processor.

One processor, two arms — dispatched via `task_data_spec.task_name`
(`search_r1_qwen_native` or `search_r1_paper`).

Both arms read `messages[0].content` for the bare question and stuff
`golden_answers` (preserved column from the parquet) into
`extra_env_info["ground_truth"]` so the env actor can compute EM in `step`.

Both arms then build the rollout-side message_log fresh via
`tokenizer.apply_chat_template`, mirroring `math_hf_data_processor`'s pattern
(processors.py:467-477):

- **paper arm**: wrap the bare question in
  `task_data_spec.prompt.format(question)` (prompt file =
  `training/src/prompts/search_r1_paper.txt`). No system prompt, no tool
  registration — the paper bakes the protocol into the user message.
- **qwen_native arm**: same shape — wrap the bare question in
  `task_data_spec.prompt.format(question)` (prompt file =
  `training/src/prompts/search_r1_qwen_native_user.txt`, which carries the
  think/search/answer protocol). The system prompt is the brief role
  description from `search_r1_qwen_native_system.txt`. Pass
  `tools=[SEARCH_TOOL]` so Qwen3.5's template auto-injects the search tool's
  schema into the system area.

Both arms pass `enable_thinking=True` so the generation prefix becomes
`<|im_start|>assistant\\n<think>\\n` (open block) — the model fills its
reasoning, closes `</think>`, then emits `<search>`/`<tool_call>`/`<answer>`.
Without this, Qwen3.5's template would auto-emit `<think>\\n\\n</think>\\n\\n`
(closed empty block) before the model generates anything, conflicting with
the prompt's instruction to reason first.
"""
from __future__ import annotations

from typing import Any, cast

import torch

from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataSpec,
    TokenizerType,
)

from training_m5_1.src.chat_template.tools import SEARCH_TOOL


def _arm_from_task_name(task_name: str) -> str:
    """Recover the arm string from the task_name set by SearchR1Dataset."""
    prefix = "search_r1_"
    if not task_name.startswith(prefix):
        raise ValueError(
            f"task_name {task_name!r} does not start with {prefix!r}; "
            "did the dataset adapter set self.task_name correctly?"
        )
    return task_name[len(prefix):]


def search_r1_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process one parquet row into a DatumSpec ready for the GRPO rollout.

    Expects `datum_dict` to carry the columns produced by either of the
    repo's two dataset-prep scripts:
      - M2 NQ+HotpotQA: `training/scripts/prepare_dataset.py`
      - M5.1 MuSiQue : `training_m5_1/scripts/prep_musique.py`

    Required columns:
      - `messages`: list[{"role": "user", "content": question}]  (length 1)
      - `golden_answers`: list[str]
      - `data_source`: str ("nq" | "hotpotqa" | "musique" | ...) — optional;
        used only for per-dataset metric splitting. Absent rows get `None`.
    """
    arm = _arm_from_task_name(cast(str, datum_dict.get("task_name") or task_data_spec.task_name))

    question = datum_dict["messages"][0]["content"]
    golden_answers = list(datum_dict["golden_answers"])

    extra_env_info: dict[str, Any] = {
        "ground_truth": golden_answers,
        "data_source": datum_dict.get("data_source"),
        "turn_count": 0,  # env increments each step
    }

    # Build the conversation that `apply_chat_template` sees.
    convo: list[dict[str, str]] = []
    if task_data_spec.system_prompt:
        convo.append({"role": "system", "content": task_data_spec.system_prompt})

    if arm == "paper":
        # Wrap the bare question with the paper instructions.
        if task_data_spec.prompt is None:
            raise ValueError(
                "paper arm requires task_data_spec.prompt_file to be set "
                "(typically training/src/prompts/search_r1_paper.txt)."
            )
        user_content = task_data_spec.prompt.format(question)
        convo.append({"role": "user", "content": user_content})
        rendered = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=True,  # open <think>\n so model fills reasoning before searching/answering
        )
    elif arm == "qwen_native":
        # Protocol (think/search/answer instructions) lives in the user message,
        # mirroring the paper arm. System message carries only the brief role
        # description; the tool schema is auto-injected by the chat template.
        if task_data_spec.prompt is None:
            raise ValueError(
                "qwen_native arm requires task_data_spec.prompt_file to be set "
                "(typically training/src/prompts/search_r1_qwen_native_user.txt)."
            )
        user_content = task_data_spec.prompt.format(question)
        convo.append({"role": "user", "content": user_content})
        rendered = tokenizer.apply_chat_template(
            convo,
            tools=[SEARCH_TOOL],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=True,  # open <think>\n so model fills reasoning before tool calls/answer
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    token_ids = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    message_log: LLMMessageLogType = [
        {"role": "user", "content": rendered, "token_ids": token_ids}
    ]
    length = int(token_ids.shape[0])

    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Mirror `math_hf_data_processor`'s overlength handling: shrink + mask out.
        message_log[0]["token_ids"] = token_ids[: min(4, max_seq_length)]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": cast(str, datum_dict.get("task_name") or task_data_spec.task_name),
    }
    return output
