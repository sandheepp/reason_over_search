"""Search-R1 retrieval environment for M5.1 (Ray actor).

Implements `EnvironmentInterface.step()` for the multi-turn rollout loop in
`run_multi_turn_rollout` (rollouts.py:403). Each turn:

1. Look at the latest assistant message in each sample's message_log.
2. If it contains `<answer>...</answer>` -> terminal: compute reward against
   `golden_answers` (carried in metadata as `ground_truth`), emit empty obs.
3. Else if it contains a search query (`<tool_call><function=search>
   <parameter=query>...` for `qwen_native`; `<search>...</search>` for `paper`)
   -> POST to the local retriever, format docs as a tool/observation message,
   continue.
4. Else if turn_count >= max_turns -> terminate with reward=0.
5. Else -> no tool call and no answer in this turn; we don't have the model's
   output yet so we let the rollout continue (turn_count++) and the next
   `env.step` decides.

Retrieval is batched: all samples that emitted a search query in this turn go
to the retriever in one HTTP call (via the `query: list[str]` shape of
`/batch_search`; VERL_REFERENCE.md §1, retriever_serving.py:111).

The observation `content` is appended raw to the message log without re-running
the chat template (rollouts.py:498-501). For `paper` we emit
`<information>{docs}</information>` (mirrors `CodeEnvironment.format_result`'s
`<result>...</result>` raw-text pattern). For `qwen_native` we hand-construct
the Qwen3.5 chat-template markers: `<|im_end|>\n<|im_start|>user\n
<tool_response>...\n</tool_response><|im_end|>\n<|im_start|>assistant\n` so
the next-turn generation context is well-formed (byte-aligned with M4's
evaluation_qwen35 pipeline; docs/milestone_5/MILESTONE_5.md §"Prompt + tag
scheme (carry from M4)").

Reward uses the **F1-only** scorer at
`training_m5_6/src/rewards/search_r1.py`. The two intentional divergences
from the ReSearch paper (no format reward, no `\\boxed{}` wrap) live there;
the env stays oblivious to those choices and only consumes `reward_info['reward']`.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

import ray
import requests
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from training_m5_6.src.environments.parsers import (
    DEFAULT_MAX_OBS_CHARS,
    VALID_ARMS,
    format_docs_paper,
    format_docs_qwen_native,
    parse_query,
)
from training_m5_6.src.rewards.search_r1 import (
    compute_search_r1_reward,
    em_check,
    extract_solution,
)


class SearchR1EnvConfig(TypedDict, total=False):
    arm: str
    retriever_url: str
    top_n: int
    max_turns: int
    max_obs_chars: int  # per-observation char cap (verl's max_obs_length=500 tokens, char proxy)
    request_timeout_s: float


class SearchR1Env(EnvironmentInterface):
    """Multi-turn search-augmented env for Search-R1 GRPO.

    Plain class so unit tests can instantiate it without spinning up Ray. The
    ray-remote-wrapped version exported as `SearchR1Environment` (below) is
    what `register_env` should point at — that's the FQN consumed by
    NeMo-RL's `create_env` (environments/utils.py:106-129).
    """

    def __init__(self, cfg: SearchR1EnvConfig) -> None:
        arm = cfg.get("arm", "qwen_native")
        if arm not in VALID_ARMS:
            raise ValueError(f"cfg.arm must be in {VALID_ARMS}, got {arm!r}")
        self.arm = arm
        self.retriever_url = cfg.get("retriever_url", "http://127.0.0.1:3005").rstrip("/")
        self.top_n = int(cfg.get("top_n", 3))
        self.max_turns = int(cfg.get("max_turns", 4))
        self.max_obs_chars = int(cfg.get("max_obs_chars", DEFAULT_MAX_OBS_CHARS))
        self.request_timeout_s = float(cfg.get("request_timeout_s", 30.0))

    # ----- retrieval -----

    def _batch_retrieve(self, queries: list[Optional[str]]) -> list[Optional[list[str]]]:
        """POST {url}/batch_search for the non-None queries; return per-input docs.

        None entries (no query this turn) get None docs, preserving order.
        """
        idx_with_q = [(i, q) for i, q in enumerate(queries) if q]
        if not idx_with_q:
            return [None] * len(queries)

        valid_queries = [q for _, q in idx_with_q]
        try:
            response = requests.post(
                f"{self.retriever_url}/batch_search",
                json={"query": valid_queries, "top_n": self.top_n, "return_score": False},
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            docs_batch = response.json()
        except requests.RequestException as e:
            # Graceful: every active sample gets the failure string back.
            err = str(e)
            return [
                [f"Retriever failed: {err}"] if i in {idx for idx, _ in idx_with_q} else None
                for i in range(len(queries))
            ]

        out: list[Optional[list[str]]] = [None] * len(queries)
        for (orig_idx, _), docs in zip(idx_with_q, docs_batch):
            out[orig_idx] = [d["contents"] for d in docs]
        return out

    # ----- step -----

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[dict[str, Any]],
    ) -> EnvironmentReturn:
        """One turn of the rollout for a batch of trajectories."""
        batch_size = len(message_log_batch)

        # Per-sample classification: "answer", "search", or "exhausted".
        kinds: list[str] = []
        queries: list[Optional[str]] = []
        extracted_answers: list[Optional[str]] = []

        for log, meta in zip(message_log_batch, metadata_batch):
            assistant_text = log[-1]["content"]  # latest message — the assistant just spoke

            # 1. Answer takes precedence (terminal).
            if (ans := extract_solution(assistant_text)) is not None:
                kinds.append("answer")
                queries.append(None)
                extracted_answers.append(ans)
                continue

            # 2. Search query — but only if we still have turns left.
            turn = int(meta.get("turn_count", 0))
            if turn + 1 >= self.max_turns:
                kinds.append("exhausted")
                queries.append(None)
                extracted_answers.append(None)
                continue

            q = parse_query(self.arm, assistant_text)
            if q is None:
                # Neither answer nor search — model output is malformed.
                # Treat as exhausted (no good observation to feed back).
                kinds.append("exhausted")
                queries.append(None)
                extracted_answers.append(None)
                continue

            kinds.append("search")
            queries.append(q)
            extracted_answers.append(None)

        # 3. Single batched HTTP call for all samples that asked to search.
        retrieved = self._batch_retrieve(queries)

        # 4. Assemble per-sample outputs.
        observations: list[dict[str, str]] = []
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        terminateds = torch.zeros(batch_size, dtype=torch.bool)
        next_stop_strings: list[Optional[list[str]]] = []
        new_metadata: list[dict[str, Any]] = []
        answers_out: list[Optional[str]] = []

        for i, (log, meta, kind) in enumerate(zip(message_log_batch, metadata_batch, kinds)):
            # had_valid_answer: did the *final* assistant turn emit a parseable <answer>?
            # Default False; flipped True only when kind == "answer". Propagates turn-to-turn
            # via metadata, so the value at end-of-rollout reflects the terminal turn.
            updated_meta = {
                **meta,
                "turn_count": int(meta.get("turn_count", 0)) + 1,
                "had_valid_answer": False,
            }

            if kind == "answer":
                # Reconstruct the full solution string for the reward function.
                # Skip the initial user prompt; include all assistant + env turns.
                solution_str = "".join(
                    m["content"] for m in log if m["role"] != "user"
                )
                gold = meta.get("ground_truth") or []
                reward_info = compute_search_r1_reward(solution_str, gold)
                rewards[i] = float(reward_info["reward"])
                terminateds[i] = True
                updated_meta["had_valid_answer"] = True
                observations.append({"role": "tool", "content": ""})
                next_stop_strings.append(None)
                answers_out.append(extracted_answers[i])

            elif kind == "exhausted":
                # No search, no answer (or out of turns). Score whatever we have.
                solution_str = "".join(
                    m["content"] for m in log if m["role"] != "user"
                )
                gold = meta.get("ground_truth") or []
                # Best-effort: maybe an unclosed <answer was emitted; em_check
                # against an empty string yields 0, which is what we want.
                # EM (not F1) here to stay consistent with M5.6's EM-only reward.
                ans = extract_solution(solution_str) or ""
                rewards[i] = float(em_check(ans, gold)) if ans else 0.0
                terminateds[i] = True
                observations.append({"role": "tool", "content": ""})
                next_stop_strings.append(None)
                answers_out.append(ans or None)

            else:  # "search"
                docs = retrieved[i] or [f"Retriever returned no documents for query: {queries[i]}"]
                # kwarg name differs by arm: qwen_native caps PER chunk, paper
                # caps the TOTAL body (see parsers.py docstrings + the M4.1 v3
                # multi-block semantics doc, MILESTONE_4.md §"M4.1 v3"). M2's
                # source had this as `max_chars=` for both, which TypeError'd
                # on the qwen_native search path — only the qwen_native answer-
                # direct and exhausted paths were reached in the M2 smoke. Fixed
                # here so M5.1's retrieval path doesn't trip the same bug.
                content = (
                    format_docs_qwen_native(docs, max_chars_per_chunk=self.max_obs_chars)
                    if self.arm == "qwen_native"
                    else format_docs_paper(docs, max_chars=self.max_obs_chars)
                )
                observations.append({"role": "tool", "content": content})
                next_stop_strings.append(self._stop_strings_for_arm())
                answers_out.append(None)
                # rewards[i], terminateds[i] stay 0 / False

            new_metadata.append(updated_meta)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=answers_out,
        )

    def _stop_strings_for_arm(self) -> list[str]:
        if self.arm == "qwen_native":
            return ["</tool_call>", "</answer>"]
        return ["</search>", "</answer>"]

    def shutdown(self) -> None:
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, float | int]]:
        """Compute rollup metrics for one rollout batch.

        Mirrors `MathEnvironment.global_post_process_and_metrics`'s shape so
        `checkpointing.metric_name = "val:accuracy"` resolves correctly. The
        rollout loop fills `batch` with `rewards`, `is_end`, `prompt_lengths`,
        `generation_lengths`, `text` per sample (rollouts.py).
        """
        rewards = (
            batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        )

        # Zero-out rewards for rollouts that didn't end with a real stop token
        # (truncated at max_response_length). Matches MathEnvironment.
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]
            properly_ended = batch["is_end"].float().mean().item()
        else:
            properly_ended = float("nan")

        accuracy = rewards.mean().item()
        # Under M5.1's F1-only reward, `accuracy` is the mean F1 over the batch
        # (continuous in [0, 1]). Track a near-EM rate too: fraction of samples
        # with F1 >= 0.8, i.e. predicted answer is close enough to gold that
        # it's almost-certainly the intended entity. Useful sanity dichotomy
        # alongside the continuous reward, since GRPO advantage is computed off
        # the continuous signal but humans read EM/near-EM rates.
        near_em_rate = (rewards >= 0.8).float().mean().item()

        metrics: dict[str, float | int] = {
            "accuracy": accuracy,
            "near_em_rate": near_em_rate,
            "fraction_of_samples_properly_ended": properly_ended,
            "num_problems_in_batch": int(rewards.shape[0]),
        }

        if "generation_lengths" in batch and "prompt_lengths" in batch:
            metrics["generation_lengths"] = batch["generation_lengths"].float().mean().item()
            metrics["prompt_lengths"] = batch["prompt_lengths"].float().mean().item()

        return batch, metrics


# Ray-remote-wrapped class. `register_env` should point at this FQN.
SearchR1Environment = ray.remote(SearchR1Env)  # pragma: no cover
