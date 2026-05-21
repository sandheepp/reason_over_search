#!/usr/bin/env python3
"""Spot-check a training rollout step for tool-collapse / reward-hacking.

Designed to be run ad-hoc by an operator (or auto-invoked every 10 steps
by watch_resources.sh) during a live training run. Emits a ~20-line digest
that surfaces every failure mode we've seen in production:

  - Tool-call usage rate (collapsed → model abandoned search)
  - Tool-response error rate (high → retriever is broken)
  - Reward distribution (% at 0.1 floor → reward-hacking the floor)
  - Generation length (collapsed → tool-collapse pattern)
  - One sample rollout per reward bucket (so operator can eyeball quality)

The lessons baked in here are from RESULTS_M5_5_B300_seed42 post-mortem
(2026-05-16): retriever was OOM-killed during prod warmup, rollouts got
"Errno 111 Connection refused" tool_responses for the entire run, and the
model learned a zero-tool policy. Catching that mid-run (instead of in a
post-mortem) is the whole point of this script.

Usage:
    # Latest rollout step (auto-detect)
    python training_m5_5/scripts/check_trace.py

    # Specific step
    python training_m5_5/scripts/check_trace.py --step 50

    # Custom rollout dir
    python training_m5_5/scripts/check_trace.py --rollouts /root/reason_over_search/logs/exp_001

    # Just the digest, no rollout samples (useful for periodic logging)
    python training_m5_5/scripts/check_trace.py --no-samples

Exit codes:
    0 - all signals look healthy
    1 - one or more signals tripped a red-flag threshold (see flag descriptions)
    2 - couldn't read rollouts (file missing, malformed, etc.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter
from typing import List, Tuple

# Red-flag thresholds. Tuned on M5.5 B300 post-mortem data.
TOOL_CALL_FLOOR_PCT = 50          # % of rollouts that should have ≥1 tool_call (search task)
RETRIEVER_ERROR_CEILING_PCT = 10  # % of tool_responses that may be connection-refused
FLOOR_DOMINANCE_CEILING_PCT = 90  # % of rollouts at exactly reward=0.1 (signal compression)
MIN_GEN_TOKENS = 100              # generations below this suggest model is degenerate
GENERIC_ANSWER_CEILING_PCT = 20   # % of "United States"-class single-token country answers

GENERIC_ANSWER_RE = re.compile(
    r"^\s*(United States|United Kingdom|UK|USA|US|England|France|Germany|"
    r"China|India|Russia|Japan|Canada|Australia|Brazil)\s*$", re.IGNORECASE)


def find_latest_step(rollout_dir: str) -> int:
    files = glob.glob(os.path.join(rollout_dir, "train_data_step*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rollout JSONLs in {rollout_dir}")
    nums = []
    for f in files:
        try:
            nums.append(int(f.split("step")[-1].split(".")[0]))
        except ValueError:
            continue
    return max(nums)


def decode_assistant_turns(token_ids: List[int]) -> str:
    """Decode token_ids to text. Caches the tokenizer across calls."""
    if not hasattr(decode_assistant_turns, "_tok"):
        # Lazy import: this script may be invoked from a venv where
        # transformers isn't installed, in which case we fall back to
        # the raw bytes — still useful for counting tool_call markers.
        try:
            from transformers import AutoTokenizer
            decode_assistant_turns._tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B", trust_remote_code=True
            )
        except Exception:
            decode_assistant_turns._tok = None
    tok = decode_assistant_turns._tok
    if tok is None:
        return ""
    return tok.decode(token_ids, skip_special_tokens=False)


def analyze(rollout_dir: str, step: int, want_samples: bool) -> Tuple[bool, List[str]]:
    """Return (healthy, lines_to_print)."""
    fp = os.path.join(rollout_dir, f"train_data_step{step}.jsonl")
    if not os.path.exists(fp):
        return False, [f"❌ no such file: {fp}"]

    rewards: List[float] = []
    tool_call_counts: List[int] = []          # # tool_call blocks in assistant turns
    tool_response_total = 0
    tool_response_errors = 0
    gen_lens: List[int] = []
    generic_count = 0
    has_answer = 0
    decode_samples = {"floor": None, "perfect": None, "partial": None, "broken": None}
    seq_too_short_errs = 0

    try:
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rs = r.get("rewards", [])
                if not rs:
                    continue
                rew = rs[0]
                rewards.append(rew)
                tids = r.get("token_ids", [[]])[0]
                if tids:
                    gen_lens.append(len(tids))
                # Cheap regex pass on the decoded text (only for the samples
                # and for tool/response counting). Decode lazily.
                text = decode_assistant_turns(tids) if tids else ""
                if not text:
                    seq_too_short_errs += 1
                    continue
                # Strip system prompt: tool_call examples there shouldn't count.
                first_assistant = text.find("<|im_start|>assistant")
                conv = text[first_assistant:] if first_assistant >= 0 else text
                tc = len(re.findall(r"<tool_call>", conv))
                tool_call_counts.append(tc)
                for trm in re.finditer(r"<tool_response>(.*?)</tool_response>",
                                       conv, re.DOTALL):
                    tool_response_total += 1
                    body = trm.group(1).strip()[:300]
                    if any(s in body for s in ("Max retries", "Errno 111",
                                                "Connection refused",
                                                "Retriever failed")):
                        tool_response_errors += 1
                am = re.search(r"<answer>(.+?)</answer>", conv, re.DOTALL)
                if am:
                    has_answer += 1
                    ans = am.group(1).strip()
                    if GENERIC_ANSWER_RE.match(ans):
                        generic_count += 1
                # Save one sample per bucket (first one we see)
                bucket = ("perfect" if rew > 0.95 else
                          "partial" if 0.2 <= rew <= 0.9 else
                          "floor" if 0.09 < rew < 0.11 else
                          "broken" if rew < 0.01 else None)
                if bucket and decode_samples[bucket] is None:
                    decode_samples[bucket] = (rew, conv)
    except Exception as e:
        return False, [f"❌ failed to parse {fp}: {e}"]

    n = len(rewards)
    if n == 0:
        return False, [f"❌ {fp}: zero rollouts"]

    rmean = sum(rewards) / n
    rstd = statistics.stdev(rewards) if n > 1 else 0
    pct_fmt = 100 * sum(1 for r in rewards if r > 0) / n
    pct_floor = 100 * sum(1 for r in rewards if 0.09 < r < 0.11) / n
    pct_partial = 100 * sum(1 for r in rewards if 0.11 < r < 0.95) / n
    pct_perfect = 100 * sum(1 for r in rewards if r > 0.95) / n
    pct_with_tool = 100 * sum(1 for t in tool_call_counts if t >= 1) / max(len(tool_call_counts), 1)
    mean_tc = statistics.mean(tool_call_counts) if tool_call_counts else 0
    tr_err_pct = 100 * tool_response_errors / max(tool_response_total, 1)
    mean_gen = statistics.mean(gen_lens) if gen_lens else 0
    pct_generic = 100 * generic_count / max(has_answer, 1)

    # Red flags
    flags: List[str] = []
    if pct_with_tool < TOOL_CALL_FLOOR_PCT:
        flags.append(f"TOOL_COLLAPSE: only {pct_with_tool:.0f}% of rollouts call search "
                     f"(threshold ≥{TOOL_CALL_FLOOR_PCT}%) — model abandoning retrieval")
    if tr_err_pct > RETRIEVER_ERROR_CEILING_PCT:
        flags.append(f"RETRIEVER_BROKEN: {tr_err_pct:.0f}% of tool_responses are connection errors "
                     f"(threshold ≤{RETRIEVER_ERROR_CEILING_PCT}%) — check dmesg for OOM-killed FAISS workers")
    if pct_floor > FLOOR_DOMINANCE_CEILING_PCT:
        flags.append(f"FLOOR_DOMINANCE: {pct_floor:.0f}% of rollouts at exact 0.1 floor "
                     f"(threshold ≤{FLOOR_DOMINANCE_CEILING_PCT}%) — GRPO advantage signal compressed")
    if mean_gen < MIN_GEN_TOKENS:
        flags.append(f"GEN_DEGENERATE: mean generation only {mean_gen:.0f} tokens "
                     f"(threshold ≥{MIN_GEN_TOKENS}) — likely outputting padding / collapsing")
    if pct_generic > GENERIC_ANSWER_CEILING_PCT:
        flags.append(f"GENERIC_ANSWERS: {pct_generic:.0f}% of answers are 'United States'-style country guesses "
                     f"(threshold ≤{GENERIC_ANSWER_CEILING_PCT}%) — reward-hacking by guessing short")

    out: List[str] = []
    out.append("=" * 78)
    out.append(f"  step {step}  n={n}  ({os.path.basename(fp)})")
    out.append("=" * 78)
    out.append(f"  reward         mean={rmean:.4f}  std={rstd:.4f}")
    out.append(f"  distribution   broken={100-pct_fmt:4.0f}%  floor={pct_floor:4.0f}%  "
               f"partial={pct_partial:4.0f}%  perfect={pct_perfect:4.0f}%")
    out.append(f"  tool usage     ≥1_call={pct_with_tool:4.0f}%  mean_calls/rollout={mean_tc:.2f}")
    out.append(f"  retriever      tool_responses={tool_response_total}  "
               f"errors={tool_response_errors} ({tr_err_pct:.0f}%)")
    out.append(f"  generation     mean_tokens={mean_gen:.0f}  generic_country_answers={pct_generic:.0f}%")
    if flags:
        out.append("")
        out.append("  🚩 RED FLAGS:")
        for f_msg in flags:
            out.append(f"     • {f_msg}")
    else:
        out.append("")
        out.append("  ✅ all health signals within thresholds")

    if want_samples:
        out.append("")
        out.append("  ─ sample rollouts ─")
        for bucket_name, sample in decode_samples.items():
            if sample is None:
                continue
            rew, conv = sample
            # Show just the last assistant block + answer
            last_assistant = conv.rfind("<|im_start|>assistant")
            tail = conv[last_assistant:] if last_assistant >= 0 else conv
            tail = re.sub(r"<\|endoftext\|>+", "", tail)[-800:]
            out.append(f"   [{bucket_name.upper():7s} r={rew:.3f}]")
            for ln in tail.split("\n")[:12]:
                out.append(f"     | {ln[:120]}")

    return len(flags) == 0, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", default="/root/reason_over_search/logs/exp_001",
                   help="dir with train_data_step*.jsonl (default: /root/reason_over_search/logs/exp_001)")
    p.add_argument("--step", type=int, default=None,
                   help="step to check (default: latest)")
    p.add_argument("--no-samples", action="store_true",
                   help="suppress per-bucket rollout samples (digest only)")
    args = p.parse_args()

    # Auto-locate exp_NNN if the default doesn't exist (NeMo-RL may have used exp_002, etc.)
    if not os.path.isdir(args.rollouts):
        candidates = sorted(glob.glob("/root/reason_over_search/logs/exp_*"),
                            key=os.path.getmtime, reverse=True)
        if candidates:
            args.rollouts = candidates[0]
        else:
            print(f"❌ no rollout dir found (looked in {args.rollouts} and /root/reason_over_search/logs/exp_*)")
            sys.exit(2)

    try:
        step = args.step if args.step is not None else find_latest_step(args.rollouts)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(2)

    healthy, lines = analyze(args.rollouts, step, want_samples=not args.no_samples)
    for ln in lines:
        print(ln)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
