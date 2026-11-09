#!/usr/bin/env python3
"""Analyze rollout JSONLs from a window of training steps and produce
Markdown for direct appending to RESULTS_M5_1_B200.md cadence sections.

Usage:
    python3 analyze_rollouts.py <exp_dir> <step_start> <step_end> [--sample-from STEP]

Output (stdout): Markdown analysis block. Caller appends to the results doc.

What it computes per-window (steps in [start, end] inclusive):
  - reward stats: mean, std, %zero, %nonzero, max
  - turns per trajectory: mean, p50, p95, max
  - tool calls per trajectory: mean, p50, p95, max
  - completion rate: % trajectories ending with </answer> tag
  - truncation rate: % trajectories not ending with </answer>
  - response length: mean, p50, p95 in chars
  - input length distribution from input_lengths field
  - per-step reward trajectory (for plotting later)

Plus 2 example trajectories from the `sample-from` step (default = step_end):
  - one with the highest reward (if any > 0)
  - one with reward 0 (random-ish; first in file)
  Truncated to ~300 lines each.

The JSONL schema (verified 2026-05-14):
  Each line is one trajectory:
    content: [[chunk_0, chunk_1, ...]]    # outer list len 1; inner list of strs
                                          # alternating user/model in chat-rendered form
    rewards: [float]
    input_lengths: [int]
    idx: int                              # rollout index within step
    token_ids: [[int...]]                 # not used here
    advantages, generation_logprobs, prev_logprobs: not used here
"""
import argparse
import json
import statistics
import sys
import re
from pathlib import Path


def percentile(values, p):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = max(0, min(len(sorted_v) - 1, int(round(p * (len(sorted_v) - 1)))))
    return sorted_v[k]


def parse_trajectory(row):
    """Extract per-trajectory metrics from one JSONL row."""
    content = row.get("content", [[]])[0] if row.get("content") else []
    reward = row.get("rewards", [0.0])[0]
    input_len = row.get("input_lengths", [0])[0]

    # Multi-turn structure: chunk 0 = system + first user question,
    # then alternating model chunks (1, 3, 5, ...) and user/tool chunks (2, 4, 6, ...).
    # Model chunks are the ones the policy emitted; tool chunks are retriever responses.
    model_chunks = []
    for i in range(1, len(content), 2):
        if i < len(content):
            model_chunks.append(content[i])

    response_text = "".join(model_chunks)
    response_len = len(response_text)
    n_turns = len(model_chunks)
    n_tool_calls = response_text.count("<tool_call>")
    has_answer = "</answer>" in response_text
    has_think = response_text.count("<think>")

    return {
        "reward": reward,
        "input_len": input_len,
        "response_len": response_len,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "has_answer": has_answer,
        "n_think": has_think,
        "content": content,
    }


def find_step_files(exp_dir, step_start, step_end):
    files = []
    for s in range(step_start, step_end + 1):
        f = exp_dir / f"train_data_step{s}.jsonl"
        if f.exists():
            files.append((s, f))
    return files


def aggregate(rows):
    if not rows:
        return None
    rewards = [r["reward"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.stdev(rewards) if n > 1 else 0.0,
        "reward_max": max(rewards),
        "reward_pct_zero": 100.0 * sum(1 for r in rewards if r == 0.0) / n,
        "reward_pct_nonzero": 100.0 * sum(1 for r in rewards if r > 0.0) / n,
        "turns_mean": statistics.mean(r["n_turns"] for r in rows),
        "turns_p50": percentile([r["n_turns"] for r in rows], 0.5),
        "turns_p95": percentile([r["n_turns"] for r in rows], 0.95),
        "turns_max": max(r["n_turns"] for r in rows),
        "tools_mean": statistics.mean(r["n_tool_calls"] for r in rows),
        "tools_p50": percentile([r["n_tool_calls"] for r in rows], 0.5),
        "tools_p95": percentile([r["n_tool_calls"] for r in rows], 0.95),
        "tools_max": max(r["n_tool_calls"] for r in rows),
        "pct_with_answer": 100.0 * sum(1 for r in rows if r["has_answer"]) / n,
        "pct_truncated": 100.0 * sum(1 for r in rows if not r["has_answer"]) / n,
        "resp_len_mean": statistics.mean(r["response_len"] for r in rows),
        "resp_len_p50": percentile([r["response_len"] for r in rows], 0.5),
        "resp_len_p95": percentile([r["response_len"] for r in rows], 0.95),
        "input_len_mean": statistics.mean(r["input_len"] for r in rows),
    }


def truncate_chunk(s, head_chars=400, tail_chars=200):
    if len(s) <= head_chars + tail_chars + 50:
        return s
    return s[:head_chars] + f"\n\n[... {len(s) - head_chars - tail_chars} chars elided ...]\n\n" + s[-tail_chars:]


def format_example(row, label):
    """Format one trajectory as Markdown."""
    out = [f"##### Example: {label} (reward={row['reward']:.4f}, turns={row['n_turns']}, tool_calls={row['n_tool_calls']}, has_answer={row['has_answer']})", ""]
    content = row["content"]
    for i, chunk in enumerate(content):
        speaker = "system+user" if i == 0 else ("model" if i % 2 == 1 else "tool")
        truncated = truncate_chunk(chunk)
        out.append(f"```")
        out.append(f"--- chunk {i} ({speaker}, {len(chunk)} chars) ---")
        out.append(truncated)
        out.append("```")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="Path to logs/exp_NNN/ directory")
    ap.add_argument("step_start", type=int, help="Inclusive start step number")
    ap.add_argument("step_end", type=int, help="Inclusive end step number")
    ap.add_argument("--sample-from", type=int, default=None, help="Step to sample examples from (default: step_end)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    sample_step = args.sample_from or args.step_end

    files = find_step_files(exp_dir, args.step_start, args.step_end)
    if not files:
        print(f"# No train_data_stepN.jsonl files found in {exp_dir} for steps [{args.step_start}, {args.step_end}]")
        sys.exit(0)

    # Aggregate window
    all_rows = []
    per_step_summary = []
    for step, f in files:
        step_rows = []
        with open(f) as fp:
            for line in fp:
                step_rows.append(parse_trajectory(json.loads(line)))
        all_rows.extend(step_rows)
        agg = aggregate(step_rows)
        if agg:
            per_step_summary.append((step, agg))

    window_agg = aggregate(all_rows)

    print(f"#### Cadence analysis — steps {args.step_start}–{args.step_end} ({len(files)} step files, {len(all_rows)} trajectories total)")
    print()
    print(f"**Window aggregate** ({window_agg['n']} trajectories across {len(files)} steps):")
    print()
    print(f"| Metric | Value |")
    print(f"|---|---:|")
    print(f"| Reward — mean | {window_agg['reward_mean']:.4f} |")
    print(f"| Reward — std | {window_agg['reward_std']:.4f} |")
    print(f"| Reward — max | {window_agg['reward_max']:.4f} |")
    print(f"| Reward — % zero | {window_agg['reward_pct_zero']:.1f}% |")
    print(f"| Reward — % nonzero | {window_agg['reward_pct_nonzero']:.1f}% |")
    print(f"| Turns — mean / p50 / p95 / max | {window_agg['turns_mean']:.2f} / {window_agg['turns_p50']} / {window_agg['turns_p95']} / {window_agg['turns_max']} |")
    print(f"| Tool calls — mean / p50 / p95 / max | {window_agg['tools_mean']:.2f} / {window_agg['tools_p50']} / {window_agg['tools_p95']} / {window_agg['tools_max']} |")
    print(f"| **Completion rate** (% with `</answer>`) | **{window_agg['pct_with_answer']:.1f}%** |")
    print(f"| **Truncation rate** (% without `</answer>`) | **{window_agg['pct_truncated']:.1f}%** |")
    print(f"| Response chars — mean / p50 / p95 | {window_agg['resp_len_mean']:.0f} / {window_agg['resp_len_p50']:.0f} / {window_agg['resp_len_p95']:.0f} |")
    print(f"| Input length — mean | {window_agg['input_len_mean']:.0f} |")
    print()

    # Per-step trajectory of rewards
    print(f"**Per-step reward / completion**:")
    print()
    print(f"| Step | n | Reward mean | Reward max | % w/ answer | Avg turns | Avg tools |")
    print(f"|---:|---:|---:|---:|---:|---:|---:|")
    for step, agg in per_step_summary:
        print(f"| {step} | {agg['n']} | {agg['reward_mean']:.4f} | {agg['reward_max']:.4f} | {agg['pct_with_answer']:.1f}% | {agg['turns_mean']:.1f} | {agg['tools_mean']:.1f} |")
    print()

    # Example trajectories from sample_step — 5 samples
    sample_file = exp_dir / f"train_data_step{sample_step}.jsonl"
    if sample_file.exists():
        sample_rows = []
        with open(sample_file) as fp:
            for i, line in enumerate(fp):
                row = parse_trajectory(json.loads(line))
                row["_file_idx"] = i
                sample_rows.append(row)

        n_examples = min(5, len(sample_rows))
        # Selection strategy: highest reward first, then diverse by turn count.
        # Stable tiebreak by file order so the same call returns the same examples.
        nonzero = [r for r in sample_rows if r["reward"] > 0]
        zero = [r for r in sample_rows if r["reward"] == 0]
        nonzero.sort(key=lambda r: (-r["reward"], r["_file_idx"]))
        # For zero-reward picks, prefer variety: spread by turn count.
        # Pick first, then ones with most/fewest turns/tool_calls.
        if zero:
            zero_first = zero[0]
            zero_by_turns_desc = sorted(zero, key=lambda r: (-r["n_turns"], -r["n_tool_calls"], r["_file_idx"]))
            zero_by_turns_asc = sorted(zero, key=lambda r: (r["n_turns"], r["n_tool_calls"], r["_file_idx"]))
            # Try to assemble diverse zero picks: first, longest, shortest, then fillers in file order
            zero_picks = []
            seen = set()
            for pick in [zero_first, zero_by_turns_desc[0], zero_by_turns_asc[0]]:
                if pick["_file_idx"] not in seen:
                    zero_picks.append(pick)
                    seen.add(pick["_file_idx"])
            for r in zero:
                if r["_file_idx"] not in seen:
                    zero_picks.append(r)
                    seen.add(r["_file_idx"])
                if len(zero_picks) >= n_examples:
                    break
        else:
            zero_picks = []

        # Combine: nonzero (best) first, fill with zero picks
        picks = nonzero[: n_examples] + zero_picks
        picks = picks[: n_examples]

        if not picks:
            print(f"_(no trajectories found in step {sample_step})_")
        else:
            print(f"**Example trajectories from step {sample_step}** ({len(picks)} sampled, chunks truncated for readability):")
            print()
            for i, row in enumerate(picks, 1):
                if row["reward"] > 0:
                    label = f"#{i} — reward {row['reward']:.4f} (nonzero)"
                else:
                    label = f"#{i} — reward 0.0000 (zero, idx={row['_file_idx']})"
                print(format_example(row, label))

    print()
    print(f"_Analysis generated by `analyze_rollouts.py` over `train_data_step{args.step_start}.jsonl` through `train_data_step{args.step_end}.jsonl`._")


if __name__ == "__main__":
    main()
