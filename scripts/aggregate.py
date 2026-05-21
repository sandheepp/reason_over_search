#!/usr/bin/env python3
"""Aggregate metric_score.txt files across runs and write a markdown report.

Walks `evaluation_search_r1/results/<dataset>/<dataset>_<timestamp>_<save_note>/metric_score.txt`,
groups by (dataset, variant) using the `save_note` convention
`search_r1_<variant>_seed<N>`, and reports per-seed scores plus mean over seeds.
Also surfaces trace-health stats (close-rate, length-truncation rate, mean
completion tokens) per (dataset, variant) from each run's intermediate_data.json.

Usage:
  scripts/aggregate.py [--output RESULTS.md] [--results-dir DIR]
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "evaluation_search_r1" / "results"

DATASETS = ["bamboogle", "nq", "triviaqa", "popqa", "musique", "2wikimultihopqa", "hotpotqa"]
METRICS = ["em", "f1", "acc"]
SAVE_NOTE_RE = re.compile(r"search_r1_(?P<variant>base|instruct)_(?:seed|run)(?P<seed>\d+)$")


def parse_metric_file(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def parse_trace_health(path: Path) -> dict[str, float] | None:
    """Compute close-rate / length-truncation / mean tokens from intermediate_data.json.

    close-rate: fraction of examples whose `final_response` contains `</answer>`.
    length-truncated: fraction with stop_reason indicating length cap (anything other
    than 'stop'/'eos'). Both are in [0,1]; close-rate is what docs/archive/COMPARISON_PLAN_B_v0.md
    asked to surface per (dataset, variant).
    """
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    total = len(data)
    closed = 0
    length_trunc = 0
    completion_tokens: list[int] = []
    for rec in data:
        out = rec.get("output") or {}
        if "</answer>" in (out.get("final_response") or ""):
            closed += 1
        # SGLang stop_reason of "stop" / "eos" = clean stop; anything else (e.g. "length") = truncation.
        sr = out.get("stop_reason")
        if sr is not None and sr not in ("stop", "eos", "stop_str"):
            length_trunc += 1
        ct = out.get("completion_tokens")
        if isinstance(ct, (int, float)):
            completion_tokens.append(int(ct))
    return {
        "n": total,
        "close_rate": closed / total,
        "length_trunc_rate": length_trunc / total,
        "mean_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0.0,
    }


def collect(results_dir: Path):
    """Return ({(dataset, variant, seed): metrics}, {(dataset, variant, seed): trace_health})."""
    runs: dict[tuple[str, str, int], dict[str, float]] = {}
    trace: dict[tuple[str, str, int], dict[str, float]] = {}
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        ds = ds_dir.name
        if ds not in DATASETS:
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metric_file = run_dir / "metric_score.txt"
            if not metric_file.exists():
                continue
            # Extract trailing save_note from the run_dir name.
            # Format: <dataset>_<YYYY>_<MM>_<DD>_<HH>_<MM>_<save_note>
            name = run_dir.name
            if not name.startswith(f"{ds}_"):
                continue
            tail = name[len(ds) + 1:]
            # The timestamp has 5 numeric parts; everything after is the save_note.
            parts = tail.split("_")
            if len(parts) <= 5:
                continue
            save_note = "_".join(parts[5:])
            m = SAVE_NOTE_RE.match(save_note)
            if not m:
                continue
            variant = m.group("variant")
            seed = int(m.group("seed"))
            key = (ds, variant, seed)
            runs[key] = parse_metric_file(metric_file)
            health = parse_trace_health(run_dir / "intermediate_data.json")
            if health is not None:
                trace[key] = health
    return runs, trace


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render(runs, metric: str) -> str:
    """Markdown table for one metric."""
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (ds, variant, seed), m in runs.items():
        if metric in m:
            grouped[(ds, variant)].append((seed, m[metric]))

    seeds_present = sorted({seed for v in grouped.values() for seed, _ in v})
    if not seeds_present:
        return f"### {metric.upper()}\n\n_no runs found_\n"

    lines = [f"### {metric.upper()}", ""]
    header = ["Dataset", "Variant", *[f"seed={s}" for s in seeds_present], "mean", "std", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for ds in DATASETS:
        for variant in ("base", "instruct"):
            row = [ds, variant]
            scores = dict(grouped.get((ds, variant), []))
            seed_vals = [scores.get(s) for s in seeds_present]
            row.extend(fmt(v) for v in seed_vals)
            present = [v for v in seed_vals if v is not None]
            row.append(fmt(statistics.mean(present)) if present else "—")
            row.append(fmt(statistics.stdev(present)) if len(present) > 1 else "—")
            row.append(str(len(present)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def grand_average(runs, metric: str) -> str:
    """Mean across (datasets × seeds) per variant."""
    by_variant: dict[str, list[float]] = defaultdict(list)
    for (ds, variant, seed), m in runs.items():
        if metric in m:
            by_variant[variant].append(m[metric])
    if not by_variant:
        return ""
    lines = [f"### Grand average {metric.upper()} across all runs", ""]
    lines.append("| Variant | mean | n_runs |")
    lines.append("|---|---|---|")
    for variant in ("base", "instruct"):
        vals = by_variant.get(variant, [])
        if vals:
            lines.append(f"| {variant} | {statistics.mean(vals):.3f} | {len(vals)} |")
        else:
            lines.append(f"| {variant} | — | 0 |")
    lines.append("")
    return "\n".join(lines)


def render_trace_health(trace) -> str:
    """Markdown table for close-rate, length-truncation, and mean completion tokens."""
    if not trace:
        return ""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for (ds, variant, seed), h in trace.items():
        grouped[(ds, variant)].append((seed, h))
    seeds_present = sorted({seed for v in grouped.values() for seed, _ in v})

    lines = ["## Trace health", ""]
    lines.append("Close-rate = fraction of examples whose `final_response` contains `</answer>`. ")
    lines.append("Length-truncated = fraction whose SGLang `stop_reason` was anything other than `stop`/`eos`/`stop_str` (typically the per-step token cap firing).")
    lines.append("")

    # Table 1: close-rate per (dataset, variant), per seed + mean
    lines.append("### Close-rate")
    lines.append("")
    header = ["Dataset", "Variant", *[f"seed={s}" for s in seeds_present], "mean", "n_examples"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds in DATASETS:
        for variant in ("base", "instruct"):
            scores = {seed: h for seed, h in grouped.get((ds, variant), [])}
            seed_vals = [scores.get(s, {}).get("close_rate") for s in seeds_present]
            present = [v for v in seed_vals if v is not None]
            if not present:
                continue
            row = [ds, variant]
            row.extend(f"{v*100:.1f}%" if v is not None else "—" for v in seed_vals)
            row.append(f"{statistics.mean(present)*100:.1f}%")
            n_total = sum(int(scores.get(s, {}).get("n", 0)) for s in seeds_present)
            row.append(str(n_total))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table 2: length-truncation rate
    lines.append("### Length-truncation rate")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds in DATASETS:
        for variant in ("base", "instruct"):
            scores = {seed: h for seed, h in grouped.get((ds, variant), [])}
            seed_vals = [scores.get(s, {}).get("length_trunc_rate") for s in seeds_present]
            present = [v for v in seed_vals if v is not None]
            if not present:
                continue
            row = [ds, variant]
            row.extend(f"{v*100:.1f}%" if v is not None else "—" for v in seed_vals)
            row.append(f"{statistics.mean(present)*100:.1f}%")
            n_total = sum(int(scores.get(s, {}).get("n", 0)) for s in seeds_present)
            row.append(str(n_total))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table 3: mean completion tokens
    lines.append("### Mean completion tokens (whole trace, summed over turns)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds in DATASETS:
        for variant in ("base", "instruct"):
            scores = {seed: h for seed, h in grouped.get((ds, variant), [])}
            seed_vals = [scores.get(s, {}).get("mean_completion_tokens") for s in seeds_present]
            present = [v for v in seed_vals if v is not None]
            if not present:
                continue
            row = [ds, variant]
            row.extend(f"{v:.0f}" if v is not None else "—" for v in seed_vals)
            row.append(f"{statistics.mean(present):.0f}")
            n_total = sum(int(scores.get(s, {}).get("n", 0)) for s in seeds_present)
            row.append(str(n_total))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "RESULTS.md")
    args = ap.parse_args()

    runs, trace = collect(args.results_dir)

    md = ["# Search-R1 evaluation results", ""]
    md.append(f"_Source: `{args.results_dir}` ({len(runs)} runs)_")
    md.append("")
    md.append("## Per-seed scores")
    md.append("")
    for metric in METRICS:
        md.append(render(runs, metric))
    md.append("## Grand averages")
    md.append("")
    for metric in METRICS:
        ga = grand_average(runs, metric)
        if ga:
            md.append(ga)
    th = render_trace_health(trace)
    if th:
        md.append(th)

    args.output.write_text("\n".join(md))
    print(f"wrote {args.output} ({len(runs)} runs aggregated, {len(trace)} with trace health)")


if __name__ == "__main__":
    main()
