#!/usr/bin/env python3
"""Aggregate held-out eval results from an HF dataset into per-run tables.

Pulls metric_score.txt for each (run_tag, step, dataset) combo from the results
dataset under m9_5/<run_tag>/step_<N>/<dataset>/, parses em/acc/f1, and produces:

  * Per-run CSVs: m9_5_<run_tag>.csv with columns
      step, ds_em, ds_f1, ds_acc for each dataset, then avg_em, avg_f1, avg_acc across the 7.
      (acc = FlashRAG substring-cover EM / Sub-ExactMatch.)
  * Markdown tables (printed) for direct paste into the results docs.

Usage:
    python scripts/aggregate_m9_5.py [--repo <your-hf-repo>]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import urllib.request
from pathlib import Path


RUN_TAGS = ["m5_1_s43", "m5_5_s43", "m5_6_s43", "m5_6_s44", "m5_1_s44", "m5_5_s44"]
MAX_STEP = {
    "m5_1_s43": 230,
    "m5_5_s43": 250,
    "m5_6_s43": 240,  # EM-only seed-43 leg (HF layout m9_5/m5_6_s43/)
    "m5_6_s44": 310,
    "m5_1_s44": 310,
    "m5_5_s44": 310,
}
DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
DS_LABELS = {"nq": "NQ", "triviaqa": "TriviaQA", "popqa": "PopQA", "hotpotqa": "HotpotQA",
             "2wikimultihopqa": "2Wiki", "musique": "MuSiQue", "bamboogle": "Bamboogle"}
METRIC_RE = re.compile(r"^(em|acc|f1):\s*([0-9.]+)\s*$")


def parse_metric(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def fetch_one(repo: str, run_tag: str, step: int, ds: str, cache: Path, token: str) -> dict | None:
    rel = f"m9_5/{run_tag}/step_{step}/{ds}/metric_score.txt"
    local = cache / rel
    if local.exists():
        with open(local) as fh:
            return parse_metric(fh.read())
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode("utf-8")
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "w") as fh:
            fh.write(data)
        return parse_metric(data)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="pantomiman/reason-over-search-eval-results")
    ap.add_argument("--out-dir", default="docs/results")
    ap.add_argument("--cache", default=".cache/m9_5_aggregate")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("WARN: HF_TOKEN not set; only public-no-auth fetches will work", file=sys.stderr)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    # Per-run aggregation
    for run_tag in RUN_TAGS:
        max_step = MAX_STEP[run_tag]
        steps = list(range(10, max_step + 1, 10))
        rows = []
        missing = 0
        for s in steps:
            row = {"step": s}
            ems = []; f1s = []; accs = []
            for ds in DATASETS:
                m = fetch_one(args.repo, run_tag, s, ds, cache, token)
                if m is None:
                    row[f"{ds}_em"] = None
                    row[f"{ds}_f1"] = None
                    row[f"{ds}_acc"] = None
                    missing += 1
                else:
                    row[f"{ds}_em"] = m.get("em")
                    row[f"{ds}_f1"] = m.get("f1")
                    row[f"{ds}_acc"] = m.get("acc")
                    if m.get("em") is not None: ems.append(m["em"])
                    if m.get("f1") is not None: f1s.append(m["f1"])
                    if m.get("acc") is not None: accs.append(m["acc"])
            row["avg_em"] = sum(ems) / len(ems) if ems else None
            row["avg_f1"] = sum(f1s) / len(f1s) if f1s else None
            row["avg_acc"] = sum(accs) / len(accs) if accs else None
            row["n_complete"] = len(ems)
            rows.append(row)

        # CSV
        csv_path = out_dir / f"m9_5_{run_tag}.csv"
        with open(csv_path, "w", newline="") as fh:
            cols = ["step"]
            for ds in DATASETS:
                cols += [f"{ds}_em", f"{ds}_f1", f"{ds}_acc"]
            cols += ["avg_em", "avg_f1", "avg_acc", "n_complete"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[{run_tag}] {len(steps)} steps, {missing} missing cells -> {csv_path}")

        # Best ckpt by avg_em
        best = max((r for r in rows if r["avg_em"] is not None), key=lambda r: r["avg_em"], default=None)
        if best:
            print(f"  best avg_em = {best['avg_em']:.3f} at step_{best['step']} ({best['n_complete']}/7 datasets)")

    # Cross-run summary table (avg_em at common step horizons)
    print("\n=== cross-run avg_em (step_50, 100, 150, 200, last) ===")
    print(f"{'run':<11} step_50  step_100 step_150 step_200 last")
    for run_tag in RUN_TAGS:
        csv_path = out_dir / f"m9_5_{run_tag}.csv"
        steps_data = {}
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                steps_data[int(row["step"])] = float(row["avg_em"]) if row["avg_em"] else None
        last_step = MAX_STEP[run_tag]
        cells = []
        for s in [50, 100, 150, 200, last_step]:
            v = steps_data.get(s)
            cells.append(f"{v:.3f}" if v is not None else "  -  ")
        print(f"{run_tag:<11} {cells[0]:8} {cells[1]:8} {cells[2]:8} {cells[3]:8} {cells[4]}")


if __name__ == "__main__":
    main()
