#!/usr/bin/env python3
"""Held-out search-count profile + termination gap (Appendix figure).

Builds heldout_em_vs_searchcount.png and the supporting
table: at each run's best-EM checkpoint, per held-out question, recover the number
of search calls (count of <function=search> in the rollout transcript) and the
per-item exact-match (the eval cell's own metric_score), pooled over 3 seeds x 4
multi-hop benchmarks. Left panel: mean EM vs search count for completed (non-aborted)
rollouts. Right panel: search-count distribution + per-shape abort rate.

Data source: the public eval-results dataset (per-question intermediate_data.json).
Best-EM checkpoints are the seed-horizon best-avg-EM steps from the held-out CSVs
(seed-horizon caps 42:180 / 43:230 / 44:310); recompute with
  python scripts/print_best_em_checkpoints.py   (or read them off plot_heldout.py).

Run: python3 scripts/plot_searchcount.py
Needs network (downloads ~36 eval cells, ~25 MB each, parsed then deleted).
"""
import csv, json, os, re, subprocess, tempfile, urllib.request
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "figures", "heldout_em_vs_searchcount.png")
BASE = "https://huggingface.co/datasets/pantomiman/reason-over-search-eval-results/resolve/main"
API = "https://huggingface.co/api/datasets/pantomiman/reason-over-search-eval-results/tree/main"

DATASETS = ["musique", "2wikimultihopqa", "hotpotqa", "bamboogle"]   # multi-hop only
SHAPES = ["F1-only", "F1+format", "EM-only"]
COL = {"F1-only": "#1f77b4", "F1+format": "#2ca02c", "EM-only": "#ff7f0e"}
RUNCODE = {"F1-only": "m5_1", "F1+format": "m5_5", "EM-only": "m5_6"}
S42DIR = {"F1-only": "m9", "F1+format": "m9_1", "EM-only": "m9_2"}     # seed-42 lives in m9/m9_1/m9_2
# best-avg-EM checkpoint per run within the seed horizon (D1 caps); from the held-out CSVs.
BEST = {("F1-only", 42): 110, ("F1-only", 43): 210, ("F1-only", 44): 310,
        ("F1+format", 42): 100, ("F1+format", 43): 210, ("F1+format", 44): 270,
        ("EM-only", 42): 170, ("EM-only", 43): 120, ("EM-only", 44): 310}
NCALL = re.compile(r"<function=search>")


def api(path):
    with urllib.request.urlopen(f"{API}/{path}?recursive=false", timeout=30) as fh:
        return json.load(fh)


def cell_url(shape, sd, step, ds):
    if sd in (43, 44):
        return f"{BASE}/m9_5/{RUNCODE[shape]}_s{sd}/step_{step}/{ds}/intermediate_data.json"
    base = S42DIR[shape]                      # seed 42: step is embedded in the dir name
    entries = api(f"{base}/{ds}")
    match = [e["path"] for e in entries
             if e["type"] == "directory" and e["path"].endswith(f"_step{step}_seed1")]
    return f"{BASE}/{match[0]}/intermediate_data.json" if match else None


def parse(fp):
    out = []
    for it in json.load(open(fp)):
        o = it["output"]
        nc = len(NCALL.findall(o["final_response"]))
        ms = o.get("metric_score", {}) or {}
        em = float(ms.get("em", 0) or 0)
        stop = o.get("stop_reason", "")
        pred = (o.get("pred") or "").strip()
        aborted = (stop != "stop") or (pred == "")
        out.append((nc, em, 1 if aborted else 0))
    return out


def collect():
    recs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for shape in SHAPES:
            for sd in (42, 43, 44):
                step = BEST[(shape, sd)]
                for ds in DATASETS:
                    url = cell_url(shape, sd, step, ds)
                    if not url:
                        print(f"  no url: {shape} s{sd} {ds}"); continue
                    dst = os.path.join(tmp, f"{shape}_{sd}_{ds}.json")
                    subprocess.run(["curl", "-sL", "--max-time", "180", url, "-o", dst], check=True)
                    recs[(shape, sd, ds)] = parse(dst)
                    os.remove(dst)
                    print(f"  ok {shape} s{sd} {ds} step{step}")
    return recs


def pooled(recs, shape):
    out = []
    for sd in (42, 43, 44):
        for ds in DATASETS:
            out += recs.get((shape, sd, ds), [])
    return out


def main():
    recs = collect()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for shape in SHAPES:
        nb = defaultdict(list)
        for c, em, a in pooled(recs, shape):
            if not a:
                nb[min(c, 5)].append(em)
        xs = [k for k in sorted(nb) if k < 5]
        ax.plot(xs, [sum(nb[k]) / len(nb[k]) for k in xs], "o-", color=COL[shape], label=shape)
    ax.axvline(2, ls=":", c="grey", alpha=.6)
    ax.set_xlabel("# search calls (completed rollouts only)")
    ax.set_ylabel("held-out EM")
    ax.set_title("EM vs search count (non-aborted rollouts)\npooled 3 seeds x 4 multi-hop, best-EM ckpt")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    bins = [0, 1, 2, 3, 4, 5]
    width = 0.26
    for i, shape in enumerate(SHAPES):
        R = pooled(recs, shape); n = len(R)
        dist = defaultdict(int); ab = 0
        for c, em, a in R:
            dist[min(c, 5)] += 1; ab += a
        ax.bar([b + (i - 1) * width for b in bins], [100 * dist[b] / n for b in bins],
               width, color=COL[shape], label=f"{shape} (abort {100 * ab / n:.0f}%)")
    ax.set_xlabel("# search calls (5 = turn cap, ~100% aborted)")
    ax.set_ylabel("% of held-out items")
    ax.set_title("Search-count distribution by reward shape")
    ax.set_xticks(bins); ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
