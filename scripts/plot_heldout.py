#!/usr/bin/env python3
"""Held-out figures from the held-out eval CSVs.

Generates into docs/figures, for each metric in {EM, F1, ACC}:
  heldout_seed42[_<m>].png, heldout_seed43[_<m>].png, heldout_seed44[_<m>].png  (per-seed panels)
  heldout_combined_matched[_<m>].png                                            (cross-seed mean +/- range, h<=180)
  heldout_bar_matched[_<m>].png                                                 (matched-horizon best, grouped bars)
EM keeps the unsuffixed filenames (used by the Results body); F1/ACC get _f1 / _acc suffixes (Appendix).
Each metric uses its own untrained terse floor as the dashed reference (EM 0.092, F1 0.123, ACC 0.146).

Per-seed truncation (within-seed uniformity): seed 42 -> step 180, seed 43 -> 230, seed 44 -> 310.
Combined / bar figures are matched to the common horizon (step <= 180).

Also prints best-of-run / matched-horizon numbers and a consolidated EM/F1/ACC table for the Appendix.
Run: python3 scripts/plot_heldout.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "docs", "results")
OUT = os.path.join(REPO, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# untrained terse Qwen3.5-0.8B floor per metric (mean row: EM 0.092 / ACC 0.146 / F1 0.123)
METRIC_FLOOR = {"em": 0.092, "f1": 0.123, "acc": 0.146}
METRIC_LABEL = {"em": "avg EM", "f1": "avg token-F1", "acc": "avg ACC (substring-cover)"}
# compact form for y-axis labels (the parenthetical lives in titles/captions, not on the axis)
AXIS_METRIC = {"em": "avg EM", "f1": "avg token-F1", "acc": "avg ACC"}
METRIC_YLIM = {"em": 0.40, "f1": 0.50, "acc": 0.50}
METRICS = ("em", "f1", "acc")

# reward -> {seed: csv basename}
RUNS = {
    "F1-only":      {42: "m9_m5_1_s42",  43: "m9_5_m5_1_s43", 44: "m9_5_m5_1_s44"},
    "F1+format":    {42: "m9_1_m5_5_s42", 43: "m9_5_m5_5_s43", 44: "m9_5_m5_5_s44"},
    "EM-only":      {42: "m9_2_m5_6_s42", 43: "m9_5_m5_6_s43", 44: "m9_5_m5_6_s44"},
}
COLOR = {"F1-only": "#1f77b4", "F1+format": "#ff7f0e", "EM-only": "#2ca02c"}
SEED_HORIZON = {42: 180, 43: 230, 44: 310}
MATCHED_H = 180


DS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def _suffix(metric):
    return "" if metric == "em" else f"_{metric}"


def load(basename, metric="em"):
    """return list of (step:int, avg_<metric>:float) sorted by step."""
    rows = []
    with open(os.path.join(DATA, basename + ".csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((int(r["step"]), float(r["avg_" + metric])))
            except (ValueError, KeyError):
                continue
    return sorted(rows)


def load_rows(basename):
    """return list of full row dicts sorted by step."""
    with open(os.path.join(DATA, basename + ".csv")) as fh:
        rows = list(csv.DictReader(fh))
    return sorted(rows, key=lambda r: int(r["step"]))


def _f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return float("nan")


def per_seed_panel(seed, metric="em"):
    h = SEED_HORIZON[seed]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for reward, seeds in RUNS.items():
        data = [(s, v) for s, v in load(seeds[seed], metric) if s <= h]
        xs = [s for s, _ in data]
        ys = [v for _, v in data]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=COLOR[reward], label=reward)
    floor = METRIC_FLOOR[metric]
    ax.axhline(floor, ls="--", color="grey", lw=1.2, label=f"untrained floor ({floor:.3f})")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel(f"held-out {AXIS_METRIC[metric]} (7 benchmarks)")
    ax.set_title(f"Seed {seed}: held-out {metric.upper()} by reward shape (to step {h})")
    ax.set_ylim(0, METRIC_YLIM[metric])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, f"heldout_seed{seed}{_suffix(metric)}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))


def combined_matched(metric="em"):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    # common step grid <= MATCHED_H present in all three seeds of a reward
    for reward, seeds in RUNS.items():
        per_seed = {sd: dict(load(bn, metric)) for sd, bn in seeds.items()}
        steps = sorted(s for s in per_seed[42] if s <= MATCHED_H
                       and s in per_seed[43] and s in per_seed[44])
        means, lo, hi = [], [], []
        for s in steps:
            vals = [per_seed[42][s], per_seed[43][s], per_seed[44][s]]
            means.append(sum(vals) / 3.0)
            lo.append(min(vals)); hi.append(max(vals))
        ax.plot(steps, means, marker="o", ms=3, lw=1.8, color=COLOR[reward], label=reward)
        ax.fill_between(steps, lo, hi, color=COLOR[reward], alpha=0.18)
    floor = METRIC_FLOOR[metric]
    ax.axhline(floor, ls="--", color="grey", lw=1.2, label=f"untrained floor ({floor:.3f})")
    ax.set_xlabel("checkpoint step (matched horizon, <= 180)")
    ax.set_ylabel(f"held-out {AXIS_METRIC[metric]} (7 benchmarks)")
    ax.set_title(f"Cross-seed mean +/- range by reward shape ({metric.upper()}, matched horizon)")
    ax.set_ylim(0, METRIC_YLIM[metric])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, f"heldout_combined_matched{_suffix(metric)}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))


def combined_bar_matched(metric="em"):
    """Grouped bars: matched-horizon (<=180) best avg <metric>, grouped by reward shape, one bar per seed."""
    rewards = ["F1-only", "F1+format", "EM-only"]
    seeds = [42, 43, 44]
    seed_color = {42: "#4c72b0", 43: "#dd8452", 44: "#55a868"}
    vals = {r: {} for r in rewards}
    for r in rewards:
        for sd in seeds:
            md = [(s, v) for s, v in load(RUNS[r][sd], metric) if s <= MATCHED_H]
            vals[r][sd] = max(md, key=lambda t: t[1])[1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    width = 0.26
    xbase = list(range(len(rewards)))
    for i, sd in enumerate(seeds):
        xs = [x + (i - 1) * width for x in xbase]
        ys = [vals[r][sd] for r in rewards]
        ax.bar(xs, ys, width=width, color=seed_color[sd], label=f"seed {sd}")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
    floor = METRIC_FLOOR[metric]
    ax.axhline(floor, ls="--", color="grey", lw=1.2, label=f"untrained floor ({floor:.3f})")
    ax.set_xticks(xbase)
    ax.set_xticklabels(rewards)
    ax.set_ylabel(f"best avg {metric.upper()} (matched horizon, step <= 180)")
    ax.set_title(f"Matched-horizon held-out {metric.upper()} by reward shape and seed")
    ax.set_ylim(0, METRIC_YLIM[metric])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, f"heldout_bar_matched{_suffix(metric)}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))


def combined_bar_three_metric():
    """Body metric-robustness visual: grouped bars of the matched-horizon (<=180) cross-seed
    mean of EM/F1/ACC at each run's best-avg-EM checkpoint, grouped by reward shape.

    One checkpoint per run (the best-EM / deployment ckpt) read on all three metrics, then
    averaged over the three seeds. This is exactly the cross-seed mean of the Appendix B.6
    consolidated matched-horizon rows, so the figure and that table share provenance. The
    point it makes: EM-only is lowest on all three metrics (the "not an EM artifact" check).
    """
    rewards = ["F1-only", "F1+format", "EM-only"]
    seeds = [42, 43, 44]
    metric_color = {"em": "#4c72b0", "f1": "#dd8452", "acc": "#55a868"}
    metric_name = {"em": "EM", "f1": "F1", "acc": "ACC"}
    means = {r: {} for r in rewards}
    for r in rewards:
        acc = {m: [] for m in METRICS}
        for sd in seeds:
            rows = [row for row in load_rows(RUNS[r][sd]) if int(row["step"]) <= MATCHED_H]
            b = max(rows, key=lambda row: _f(row, "avg_em"))
            for m in METRICS:
                acc[m].append(_f(b, "avg_" + m))
        for m in METRICS:
            means[r][m] = sum(acc[m]) / len(acc[m])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    width = 0.26
    xbase = list(range(len(rewards)))
    for i, m in enumerate(METRICS):
        xs = [x + (i - 1) * width for x in xbase]
        ys = [means[r][m] for r in rewards]
        ax.bar(xs, ys, width=width, color=metric_color[m], label=metric_name[m])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(METRIC_FLOOR[m], ls="--", lw=1.0, color=metric_color[m], alpha=0.7)
    ax.set_xticks(xbase)
    ax.set_xticklabels(rewards)
    ax.set_ylabel("cross-seed mean (matched horizon)")
    ax.set_title("Held-out EM / F1 / ACC by reward shape (matched horizon, cross-seed mean)")
    ax.set_ylim(0, 0.50)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", ncol=3, title="metric (dashed line = its untrained floor)")
    fig.tight_layout()
    p = os.path.join(OUT, "heldout_bar_three_metric_matched.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))
    print("\n=== matched-horizon cross-seed mean EM/F1/ACC at best-EM ckpt (body Fig 4.1f) ===")
    for r in rewards:
        print(f"{r:12} EM {means[r]['em']:.3f}  F1 {means[r]['f1']:.3f}  ACC {means[r]['acc']:.3f}")


def print_numbers():
    print("\n=== avg-EM per run: matched<=180 / best at seed horizon (caps 42:180/43:230/44:310) ===")
    print(f"{'reward':12}{'seed':>5}{'matched':>9}{'@step':>7}{'seedHzn':>9}{'@step':>7}{'accBest':>9}")
    for reward, seeds in RUNS.items():
        for sd, bn in seeds.items():
            rows = load_rows(bn)
            md = [r for r in rows if int(r["step"]) <= MATCHED_H]
            sd_rows = [r for r in rows if int(r["step"]) <= SEED_HORIZON[sd]]
            mbest = max(md, key=lambda r: _f(r, "avg_em"))
            sbest = max(sd_rows, key=lambda r: _f(r, "avg_em"))
            print(f"{reward:12}{sd:>5}{_f(mbest,'avg_em'):>9.3f}{mbest['step']:>7}"
                  f"{_f(sbest,'avg_em'):>9.3f}{sbest['step']:>7}{_f(sbest,'avg_acc'):>9.3f}")


def print_three_metric():
    """EM / F1 / ACC together at the best-avg-EM checkpoint (one ckpt, three metrics), both horizons.

    For the Appendix consolidated three-metric table; selecting one checkpoint (best avg EM) avoids the
    per-metric-different-checkpoint confusion and shows the metrics are concordant.
    """
    for horizon_name, cap in (("matched (<=180)", lambda sd: MATCHED_H),
                              ("seed-horizon", lambda sd: SEED_HORIZON[sd])):
        print(f"\n=== three-metric (EM/F1/ACC) at best-avg-EM ckpt, {horizon_name} ===")
        print(f"{'reward':12}{'seed':>5}{'step':>7}{'EM':>8}{'F1':>8}{'ACC':>8}")
        for reward, seeds in RUNS.items():
            for sd, bn in seeds.items():
                rows = [r for r in load_rows(bn) if int(r["step"]) <= cap(sd)]
                b = max(rows, key=lambda r: _f(r, "avg_em"))
                print(f"{reward:12}{sd:>5}{b['step']:>7}{_f(b,'avg_em'):>8.3f}"
                      f"{_f(b,'avg_f1'):>8.3f}{_f(b,'avg_acc'):>8.3f}")


def print_spread():
    """per-dataset best-ckpt spread within seed horizon (appendix B.4)."""
    print("\n=== per-dataset best-ckpt spread (within seed horizon; EM=F1 best-ckpt check) ===")
    for reward, seeds in RUNS.items():
        for sd, bn in seeds.items():
            rows = [r for r in load_rows(bn) if int(r["step"]) <= SEED_HORIZON[sd]]
            best_em = max(rows, key=lambda r: _f(r, "avg_em"))
            best_f1 = max(rows, key=lambda r: _f(r, "avg_f1"))
            peaks = {d: int(max(rows, key=lambda r: _f(r, d + "_em"))["step"]) for d in DS}
            n_at = sum(1 for d in peaks if peaks[d] == int(best_em["step"]))
            rng = f"{min(peaks.values())}-{max(peaks.values())}"
            same = "EM=F1" if best_em["step"] == best_f1["step"] else f"EM@{best_em['step']}!=F1@{best_f1['step']}"
            print(f"{reward}-s{sd}: best-avg step_{best_em['step']}  {n_at}/7 peak there  range {rng}  [{same}]")


if __name__ == "__main__":
    for metric in METRICS:
        for seed in (42, 43, 44):
            per_seed_panel(seed, metric)
        combined_matched(metric)
        combined_bar_matched(metric)
    combined_bar_three_metric()
    print_numbers()
    print_three_metric()
    print_spread()
