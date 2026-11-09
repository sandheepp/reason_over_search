#!/usr/bin/env python3
"""GRPO training-dynamics figures from the per-step training CSVs.

Generates into docs/figures:
  Body:
    dyn_seed42.png / dyn_seed43.png / dyn_seed44.png   per-seed, 5 body panels, 3 reward shapes overlaid
    dyn_xseed_f1only.png / _f1format.png / _emonly.png  cross-seed: one reward shape, seeds 42/43/44 overlaid
    dyn_avg_matched.png                                  cross-seed mean +/- range per reward shape (matched <=180)
  Appendix (training-health):
    dyn_health_seed42.png / _seed43.png / _seed44.png    per-seed, 6 health/termination panels, 3 shapes overlaid

Every line is the centred moving-average (EMA-style) drawn solid over the raw per-step series (faint), so
the smoothed trend and the background noise are both visible (the noise is the stability signal).

Body metrics: rollout reward, tool calls/sample, response length (tokens), total tokens/sample, length-clip ratio.
Health metrics: gradient norm, loss, KL penalty, approx entropy, aborted ratio, natural-termination rate.
Run: python3 scripts/plot_dynamics.py
"""
from __future__ import annotations
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print legibility: the 2x3 / 2x2 / 1x3 grids render at \linewidth (~6.3 in)
# from 13-17 in figures, a ~2-2.5x shrink, so default ~10 pt type prints at
# ~4 pt. The sizes below keep tick labels >= ~7 pt at print size.
plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 17, "axes.titlesize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 15, "legend.title_fontsize": 15,
})

REPO = pathlib.Path(__file__).resolve().parent.parent
D = REPO / "docs" / "results"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# reward shape -> {seed: csv tag}; consistent colours with the held-out figures
RUNS = {
    "F1-only":   ({42: "m5_1_s42", 43: "m5_1_s43", 44: "m5_1_s44"}, "tab:blue"),
    "F1+format": ({42: "m5_5_s42", 43: "m5_5_s43", 44: "m5_5_s44"}, "tab:green"),
    "EM-only":   ({42: "m5_6_s42", 43: "m5_6_s43", 44: "m5_6_s44"}, "tab:orange"),
}
SEED_COLOR = {42: "#6a3d9a", 43: "#e31a1c", 44: "#1f78b4"}
# STANDARD HORIZONS (load-bearing decision):
# per-seed nominal cap 42->180, 43->230, 44->312 (one epoch); the last full cadence /
# checkpoint is step_310, so figures cap there (the seed-44 C32 2-step partial wrap
# 311-312 is dropped). Identical to plot_heldout.py SEED_HORIZON.
SEED_HORIZON = {42: 180, 43: 230, 44: 310}
MATCHED_H = 180     # common horizon for the cross-seed average (the seeds' shared minimum)

# (csv column, axis label, smoothing window)
BODY_PANELS = [
    ("ros_reward_mean",                "Rollout reward (mean)",          5),
    ("ros_mean_tool_calls_per_sample", "Tool calls / sample",            5),
    ("ros_resp_len_mean",              "Response length (tokens)",       5),
    ("ros_mean_total_tokens_per_sample","Total tokens / sample",         5),
    ("ros_len_clip_ratio",             "Length-clip ratio (frac at cap)",5),
]
HEALTH_PANELS = [
    ("grad_norm",                    "Gradient norm",                 9),
    ("loss",                         "Loss",                          9),
    ("kl_penalty",                   "KL penalty",                    9),
    ("approx_entropy",               "Approx. entropy",               9),
    ("ros_aborted_ratio",            "Aborted ratio (no valid answer)",5),
    ("ros_natural_termination_rate", "Natural-termination rate",      5),
]
AVG_PANELS = [
    ("ros_reward_mean",                "Rollout reward (mean)",    5),
    ("ros_mean_tool_calls_per_sample", "Tool calls / sample",      5),
    ("ros_resp_len_mean",              "Response length (tokens)", 5),
]


def load(tag):
    seed = int(tag.rsplit("_s", 1)[1])  # e.g. "m5_1_s44" -> 44
    df = pd.read_csv(D / f"train_{tag}.csv").sort_values("_step")
    return df[df["_step"] <= SEED_HORIZON[seed]]


def ema(s, w):
    return s.rolling(w, min_periods=1, center=True).mean()


def _line(ax, sub, col, color, w, label):
    """raw faint + smoothed solid, same colour; raw makes the noise/stability visible."""
    ax.plot(sub["_step"], sub[col], color=color, lw=0.7, alpha=0.22)
    ax.plot(sub["_step"], ema(sub[col], w), color=color, lw=1.9, alpha=0.95, label=label)


def _panel_grid(panels):
    """2x3 grid; returns (fig, axes-list, the spare axis for the legend)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), constrained_layout=True)
    axes = axes.flatten()
    return fig, axes


def per_seed(seed, panels, fname, title, legend_entries_from):
    fig, axes = _panel_grid(panels)
    for ax, (col, ylab, w) in zip(axes, panels):
        for reward, (seeds, color) in RUNS.items():
            df = TRAIN.get(seeds[seed])
            if df is None or col not in df.columns:
                continue
            sub = df[["_step", col]].dropna()
            if not sub.empty:
                _line(ax, sub, col, color, w, reward)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    if len(panels) < len(axes):
        # spare axis -> put the legend there
        for ax in axes[len(panels):]:
            ax.axis("off")
        axes[len(panels)].legend(h, l, loc="center", fontsize=15,
                                 title="reward shape\n(solid=smoothed, faint=raw)")
        fig.suptitle(title, fontsize=19)
    else:
        # grid full -> figure-level legend below the panels (keeps the top + panels clear)
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3,
                   fontsize=15, frameon=False, title="reward shape (solid = smoothed, faint = raw)")
        fig.suptitle(title, fontsize=19)
    p = OUT / fname
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(REPO))


def cross_seed(reward, fname):
    seeds, _ = RUNS[reward]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    axes = axes.flatten()
    panels = BODY_PANELS[:2] + BODY_PANELS[2:3] + BODY_PANELS[4:5]  # reward, tool, resp_len, len_clip
    for ax, (col, ylab, w) in zip(axes, panels):
        for sd in (42, 43, 44):
            df = TRAIN.get(seeds[sd])
            if df is None or col not in df.columns:
                continue
            sub = df[["_step", col]].dropna()
            if not sub.empty:
                _line(ax, sub, col, SEED_COLOR[sd], w, f"seed {sd}")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=14, title="(solid=smoothed, faint=raw)")
    fig.suptitle(f"Cross-seed training dynamics: {reward} recipe, seeds 42 / 43 / 44", fontsize=19)
    p = OUT / fname
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(REPO))


def average_matched(fname):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for ax, (col, ylab, w) in zip(axes, AVG_PANELS):
        for reward, (seeds, color) in RUNS.items():
            # per-step mean +/- min-max across the 3 seeds, steps present in all three (<=180)
            series = {}
            for sd in (42, 43, 44):
                df = TRAIN.get(seeds[sd])
                if df is None or col not in df.columns:
                    continue
                s = df[df["_step"] <= MATCHED_H][["_step", col]].dropna().set_index("_step")[col]
                series[sd] = s
            if len(series) < 3:
                continue
            common = sorted(set(series[42].index) & set(series[43].index) & set(series[44].index))
            if not common:
                continue
            mat = pd.DataFrame({sd: series[sd].reindex(common) for sd in (42, 43, 44)})
            mean = ema(mat.mean(axis=1), w)
            lo = ema(mat.min(axis=1), w)
            hi = ema(mat.max(axis=1), w)
            ax.plot(common, mean, color=color, lw=2.0, label=reward)
            ax.fill_between(common, lo, hi, color=color, alpha=0.15)
        ax.set_xlabel("training step (matched horizon, <= 180)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, MATCHED_H)
    axes[0].legend(fontsize=14)
    fig.suptitle("Cross-seed mean +/- range by reward shape (matched horizon, step <= 180)", fontsize=19)
    p = OUT / fname
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(REPO))


# load all 9 runs
TRAIN = {}
for _, (seeds, _) in RUNS.items():
    for tag in seeds.values():
        TRAIN[tag] = load(tag)

if __name__ == "__main__":
    for sd in (42, 43, 44):
        per_seed(sd, BODY_PANELS, f"dyn_seed{sd}.png",
                 f"Seed {sd}: GRPO training dynamics by reward shape (Qwen3.5-0.8B, MuSiQue)", BODY_PANELS)
    cross_seed("F1-only",  "dyn_xseed_f1only.png")
    cross_seed("F1+format","dyn_xseed_f1format.png")
    cross_seed("EM-only",  "dyn_xseed_emonly.png")
    average_matched("dyn_avg_matched.png")
    for sd in (42, 43, 44):
        per_seed(sd, HEALTH_PANELS, f"dyn_health_seed{sd}.png",
                 f"Seed {sd}: GRPO training-health diagnostics by reward shape", HEALTH_PANELS)
    print("DONE")
