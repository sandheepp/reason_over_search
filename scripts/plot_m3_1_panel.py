"""Side-by-side training-curve comparison: with-example (z7kcxfof) vs no-example (el6s2d2h).

Inputs (already exported):
  docs/report/archive/m0_a/csv/p1_basic_w_ex_z7kcxfof.csv  (M3   — heavy-tool 2/4)
  docs/report/archive/m0_a/csv/p3_decide_no_ex_el6s2d2h.csv  (M3.1 — standard 1/3)

Output:
  docs/report/archive/m0_a/comparison_z7kcxfof_vs_el6s2d2h.png

Four panels overlaid:
  reward (critic/rewards/mean)
  tool calls (re_search/search_calls/mean)
  num_turns (num_turns/mean)
  response length (response_length/mean)

Smoothing: simple moving-average (window=20) over the 1000 logged points per run.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
CSV_DIR = REPO / "docs/report/archive/m0_a/csv"
OUT = REPO / "docs/report/archive/m0_a/comparison_z7kcxfof_vs_el6s2d2h.png"

RUNS = {
    "z7kcxfof (M3, with example) — heavy-tool 2/4, end reward 0.190 (+28 %), 1046 steps": {
        "csv": CSV_DIR / "p1_basic_w_ex_z7kcxfof.csv",
        "color": "tab:orange",
    },
    "el6s2d2h (M3.1, no example) — standard 1/3, end reward 0.215 (+43 %), 2280 steps": {
        "csv": CSV_DIR / "p3_decide_no_ex_el6s2d2h.csv",
        "color": "tab:blue",
    },
}

PANELS = [
    ("critic/rewards/mean",       "Reward (rollout mean)",        "reward"),
    ("re_search/search_calls/mean", "Tool calls per episode",     "calls"),
    ("num_turns/mean",            "Turns per episode",            "turns"),
    ("response_length/mean",      "Response length (tokens)",     "tokens"),
]

WINDOW = 20


def load(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["_step"]).sort_values("_step").reset_index(drop=True)
    df["_step"] = df["_step"].astype(int)
    return df


def smooth(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=1, center=False).mean()


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes = axes.flatten()

    runs = {label: load(meta["csv"]) for label, meta in RUNS.items()}

    for ax, (col, ylab, _key) in zip(axes, PANELS):
        for label, meta in RUNS.items():
            df = runs[label]
            if col not in df.columns:
                continue
            sub = df[["_step", col]].dropna()
            if sub.empty:
                continue
            x = sub["_step"].to_numpy()
            y = smooth(sub[col], WINDOW).to_numpy()
            ax.plot(x, y, color=meta["color"], lw=1.6, label=label, alpha=0.95)

        ax.set_xlabel("Training step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=1,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Phase-1 v0: with-example (z7kcxfof) vs no-example (el6s2d2h) — training curves "
        "(SMA window = 20, ~1000 logged points / run)",
        fontsize=12,
        y=1.13,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
