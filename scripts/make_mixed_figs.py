#!/usr/bin/env python3
"""Chapter-4 figures for the mixed PAI blocks (whole + fractional gangs).

Reads only committed aggregates (run make_mixed_stats.py first):
  results/aggregates/mixed/{mixed_domain_means,mixed_contrasts,mixed_m_levels}.csv
Writes results/figures/fig4_{mixed_domains,mixed_split,
m_cost}.{png,pdf}.

Palette: validated categorical slots (dataviz reference palette, light):
DP-FGD blue, FGD-посл. orange, BestFit aqua; all bars carry value labels
(aqua is below 3:1 contrast on white, so labels are required).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "results" / "aggregates" / "mixed"
OUT = ROOT / "results" / "figures"
SIG = ["MISC", "V100", "V100M32"]

C_DP, C_SEQ, C_ABF = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POL = [("abf_stream", "BestFit", C_ABF), ("seq_stream", "FGD-посл.", C_SEQ),
       ("dp_stream", "DP-FGD", C_DP)]

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def contrast(res, label, scope="ALL"):
    return res[(res.contrast == label) & (res.scope == scope)].iloc[0]


def fig_domains(means):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(SIG))
    w = 0.26
    for i, (arm, name, col) in enumerate(POL):
        vals = [means[(means.arm == arm) & (means.domain == d)].unalloc100.iloc[0] for d in SIG]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11.5, color=INK)
    ax.set_xticks(x, SIG)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(r"$U_{100}$, % ёмкости")
    ax.set_ylim(0, 19)
    ax.legend(ncol=3, loc="upper right")
    save(fig, "fig4_mixed_domains")


def fig_split(res):
    scopes = ["ALL"] + SIG
    names = {"ALL": "все блоки", "MISC": "MISC", "V100": "V100", "V100M32": "V100M32"}
    fgd = [-contrast(res, "seq - aBF", s).mean_pp for s in scopes]
    joint = [-contrast(res, "DP - seq", s).mean_pp for s in scopes]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    y = np.arange(len(scopes))[::-1]
    ax.barh(y, fgd, 0.55, color=C_SEQ, label="мера FGD (BestFit → FGD-посл.)")
    ax.barh(y, joint, 0.55, left=np.array(fgd) + 0.06, color=C_DP,
            label="совместное размещение (FGD-посл. → DP-FGD)")
    for yi, f, j in zip(y, fgd, joint):
        tot = f + j
        ax.text(tot + 0.2, yi, f"{tot:.2f} п.п. ({100 * j / tot:.0f}% — совместность)",
                va="center", fontsize=11.5, color=INK)
    ax.set_yticks(y, [names[s] for s in scopes])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 14)
    ax.set_xlabel("снижение нераспределённой ёмкости относительно BestFit, п.п.")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=11.5)
    save(fig, "fig4_mixed_split")


def fig_m_cost(levels):
    """unalloc@100 of DP-FGD and FGD-посл. for three workload estimates M
    (all blocks, same blocks per row). Dumbbell: dots need no zero baseline,
    so the axis is zoomed and the gap between the policies is the message."""
    lv = levels[levels.scope == "ALL"]
    kinds = [("oracle", "точное M"),
             ("prev", "M прошлого блока"),
             ("uniform", "все классы поровну")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for row, (k, _) in enumerate(kinds):
        y = len(kinds) - 1 - row
        dp = lv[(lv.policy == "dp") & (lv.m == k)].unalloc100.iloc[0]
        sq = lv[(lv.policy == "seq_fgd") & (lv.m == k)].unalloc100.iloc[0]
        ax.plot([dp, sq], [y, y], color="#c9c8c2", lw=6, solid_capstyle="butt", zorder=1)
        ax.text((dp + sq) / 2, y + 0.18, f"разница {sq - dp:.2f} п.п.", ha="center",
                va="bottom", fontsize=12, color=INK, fontweight="bold")
        for v, col, name, ha, dx in ((dp, C_DP, "DP-FGD", "right", -0.12),
                                     (sq, C_SEQ, "FGD-посл.", "left", 0.12)):
            ax.plot(v, y, "o", ms=12, color=col, mec="white", mew=2, zorder=3,
                    label=name if row == 0 else None)
            ax.text(v + dx, y, f"{v:.2f}", ha=ha, va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(kinds))[::-1], [n for _, n in kinds])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, len(kinds) - 0.3)
    ax.set_xlim(5.5, 10)
    ax.set_xlabel(r"$U_{100}$, % ёмкости (меньше — лучше)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig4_m_cost")


def main() -> int:
    means = pd.read_csv(MIX / "mixed_domain_means.csv")
    res = pd.read_csv(MIX / "mixed_contrasts.csv")
    levels = pd.read_csv(MIX / "mixed_m_levels.csv")
    fig_domains(means)
    fig_split(res)
    fig_m_cost(levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
