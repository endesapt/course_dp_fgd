#!/usr/bin/env python3
"""F1 analysis as pre-registered in PREREG.md.

rsync first (summaries only):
  rsync -a server:fgdg2-run/results/pai_f1/ out/pai_f1/ \
        --include='*/' --include='summary.csv' --exclude='*'
The `full` variant is read from ../2026-11-18_mixed_gangs/out/pai_mixed
({dp,seq,abf}_stream). Partial results are fine: every contrast uses matched
blocks only (and reports n).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F1 = HERE / "out/pai_f1"
MIX = HERE.parent / "2026-11-18_mixed_gangs/out/pai_mixed"
SIG = ["MISC", "V100", "V100M32"]
N_BOOT, SEED = 4000, 0
METRICS = ["unalloc100", "unalloc_end"]


def read_tree(root: Path, arm_map: dict) -> list[dict]:
    rows = []
    for f in root.glob("*/*/block*/summary.csv"):
        arm, dom, blk = f.parts[-4], f.parts[-3], f.parts[-2]
        if arm not in arm_map:
            continue
        variant, pol = arm_map[arm]
        s = pd.read_csv(f).iloc[-1]
        u100 = str(s["unalloc_pct_at_100"]).rstrip("%")
        rows.append({"variant": variant, "pol": pol, "domain": dom, "block": int(blk[5:]),
                     "unalloc100": float(u100) if u100 not in ("n/a", "nan") else np.nan,
                     "unalloc_end": float(s["unalloc_pct_at_end"])})
    return rows


def load() -> pd.DataFrame:
    vmap = {"sfg": "splitfracgang", "swg": "splitwholegang", "sall": "splitall",
            "sfgbg": "splitfracgang_fgdbg"}
    f1_arms = {f"{k}__{p}": (v, p) for k, v in vmap.items() for p in ("dp", "seq", "abf")}
    mix_arms = {f"{p}_stream": ("full", p) for p in ("dp", "seq", "abf")}
    return pd.DataFrame(read_tree(F1, f1_arms) + read_tree(MIX, mix_arms))


def boot(d: pd.Series):
    """Cluster bootstrap over 10 consecutive blocks within a domain (same scheme
    as analyze_flat_m.cluster_ci); returns (mean, lo, hi, two-sided p)."""
    rng = np.random.default_rng(SEED)
    keys = [f"{dom}:{b // 10}" for dom, b in d.index]
    sums = pd.DataFrame({"d": d.values, "c": keys}).groupby("c")["d"].agg(["sum", "count"])
    s, n = sums["sum"].values, sums["count"].values
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(p)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def gap(w: pd.DataFrame, variant: str, a: str, b: str) -> pd.Series:
    ca, cb = (variant, a), (variant, b)
    if ca not in w or cb not in w:
        return pd.Series(index=w.index[:0], dtype=float)
    return (w[ca] - w[cb]).dropna()


def fmt(label, scope, r, n, extra=""):
    m, lo, hi, p = r
    return (f"  {label:44s} {scope:8s} {m:+.3f} pp CI[{lo:+.3f},{hi:+.3f}] "
            f"p={p:.4f}{extra} n={n}")


def scoped(d: pd.Series, scope: str) -> pd.Series:
    return d if scope == "ALL" else d[d.index.get_level_values(0) == scope]


def main():
    df = load()
    if df.empty:
        raise SystemExit(f"no summaries under {F1} / {MIX}")
    print("cells per variant x policy:")
    print(df.groupby(["variant", "pol"]).size().unstack().to_string(), "\n")

    for metric in METRICS:
        w = df.pivot_table(index=["domain", "block"], columns=["variant", "pol"], values=metric)
        print(f"################ metric = {metric}")
        print("mean by domain (each cell over its own available blocks):")
        print(w.groupby(level=0).mean().round(3).T.to_string(), "\n")

        G = {v: gap(w, v, "dp", "seq") for v in
             ["full", "splitfracgang", "splitwholegang", "splitall", "splitfracgang_fgdbg"]}
        A = {v: gap(w, v, "dp", "abf") for v in G}
        S = {v: gap(w, v, "seq", "abf") for v in G}

        print("Per-variant gaps G = DP-seq, A = DP-aBF, S = seq-aBF:")
        for v in G:
            for name, X in (("G", G), ("A", A), ("S", S)):
                for scope in ["ALL"] + SIG:
                    d = scoped(X[v], scope)
                    if len(d) >= 5:
                        print(fmt(f"{name}({v})", scope, boot(d), len(d)))
        print()

        print("SANITY |G(splitall)| < 0.05 pp per domain:")
        for scope in SIG:
            d = scoped(G["splitall"], scope)
            if len(d):
                ok = abs(d.mean()) < 0.05
                print(f"  {scope:8s} mean={d.mean():+.4f} max|d|={d.abs().max():.4f} "
                      f"n={len(d)} -> {'OK' if ok else 'VIOLATED'}")
        print()

        contrasts = {
            "H1 dG_fg = G(full) - G(splitfracgang)": (G["full"], G["splitfracgang"]),
            "H2 dG_wg = G(full) - G(splitwholegang)": (G["full"], G["splitwholegang"]),
            "H3 dG_bg = G(sfg_fgdbg) - G(splitfracgang)": (G["splitfracgang_fgdbg"],
                                                          G["splitfracgang"]),
        }
        print("H1-H3 (paired by block; Holm within scope):")
        for scope in ["ALL"] + SIG:
            res = []
            for label, (x, y) in contrasts.items():
                d = scoped((x - y).dropna(), scope)
                if len(d) >= 5:
                    res.append((label, boot(d), len(d)))
            for (label, r, n), padj in zip(res, holm([r[3] for _, r, _ in res])):
                share = ""
                if label.startswith(("H1", "H2")):
                    base = scoped(G["full"], scope).reindex(
                        scoped((contrasts[label][0] - contrasts[label][1]).dropna(), scope).index)
                    if base.mean() != 0:
                        share = f" share={r[0] / base.mean():.2f}"
                print(fmt(label, scope, r, n, f" p_holm={padj:.4f}{share}"))
        print()

        print("Descriptive: same decomposition for A = DP-aBF and S = seq-aBF:")
        for name, X in (("A", A), ("S", S)):
            for v in ["splitfracgang", "splitwholegang", "splitall"]:
                for scope in ["ALL"] + SIG:
                    d = scoped((X["full"] - X[v]).dropna(), scope)
                    if len(d) >= 5:
                        print(fmt(f"{name}(full) - {name}({v})", scope, boot(d), len(d)))
        print()


if __name__ == "__main__":
    main()
