#!/usr/bin/env python3
"""Mixed blocks (whole + fractional gangs) analysis as in PREREG.md.

rsync first:
  rsync -a server:fgdg2-run/results/pai_mixed/ out/pai_mixed/ \
        --include='*/' --include='summary.csv' --exclude='*'
Partial results are fine: every comparison uses matched blocks only.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-11-17_flat_m"))
from analyze_flat_m import cluster_ci  # noqa: E402

MIX = HERE / "out/pai_mixed"
SIG = ["MISC", "V100", "V100M32"]
WHOLE_REF = {"dp-aBF": -1.982, "dp-seq": -0.209}  # whole-only core, signal3


def load() -> pd.DataFrame:
    rows = []
    for f in MIX.glob("*/*/block*/summary.csv"):
        arm, dom, blk = f.parts[-4], f.parts[-3], f.parts[-2]
        s = pd.read_csv(f).iloc[-1]
        rows.append({"arm": arm, "domain": dom, "block": int(blk[5:]),
                     "unalloc100": float(str(s["unalloc_pct_at_100"]).rstrip("%")),
                     "unalloc_end": float(s["unalloc_pct_at_end"]),
                     "rate": float(s["rate"])})
    return pd.DataFrame(rows)


def cmp(p, a, b, label, scope="ALL"):
    if scope != "ALL" and scope not in p.index.get_level_values(0):
        return
    sub = p if scope == "ALL" else p.loc[[scope]]
    if a not in sub or b not in sub:
        return
    d = (sub[a] - sub[b]).dropna()
    if len(d) < 5:
        return
    lo, hi = cluster_ci(d)
    print(f"  {label:34s} {scope:8s} {d.mean():+.3f} pp CI[{lo:+.3f},{hi:+.3f}] "
          f"W/T/L {(d < 0).sum()}/{(d == 0).sum()}/{(d > 0).sum()} n={len(d)}")


def main():
    df = load()
    if df.empty:
        raise SystemExit(f"no summaries under {MIX}")
    print("cells per arm:", df.groupby("arm").size().to_dict(), "\n")
    p = df.pivot_table(index=["domain", "block"], columns="arm", values="unalloc100")
    print("mean unalloc@100 by domain (each arm over its own available blocks):")
    print(p.groupby(level=0).mean().round(3).T.to_string(), "\n")

    print("P1 — policy gaps (oracle M = block's own worker histogram):")
    for scope in ["ALL"] + SIG:
        cmp(p, "dp_stream", "abf_stream", "DP - atomic BestFit", scope)
        cmp(p, "dp_stream", "seq_stream", "DP - seq-FGD", scope)
        cmp(p, "seq_stream", "abf_stream", "seq-FGD - atomic BestFit", scope)
    print(f"  (whole-only reference: DP-aBF {WHOLE_REF['dp-aBF']}, DP-seq {WHOLE_REF['dp-seq']})\n")

    print("Baselines vs atomic BestFit (CAFGD-paper policies, gang setting):")
    for arm in ["aff_stream", "abfgpu_stream", "adot_stream", "apack_stream", "arand_stream"]:
        cmp(p, arm, "abf_stream", f"{arm} - aBF")
    print()

    print("P2/P3 — value of M (loss vs own-block oracle), and DP - seq difference in loss:")
    for m in ["prev", "uniform"]:
        for pol in ["dp", "seq"]:
            cmp(p, f"{pol}_{m}", f"{pol}_stream", f"{pol} loss with {m}")
        if {f"dp_{m}", "dp_stream", f"seq_{m}", "seq_stream"} <= set(p):
            q = p[[f"dp_{m}", "dp_stream", f"seq_{m}", "seq_stream"]].dropna()
            q = q.assign(dd=(q[f"dp_{m}"] - q["dp_stream"]) - (q[f"seq_{m}"] - q["seq_stream"]))
            for scope in ["ALL"] + SIG:
                cmp(q.assign(z=0.0), "dd", "z", f"(DP loss - seq loss) {m}", scope)
        cmp(p, f"dp_{m}", "seq_stream", f"cross-M DP({m}) - seq(oracle)")
    print()
    print("Secondary: placement rate by arm:", df.groupby("arm").rate.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
