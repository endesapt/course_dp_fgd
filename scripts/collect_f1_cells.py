#!/usr/bin/env python3
"""Collect the F1 (workload composition) runs into one committed aggregate.

F1 = experiment 2026-11-19_f1_composition: the same 778 mixed blocks with
demand preserved, gangs split into single tasks (fractional gangs, whole
gangs, all gangs), plus fractional singles placed by an FGD background.
The `full` variant is NOT collected here: it is the oracle-M arms of
results/aggregates/mixed/mixed_cells.csv ({dp,seq,abf}_stream).

Reads:  <runs>/<arm>/<domain>/block<k>/summary.csv, arm = <variant>__<pol>
Writes: results/aggregates/f1/f1_cells.csv
        (variant, policy, domain, block, unalloc100, unalloc_end)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aggregates" / "f1" / "f1_cells.csv"
VARIANTS = {"sfg": "splitfracgang", "swg": "splitwholegang", "sall": "splitall",
            "sfgbg": "splitfracgang_fgdbg"}
POLICIES = {"dp": "dp", "seq": "seq_fgd", "abf": "atomic_bestfit"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="pai_f1 results tree (summary.csv only is enough)")
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.runs).glob("*/*/block*/summary.csv")):
        arm, dom, blk = f.parts[-4], f.parts[-3], f.parts[-2]
        var, _, pol = arm.partition("__")
        if var not in VARIANTS or pol not in POLICIES:
            continue
        s = pd.read_csv(f).iloc[-1]
        u100 = str(s["unalloc_pct_at_100"]).rstrip("%")
        rows.append({"variant": VARIANTS[var], "policy": POLICIES[pol], "domain": dom,
                     "block": int(blk[5:]),
                     "unalloc100": float(u100) if u100 not in ("n/a", "nan") else np.nan,
                     "unalloc_end": float(s["unalloc_pct_at_end"])})
    df = pd.DataFrame(rows).sort_values(["variant", "policy", "domain", "block"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{OUT.relative_to(ROOT)}: {len(df)} cells")
    print(df.groupby(["variant", "policy"]).size().unstack().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
