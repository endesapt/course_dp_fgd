"""Compare our calibration artifacts against the authors' cached results.

Reference: data/reference/analysis_allo_discrete.csv (hkust-adsl repo, Apache-2.0),
workload openb_pod_list_default, tune=1.3, median over 10 seeds.
Our side: results/experiments/calib_<policy>[_sf1.3]_seed*.json — median over seeds.
Base (sf=1.0) runs provide the @100% numbers; sf=1.3 runs add @130% and are used as a
fallback only when a seed has no base run (both modes record the same @100% metric).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _artifacts as art
import pandas as pd


def collect_ours(policy: str):
    runs: dict[int, dict] = {}
    for suffix in ("", "_sf1.3"):
        for path in sorted(Path(art.RESULTS_DIR).glob(f"calib_{policy}{suffix}_seed*.json")):
            seed = int(path.stem.split("seed")[-1])
            m = json.loads(path.read_text())["metrics"]
            entry = runs.setdefault(seed, {"at100": None, "at130": None, "suffix": None})
            if suffix == "":
                entry["at100"] = art.at_capacity_pct(m)
                entry["suffix"] = "base"
            else:
                if entry["at100"] is None:
                    entry["at100"] = art.at_capacity_pct(m)
                    entry["suffix"] = "sf1.3"
                entry["at130"] = art.stop_pct(m)
    at100 = [r["at100"] for r in runs.values() if r["at100"] is not None]
    at130 = [r["at130"] for r in runs.values() if r["at130"] is not None]
    return runs, at100, at130


def main():
    ap = argparse.ArgumentParser(description="Our medians vs the authors' cached medians.")
    ap.add_argument("--reference", default=str(art.DEFAULT_REFERENCE_CSV))
    ap.add_argument("--workload", default="openb_pod_list_default")
    ap.add_argument("--tune", type=float, default=1.3)
    ap.add_argument("--out",
                    default=str(art.RESULTS_DIR.parent / "reference_compare.csv"))
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    ref = ref[(ref["workload"] == args.workload) & (ref["tune"] == args.tune)]

    rows = []
    for policy in art.PAPER_ORDER:
        runs, ours100, ours130 = collect_ours(policy)
        sub = ref[ref["sc_policy"] == art.REF_POLICY_MAP[policy]]
        paper100 = float(sub["100"].median()) if not sub.empty else None
        paper130 = float(sub["130"].median()) if not sub.empty else None
        unreadable = [s for s, r in runs.items() if r["at100"] is None]
        if unreadable:
            print(f"warning: {policy}: seeds {unreadable} lack unallocated_pct_at_capacity; skipped")
        rows.append(
            {
                "policy": policy,
                "seeds": len(ours100),
                "ours_unalloc_100": round(pd.Series(ours100).median(), 2) if ours100 else None,
                "paper_unalloc_100": None if paper100 is None else round(100 - paper100, 2),
                "ours_unalloc_130": round(pd.Series(ours130).median(), 2) if ours130 else None,
                "paper_unalloc_130": None if paper130 is None else round(100 - paper130, 2),
            }
        )

    df = pd.DataFrame(rows)
    df["diff_100_pp"] = (df["ours_unalloc_100"] - df["paper_unalloc_100"]).round(2)
    print(df.to_string(index=False))
    print()
    base = df.set_index("policy")
    fgd_ours = base.loc["fgd_q", "ours_unalloc_100"]
    bf_ours = base.loc["bestfit", "ours_unalloc_100"]
    fgd_paper = base.loc["fgd_q", "paper_unalloc_100"]
    bf_paper = base.loc["bestfit", "paper_unalloc_100"]
    if None not in (fgd_ours, bf_ours, fgd_paper, bf_paper):
        ours = 100 * (1 - fgd_ours / bf_ours)
        paper = 100 * (1 - fgd_paper / bf_paper)
        print(f"FGD vs BestFit reduction @100%: ours {ours:.1f}%  paper {paper:.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
