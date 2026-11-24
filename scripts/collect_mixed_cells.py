#!/usr/bin/env python3
"""Collect the mixed-PAI-block runs into small committed aggregates.

Inputs (not committed — raw runs and blocks):
  --runs   tree <arm>/<domain>/block<N>/summary.csv  (results/pai_mixed, written
           by scripts/run_mixed.sh; design: docs/prereg/PREREG_mixed_gangs.md)
  --blocks blocks_mixed_l1v2 directory (make_pai_blocks.py --mixed --load 1.2)

Outputs:
  results/aggregates/mixed/mixed_cells.csv   one row per (arm, domain, block)
  results/aggregates/mixed/mixed_blocks.csv  workload composition per block
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aggregates" / "mixed"

ARM_POLICY = {
    "dp": "dp", "seq": "seq_fgd", "abf": "atomic_bestfit", "aff": "atomic_firstfit",
    "abfgpu": "atomic_bestfit_gpu", "adot": "atomic_dotprod",
    "apack": "atomic_packing", "arand": "atomic_random",
}


def collect_cells(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/*/block*/summary.csv")):
        arm, dom, blk = f.parts[-4], f.parts[-3], f.parts[-2]
        pol_tag, m = arm.split("_", 1)
        s = pd.read_csv(f).iloc[-1]
        u100 = str(s["unalloc_pct_at_100"]).rstrip("%")
        rows.append({
            "arm": arm, "policy": ARM_POLICY[pol_tag],
            "m": {"stream": "oracle"}.get(m, m),
            "domain": dom, "block": int(blk[5:]),
            "unalloc100": float(u100) if u100 not in ("n/a", "nan") else None,
            "unalloc_end": float(s["unalloc_pct_at_end"]),
            "gangs": int(s["gangs"]), "gangs_placed": int(s["gangs_placed"]),
            "workers_placed": int(s["workers_placed"]),
        })
    return pd.DataFrame(rows).sort_values(["arm", "domain", "block"])


def collect_blocks(blocks: Path) -> pd.DataFrame:
    rows = []
    for ln in (blocks / "blocks.tsv").read_text().splitlines():
        if not ln.strip():
            continue
        dom, bi, stream, _pods, _nodes, reached, gangs, items, demand = ln.split("\t")
        with open(blocks / stream, encoding="utf-8") as fh:
            h = json.loads(fh.readline())
        cfg, rz = h["config"], h["realized"]
        frac_gang = sum(int(k) * float(d) * c for key, c in rz["realized_shape_counts"].items()
                        for k, d in [key.split("x")] if float(d) < 1.0)
        total = float(demand)
        rows.append({
            "domain": dom, "block": int(bi), "reached_target": int(reached),
            "capacity_gpus": cfg["capacity"], "demand_gpus": total,
            "items": int(items), "gangs": int(gangs),
            "fractional_singletons": rz["n_fractional_singletons"],
            "gang_demand_share": rz["realized_gang_demand_share"],
            "fractional_gang_demand_share": frac_gang / total if total else 0.0,
            "fractional_singleton_demand_share": rz["realized_fractional_demand_share"],
            "mean_gang_size": cfg["mean_gang_size"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--blocks", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = collect_cells(Path(args.runs))
    cells.to_csv(OUT / "mixed_cells.csv", index=False)
    print(f"mixed_cells.csv: {len(cells)} rows, arms={cells.arm.nunique()}")
    blocks = collect_blocks(Path(args.blocks))
    blocks.to_csv(OUT / "mixed_blocks.csv", index=False, float_format="%.6g")
    print(f"mixed_blocks.csv: {len(blocks)} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
