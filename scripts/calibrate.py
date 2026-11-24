"""Layer-1 inflation calibration (thin CLI over fgdg2.experiments).

    python scripts/calibrate.py --policy fgd_q --seeds 10 --fast
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdg2.experiments.config import InflationConfig
from fgdg2.experiments.runner import run_inflation
from fgdg2.policies.registry import INFLATION_POLICIES


def main():
    ap = argparse.ArgumentParser(description="Run one policy on the trace inflation benchmark.")
    ap.add_argument("--policy", required=True, choices=INFLATION_POLICIES)
    ap.add_argument("--seeds", type=int, default=1, help="runs seeds 0..N-1")
    ap.add_argument("--fast", action="store_true", help="use the batched decision path (FGD policies)")
    ap.add_argument("--sampling", choices=["replacement", "sequential"], default="replacement",
                    help="replacement = paper-style inflation; sequential = one trace pass (debug)")
    ap.add_argument("--stop-fraction", type=float, default=1.0,
                    help="stop at this fraction of capacity; 1.3 replicates the paper's -tune 1.3")
    ap.add_argument("--out", default="results/experiments", help="output directory")
    ap.add_argument("--pods-csv", default=InflationConfig.pods_csv)
    ap.add_argument("--nodes-csv", default=InflationConfig.nodes_csv)
    args = ap.parse_args()
    run_inflation(InflationConfig(
        policy=args.policy, seeds=args.seeds, fast=args.fast,
        sampling=args.sampling, stop_fraction=args.stop_fraction, out=args.out,
        pods_csv=args.pods_csv, nodes_csv=args.nodes_csv,
    ))


if __name__ == "__main__":
    main()
