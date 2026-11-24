"""Layer-1 gang streams (thin CLI over fgdg2.experiments).

    python scripts/calibrate_gang.py --policy dp --streams 'data/gang/*.jsonl'
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdg2.experiments.config import GangConfig
from fgdg2.experiments.runner import run_gang
from fgdg2.policies.registry import BACKGROUND_POLICIES, GANG_POLICIES
from fgdg2.workload.spec import parse_mspec


def main():
    ap = argparse.ArgumentParser(description="Run one gang policy on gang streams.")
    ap.add_argument("--policy", required=True, choices=GANG_POLICIES)
    ap.add_argument("--background", default="fgd_q", choices=BACKGROUND_POLICIES,
                    help="policy for fractional singleton items (schema-v2 streams)")
    ap.add_argument("--streams", default=GangConfig.streams,
                    help="glob of stream files")
    ap.add_argument("--seeds", type=int, default=1, help="runs seeds 0..N-1")
    ap.add_argument("--out", default="results/gang", help="output directory")
    ap.add_argument("--pods-csv", default=GangConfig.pods_csv,
                    help="pod/jobs table for M construction (v2023 schema)")
    ap.add_argument("--nodes-csv", default=GangConfig.nodes_csv,
                    help="node table for the cluster (v2023 schema)")
    ap.add_argument("--policy-m", default="stream",
                    help="M the policies score against. Layer-1 supports STATIC specs "
                         "only: stream | empirical | occupancy | tilted:<gamma> | "
                         "stale:<file> | stale_occ:<file>. Tracking specs "
                         "(slide_*, queue*, running*, mix*) need a time dimension and "
                         "are rejected by the Layer-1 runner. 'stale:<file>' accepts a "
                         "past pods .csv or stream .jsonl.")
    ap.add_argument("--consolidation-beta", type=float, default=0.0,
                    help="Phase-2 elasticity knob (dp/atomic_fgd/seq_fgd only; 0 = pure FGD)")
    args = ap.parse_args()
    try:
        mspec = parse_mspec(args.policy_m)
    except ValueError as exc:
        ap.error(str(exc))
    if mspec.tracking:
        ap.error(f"--policy-m {args.policy_m!r} is a tracking spec; Layer-1 gang "
                 "runs have no time dimension (use stream|empirical|stale:<file>)")
    run_gang(GangConfig(
        policy=args.policy, background=args.background, streams=args.streams,
        seeds=args.seeds, out=args.out, pods_csv=args.pods_csv,
        nodes_csv=args.nodes_csv, policy_m=args.policy_m,
        consolidation_beta=args.consolidation_beta,
    ))


if __name__ == "__main__":
    main()
