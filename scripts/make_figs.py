"""Fig. 5/9-style curves from calibration artifacts (verbatim port)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import _artifacts as art
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[1] / "results" / "figs"


def curves(record):
    cap = record["config"]["capacity"]
    t = record["timeline"]
    x = [100.0 * d / cap for d in t["cumulative_demand"]]
    y_unalloc = [100.0 * u / cap for u in t["unallocated_shares"]]
    return x, y_unalloc


def main():
    ap = argparse.ArgumentParser(description="Fig. 5/9-style curves from calibration artifacts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", nargs="*", default=art.PAPER_ORDER)
    ap.add_argument("--out", default=str(FIGS / "fig5_unallocated_vs_demand.png"), help="output PNG (file)")
    ap.add_argument("--zoom-from", type=float, default=90.0)
    ap.add_argument("--xmax", type=float, default=100.0)
    ap.add_argument("--tag-suffix", default="", help="e.g. _sf1.3 for stop-fraction != 1 runs")
    ap.add_argument("--reference", default=None, help="authors' analysis_allo_discrete.csv for overlay")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=False, dpi=160)

    missing = []
    plotted = 0
    y_max = 0.0
    for policy in args.policies:
        record = art.load_run(policy, args.seed, args.tag_suffix)
        if record is None:
            missing.append(policy)
            continue
        label, color, lw, ls = art.STYLE.get(policy, (policy, None, 1.5, "-"))
        x, y_unalloc = curves(record)
        ax1.plot(x, y_unalloc, color=color, lw=lw, ls=ls, label=label)
        ax2.plot(x, y_unalloc, color=color, lw=lw, ls=ls, label=label)
        ax1.plot(x[-1], y_unalloc[-1], "o", color=color, ms=4)
        ax2.plot(x[-1], y_unalloc[-1], "o", color=color, ms=5)
        plotted += 1
        y_max = max([y_max] + [y for xv, y in zip(x, y_unalloc) if xv >= args.zoom_from])

        m = record["metrics"]
        at_cap = art.at_capacity_pct(m)
        at_cap_s = "n/a" if at_cap is None else f"{at_cap:.2f}%"
        print(f"{policy:10s} seed={args.seed} unalloc@100%={at_cap_s} unplaced={m['unplaced']}")

        if args.reference:
            ref = art.load_reference(policy, path=args.reference)
            if ref is not None:
                rxs, rys = ref
                ax1.plot(rxs, rys, color=color, lw=1.1, ls="--", alpha=0.85)
                ax2.plot(rxs, rys, color=color, lw=1.1, ls="--", alpha=0.85, label=f"{label} (paper)")
                y_max = max([y_max] + [y for xv, y in zip(rxs, rys) if args.zoom_from <= xv <= args.xmax])

    if plotted == 0:
        print(f"no artifacts found for seed {args.seed} (suffix={args.tag_suffix!r}); nothing to plot")
        return 1

    ax1.set_ylabel("unallocated GPU shares (% of capacity)")
    ax1.set_xlabel("cumulative demanded GPUs (% of capacity)")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, args.xmax)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("full range")

    ax2.set_ylabel("unallocated GPU shares (% of capacity)")
    ax2.set_xlabel("cumulative demanded GPUs (% of capacity)")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(args.zoom_from, args.xmax)
    ax2.set_ylim(0, max(5.0, y_max * 1.15))
    ax2.set_title(f"zoom: {args.zoom_from:.0f}%..{args.xmax:.0f}% of demand")
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Workload inflation, trace-gpu-v2023 (seed {args.seed})")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}")
    if missing:
        print(f"missing/incomplete artifacts: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
