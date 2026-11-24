"""Generate per-domain stream files + v2023-schema CSVs from Alibaba PAI v2020.

Each ``gpu_type`` is an independent scheduling domain (the trace assigns a GPU
type to every task; the FGD argmin only compares feasible nodes, so per-domain
replay is equivalent to a unified cluster with model filtering).

Outputs under --out:
  nodes_<gpu_type>.csv             v2023 node schema
  pods_<gpu_type>.csv              v2023 pod schema (one row per worker instance)
  stream_<gpu_type>_load<L>.jsonl  schema-v2 stream (real order, demand-capped)
  manifest.json                    counts, drops, windows, capacities
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdg2.datasets.pai import (
    domain_capacity,
    load_job_tasks,
    load_machine_spec,
    max_worker_gpus,
    nodes_to_node_frame,
    split_tasks,
    stream_header,
    tasks_to_items,
    tasks_to_pod_frame,
)
from fgdg2.datasets.spot import select_peak_window, select_window, window_stats
from fgdg2.datasets.v2023 import build_m_fractional, build_m_whole
from fgdg2.io.provenance import git_sha
from fgdg2.streams.io import file_md5, write_stream

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "trace-gpu-v2020"
DEFAULT_OUT = SRC / "derived"


def _window_by_days(tasks, window_range: str, capacity: float):
    d0, d1 = (float(x) for x in window_range.split(":"))
    t_lo, t_hi = d0 * 86400.0, d1 * 86400.0
    window = tasks[(tasks["submit_time"] >= t_lo) & (tasks["submit_time"] < t_hi)].copy()
    return window, window_stats(window, t_lo, t_hi, capacity)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build PAI-v2020 per-domain streams.")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--domains", nargs="*", default=None,
                    help="gpu_type domains (default: all GPU-bearing types)")
    ap.add_argument("--load", type=float, default=1.2,
                    help="Layer-1 demand cap = load * domain capacity")
    ap.add_argument("--window", default="representative", choices=["prefix", "representative"],
                    help="Layer-1 demand-capped slice rule")
    ap.add_argument("--layer2", action="store_true",
                    help="emit temporal streams (real submit_time_s/duration_s)")
    ap.add_argument("--days", type=float, default=7.0,
                    help="Layer-2 window length in days")
    ap.add_argument("--window-range", default=None, metavar="START:END",
                    help="Layer-2 explicit window in DAYS (e.g. 10:17)")
    ap.add_argument("--whole-only", action="store_true",
                    help="drop fractional singletons too: pure whole-GPU gang streams "
                         "(no background policy is ever invoked)")
    args = ap.parse_args()
    if args.window_range is not None and not args.layer2:
        ap.error("--window-range requires --layer2")

    src = Path(args.src)
    raw = src / "raw"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = load_job_tasks(raw / "pai_job_table.csv", raw / "pai_task_table.csv")
    spec = load_machine_spec(raw / "pai_machine_spec.csv")
    max_gpus = max_worker_gpus(spec)
    domains = args.domains or sorted(
        d for d in spec["gpu_type"].dropna().unique() if max_gpus.get(str(d), 0) > 0
    )
    meta = {
        "git_sha": git_sha(),
        "source_task_csv": str(raw / "pai_task_table.csv").replace("\\", "/"),
        "source_task_md5": file_md5(raw / "pai_task_table.csv"),
        "source_job_md5": file_md5(raw / "pai_job_table.csv"),
        "source_machine_md5": file_md5(raw / "pai_machine_spec.csv"),
    }
    suffix = "_whole" if args.whole_only else ""
    man_path = outdir / (
        f"manifest_l2{suffix}.json" if args.layer2 else f"manifest{suffix}.json"
    )
    manifest = {
        "source": str(src).replace("\\", "/"),
        "tables": meta,
        "load": None if args.layer2 else args.load,
        "whole_only": bool(args.whole_only),
        "git_sha": meta["git_sha"],
        "domains": {},
    }
    if man_path.exists():
        old = json.loads(man_path.read_text(encoding="utf-8"))
        manifest["domains"].update(old.get("domains", {}))

    for domain in domains:
        domain = str(domain)
        domain_tasks = tasks[tasks["gpu_type"] == domain].copy()
        kept, drops = split_tasks(domain_tasks, max_gpus)
        if args.whole_only:
            frac_singletons = int((kept["kind"] == "fractional").sum())
            kept = kept[kept["kind"] == "whole"].copy()
            drops["dropped_fractional_singletons_whole_only"] = frac_singletons
        node_df = nodes_to_node_frame(spec, domain)
        capacity = float(domain_capacity(spec, domain))
        if kept.empty:
            print(f"{domain}: no replayable tasks after filters", flush=True)
            continue

        if args.layer2:
            valid = kept[
                kept["duration"].notna()
                & (kept["duration"] > 0.0)
                & kept["gpu_type"].notna()
            ].copy()
            if args.window_range is not None:
                window, winfo = _window_by_days(valid, args.window_range, capacity)
                winfo["rule"] = f"absolute {args.window_range}d"
            else:
                window, winfo = select_peak_window(valid, capacity, days=args.days)
            items = tasks_to_items(window)
        else:
            window = select_window(kept, capacity, args.load, rule=args.window)
            winfo = None
            items = tasks_to_items(window, capacity, args.load)

        pod_df = tasks_to_pod_frame(window)
        m_whole, wstats = build_m_whole(pod_df)
        m_frac, fstats = build_m_fractional(pod_df)
        header = stream_header(
            gpu_type=domain, capacity=capacity,
            load=None if args.layer2 else args.load,
            items=items, m_whole=m_whole, m_frac=m_frac, meta=meta,
            window=winfo, filter_stats=drops, whole_only=args.whole_only,
        )

        tag = domain.replace("/", "-")
        wr = f"_w{args.window_range.replace(':', '-')}" if args.window_range else ""
        node_path = outdir / f"nodes_{tag}.csv"
        pod_path = outdir / (
            f"pods_{tag}_l2{wr}{suffix}.csv" if args.layer2
            else f"pods_{tag}{suffix}.csv"
        )
        stream_path = outdir / (
            f"stream_{tag}_l2_d{args.days:g}{wr}{suffix}.jsonl" if args.layer2
            else f"stream_{tag}_load{args.load:g}{suffix}.jsonl"
        )
        duration_path = outdir / (
            f"durations_{tag}_l2{wr}{suffix}.csv" if args.layer2
            else f"durations_{tag}{suffix}.csv"
        )
        node_df.to_csv(node_path, index=False)
        pod_df.to_csv(pod_path, index=False)
        write_stream(stream_path, header, items)
        dur = window.loc[window["duration"] > 0.0, ["duration", "cpu_request", "gpu_request"]]
        dur.to_csv(duration_path, index=False)

        rz = header["realized"]
        manifest["domains"][domain] = {
            "nodes": len(node_df), "gpus": int(capacity),
            "tasks_total": len(domain_tasks), "tasks_kept": len(kept),
            "window_rule": (
                (f"abs-{args.window_range}d" if args.window_range
                 else f"peak-{args.days:g}d") if args.layer2 else args.window
            ),
            "window_submit_range_s": [
                float(window["submit_time"].min()), float(window["submit_time"].max())
            ] if len(window) else [None, None],
            "window_rho_c": None if winfo is None else winfo["window_rho_c"],
            "whole_only": bool(args.whole_only),
            **drops,
            "stream_items": rz["n_items"],
            "gang_demand_share": rz["realized_gang_demand_share"],
            "frac_demand_share": rz["realized_fractional_demand_share"],
            "n_whole_classes": wstats["n_classes"],
            "n_frac_classes": fstats["n_classes"],
            "files": [node_path.name, pod_path.name, stream_path.name, duration_path.name],
        }
        print(
            f"{domain}: nodes={len(node_df)} gpus={int(capacity)} "
            f"tasks={len(kept)}/{len(domain_tasks)} items={rz['n_items']} "
            f"gang_share={rz['realized_gang_demand_share']:.3f} "
            f"classes={wstats['n_classes']}+{fstats['n_classes']}frac"
            + (f" rho_c={winfo['window_rho_c']:.2f}" if winfo else "")
            + f" -> {stream_path.name}",
            flush=True,
        )

    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
