"""Partition whole-GPU PAI v2020 arrivals into contiguous demand blocks.

Each block is an independent Layer-1 replay unit: it starts from an empty
cluster and accumulates demand until it reaches ``load * capacity`` (the last
arrival may overshoot). Blocks are contiguous and cover every whole-GPU arrival
in the domain, so the census is not a hand-picked window.

Outputs under --out:
  nodes_<domain>.csv
  pods_<domain>_block<NNNN>.csv
  pods_<domain>_block<NNNN>_prefix<NN>.csv   # only with --prefix-fracs
  stream_<domain>_block<NNNN>.jsonl
  manifest_blocks.json
  blocks.tsv                 # domain, block, stream, pods, nodes, reached, gangs, items, demand

`--prefix-fracs 0.5` writes a causal M source per block: the pod histogram of
the first 50% of that block's tasks (arrival order). Use it as
``--policy-m stale:<prefix-pods.csv>`` in the Layer-1 runner; the evaluated
stream still contains the full block, so M never sees the future half.
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
from fgdg2.datasets.v2023 import build_m_fractional, build_m_whole
from fgdg2.io.provenance import git_sha
from fgdg2.streams.io import file_md5, write_stream

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "trace-gpu-v2020"
DEFAULT_OUT = SRC / "blocks_whole"


def demand_blocks(tasks, target: float):
    """Contiguous blocks whose cumulative demand reaches target (last block may not)."""
    demand = (tasks["worker_num"].astype(float) * tasks["gpu_request"].astype(float)).to_numpy()
    blocks = []
    start = 0
    acc = 0.0
    for i, d in enumerate(demand):
        acc += float(d)
        if acc >= target:
            blocks.append((start, i + 1, acc, True))
            start = i + 1
            acc = 0.0
    if start < len(tasks):
        blocks.append((start, len(tasks), acc, False))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description="Build whole-only PAI demand-block streams.")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--domains", nargs="*", default=None,
                    help="gpu_type domains (default: all GPU-bearing types)")
    ap.add_argument("--load", type=float, default=1.2,
                    help="block demand target = load * domain capacity")
    ap.add_argument("--mixed", action="store_true",
                    help="keep fractional tasks: fractional gangs AND fractional "
                         "singletons together with whole tasks (default: whole-only)")
    ap.add_argument("--min-gangs", type=int, default=1,
                    help="mark blocks with fewer gangs as sparse (still emitted)")
    ap.add_argument("--prefix-fracs", nargs="*", type=float, default=(),
                    help="emit causal prefix pods for each fraction, e.g. 0.5 "
                         "(files pods_<domain>_block<NNNN>_prefix50.csv)")
    args = ap.parse_args()
    for frac in args.prefix_fracs:
        if not 0.0 < float(frac) < 1.0:
            ap.error(f"--prefix-fracs must be in (0, 1), got {frac}")

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

    manifest = {
        "source": str(src).replace("\\", "/"),
        "tables": meta,
        "load": float(args.load),
        "whole_only": not args.mixed,
        "min_gangs": int(args.min_gangs),
        "prefix_fracs": [float(f) for f in args.prefix_fracs],
        "domains": {},
    }
    tsv_lines = []

    for domain in domains:
        domain = str(domain)
        domain_tasks = tasks[tasks["gpu_type"] == domain].copy()
        kept, drops = split_tasks(domain_tasks, max_gpus,
                                  keep_fractional_gangs=args.mixed)
        if not args.mixed:
            kept = kept[kept["kind"] == "whole"].copy()
        if kept.empty:
            print(f"{domain}: no whole-GPU tasks", flush=True)
            continue
        node_df = nodes_to_node_frame(spec, domain)
        capacity = float(domain_capacity(spec, domain))
        target = float(args.load) * capacity
        blocks = demand_blocks(kept, target)

        tag = domain.replace("/", "-")
        node_path = outdir / f"nodes_{tag}.csv"
        node_df.to_csv(node_path, index=False)

        dom_man = {
            "gpus": int(capacity),
            "target_demand": target,
            "tasks_whole": len(kept),
            "blocks": 0,
            "sparse_blocks": 0,
            "block_files": [],
        }
        for bi, (lo, hi, dem, reached) in enumerate(blocks):
            block = kept.iloc[lo:hi].copy()
            items = tasks_to_items(block)
            pod_df = tasks_to_pod_frame(block)
            m_whole, wstats = build_m_whole(pod_df)
            m_frac, fstats = build_m_fractional(pod_df)
            header = stream_header(
                gpu_type=domain, capacity=capacity, load=args.load,
                items=items, m_whole=m_whole, m_frac=m_frac, meta=meta,
                filter_stats={
                    "whole_only": not args.mixed,
                    "block": bi,
                    "block_lo": lo,
                    "block_hi": hi,
                    "reached_target": reached,
                    "drops": drops,
                },
                whole_only=not args.mixed,
            )
            stream_path = outdir / f"stream_{tag}_block{bi:04d}.jsonl"
            pod_path = outdir / f"pods_{tag}_block{bi:04d}.csv"
            write_stream(stream_path, header, items)
            pod_df.to_csv(pod_path, index=False)

            prefix_pods = {}
            for frac in args.prefix_fracs:
                cut = max(1, round(len(block) * float(frac)))
                prefix = tasks_to_pod_frame(block.iloc[:cut].copy())
                pct = round(float(frac) * 100)
                prefix_path = outdir / (
                    f"pods_{tag}_block{bi:04d}_prefix{pct:02d}.csv"
                )
                prefix.to_csv(prefix_path, index=False)
                prefix_pods[str(float(frac))] = prefix_path.name

            gangs = int(sum(1 for it in items if it.is_gang))
            sparse = gangs < args.min_gangs
            if sparse:
                dom_man["sparse_blocks"] += 1
            rec = {
                "block": bi,
                "reached_target": bool(reached),
                "sparse": bool(sparse),
                "t_lo": float(block["submit_time"].min()),
                "t_hi": float(block["submit_time"].max()),
                "items": len(items),
                "gangs": gangs,
                "demand": float(dem),
                "stream": stream_path.name,
                "pods": pod_path.name,
                "prefix_pods": prefix_pods,
                "nodes": node_path.name,
                "n_whole_classes": wstats["n_classes"],
                "n_frac_classes": fstats["n_classes"],
            }
            dom_man["block_files"].append(rec)
            tsv_lines.append(
                "\t".join([
                    domain, str(bi), stream_path.name, pod_path.name,
                    node_path.name, str(int(reached)), str(gangs),
                    str(len(items)), f"{dem:.6f}",
                ])
            )
        dom_man["blocks"] = len(blocks)
        manifest["domains"][domain] = dom_man
        print(
            f"{domain}: blocks={len(blocks)} sparse={dom_man['sparse_blocks']} "
            f"tasks={len(kept)} gpus={int(capacity)} target={target:.0f}",
            flush=True,
        )

    (outdir / "manifest_blocks.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (outdir / "blocks.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    print(f"manifest -> {outdir / 'manifest_blocks.json'}")
    print(f"blocks   -> {outdir / 'blocks.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
