"""Generate the gang-phase dataset (schema v2, thin CLI over fgdg2.datasets).

Default: writes data/gang/trace_coarrival_stats.json + 6 committed demo streams
(gang-shape mix from PAI v2020, fractional background with the real v2023 share):
  f00_mix, f25_mix, f50_mix, f50_mix_frac50, f50_k16 (legacy fixed-k), f50_k16_g50.
--full: the matrix (fractions x shapes/sizes x loads x seeds) — NOT committed to git
(growth policy); expect hundreds of MB.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdg2.datasets.synth import (
    PAI_GANG_SHAPES,
    normalize_shapes,
    sample_stream,
    tilt_multi_gpu_share,
)
from fgdg2.datasets.v2023 import (
    build_m_fractional,
    build_m_whole,
    coarrival_stats,
    load_nodes,
    load_pods,
    whole_gpu_pods,
)
from fgdg2.io.provenance import git_sha
from fgdg2.streams.io import file_md5, write_stream

ROOT = Path(__file__).resolve().parents[1]  # repo root — self-contained
TRACE_DIR = ROOT / "data" / "trace-gpu-v2023" / "csv"
PODS_CSV = TRACE_DIR / "openb_pod_list_default.csv"
NODES_CSV = TRACE_DIR / "openb_node_list_gpu_node.csv"
DEFAULT_OUT = ROOT / "data" / "gang"

DEMO = [
    {"fraction": 0, "tag": "mix", "shapes": "pai", "frac": "real"},
    {"fraction": 25, "tag": "mix", "shapes": "pai", "frac": "real"},
    {"fraction": 50, "tag": "mix", "shapes": "pai", "frac": "real"},
    {"fraction": 50, "tag": "mix", "shapes": "pai", "frac": 0.5},
    {"fraction": 50, "tag": "k16", "k": 16, "frac": 0},
    {"fraction": 50, "tag": "k16", "k": 16, "frac": 0, "g": 0.5},
]
FULL_FRACTIONS = [0, 10, 25, 50]
FULL_SIZES = [2, 4, 8, 16]
FULL_LOADS = [0.8, 1.0, 1.2]


def parse_shapes(spec: str):
    """'2x4:0.33,4x1:0.2' -> ((2, 4.0, 0.33), (4, 1.0, 0.2))"""
    shapes = []
    for part in spec.split(","):
        shape, weight = part.strip().split(":")
        k, d = shape.lower().split("x")
        shapes.append((int(k), float(d), float(weight)))
    return normalize_shapes(shapes)


def stream_name(seed: int, fraction: int, tag: str, load: float, g, frac) -> str:
    suffix = ""
    if g is not None:
        suffix += f"_g{round(g * 100):02d}"
    if frac is not None and frac > 0:
        suffix += f"_frac{round(frac * 100):02d}"
    return f"synth_gangs_seed{seed}_f{fraction:02d}_{tag}_load{load:g}{suffix}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the gang-phase dataset.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fractions", nargs="*", type=int, default=None, help="percents")
    ap.add_argument("--sizes", nargs="*", type=int, default=None, help="legacy fixed gang sizes k")
    ap.add_argument("--shapes", default=None, help='gang-shape mix, e.g. "2x4:0.4,4x1:0.3"')
    ap.add_argument("--loads", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--fractional-share", type=float, default=None,
                    help="fractional background share of GPU demand (default: real v2023 share)")
    ap.add_argument("--multi-gpu-share", type=float, default=None,
                    help="tilt whole-class M so multi-GPU classes supply this demand share")
    ap.add_argument("--full", action="store_true", help="generate the whole matrix (large!)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    pods = load_pods(PODS_CSV)
    m_whole, stats = build_m_whole(pods)
    m_frac, frac_stats = build_m_fractional(pods)
    capacity = load_nodes(NODES_CSV).total_gpus
    real_frac_share = frac_stats["fractional_gpu_demand_share_of_trace"]

    if args.full:
        fractions = args.fractions or FULL_FRACTIONS
        loads = args.loads or FULL_LOADS
        seeds = args.seeds if args.seeds is not None else list(range(10))
        frac = args.fractional_share if args.fractional_share is not None else "real"
        if args.shapes:
            shapes = parse_shapes(args.shapes)
            entries = [{"fraction": f, "tag": "mix", "shapes": shapes,
                        "frac": frac, "load": load, "seed": s}
                       for f in fractions for load in loads for s in seeds]
        else:
            sizes = args.sizes or FULL_SIZES
            entries = [{"fraction": f, "tag": f"k{k}", "k": k,
                        "frac": frac, "load": load, "seed": s}
                       for f in fractions for k in sizes for load in loads for s in seeds]
        if args.multi_gpu_share is not None:
            for e in entries:
                e.setdefault("g", args.multi_gpu_share)
        print(f"full matrix: {len(entries)} streams -> {outdir}", flush=True)
    elif args.fractions is None and args.sizes is None and args.shapes is None:
        entries = []
        for s in (args.seeds if args.seeds is not None else [0]):
            for d in DEMO:
                entries.append(dict(d, load=(args.loads[0] if args.loads else 1.0), seed=s))
    else:
        fractions = args.fractions or FULL_FRACTIONS
        loads = args.loads or [1.0]
        seeds = args.seeds if args.seeds is not None else [0]
        frac = args.fractional_share if args.fractional_share is not None else "real"
        if args.shapes:
            shapes = parse_shapes(args.shapes)
            entries = [{"fraction": f, "tag": "mix", "shapes": shapes, "frac": frac,
                        "load": load, "seed": s}
                       for f in fractions for load in loads for s in seeds]
        else:
            sizes = args.sizes or FULL_SIZES
            entries = [{"fraction": f, "tag": f"k{k}", "k": k, "frac": frac,
                        "load": load, "seed": s}
                       for f in fractions for k in sizes for load in loads for s in seeds]
        if args.multi_gpu_share is not None:
            for e in entries:
                e["g"] = args.multi_gpu_share

    coarr = coarrival_stats(whole_gpu_pods(pods))
    (outdir / "trace_coarrival_stats.json").write_text(
        json.dumps(coarr, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    meta = {
        "git_sha": git_sha(),
        "source_pod_csv": str(PODS_CSV).replace("\\", "/"),
        "source_md5": file_md5(PODS_CSV),
    }
    print(f"nodes={len(load_nodes(NODES_CSV).nodes)} capacity={capacity}")
    print(f"whole: pods={stats['n_whole_pods']} classes={stats['n_classes']} "
          f"E[d]={stats['mean_worker_gpu_demand']:.3f} multi_gpu={stats['multi_gpu_demand_share']:.3f}")
    print(f"fractional: pods={frac_stats['n_fractional_pods']} classes={frac_stats['n_classes']} "
          f"E[d]={frac_stats['mean_worker_gpu_demand']:.3f} "
          f"real_share_of_gpu_demand={real_frac_share:.4f}")
    print(f"default shapes: {PAI_GANG_SHAPES}")

    for e in entries:
        k = e.get("k")
        shapes = PAI_GANG_SHAPES if e.get("shapes") == "pai" else e.get("shapes")
        frac = e.get("frac")
        if frac == "real":
            frac = real_frac_share
        if frac is None:
            frac = 0.0
        g = e.get("g")
        base = m_whole if g is None else tilt_multi_gpu_share(m_whole, g)
        header, items = sample_stream(
            base, e["fraction"] / 100.0, e["load"], capacity, e["seed"], meta,
            k=k, shapes=shapes, multi_gpu_share=g,
            fractional_workload=(m_frac if frac > 0 else None),
            fractional_share=frac,
        )
        path = outdir / stream_name(e["seed"], e["fraction"], e["tag"], e["load"], g,
                                    None if frac == 0 else frac)
        write_stream(path, header, items)
        r = header["realized"]
        print(
            f"{path.name}: items={r['n_items']} gangs={r['n_gangs']} "
            f"frac_singles={r['n_fractional_singletons']} "
            f"share_g={r['realized_gang_demand_share']:.4f} "
            f"share_f={r['realized_fractional_demand_share']:.4f} "
            f"shapes={r['realized_shape_counts']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
