#!/usr/bin/env python3
"""F1: derive demand-preserving stream variants from mixed PAI blocks.

Same block boundaries, same items, same order, same pods (hence the same M and
the same M fingerprints) — only the gang structure changes. See PREREG.md.

  splitfracgang   fractional gangs (k>1, d<1)  -> k fractional singletons
  splitwholegang  whole gangs      (k>1, d>=1) -> k whole singletons
  splitall        every gang                    -> singletons

Run from the fgdg2 checkout root (needs `src/` on the path):
  python derive_variants.py --src data/trace-gpu-v2020/blocks_mixed_l1v2 \
      --out-root data/trace-gpu-v2020 [--variants ...] [--domains ...] [--max-blocks N]
Writes <out-root>/blocks_f1_<variant>/{stream_*,pods_*,nodes_*,blocks.tsv,manifest_f1.json}.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import pandas as pd

from fgdg2.datasets.pai import stream_header
from fgdg2.datasets.v2023 import build_m_fractional, build_m_whole
from fgdg2.items import StreamItem
from fgdg2.streams.io import file_md5, load_stream, write_stream

VARIANTS = {
    "splitfracgang": lambda it: it["is_gang"] and it["worker_gpu_demand"] < 1.0,
    "splitwholegang": lambda it: it["is_gang"] and it["worker_gpu_demand"] >= 1.0,
    "splitall": lambda it: it["is_gang"],
}


def split_items(items: list[dict], should_split) -> list[dict]:
    out = []
    for it in items:
        if should_split(it):
            for i in range(int(it["gang_size"])):
                w = dict(it)
                w["gang_id"] = f"{it['gang_id']}:{i}"
                w["gang_size"] = 1
                w["is_gang"] = False
                out.append(w)
        else:
            out.append(dict(it))
    for i, it in enumerate(out):
        it["arrival_index"] = i
    return out


def demand(items) -> float:
    return sum(int(i["gang_size"]) * float(i["worker_gpu_demand"]) for i in items)


def derive_block(src: Path, dst: Path, stream_name: str, pods_name: str, variant: str):
    lines = (src / stream_name).read_text(encoding="utf-8").splitlines()
    orig_header = json.loads(lines[0])
    items = [json.loads(x) for x in lines[1:] if x.strip()]
    new_items = split_items(items, VARIANTS[variant])

    # invariants: same demand, same worker multiset, same order of workers
    assert abs(demand(new_items) - demand(items)) < 1e-6, "demand changed"
    assert sum(int(i["gang_size"]) for i in new_items) == sum(int(i["gang_size"]) for i in items)

    pods = pd.read_csv(src / pods_name)
    m_whole, _ = build_m_whole(pods)
    m_frac, _ = build_m_fractional(pods)
    cfg = orig_header["config"]
    fstats = dict(cfg.get("filter_stats") or {})
    fstats["f1_variant"] = variant
    fstats["f1_source_stream_md5"] = file_md5(src / stream_name)
    header = stream_header(
        gpu_type=cfg["gpu_model_domain"],
        capacity=float(cfg["capacity"]),
        load=cfg["load"],
        items=[StreamItem.from_dict(i) for i in new_items],
        m_whole=m_whole,
        m_frac=m_frac,
        meta={"git_sha": orig_header.get("git_sha", "unknown"),
              "source_task_csv": orig_header.get("source_pod_csv", ""),
              "source_task_md5": orig_header.get("source_pod_md5", "")},
        filter_stats=fstats,
        whole_only=bool(cfg.get("whole_only", False)),
    )
    for key in ("m_whole_sha256", "m_fractional_sha256"):
        assert header[key] == orig_header[key], f"{stream_name}: {key} changed"

    write_stream(dst / stream_name, header, new_items)
    shutil.copyfile(src / pods_name, dst / pods_name)
    # round-trip through the real loader with class-membership checks
    load_stream(dst / stream_name, m_whole, m_frac)
    rz = header["realized"]
    return rz["n_gangs"], rz["n_items"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--max-blocks", type=int, default=0, help="per domain; 0 = all")
    args = ap.parse_args()

    src = Path(args.src)
    rows = [ln.split("\t") for ln in (src / "blocks.tsv").read_text().splitlines() if ln.strip()]
    if args.domains:
        rows = [r for r in rows if r[0] in args.domains]
    if args.max_blocks > 0:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            seen[r[0]] = seen.get(r[0], 0) + 1
            if seen[r[0]] <= args.max_blocks:
                kept.append(r)
        rows = kept

    for variant in args.variants:
        if variant not in VARIANTS:
            ap.error(f"unknown variant {variant!r}")
        dst = Path(args.out_root) / f"blocks_f1_{variant}"
        dst.mkdir(parents=True, exist_ok=True)
        tsv = []
        for r in rows:
            dom, bi, stream_name, pods_name, nodes_name, reached = r[:6]
            dem = r[8]
            gangs, n_items = derive_block(src, dst, stream_name, pods_name, variant)
            if not (dst / nodes_name).exists():
                shutil.copyfile(src / nodes_name, dst / nodes_name)
            tsv.append("\t".join([dom, bi, stream_name, pods_name, nodes_name, reached,
                                  str(gangs), str(n_items), dem]))
        (dst / "blocks.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
        (dst / "manifest_f1.json").write_text(json.dumps({
            "variant": variant,
            "source_blocks": str(src),
            "source_blocks_tsv_md5": file_md5(src / "blocks.tsv"),
            "n_blocks": len(tsv),
        }, indent=2), encoding="utf-8")
        print(f"{variant}: {len(tsv)} blocks -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
