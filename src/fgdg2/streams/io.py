"""Stream I/O — the single home of the stream schema, header contract and
workload fingerprints.

Schema v2 (unchanged on disk): line 0 = JSON header {"type": "header",
"schema_version": 2, "config": {...}, "m_whole_sha256": ..., "seed": ...},
then one JSON item per line (StreamItem fields). `load_stream` enforces the
version (a real check, not just field presence) and validates every item's
class membership against the M the stream was generated from when the
caller supplies it (fingerprint gate).
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from ..core.tasks import Workload
from ..items import StreamItem

STREAM_SCHEMA_VERSION = 2

_HEADER_REQUIRED = ("schema_version", "config", "m_whole_sha256", "seed")
_ITEM_REQUIRED = (
    "gang_id", "gang_size", "worker_cpu_milli", "worker_gpu_demand",
    "arrival_index", "is_gang",
)


def workload_fingerprint(workload: Workload) -> str:
    """sha256 over canonical JSON of the class list + popularities."""
    payload = [
        {
            "name": c.name,
            "cpu_milli": int(c.cpu_milli),
            "gpu_demand": float(c.gpu_demand),
            "popularity": round(float(p), 12),
        }
        for c, p in zip(workload.classes, workload.popularities)
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest().upper()


def write_stream(path, header: dict, items) -> None:
    """Atomic write (tmp + os.replace); items are StreamItems or dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    lines += [
        json.dumps(it.to_dict() if isinstance(it, StreamItem) else it,
                   sort_keys=True, separators=(",", ":"))
        for it in items
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_stream(
    path,
    workload: Workload | None = None,
    fractional_workload: Workload | None = None,
) -> tuple[dict, list[StreamItem]]:
    """Load and validate one stream -> (header, items as StreamItems).

    `workload` / `fractional_workload` enable the class-membership checks
    (whole classes for demand >= 1, fractional classes for demand < 1).
    Callers that know the generating M should also verify
    header["m_whole_sha256"] == workload_fingerprint(m) — see
    experiments.runner which does this unconditionally (audit A1).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty stream file {path}")
    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError(f"{path}: first line is not a stream header")
    for field in _HEADER_REQUIRED:
        if field not in header:
            raise ValueError(f"{path}: header missing field {field!r}")
    version = header["schema_version"]
    if version != STREAM_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported stream schema_version {version} "
            f"(this reader knows {STREAM_SCHEMA_VERSION})"
        )
    items = [json.loads(line) for line in lines[1:] if line.strip()]
    allowed_whole = None
    if workload is not None:
        allowed_whole = {(int(c.cpu_milli), float(c.gpu_demand)) for c in workload.classes}
    allowed_frac = None
    if fractional_workload is not None:
        allowed_frac = {
            (int(c.cpu_milli), float(c.gpu_demand)) for c in fractional_workload.classes
        }
    cfg = header["config"]
    k = cfg.get("gang_size")
    shape_ks = {int(s["k"]) for s in cfg.get("gang_shapes") or []}
    out: list[StreamItem] = []
    for expected, it in enumerate(items):
        for field in _ITEM_REQUIRED:
            if field not in it:
                raise ValueError(f"{path}: item missing field {field!r}")
        if it["arrival_index"] != expected:
            raise ValueError(f"{path}: arrival_index {it['arrival_index']} != {expected}")
        if it["is_gang"]:
            if k is not None and it["gang_size"] != k:
                raise ValueError(f"{path}: gang_size {it['gang_size']} != {k}")
            if shape_ks and it["gang_size"] not in shape_ks:
                raise ValueError(f"{path}: gang_size {it['gang_size']} not in {sorted(shape_ks)}")
        elif it["gang_size"] != 1:
            raise ValueError(f"{path}: singleton gang_size must be 1, got {it['gang_size']}")
        key = (it["worker_cpu_milli"], it["worker_gpu_demand"])
        if it["worker_gpu_demand"] < 1.0:
            if allowed_frac is not None and key not in allowed_frac:
                raise ValueError(f"{path}: fractional class {key} not in M_frac")
        elif allowed_whole is not None and key not in allowed_whole:
            raise ValueError(f"{path}: worker class {key} not in M_whole")
        out.append(StreamItem.from_dict(it))
    return header, out


def realized_stats(items) -> dict:
    """Single home of the stream realized-stats block (was duplicated in the
    old gang_dataset and spot_gpu modules — audit A10)."""
    total = sum(it.gang_size * it.worker_gpu_demand for it in items)
    gang = sum(
        it.gang_size * it.worker_gpu_demand for it in items if it.is_gang
    )
    frac = sum(
        it.gang_size * it.worker_gpu_demand
        for it in items
        if not it.is_gang and it.worker_gpu_demand < 1.0
    )
    shapes = Counter(
        f"{it.gang_size}x{it.worker_gpu_demand:g}" for it in items if it.is_gang
    )
    return {
        "n_items": len(items),
        "n_gangs": sum(1 for it in items if it.is_gang),
        "n_singletons": sum(1 for it in items if not it.is_gang),
        "n_fractional_singletons": sum(
            1 for it in items if not it.is_gang and it.worker_gpu_demand < 1.0
        ),
        "realized_gang_demand_share": float(gang / total) if total else 0.0,
        "realized_fractional_demand_share": float(frac / total) if total else 0.0,
        "realized_shape_counts": dict(sorted(shapes.items())),
    }


def realized_shape_mix(items) -> list[dict]:
    """(k, d, weight) histogram over gang items — per-arrival weights, same role
    as config.gang_shapes in synthetic streams (drives load_stream's size check
    and the policy_workload mix when policy_m='stream')."""
    gangs = [it for it in items if it.is_gang]
    counts = Counter((it.gang_size, it.worker_gpu_demand) for it in gangs)
    total = sum(counts.values()) or 1
    return [
        {"k": int(k), "d": float(d), "weight": c / total}
        for (k, d), c in sorted(counts.items())
    ]


def verify_fingerprints(path, header: dict, stream_whole: Workload,
                        m_fractional: Workload | None = None) -> None:
    """Hard-fail when the stream's recorded M fingerprints do not match the
    workloads it will be replayed against (the pods↔stream pairing check,
    audit A1).

    `stream_whole` must be the STREAM-resolved whole-class M — i.e.
    `workload_for_stream(base_m_whole, header)`, not the raw base workload
    (a header-level multi_gpu_share tilt changes the fingerprint)."""
    got = header["m_whole_sha256"]
    want = workload_fingerprint(stream_whole)
    if got != want:
        raise ValueError(
            f"{path}: m_whole_sha256 mismatch — stream was generated from a "
            "different whole-class M"
        )
    frac_fp = header.get("m_fractional_sha256")
    if frac_fp is not None:
        if m_fractional is None:
            raise ValueError(f"{path}: stream has fractional classes but no M_frac given")
        if frac_fp != workload_fingerprint(m_fractional):
            raise ValueError(f"{path}: m_fractional_sha256 mismatch")
