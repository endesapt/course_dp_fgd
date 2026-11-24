"""Synthetic gang-stream generator (schema v2).

Real vs synthetic (claims discipline):
- worker classes are REAL: whole-GPU classes and fractional classes both come from
  the Alibaba v2023 trace (`build_m_whole` / `build_m_fractional`);
- the DEFAULT gang-shape mix is REAL: the joint (k, GPU-per-worker) distribution of
  the 4,015 whole-worker multi-instance tasks in Alibaba PAI v2020 (`PAI_GANG_SHAPES`);
- synthetic: gang grouping, shape mixing, arrival order, fractional-background share
  and the demand horizon.

Streams are EXPERIMENT data; never present them as observed production gangs.
"""

import numpy as np

from ..core.tasks import TaskClass, Workload
from ..items import StreamItem
from ..streams.io import STREAM_SCHEMA_VERSION, realized_stats, workload_fingerprint

# Real gang-shape mix from Alibaba PAI v2020 (CC BY 4.0), derived from the 4,015
# multi-instance whole-worker tasks (inst_num >= 2, integer >= 1 GPU per instance):
# task-count weights, shape (2,3) dropped (2 tasks, 0.05% — no 3-GPU class exists in
# the v2023 whole-GPU distribution), renormalized. Tuple = (workers k, GPU/worker d,
# weight). Provenance and the full joint table: data/gang/README.md.
PAI_GANG_SHAPES = (
    (2, 4.0, 0.3316),
    (2, 1.0, 0.2796),
    (4, 2.0, 0.2609),
    (8, 1.0, 0.0531),
    (2, 2.0, 0.0483),
    (4, 1.0, 0.0207),
    (3, 1.0, 0.0040),
    (5, 1.0, 0.0010),
    (6, 1.0, 0.0007),
)

STREAM_NOTE = (
    "Synthetic gang experiment layer: gang grouping/shape mixing, arrival order, "
    "fractional-background share and the demand horizon are generated, not observed. "
    "Worker classes are real (v2023 whole-GPU and fractional classes); the default "
    "gang-shape mix is the real PAI-v2020 distribution. Never present these gangs as "
    "production data (AGENTS §4.7-4.8)."
)


def tilt_multi_gpu_share(workload: Workload, share: float) -> Workload:
    """Tilt popularities so multi-GPU (d > 1) classes supply `share` of GPU demand.

    Workload-CONSTRUCTION knob for the gang experiment: arrivals AND the FGD
    objective both use the tilted M. This is NOT the "tilted-M" experiment
    (which tilts the objective's M while arrivals follow the true M) — keep the two
    apart in all write-ups.
    """
    if not 0.0 < share < 1.0:
        raise ValueError(f"multi_gpu_share must be in (0, 1), got {share}")
    d = workload.demand_array
    p = workload.popularities
    multi = d > 1.0
    if not multi.any():
        raise ValueError("workload has no multi-GPU classes to tilt")
    A = float((p * d * multi).sum())
    B = float((p * d * ~multi).sum())
    P_multi = float(p[multi].sum())
    P_single = float(p[~multi].sum())
    r = share * B / (A * (1.0 - share))
    beta = 1.0 / (r * P_multi + P_single)
    new_p = np.where(multi, p * r * beta, p * beta)
    return workload.with_popularities(new_p)


def workload_for_stream(base_workload: Workload, header: dict) -> Workload:
    """The whole-class M a stream was generated from: base, or tilted per its header."""
    share = header["config"].get("multi_gpu_share")
    if share is None:
        return base_workload
    return tilt_multi_gpu_share(base_workload, float(share))


def normalize_shapes(shapes) -> tuple[tuple[int, float | None, float], ...]:
    """Normalize a shape mix: iterable of (k, d-or-None, weight) -> weights sum to 1."""
    if not shapes:
        raise ValueError("empty gang-shape mix")
    out = []
    total = 0.0
    for k, d, w in shapes:
        if int(k) < 1:
            raise ValueError(f"shape workers k must be >= 1, got {k}")
        if d is not None and float(d) < 1.0:
            raise ValueError(f"shape GPU/worker must be >= 1 (whole workers), got {d}")
        if float(w) <= 0.0:
            raise ValueError(f"shape weight must be > 0, got {w}")
        out.append((int(k), (None if d is None else float(d)), float(w)))
        total += float(w)
    return tuple((k, d, w / total) for k, d, w in out)


def shapes_mean_gang_demand(shapes, m_whole: Workload) -> float:
    """Expected GPU demand of one gang arrival under the shape mix (E[k*d])."""
    demands = m_whole.demand_array
    pops = m_whole.popularities
    mean_whole = float((pops * demands).sum())
    return float(
        sum(w * k * (mean_whole if d is None else d) for k, d, w in shapes)
    )


def _group_probabilities(s_gang, s_frac, mean_gang, mean_whole, mean_frac):
    """Arrival-group probabilities so the expected DEMAND shares equal the targets.

    For a group with demand share s and mean demand m, p ∝ s/m; normalizing gives
    E[share_i] = s_i exactly in expectation."""
    entries = []
    if s_gang > 0.0 and mean_gang > 0.0:
        entries.append(("gang", s_gang / mean_gang))
    if s_frac > 0.0 and mean_frac > 0.0:
        entries.append(("fractional", s_frac / mean_frac))
    s_whole = 1.0 - s_gang - s_frac
    if s_whole > 0.0 and mean_whole > 0.0:
        entries.append(("whole", s_whole / mean_whole))
    if not entries:
        raise ValueError(
            f"no arrival group has positive demand (gang={s_gang}, frac={s_frac})"
        )
    total = sum(v for _, v in entries)
    return tuple(g for g, _ in entries), np.array([v / total for _, v in entries])


def _sample_whole_class(rng, m_whole: Workload, d: float | None) -> int:
    """Class index: from the full M_whole, or conditioned on gpu_demand == d."""
    demands = m_whole.demand_array
    if d is None:
        return int(rng.choice(len(demands), p=m_whole.popularities))
    mask = np.isclose(demands, float(d))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"M_whole has no class with gpu_demand={d}")
    pops = m_whole.popularities[idx]
    return int(idx[int(rng.choice(idx.size, p=pops / pops.sum()))])


def _shapes_entropy(shapes) -> int:
    return int(sum(k * 1000 + int((d or 0.0) * 100) for k, d, _ in shapes) % (2**31))


def sample_stream(
    m_whole: Workload,
    fraction: float,
    load: float,
    capacity: float,
    seed: int,
    meta: dict,
    k: int | None = None,
    shapes: tuple | None = None,
    multi_gpu_share: float | None = None,
    fractional_workload: Workload | None = None,
    fractional_share: float = 0.0,
) -> tuple[dict, list[StreamItem]]:
    """Generate one synthetic stream (schema v2).

    Arrival groups: gangs (shape mix; k workers of ONE class), whole singletons and
    fractional singletons (background). Group draw probabilities are set so the
    EXPECTED demand shares equal `fraction` (gangs) and `fractional_share`
    (fractional), the rest whole singletons (see `_group_probabilities`).
    Either `k` (legacy: all gangs of size k, worker class from M_whole) or `shapes`
    (explicit (k, d, weight) mix) must be given.

    The stream stops when cumulative GPU demand reaches `load * capacity` (the last
    arrival may overshoot, same as the inflation benchmark). `meta` supplies
    provenance (git_sha, source csv/md5); `multi_gpu_share` records a tilt already
    applied by the caller to `m_whole`; `fractional_workload` is the fractional-class
    M (its fingerprint goes into the header).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"gang_fraction must be within [0, 1], got {fraction}")
    if not 0.0 <= fractional_share <= 1.0:
        raise ValueError(f"fractional_share must be within [0, 1], got {fractional_share}")
    if fraction + fractional_share > 1.0 + 1e-12:
        raise ValueError(
            f"gang_fraction + fractional_share must be <= 1, got {fraction + fractional_share}"
        )
    if (k is None) == (shapes is None):
        raise ValueError("exactly one of `k` or `shapes` must be given")
    if shapes is not None:
        shapes = normalize_shapes(shapes)
    if fractional_share > 0.0 and fractional_workload is None:
        raise ValueError("fractional_share > 0 requires a fractional_workload")

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                round(fraction * 100),
                int(k) if k is not None else 0,
                _shapes_entropy(shapes) if shapes is not None else 0,
                round(load * 100),
                round(fractional_share * 100),
                round((multi_gpu_share if multi_gpu_share is not None else 0.0) * 100),
            ]
        )
    )
    target = float(load) * float(capacity)

    whole_demands = m_whole.demand_array
    whole_pops = m_whole.popularities
    mean_whole = float((whole_pops * whole_demands).sum())
    mean_frac = (
        0.0
        if fractional_workload is None
        else float((fractional_workload.popularities * fractional_workload.demand_array).sum())
    )
    mean_gang = (
        float(k) * mean_whole
        if shapes is None
        else shapes_mean_gang_demand(shapes, m_whole)
    )
    groups, probs = _group_probabilities(fraction, fractional_share, mean_gang, mean_whole, mean_frac)
    cum_probs = np.cumsum(probs)
    shape_weights = None if shapes is None else np.cumsum([w for _, _, w in shapes])

    items: list[StreamItem] = []
    cumulative = 0.0
    idx = 0
    while cumulative < target:
        group = groups[int(np.searchsorted(cum_probs, rng.random()))]
        if group == "gang":
            if shapes is None:
                size = int(k)
                m = _sample_whole_class(rng, m_whole, None)
                source = m_whole
            else:
                si = int(np.searchsorted(shape_weights, rng.random()))
                k_s, d_s, _ = shapes[si]
                size = k_s
                m = _sample_whole_class(rng, m_whole, d_s)
                source = m_whole
        elif group == "whole":
            size = 1
            m = _sample_whole_class(rng, m_whole, None)
            source = m_whole
        else:
            size = 1
            m = int(rng.choice(len(fractional_workload.demand_array), p=fractional_workload.popularities))
            source = fractional_workload
        d = float(source.demand_array[m])
        items.append(
            StreamItem(
                gang_id=f"g{idx:06d}",
                gang_size=size,
                worker_cpu_milli=int(source.classes[m].cpu_milli),
                worker_gpu_demand=d,
                arrival_index=idx,
                is_gang=group == "gang",
            )
        )
        cumulative += size * d
        idx += 1

    header = {
        "type": "header",
        "schema_version": STREAM_SCHEMA_VERSION,
        "git_sha": meta.get("git_sha", "unknown"),
        "seed": int(seed),
        "config": {
            "gang_fraction": float(fraction),
            "gang_size": (int(k) if k is not None else None),
            "gang_shapes": (
                None
                if shapes is None
                else [
                    {"k": k_s, "d": d_s, "weight": round(w, 6)}
                    for k_s, d_s, w in shapes
                ]
            ),
            "load": float(load),
            "target_demand": target,
            "capacity": float(capacity),
            "multi_gpu_share": (None if multi_gpu_share is None else float(multi_gpu_share)),
            "fractional_share": float(fractional_share),
        },
        "m_whole_sha256": workload_fingerprint(m_whole),
        "m_fractional_sha256": (
            None if fractional_workload is None else workload_fingerprint(fractional_workload)
        ),
        "source_pod_csv": meta.get("source_pod_csv", ""),
        "source_pod_md5": meta.get("source_md5", ""),
        "note": STREAM_NOTE,
        "realized": realized_stats(items),
    }
    return header, items


def policy_workload(
    base_whole: Workload, header: dict, fractional_workload: Workload | None = None
) -> Workload:
    """The M the policies should score against for this stream (schema v2).

    Mixes whole classes (gangs + whole singletons) and fractional classes by their
    ARRIVAL probabilities (the FGD objective's p_m is the task-class popularity).
    Deterministic given the base M, the fractional M and the header config.
    """
    whole = workload_for_stream(base_whole, header)
    cfg = header["config"]
    fshare = float(cfg.get("fractional_share") or 0.0)
    if fshare <= 0.0 or fractional_workload is None:
        return whole
    if cfg.get("gang_shapes"):
        shapes = normalize_shapes(
            (s["k"], s["d"], s["weight"]) for s in cfg["gang_shapes"]
        )
    elif cfg.get("gang_size") is not None:
        shapes = ((int(cfg["gang_size"]), None, 1.0),)
    else:
        shapes = ()
    mean_whole = float((whole.popularities * whole.demand_array).sum())
    mean_frac = float(
        (fractional_workload.popularities * fractional_workload.demand_array).sum()
    )
    mean_gang = shapes_mean_gang_demand(shapes, whole) if shapes else 0.0
    groups, probs = _group_probabilities(
        float(cfg["gang_fraction"]), fshare, mean_gang, mean_whole, mean_frac
    )
    p = dict(zip(groups, probs))

    p_arr_whole = p.get("whole", 0.0) * whole.popularities.copy()
    if p.get("gang", 0.0) > 0.0 and shapes:
        demands = whole.demand_array
        for _, d_s, w in shapes:
            if d_s is None:
                p_arr_whole += p["gang"] * w * whole.popularities
            else:
                mask = np.isclose(demands, d_s)
                sub = np.where(mask, whole.popularities, 0.0)
                if sub.sum() <= 0.0:
                    raise ValueError(f"M has no class with gpu_demand={d_s}")
                p_arr_whole += p["gang"] * w * (sub / sub.sum())
    p_arr_frac = p.get("fractional", 0.0) * fractional_workload.popularities
    pops = np.concatenate([p_arr_whole, p_arr_frac])
    classes: tuple[TaskClass, ...] = tuple(whole.classes) + tuple(fractional_workload.classes)
    return Workload(classes, pops / pops.sum())
