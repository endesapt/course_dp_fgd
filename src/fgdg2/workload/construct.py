"""Pure M (workload-distribution) constructors and distances.

These build a `Workload` from observed data (stream items, pod frames, files)
or transform one (tilt). Nothing here touches the cluster or the clock —
tracking estimators live in `workload/tracking.py`.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.tasks import Workload, workload_from_counts
from ..datasets.durations import durations_from_file
from ..datasets.v2023 import gpu_demand, load_pods
from ..items import StreamItem


def empirical_workload(items) -> Workload:
    """The exact per-item worker-class distribution (whole + fractional classes
    in one Workload) — the correct policy M for a real trace, where classes are
    observed, not sampled from a generative mix. This is the window ORACLE for
    Layer-2 experiments."""
    return workload_from_counts(Counter(it.class_key for it in items))


def empirical_workload_from_file(path) -> Workload:
    """Stale-M source: a stream .jsonl (item histogram) or a v2023-schema
    pod/jobs .csv (row histogram). Light parse — no stream validation; the
    file only supplies class frequencies."""
    path = Path(path)
    if path.suffix == ".jsonl":
        items = [
            StreamItem.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()[1:]
            if line.strip()
        ]
        return empirical_workload(items)
    if path.suffix == ".csv":
        return empirical_workload_from_pods(load_pods(path))
    raise ValueError(f"unsupported stale-M source {path} (want .jsonl or .csv)")


def empirical_workload_from_pods(pods: pd.DataFrame) -> Workload:
    """Class histogram of a v2023-schema pod frame — same (cpu_milli,
    gpu_demand) key as empirical_workload (stale-M source path for .csv)."""
    counts = Counter(
        (int(r.cpu_milli), float(gpu_demand(r))) for r in pods.itertuples(index=False)
    )
    return workload_from_counts(counts)


def occupancy_workload_from_file(path) -> Workload:
    """`stale_occ:` source — the duration-weighted (λ_m·E[L_m]) class mix of
    a PAST window: .jsonl streams clip at that window's layer2_window.t_hi
    (realized occupancy), .csv sources use raw durations."""
    path = Path(path)
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0]) if lines else {}
        t_end = ((header.get("config") or {}).get("layer2_window") or {}).get("t_hi")
        items = [
            StreamItem.from_dict(json.loads(line))
            for line in lines[1:]
            if line.strip()
        ]
        return occupancy_workload(items, t_end)
    if path.suffix == ".csv":
        durs = durations_from_file(path)
        return workload_from_counts({k: sum(v) for k, v in durs.items()})
    raise ValueError(f"unsupported stale_occ source {path} (want .jsonl or .csv)")


def tilted_workload(workload: Workload, gamma: float) -> Workload:
    """Path-B tilted-M arm (AGENTS §3): p~_m ∝ p_m · d_m^gamma over GPU demand.

    gamma = 0 is the identity (the true-M rule). gamma > 0 over-weights
    big-demand classes, gamma < 0 over-weights small ones. GPU-free classes
    (d = 0) are tilt-NEUTRAL (factor 1): 0^gamma is degenerate for gamma < 0
    and silently dropping them for gamma > 0 would change the support — the
    tilt family is support-preserving by design."""
    if gamma == 0.0:
        return workload
    d = workload.demand_array
    factor = np.ones_like(d)
    pos = d > 0.0
    factor[pos] = np.power(d[pos], gamma)
    p = workload.popularities * factor
    return workload.with_popularities(p / p.sum())


def occupancy_workload(items, t_end: float | None = None) -> Workload:
    """Occupancy-weighted M: p_m ∝ Σ_{items of class m} duration — which
    classes HOLD the cluster (λ_m·E[L_m] weighting) rather than which arrive
    (the FGD objective's p_m is an arrival frequency; under churn the occupancy
    mix is what the state actually sees).

    With t_end (the replay horizon, layer2_window.t_hi) each item's weight is
    min(duration_s, t_end − submit_time_s) — realized in-window occupancy;
    without t_end the raw durations are used."""
    weights = Counter()
    for it in items:
        if it.duration_s is None:
            raise ValueError(
                f"item {it.gang_id} has no duration_s — occupancy M "
                "needs real durations"
            )
        w = float(it.duration_s)
        if t_end is not None:
            w = min(w, float(t_end) - it.t_submit)
        if w > 0.0:
            weights[it.class_key] += w
    return workload_from_counts(weights)


def workload_dtv(a: Workload, b: Workload) -> float:
    """d_TV between two workloads over the union of their
    (cpu_milli, gpu_demand) supports — the B1-lemma distance."""
    pa = {
        (int(c.cpu_milli), float(c.gpu_demand)): float(p)
        for c, p in zip(a.classes, a.popularities)
    }
    pb = {
        (int(c.cpu_milli), float(c.gpu_demand)): float(p)
        for c, p in zip(b.classes, b.popularities)
    }
    keys = set(pa) | set(pb)
    return 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)


def histogram_dtv(p: dict, q: dict) -> float:
    """Total variation between two count-histograms (normalised inside)."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if sp == 0 or sq == 0:
        return float("nan")
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)
