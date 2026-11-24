"""Alibaba cluster-trace-gpu-v2023 adapter — pods/nodes CSVs -> core objects.

Also hosts the workload (M) constructors shared by every dataset source:
`build_workload` (95%-mass-truncated, reference convention), `build_m_whole`
/ `build_m_fractional` (gang-phase M, no truncation), plus pod-row decoding
(`task_class_from_row`, `gpu_demand`) and the co-arrival honesty layer.
"""

from collections import Counter

import pandas as pd

from ..core.cluster import Cluster
from ..core.node import Node
from ..core.tasks import TaskClass, Workload

POPULARITY_MASS_THRESHOLD = 0.95
WHOLE_GPU_MILLI = 1000
# No popularity truncation in the gang layer: the 95% cut is a calibration-side,
# reference-fidelity device (GetTypicalPods semantics); keeping it here would
# silently drop every multi-GPU worker class (75 pods, 1.9% of pods, 10.2% of
# whole-GPU demand).
GANG_POPULARITY_MASS = 1.0


# -- raw loaders -------------------------------------------------------------

def load_nodes(csv_path) -> Cluster:
    df = pd.read_csv(csv_path)
    df = df.sort_values("sn", kind="stable").reset_index(drop=True)
    df = df.fillna({"model": None})
    nodes = [
        Node(
            node_id=row.sn,
            num_gpus=int(row.gpu),
            model=None if row.model is None or pd.isna(row.model) else str(row.model),
            cpu_milli=int(row.cpu_milli),
            memory_mib=int(row.memory_mib),
        )
        for row in df.itertuples(index=False)
    ]
    return Cluster(nodes)


def load_pods(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["creation_time", "name"], kind="stable").reset_index(drop=True)
    return df


def gpu_demand(row) -> float:
    n = int(row.num_gpu)
    milli = int(row.gpu_milli)
    if n == 0:
        return 0.0
    if milli == WHOLE_GPU_MILLI:
        return float(n)
    if n == 1:
        return milli / 1000.0
    raise ValueError(f"unsupported demand combo num_gpu={n}, gpu_milli={milli}")


def task_class_from_row(row) -> TaskClass:
    """v2023 pod row -> TaskClass (memory included — the trace enforces it)."""
    demand = float(row["gpu_demand"])
    return TaskClass(
        name=f"t_{row['name']}",
        cpu_milli=int(row["cpu_milli"]),
        memory_mib=int(row["memory_mib"]),
        gpu_demand=demand,
        whole_gpu=(demand >= 1.0),
    )


# -- workload constructors ----------------------------------------------------

def build_workload(pods: pd.DataFrame, popularity_mass: float = POPULARITY_MASS_THRESHOLD):
    """M in the reference-impl convention: class = (cpu_milli, gpu_demand) — memory is
    NOT part of the key (reference PodResource has it commented out); popularities =
    trace-class histogram truncated at `popularity_mass` (0.95 in the paper's config)
    and renormalized on the kept mass.

    Note: a hard mass cut keeps |M| bounded; set popularity_mass=1.0 to disable.
    """
    pods = pods.copy()
    pods["gpu_demand"] = pods.apply(gpu_demand, axis=1)
    counts = pods.groupby(
        ["cpu_milli", "gpu_demand"], sort=True
    ).size().rename("cnt").reset_index()
    counts = counts.sort_values("cnt", ascending=False, kind="stable")
    total = len(pods)
    cumulative = counts["cnt"].cumsum() / total
    kept = counts[cumulative <= popularity_mass] if cumulative.iloc[0] <= popularity_mass else counts.head(1)
    kept = kept.copy()
    kept["p"] = kept["cnt"] / kept["cnt"].sum()
    classes = [
        TaskClass(
            name=f"c{row.cpu_milli}_{row.gpu_demand:g}",
            cpu_milli=int(row.cpu_milli),
            gpu_demand=float(row.gpu_demand),
            whole_gpu=(row.gpu_demand >= 1.0),
        )
        for row in kept.itertuples(index=False)
    ]
    workload = Workload(classes, kept["p"].to_numpy())
    return workload, pods


def whole_gpu_pods(pods: pd.DataFrame) -> pd.DataFrame:
    """Pods with whole-GPU requests only (num_gpu >= 1 and gpu_milli == 1000)."""
    return pods[(pods["num_gpu"] >= 1) & (pods["gpu_milli"] == WHOLE_GPU_MILLI)].copy()


def fractional_gpu_pods(pods: pd.DataFrame) -> pd.DataFrame:
    """Pods with fractional GPU requests (num_gpu == 1 and 0 < gpu_milli < 1000)."""
    return pods[(pods["num_gpu"] == 1) & (pods["gpu_milli"] > 0) & (pods["gpu_milli"] < WHOLE_GPU_MILLI)].copy()


def _truncated_counts(whole: pd.DataFrame, mass: float) -> pd.DataFrame:
    """Class histogram truncated at `mass`, same rule as build_workload."""
    counts = whole.groupby(["cpu_milli", "num_gpu"]).size().rename("cnt").reset_index()
    counts = counts.sort_values("cnt", ascending=False, kind="stable")
    cumulative = counts["cnt"].cumsum() / len(whole)
    if cumulative.iloc[0] <= mass:
        kept = counts[cumulative <= mass]
    else:
        kept = counts.head(1)
    return kept


def build_m_whole(pods: pd.DataFrame, mass: float = GANG_POPULARITY_MASS):
    """M_whole for the gang phase: the whole-GPU worker-class distribution.

    Unlike the calibrated workload (95% mass truncation), the gang layer keeps ALL
    whole-GPU classes (GANG_POPULARITY_MASS = 1.0) so the real multi-GPU worker tail
    (2/4/8-GPU classes) survives — a documented, deliberate deviation.
    """
    whole = whole_gpu_pods(pods)
    workload, _ = build_workload(whole, popularity_mass=mass)
    kept = _truncated_counts(whole, mass)
    total_demand = float((pods["num_gpu"] * pods["gpu_milli"] / 1000.0).sum())
    whole_demand = float(whole["num_gpu"].sum())
    demands = workload.demand_array
    stats = {
        "n_pods_total": len(pods),
        "n_whole_pods": len(whole),
        "n_classes": len(workload.classes),
        "mass_threshold": float(mass),
        "kept_mass": float(kept["cnt"].sum() / len(whole)),
        "whole_gpu_demand_share_of_trace": float(whole_demand / total_demand),
        "mean_worker_gpu_demand": float((workload.popularities * demands).sum()),
        "worker_gpu_demands": sorted({float(d) for d in demands}),
        "multi_gpu_demand_share": multi_gpu_demand_share(workload),
    }
    return workload, stats


def build_m_fractional(pods: pd.DataFrame, mass: float = GANG_POPULARITY_MASS):
    """M_frac: the fractional-GPU class distribution (background candidates).

    Returns (None, stats) when the population has no fractional rows — callers
    already guard on `fractional_workload is None` (e.g. spot-gpu domains with
    zero fractional jobs)."""
    frac = fractional_gpu_pods(pods)
    if frac.empty:
        return None, {
            "n_fractional_pods": 0,
            "n_classes": 0,
            "kept_mass": 0.0,
            "fractional_gpu_demand_share_of_trace": 0.0,
            "mean_worker_gpu_demand": 0.0,
            "worker_gpu_demands": [],
        }
    workload, _ = build_workload(frac, popularity_mass=mass)
    kept = _truncated_counts(frac, mass)
    total_demand = float((pods["num_gpu"] * pods["gpu_milli"] / 1000.0).sum())
    frac_demand = float((frac["num_gpu"] * frac["gpu_milli"] / 1000.0).sum())
    stats = {
        "n_fractional_pods": len(frac),
        "n_classes": len(workload.classes),
        "kept_mass": float(kept["cnt"].sum() / len(frac)) if len(frac) else 0.0,
        "fractional_gpu_demand_share_of_trace": float(frac_demand / total_demand),
        "mean_worker_gpu_demand": float(
            (workload.popularities * workload.demand_array).sum()
        ),
        "worker_gpu_demands": sorted({float(d) for d in workload.demand_array}),
    }
    return workload, stats


def multi_gpu_demand_share(workload: Workload) -> float:
    """Share of GPU demand supplied by classes with gpu_demand > 1."""
    d = workload.demand_array
    p = workload.popularities
    total = float((p * d).sum())
    return float((p * d * (d > 1.0)).sum() / total)


def empirical_workload_from_pods(pods: pd.DataFrame) -> Workload:
    """Class histogram of a v2023-schema pod frame — same (cpu_milli,
    gpu_demand) key as empirical_workload (stale-M source path for .csv)."""
    from ..core.tasks import workload_from_counts

    counts = Counter(
        (int(r.cpu_milli), float(gpu_demand(r))) for r in pods.itertuples(index=False)
    )
    return workload_from_counts(counts)


def coarrival_stats(whole: pd.DataFrame) -> dict:
    """Honesty layer: co-arrival groups of whole-GPU pods by (creation_time, class).

    Groups are a LOWER bound on real gang structure: same-second co-arrival of one
    class does not prove a production gang; its absence is evidence against one.
    """
    from ..streams.io import STREAM_SCHEMA_VERSION

    groups = whole.groupby(["creation_time", "cpu_milli", "num_gpu"]).size()
    histogram = {int(s): int(c) for s, c in groups.value_counts().sort_index().items()}
    pods_in_multi = int(groups[groups > 1].sum())
    by_class = (
        whole.groupby(["cpu_milli", "num_gpu"]).size().sort_values(ascending=False, kind="stable")
    )
    return {
        "schema_version": STREAM_SCHEMA_VERSION,
        "n_whole_pods": len(whole),
        "n_groups": len(groups),
        "group_size_histogram": histogram,
        "pods_in_groups_gt1": pods_in_multi,
        "share_pods_in_groups_gt1": float(pods_in_multi / len(whole)),
        "top_classes": [
            {"cpu_milli": int(cpu), "gpu_demand": float(g), "pods": int(n)}
            for (cpu, g), n in by_class.head(15).items()
        ],
        "note": (
            "Grouping rule = same (creation_time, cpu_milli, num_gpu). These are NOT "
            "verified production gangs: natural co-arrival is near-singleton and "
            "second-granularity grouping is a LOWER bound on real gang structure."
        ),
    }


def items_to_pod_frame(items) -> pd.DataFrame:
    """StreamItems -> a v2023-schema pod frame (arrival-level; for M builders)."""
    rows = [
        {
            "name": it.gang_id,
            "creation_time": it.t_submit,
            "cpu_milli": it.worker_cpu_milli,
            "num_gpu": int(it.worker_gpu_demand) if it.worker_gpu_demand >= 1.0 else 1,
            "gpu_milli": (
                WHOLE_GPU_MILLI if it.worker_gpu_demand >= 1.0
                else round(it.worker_gpu_demand * WHOLE_GPU_MILLI)
            ),
            "memory_mib": 0,
        }
        for it in items
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(["creation_time", "name"], kind="stable").reset_index(drop=True)
