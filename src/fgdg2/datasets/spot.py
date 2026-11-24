"""Loader for the Alibaba cluster-trace-v2026 spot-gpu dataset (GFS, ASPLOS'26).

Source: github.com/alibaba/clusterdata -> cluster-trace-v2026-spot-gpu
(job_info_df.csv, node_info_df.csv; umbrella research/study terms).

Semantics (verified in docs/research/briefs/GB_spotgpu_gate.md):
- one row = one job; `worker_num` = co-scheduled instances (the gang size k);
  `gpu_request`/`cpu_request` are PER-WORKER demands (per-worker reading is the
  feasibility-consistent one: 0 jobs exceed their model's largest node or pool);
- `worker_num == 1 and gpu_request > 1` is a PACKED job -> a singleton stream
  item with whole demand d > 1 (packed = one task, d GPUs, one node);
- `worker_num > 1` is a SPREAD gang of homogeneous workers;
- `gpu_model` is a HARD constraint: jobs pin to a GPU model, so each model is an
  independent scheduling domain (per-domain replay; the FGD argmin only compares
  feasible nodes, so domain replay is equivalent to a unified cluster with
  gpu_spec filtering);
- fractional gang workers (worker_num > 1 and gpu_request < 1) are out of scope
  (scope lock, MixedPolicy raises) and are filtered, counted in the manifest;
- `submit_time`/`duration`/`job_type`/`gpu_model` ride along as optional
  StreamItem fields for Layer 2; Layer-1 runners ignore them.
"""

import numpy as np
import pandas as pd

from ..core.tasks import Workload
from ..items import StreamItem
from ..streams.io import realized_shape_mix, realized_stats, workload_fingerprint

WHOLE_MILLI = 1000


def load_jobs(path) -> pd.DataFrame:
    """Raw spot-gpu job table, validated and sorted by submit_time."""
    df = pd.read_csv(path)
    required = {
        "job_name", "organization", "gpu_model", "cpu_request", "gpu_request",
        "worker_num", "submit_time", "duration", "job_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if (df["gpu_request"] < 0).any() or (df["worker_num"] < 1).any():
        raise ValueError(f"{path}: negative gpu_request or worker_num < 1")
    if (df["duration"] <= 0).any():
        raise ValueError(f"{path}: non-positive durations present")
    return df.sort_values(["submit_time", "job_name"], kind="stable").reset_index(drop=True)


def load_node_frame(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"node_name", "gpu_model", "gpu_capacity_num", "cpu_num"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def worker_gpu_milli(gpu_request: float) -> int:
    """gpu_request -> milli share, for fractional requests."""
    return round(float(gpu_request) * WHOLE_MILLI)


def jobs_to_pod_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """One v2023-schema pod row per JOB (arrival-level; worker class = job's
    per-worker request) so build_m_whole / build_m_fractional reproduce the
    empirical class histogram. Fractional requests map to (num_gpu=1,
    gpu_milli=round(d*1000)) — the same gpu_demand() decoding as v2023."""
    df = pd.DataFrame(
        {
            "name": jobs["job_name"].astype(str),
            "creation_time": jobs["submit_time"].astype(float),
            "cpu_milli": (jobs["cpu_request"] * WHOLE_MILLI).round().astype(int),
            "num_gpu": np.where(jobs["gpu_request"] >= 1.0, jobs["gpu_request"].astype(int), 1),
            "gpu_milli": np.where(
                jobs["gpu_request"] >= 1.0, WHOLE_MILLI,
                (jobs["gpu_request"] * WHOLE_MILLI).round().astype(int),
            ),
            "memory_mib": 0,
        }
    )
    return df.sort_values(["creation_time", "name"], kind="stable").reset_index(drop=True)


def nodes_to_node_frame(nodes: pd.DataFrame, model: str) -> pd.DataFrame:
    """v2023-schema node rows for one GPU-model domain (memory_mib=0 — the trace
    carries no memory field; TaskClass.memory_mib is 0 so the constraint is
    inert)."""
    sub = nodes[nodes["gpu_model"] == model]
    if sub.empty:
        raise ValueError(f"no nodes with gpu_model={model!r}")
    return pd.DataFrame(
        {
            "sn": sub["node_name"].astype(str).to_numpy(),
            "gpu": sub["gpu_capacity_num"].astype(int).to_numpy(),
            "model": sub["gpu_model"].astype(str).to_numpy(),
            "cpu_milli": (sub["cpu_num"] * WHOLE_MILLI).astype(int).to_numpy(),
            "memory_mib": 0,
        }
    ).sort_values("sn", kind="stable").reset_index(drop=True)


def split_jobs(jobs: pd.DataFrame):
    """(kept, filtered) — drops fractional gang jobs (out of scope) and
    zero-GPU jobs; returns the drop counts for the manifest."""
    frac_gang = (jobs["worker_num"] > 1) & (jobs["gpu_request"] < 1.0)
    zero_gpu = jobs["gpu_request"] <= 0.0
    drop = frac_gang | zero_gpu
    return jobs[~drop].copy(), {
        "dropped_fractional_gangs": int(frac_gang.sum()),
        "dropped_zero_gpu_jobs": int(zero_gpu.sum()),
    }


def select_window(jobs: pd.DataFrame, capacity: float, load: float,
                  rule: str = "representative") -> pd.DataFrame:
    """Pick the contiguous submit-order slice whose cumulative GPU demand first
    reaches load*capacity.

    rule="prefix": the head of the trace (can be unrepresentative — e.g. A10's
    early hours are singleton-only).
    rule="representative": among all contiguous windows reaching the demand
    target, the one whose gang-demand share is closest to the domain's overall
    share (ties -> earliest). Deterministic; the window's submit_time range is
    recorded in the stream header."""
    target = float(load) * float(capacity)
    demand = (jobs["worker_num"] * jobs["gpu_request"]).to_numpy(dtype=float)
    if demand.sum() <= target or rule == "prefix":
        cum = np.cumsum(demand)
        end = int(np.searchsorted(cum, target, side="left")) + 1
        return jobs.iloc[: min(end, len(jobs))]

    is_gang = (jobs["worker_num"] > 1).to_numpy()
    gang_dem = np.where(is_gang, demand, 0.0)
    domain_share = gang_dem.sum() / demand.sum()
    n = len(jobs)
    best = None  # (|share diff|, start, end)
    lo = 0
    acc = acc_g = 0.0
    for hi in range(n):
        acc += demand[hi]
        acc_g += gang_dem[hi]
        while lo <= hi and acc - demand[lo] >= target:
            acc -= demand[lo]
            acc_g -= gang_dem[lo]
            lo += 1
        if acc >= target:
            share = acc_g / acc
            diff = abs(share - domain_share)
            if best is None or diff < best[0] - 1e-12:
                best = (diff, lo, hi + 1)
    if best is None:
        return jobs
    return jobs.iloc[best[1] : best[2]]


def select_peak_window(jobs: pd.DataFrame, capacity: float,
                       days: float = 7.0,
                       gang_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Layer-2 window: the contiguous submit-time span of `days` days with the
    highest mean IN-WINDOW concurrency under replay-from-empty — i.e.
    score(W) = sum_j gpu_j * min(dur_j, W_end - submit_j) / len(W) / capacity.
    This is the honest congestion proxy: submitted gpu-hours alone mislead
    because long jobs (p99 ~ 60 days) execute mostly outside the window.
    Deterministic, candidate starts at day granularity, ties -> earliest.

    gang_only=True scores by GANG-job concurrency only (worker_num>1) — used to
    find windows where gangs actually congest a domain (e.g. GPU-series-1,
    whose total-load peak is gang-free)."""
    dem = (jobs["worker_num"].to_numpy(dtype=float)
           * jobs["gpu_request"].clip(lower=0).to_numpy(dtype=float))
    if gang_only:
        dem = dem * (jobs["worker_num"].to_numpy() > 1)
    st = jobs["submit_time"].to_numpy(dtype=float)
    du = jobs["duration"].to_numpy(dtype=float)
    w = max(1, round(days))
    d0, d1 = int(st.min() // 86400), int(st.max() // 86400)
    span = w * 86400.0
    best = None  # (mean_concurrency, day_start)
    for ds in range(d0, d1 - w + 2):
        t_lo, t_hi = ds * 86400.0, (ds + w) * 86400.0
        m = (st >= t_lo) & (st < t_hi)
        if not m.any():
            continue
        conc = float((dem[m] * np.minimum(du[m], t_hi - st[m])).sum() / span)
        if best is None or conc > best[0]:
            best = (conc, ds)
    if best is None:
        return jobs.iloc[0:0].copy(), {
            "t_lo": d0 * 86400.0, "t_hi": (d0 + w) * 86400.0, "days": w,
            "window_mean_concurrency": 0.0, "window_rho_c": 0.0,
        }
    conc, ds = best
    t_lo, t_hi = ds * 86400.0, (ds + w) * 86400.0
    mask = (st >= t_lo) & (st < t_hi)
    out = jobs[mask].copy()
    return out, window_stats(out, t_lo, t_hi, capacity, gang_only=gang_only)


def window_stats(window: pd.DataFrame, t_lo: float, t_hi: float,
                 capacity: float, gang_only: bool = False) -> dict:
    """The layer2_window header block for an explicitly chosen window —
    same fields select_peak_window emits (rho_c = mean in-window
    concurrency / capacity)."""
    span = t_hi - t_lo
    dem = (window["worker_num"].to_numpy(dtype=float)
           * window["gpu_request"].clip(lower=0).to_numpy(dtype=float))
    gang_dem = dem * (window["worker_num"].to_numpy() > 1)
    tot_conc = float(
        (dem * np.minimum(window["duration"].to_numpy(dtype=float),
                          t_hi - window["submit_time"].to_numpy(dtype=float))).sum()
        / span
    ) if len(window) else 0.0
    return {
        "t_lo": float(t_lo), "t_hi": float(t_hi), "days": span / 86400.0,
        "window_mean_concurrency": tot_conc,
        "window_rho_c": tot_conc / float(capacity),
        "gang_only_scored": bool(gang_only),
        "window_gang_concurrency_share": (
            float(gang_dem.sum() / dem.sum()) if dem.sum() else 0.0
        ),
    }


def jobs_to_items(jobs: pd.DataFrame, capacity: float | None = None,
                  load: float | None = None) -> list[StreamItem]:
    """Stream items in submit_time order. With capacity+load: capped at
    cumulative demand >= load*capacity (Layer-1 fill-up semantics). With
    capacity=None: uncapped — every job emits an item (Layer-2 replay).
    submit_time_s/duration_s/job_type/gpu_model ride along for Layer 2."""
    target = None if capacity is None else float(load) * float(capacity)
    items: list[StreamItem] = []
    cumulative = 0.0
    for row in jobs.itertuples(index=False):
        d = float(row.gpu_request)
        # match gpu_demand() decoding exactly so class keys are identical
        d = float(int(d)) if d >= 1.0 else worker_gpu_milli(d) / WHOLE_MILLI
        items.append(
            StreamItem(
                gang_id=str(row.job_name),
                gang_size=int(row.worker_num),
                worker_cpu_milli=round(float(row.cpu_request) * WHOLE_MILLI),
                worker_gpu_demand=d,
                arrival_index=len(items),
                is_gang=bool(row.worker_num > 1),
                submit_time_s=float(row.submit_time),
                duration_s=float(row.duration),
                job_type=str(row.job_type),
                gpu_model=str(row.gpu_model),
            )
        )
        cumulative += int(row.worker_num) * d
        if target is not None and cumulative >= target:
            break
    return items


def stream_header(
    *,
    model: str,
    capacity: float,
    load: float | None,
    items: list[StreamItem],
    m_whole: Workload,
    m_frac: Workload | None,
    meta: dict,
    window: dict | None = None,
) -> dict:
    """Stream header for REAL (spot-gpu) streams — note and config differ from
    the synthetic header in datasets.synth."""
    from ..streams.io import STREAM_SCHEMA_VERSION

    rz = realized_stats(items)
    shapes = realized_shape_mix(items)
    n_gangs = rz["n_gangs"] or 1
    # arrival-level gang share for the config record (demand share is in realized)
    cfg = {
        "gang_fraction": rz["realized_gang_demand_share"],
        "gang_size": None,
        "gang_shapes": shapes,
        "load": None if load is None else float(load),
        "target_demand": (
            None if load is None else float(load) * float(capacity)
        ),
        "capacity": float(capacity),
        "multi_gpu_share": None,
        "fractional_share": rz["realized_fractional_demand_share"],
        "real_trace": "spot-gpu-v2026",
        "gpu_model_domain": model,
        "gang_arrival_share": float(rz["n_gangs"] / max(len(items), 1)),
        "mean_gang_size": float(
            sum(it.gang_size for it in items if it.is_gang) / n_gangs
        ),
    }
    if window is not None:
        cfg["temporal"] = True
        cfg["layer2_window"] = window
    return {
        "type": "header",
        "schema_version": STREAM_SCHEMA_VERSION,
        "git_sha": meta.get("git_sha", "unknown"),
        "seed": 0,
        "config": cfg,
        "m_whole_sha256": workload_fingerprint(m_whole),
        "m_fractional_sha256": (
            None if m_frac is None else workload_fingerprint(m_frac)
        ),
        "source_pod_csv": meta.get("source_pod_csv", ""),
        "source_pod_md5": meta.get("source_md5", ""),
        "note": (
            "REAL trace replay (Alibaba cluster-trace-v2026-spot-gpu, GFS ASPLOS'26): "
            "items are observed jobs in submit_time order; worker_num is the "
            "co-scheduled instance count (spread gang when >1); worker_num==1 with "
            "gpu_request>1 is a packed job; fractional gang jobs filtered (scope "
            "lock). Not a synthetic mix — do not regress to synthetic-only claims."
        ),
        "realized": rz,
    }
