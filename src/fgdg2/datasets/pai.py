"""Alibaba cluster-trace-gpu-v2020 (PAI) adapter.

Source: github.com/alibaba/clusterdata -> cluster-trace-gpu-v2020
(CC BY 4.0; Weng et al., NSDI'22). The released tables are headerless CSVs
with sibling ``*.header`` files:

- ``pai_job_table.csv``      job_name, inst_id, user, status, start_time, end_time
- ``pai_task_table.csv``     job_name, task_name, inst_num, status, start_time,
                             end_time, plan_cpu, plan_mem, plan_gpu, gpu_type
- ``pai_machine_spec.csv``   machine, gpu_type, cap_cpu, cap_mem, cap_gpu

Reconstruction rules (documented in ``data/trace-gpu-v2020/README.md``):
- one task with ``inst_num = k`` is one gang of k homogeneous workers;
- ``plan_gpu`` is a percentage: 100 = one whole GPU, 50 = half a GPU;
- whole-GPU workers require ``plan_gpu >= 100`` and ``plan_gpu % 100 == 0``;
  combined requests (150, 250, ...) are outside the FGD model and are dropped;
- fractional gangs (``inst_num > 1`` and ``0 < plan_gpu < 100``) are outside
  the whole-worker gang layer and are dropped; fractional singletons stay as
  background items;
- arrival time is the JOB submission time (``job.start_time``); task launch
  time is later and encodes the observed wait, which the simulator reproduces;
- duration is ``task.end_time - task.start_time``; missing end times are
  allowed for Layer-1 fill-up but filtered before Layer-2 replay;
- ``gpu_type`` is the assigned type in the trace and is treated as a hard
  domain constraint, matching the spot-gpu adapter's per-model replay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.tasks import Workload
from ..items import StreamItem
from ..streams.io import realized_shape_mix, realized_stats, workload_fingerprint

WHOLE_MILLI = 1000
WHOLE_PLAN_GPU = 100.0

JOB_COLUMNS = ("job_name", "inst_id", "user", "status", "start_time", "end_time")
TASK_COLUMNS = (
    "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
    "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
)
MACHINE_COLUMNS = ("machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu")


def _read_headerless(path, names) -> pd.DataFrame:
    df = pd.read_csv(path, names=list(names), header=None)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def load_jobs(path) -> pd.DataFrame:
    df = _read_headerless(path, JOB_COLUMNS)
    if df["job_name"].isna().any():
        raise ValueError(f"{path}: null job_name")
    return df.sort_values(["start_time", "job_name"], kind="stable").reset_index(drop=True)


def load_tasks(path) -> pd.DataFrame:
    df = _read_headerless(path, TASK_COLUMNS)
    if df["job_name"].isna().any() or df["task_name"].isna().any():
        raise ValueError(f"{path}: null job_name/task_name")
    if (df["inst_num"].dropna() < 1).any():
        raise ValueError(f"{path}: inst_num < 1")
    if (df["plan_gpu"].dropna() < 0).any():
        raise ValueError(f"{path}: negative plan_gpu")
    return df.sort_values(["job_name", "task_name"], kind="stable").reset_index(drop=True)


def load_machine_spec(path) -> pd.DataFrame:
    df = _read_headerless(path, MACHINE_COLUMNS)
    if df["machine"].isna().any():
        raise ValueError(f"{path}: null machine")
    return df.sort_values("machine", kind="stable").reset_index(drop=True)


def load_job_tasks(job_path, task_path) -> pd.DataFrame:
    """Join tasks to their job submission times and derive canonical fields."""
    jobs = load_jobs(job_path)
    tasks = load_tasks(task_path)
    job_cols = jobs[["job_name", "start_time", "end_time", "status"]].rename(
        columns={
            "start_time": "submit_time",
            "end_time": "job_end_time",
            "status": "job_status",
        }
    )
    df = tasks.merge(job_cols, on="job_name", how="left", validate="many_to_one")
    # Task-level CPU/memory fields are occasionally missing; an unknown request
    # is treated as zero (the FGD class key is CPU+GPU, memory is inert here).
    df["plan_cpu"] = df["plan_cpu"].fillna(0.0)
    df["plan_mem"] = df["plan_mem"].fillna(0.0)
    df["worker_num"] = df["inst_num"]
    df["cpu_request"] = df["plan_cpu"] / 100.0       # percent -> cores
    df["gpu_request"] = df["plan_gpu"] / 100.0       # percent -> GPUs
    df["duration"] = df["end_time"] - df["start_time"]
    df["wait_s"] = df["start_time"] - df["submit_time"]
    df["task_status"] = df["status"]
    return df.sort_values(
        ["submit_time", "job_name", "task_name"], kind="stable"
    ).reset_index(drop=True)


def task_kind(plan_gpu: float) -> str:
    """Classify one task's GPU request under the FGD model."""
    if pd.isna(plan_gpu) or float(plan_gpu) <= 0.0:
        return "none"
    g = float(plan_gpu)
    if g < WHOLE_PLAN_GPU:
        return "fractional"
    if abs(g % WHOLE_PLAN_GPU) > 1e-9:
        return "unsupported"
    return "whole"


def split_tasks(tasks: pd.DataFrame, max_worker_gpus: dict[str, int] | None = None,
                keep_fractional_gangs: bool = False):
    """Keep replayable tasks; return (kept, drop_counts).

    Drops, with separate counters: missing submission time, zero/NaN GPU,
    unsupported combined requests, fractional gangs, missing GPU type, and
    worker demands larger than any node of their GPU-type domain.
    """
    df = tasks.copy()
    df["kind"] = df["plan_gpu"].map(task_kind)
    missing_submit = df["submit_time"].isna()
    zero_gpu = df["kind"] == "none"
    unsupported = df["kind"] == "unsupported"
    frac_gang = (df["kind"] == "fractional") & (df["worker_num"] > 1)
    if keep_fractional_gangs:  # mixed blocks: fractional gangs are replayed
        frac_gang = pd.Series(False, index=df.index)
    missing_type = df["gpu_type"].isna()
    too_large = pd.Series(False, index=df.index)
    if max_worker_gpus:
        cap = df["gpu_type"].map(max_worker_gpus)
        too_large = df["kind"].eq("whole") & cap.notna() & (df["gpu_request"] > cap)
    drop = missing_submit | zero_gpu | unsupported | frac_gang | missing_type | too_large
    counts = {
        "dropped_missing_submit": int(missing_submit.sum()),
        "dropped_zero_gpu": int(zero_gpu.sum()),
        "dropped_unsupported_combined": int(unsupported.sum()),
        "dropped_fractional_gangs": int(frac_gang.sum()),
        "dropped_missing_gpu_type": int(missing_type.sum()),
        "dropped_too_large_for_domain": int(too_large.sum()),
    }
    return df[~drop].copy(), counts


def max_worker_gpus(spec: pd.DataFrame) -> dict[str, int]:
    """Largest GPU count per gpu_type (0 for CPU-only types)."""
    out = {}
    for gpu_type, sub in spec.groupby("gpu_type", sort=True):
        out[str(gpu_type)] = int(sub["cap_gpu"].max())
    return out


def nodes_to_node_frame(spec: pd.DataFrame, gpu_type: str) -> pd.DataFrame:
    """v2023-schema node rows for one GPU-type domain."""
    sub = spec[spec["gpu_type"] == gpu_type]
    if sub.empty:
        raise ValueError(f"no nodes with gpu_type={gpu_type!r}")
    return pd.DataFrame(
        {
            "sn": sub["machine"].astype(str).to_numpy(),
            "gpu": sub["cap_gpu"].astype(int).to_numpy(),
            "model": sub["gpu_type"].astype(str).to_numpy(),
            "cpu_milli": (sub["cap_cpu"] * WHOLE_MILLI).astype(int).to_numpy(),
            "memory_mib": (sub["cap_mem"] * 1024).astype(int).to_numpy(),
        }
    ).sort_values("sn", kind="stable").reset_index(drop=True)


def domain_capacity(spec: pd.DataFrame, gpu_type: str) -> int:
    sub = spec[spec["gpu_type"] == gpu_type]
    return int(sub["cap_gpu"].sum())


def tasks_to_pod_frame(tasks: pd.DataFrame) -> pd.DataFrame:
    """v2023-schema pod rows, one per INSTANCE (worker) for M construction.

    A task with ``inst_num = k`` contributes k rows: the FGD input M is a
    worker-class distribution, so instance weighting is the faithful one.
    Whole workers map to ``num_gpu = d, gpu_milli = 1000``; fractional
    singletons map to ``num_gpu = 1, gpu_milli = round(d * 1000)``.
    """
    if tasks.empty:
        return pd.DataFrame(
            columns=["name", "creation_time", "cpu_milli", "num_gpu", "gpu_milli", "memory_mib"]
        )
    counts = tasks["worker_num"].astype(int).to_numpy()
    rep = tasks.loc[tasks.index.repeat(counts)].reset_index(drop=True)
    ix = rep.groupby(["job_name", "task_name"], sort=False).cumcount().astype(int)
    rep["name"] = (
        rep["job_name"].astype(str) + ":" + rep["task_name"].astype(str) + ":" + ix.astype(str)
    )
    whole = rep["gpu_request"] >= 1.0
    rep["num_gpu"] = np.where(whole, rep["gpu_request"].astype(int), 1)
    rep["gpu_milli"] = np.where(
        whole, WHOLE_MILLI, (rep["gpu_request"] * WHOLE_MILLI).round().astype(int)
    )
    rep["cpu_milli"] = (rep["plan_cpu"] * 10.0).round().astype(int)
    rep["memory_mib"] = (rep["plan_mem"] * 1024.0).round().astype(int)
    rep["creation_time"] = rep["submit_time"].astype(float)
    out = rep[
        ["name", "creation_time", "cpu_milli", "num_gpu", "gpu_milli", "memory_mib"]
    ].copy()
    return out.sort_values(["creation_time", "name"], kind="stable").reset_index(drop=True)


def tasks_to_items(tasks: pd.DataFrame, capacity: float | None = None,
                   load: float | None = None) -> list[StreamItem]:
    """Stream items in submit-time order; optional Layer-1 demand cap."""
    target = None if capacity is None else float(load) * float(capacity)
    items: list[StreamItem] = []
    cumulative = 0.0
    for row in tasks.itertuples(index=False):
        d = float(row.gpu_request)
        items.append(
            StreamItem(
                gang_id=f"{row.job_name}:{row.task_name}",
                gang_size=int(row.worker_num),
                worker_cpu_milli=round(float(row.plan_cpu) * 10.0),
                worker_gpu_demand=d,
                arrival_index=len(items),
                is_gang=bool(int(row.worker_num) > 1),
                submit_time_s=float(row.submit_time),
                duration_s=(
                    float(row.duration)
                    if pd.notna(row.duration) and float(row.duration) > 0.0
                    else None
                ),
                job_type=str(row.task_status),
                gpu_model=str(row.gpu_type),
            )
        )
        cumulative += int(row.worker_num) * d
        if target is not None and cumulative >= target:
            break
    return items


def stream_header(
    *,
    gpu_type: str,
    capacity: float,
    load: float | None,
    items: list[StreamItem],
    m_whole: Workload,
    m_frac: Workload | None,
    meta: dict,
    window: dict | None = None,
    filter_stats: dict | None = None,
    whole_only: bool = False,
) -> dict:
    """Stream header for REAL PAI-v2020 streams."""
    from ..streams.io import STREAM_SCHEMA_VERSION

    rz = realized_stats(items)
    shapes = realized_shape_mix(items)
    n_gangs = rz["n_gangs"] or 1
    cfg = {
        "gang_fraction": rz["realized_gang_demand_share"],
        "gang_size": None,
        "gang_shapes": shapes,
        "load": None if load is None else float(load),
        "target_demand": None if load is None else float(load) * float(capacity),
        "capacity": float(capacity),
        "multi_gpu_share": None,
        "fractional_share": rz["realized_fractional_demand_share"],
        "real_trace": "pai-v2020",
        "gpu_model_domain": gpu_type,
        "whole_only": bool(whole_only),
        "gang_arrival_share": float(rz["n_gangs"] / max(len(items), 1)),
        "mean_gang_size": float(
            sum(it.gang_size for it in items if it.is_gang) / n_gangs
        ),
    }
    if window is not None:
        cfg["temporal"] = True
        cfg["layer2_window"] = window
    if filter_stats:
        cfg["filter_stats"] = dict(filter_stats)
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
        "source_pod_csv": meta.get("source_task_csv", ""),
        "source_pod_md5": meta.get("source_task_md5", ""),
        "note": (
            "REAL trace replay (Alibaba cluster-trace-gpu-v2020, PAI, CC BY 4.0): "
            "one task with inst_num=k is a spread gang of k homogeneous workers; "
            "arrival = job submission time; gpu_type is a hard per-domain "
            "constraint. Whole-GPU workers only; combined requests and fractional "
            "workers filtered. "
            + (
                "WHOLE-ONLY stream: no fractional background items; pure gang policies."
                if whole_only else
                "Fractional singletons are kept as background items."
            )
            + " Reconstructed production structure, not a synthetic gang mix."
        ),
        "realized": rz,
    }
