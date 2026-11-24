"""Run-result collectors — the dataclasses runners fill in.

These are the metric collectors: runners own the event loop, results own the
aggregation state (counts, timelines, per-node streaks, wait samples) and the
summary projections (`wait_summary`, `decide_runtime_summary`).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def gpu_demand_bucket(demand: float) -> str:
    """Bucket label for a job's total GPU demand (per-class wait/queue plots)."""
    if demand < 1.0:
        return "<1"
    r = int(demand + 0.5)
    return str(r) if r <= 8 else ">8"


@dataclass
class GangResult:
    """Layer-1 gang run (closed-loop fill-up, no departures)."""

    gangs_total: int = 0
    gangs_placed: int = 0
    workers_placed: int = 0
    singletons_total: int = 0
    singletons_placed: int = 0
    items_placed: int = 0
    unplaced_items: int = 0
    cumulative_demand: float = 0.0
    unallocated_at_capacity: float | None = None
    unallocated_at_end: float | None = None
    failure_reasons: dict = field(default_factory=dict)
    shape_stats: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)
    decide_runtime_ms: list = field(default_factory=list)

    @property
    def gang_placement_rate(self) -> float | None:
        if self.gangs_total == 0:
            return None
        return self.gangs_placed / self.gangs_total

    @property
    def decide_runtime_summary(self) -> dict:
        rt = np.asarray(self.decide_runtime_ms)
        if rt.size == 0:
            return {}
        return {
            "mean": float(rt.mean()),
            "p50": float(np.percentile(rt, 50)),
            "p99": float(np.percentile(rt, 99)),
        }


@dataclass
class InflationResult:
    """Layer-1 paper calibration (Monte-Carlo workload inflation)."""

    placed: int = 0
    unplaced: int = 0
    crossed: bool = False
    cumulative_demand: float = 0.0
    unallocated_at_crossing: float | None = None
    fully_free_at_crossing: int | None = None
    unallocated_at_capacity: float | None = None
    fully_free_at_capacity: int | None = None
    timeline: list[dict] = field(default_factory=list)
    decide_runtime_ms: list[float] = field(default_factory=list)

    @property
    def decide_runtime_summary(self) -> dict:
        rt = np.asarray(self.decide_runtime_ms)
        if rt.size == 0:
            return {}
        return {"mean": float(rt.mean()), "p50": float(np.percentile(rt, 50)),
                "p99": float(np.percentile(rt, 99))}


@dataclass
class ChurnResult:
    """Layer-2 run: arrivals + releases + FCFS queue.

    Releasable-capacity integrals: `_window` values cover the arrival phase
    only (the honest elasticity metric — comparable across policies); `_total`
    adds the drain tail where all nodes are empty anyway (policy-independent).
    """

    items_total: int = 0
    gangs_total: int = 0
    gangs_placed: int = 0
    singletons_total: int = 0
    singletons_placed: int = 0
    items_placed: int = 0
    workers_placed: int = 0
    queued_total: int = 0
    queued_never_placed: int = 0
    max_queue_depth: int = 0
    releases: int = 0
    active_at_end: int = 0
    queue_at_end: int = 0
    waits: list = field(default_factory=list)        # placed items: t_placed - t_submit
    wait_records: list = field(default_factory=list)  # per placed item:
    #   (wait_s, gpu_demand_total, k, is_gang, t_submit) — per-class wait
    #   splits + arrival-bucketed (warm-state per-window) analysis
    queued_waits_end: list = field(default_factory=list)  # still queued: t_end - t_submit
    # queued GPU-demand integrals (full run incl. drain tail, like
    # congested_time): ∫ (queued GPU demand) dt — exposes WHICH job sizes the
    # queue is made of, not just its depth
    queue_gpu_s: float = 0.0
    queue_gpu_s_by_bucket: Counter = field(default_factory=Counter)
    # assembly starvation: ∫ (GPU demand of queued gangs that cannot assemble)
    # dt. Starved = aggregate RAW free GPU shares >= gang demand, but no
    # feasible whole-worker split exists (per-node floor loss / CPU / memory).
    # This is the M01 pilot metric.
    assembly_starved_gpu_s: float = 0.0
    assembly_starved_events: int = 0
    timeline: list = field(default_factory=list)
    decide_runtime_ms: list = field(default_factory=list)
    decide_attempts: int = 0
    span: float = 0.0
    congested_time: float = 0.0    # integral of 1[queue>0] dt over the run
    empty_node_hours: float = 0.0        # ∫ empty_nodes dt, arrival window
    empty_node_hours_total: float = 0.0  # incl. drain tail
    unalloc_gpu_hours: float = 0.0
    unalloc_gpu_hours_total: float = 0.0
    # ∫ busy-GPUs dt (capacity - fully_free); whole-GPU streams make this exact,
    # fractional free shares count as busy — documented approximation
    busy_gpu_hours: float = 0.0        # arrival window
    busy_gpu_hours_total: float = 0.0  # incl. drain tail
    arrival_span: float = 0.0          # t of last arrival (window for util)
    # GPU-demand-weighted wait: Σ w_i·d_i over placed items (d_i = job's total
    # GPU demand, k×worker_demand) — exposes policies that strand BIG jobs
    wait_gpu_s: float = 0.0
    wait_gpu_demand: float = 0.0
    # tracking-M trajectory: hourly [t, [[cpu,gpu,p],...]] snapshots of the
    # estimate the policy scored against (postmortem estimator evidence)
    m_history: list = field(default_factory=list)
    # per-node CONTINUOUS emptiness: node_id -> longest empty streak (s);
    # distinguishes "same nodes empty all week" (reclaimable) from rotating
    # emptiness (not reclaimable)
    node_max_empty_streak: dict = field(default_factory=dict)
    nodes_never_used: int = 0
    # elasticity (Stage-1): billing = ∫ ONLINE capacity dt — nodes pay from
    # node_ready until evict; launch latency costs wait, not money. Static
    # runs leave these at capacity×time equivalents.
    billed_node_hours: float = 0.0   # ∫ online_nodes dt (arrival window)
    billed_gpu_hours: float = 0.0    # ∫ online_gpus dt  (arrival window)
    billed_node_hours_total: float = 0.0   # incl. drain tail
    billed_gpu_hours_total: float = 0.0
    launches: int = 0                # node launch requests issued
    evictions: int = 0               # empty nodes deallocated
    eviction_regret: int = 0         # launches within ~launch_latency of an evict

    @property
    def gang_placement_rate(self) -> float | None:
        return None if self.gangs_total == 0 else self.gangs_placed / self.gangs_total

    @staticmethod
    def _q(ws, ps):
        a = np.asarray(ws, dtype=float)
        if a.size == 0:
            return {}
        return {f"p{int(p*100)}": float(np.quantile(a, p)) for p in ps}

    def wait_summary(self) -> dict:
        ws = np.asarray(self.waits, dtype=float)
        if ws.size == 0:
            return {}
        q = self._q(ws, (0.5, 0.9, 0.99))
        return {
            "mean": float(ws.mean()), "max": float(ws.max()),
            "n_placed": int(ws.size), "n_zero_wait": int((ws <= 0.0).sum()),
            **q,
        }

    def decide_runtime_summary(self) -> dict:
        rt = np.asarray(self.decide_runtime_ms)
        if rt.size == 0:
            return {}
        return {"mean": float(rt.mean()), "p50": float(np.percentile(rt, 50)),
                "p99": float(np.percentile(rt, 99))}


def streak_count(res: ChurnResult, threshold_s: float) -> int:
    """Nodes whose longest continuous empty streak reaches threshold_s."""
    return sum(1 for v in res.node_max_empty_streak.values() if v >= threshold_s)
