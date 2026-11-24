"""TraceEngine — v2023 pod replay with real departures (Layer-1 systems check).

Replays pods in creation_time order, releasing on deletion_time; the engine
owns state mutations and pod lifetimes, the policy only proposes
(node, gpu_idx). Pods with missing/NaN deletion_time are treated as
open-ended (never released within the run, as the reference impl does).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.node import Placement
from ..core.tasks import TaskClass
from ..datasets.v2023 import task_class_from_row

REQUIRED_COLUMNS = ("name", "cpu_milli", "memory_mib", "creation_time", "deletion_time", "gpu_demand")


@dataclass
class PlacementRecord:
    """Journal entry for one admitted pod; the release needs node+placement+cls."""
    node_id: str
    placement: Placement
    cls: TaskClass
    deadline: int


@dataclass
class RoundResult:
    arrivals: int = 0
    placed: int = 0
    unplaced: int = 0
    releases: int = 0


class TraceEngine:
    def __init__(self, cluster, policy, release_before_place: bool = True):
        self.cluster = cluster
        self.policy = policy
        self.release_before_place = release_before_place
        self._active: list[PlacementRecord] = []

    def _release_completed(self, now: int) -> int:
        out = 0
        kept = []
        for rec in self._active:
            if rec.deadline <= now:
                node = self.cluster.node_by_id(rec.node_id)
                node.release(rec.placement, rec.cls)
                if hasattr(self.policy, "observe_placement"):
                    self.policy.observe_placement(node.id, rec.cls, False)
                out += 1
            else:
                kept.append(rec)
        self._active = kept
        return out

    def flush(self) -> int:
        total = 0
        for rec in self._active:
            node = self.cluster.node_by_id(rec.node_id)
            node.release(rec.placement, rec.cls)
            if hasattr(self.policy, "observe_placement"):
                self.policy.observe_placement(node.id, rec.cls, False)
            total += 1
        self._active = []
        return total

    def step(self, row) -> RoundResult:
        now = int(row["creation_time"])
        res = RoundResult(arrivals=1)
        if self.release_before_place:
            res.releases = self._release_completed(now)
        cls = task_class_from_row(row)
        node, gpu_idx = self.policy.decide(self.cluster, cls)
        if node is None:
            res.unplaced = 1
            return res
        placement = node.place(cls, gpu_idx=gpu_idx)
        if hasattr(self.policy, "observe_placement"):
            self.policy.observe_placement(node.id, cls, True)
        deadline = row["deletion_time"]
        deadline = int(deadline) if not pd.isna(deadline) else 10**12
        self._active.append(PlacementRecord(node.id, placement, cls, deadline))
        res.placed = 1
        return res

    def run(self, pods: pd.DataFrame) -> RoundResult:
        missing = [c for c in REQUIRED_COLUMNS if c not in pods.columns]
        if missing:
            raise ValueError(f"pods DataFrame missing columns: {missing}")
        times = pods["creation_time"].to_numpy()
        if (np.diff(times) < 0).any():
            raise ValueError("pods must be sorted by creation_time")
        total = RoundResult()
        for _, row in pods.iterrows():
            r = self.step(row)
            total.arrivals += r.arrivals
            total.placed += r.placed
            total.unplaced += r.unplaced
            total.releases += r.releases
        total.releases += self.flush()
        return total
