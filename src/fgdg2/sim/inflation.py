"""InflationRunner — Monte-Carlo workload inflation (paper §6.1).

Sample pods with replacement, place while capacity lasts; the key metric is
unallocated GPUs at the moment cumulative GPU requests reach 100% of cluster
capacity. No departures: the inflation question is pure packing quality.
"""

import time

import numpy as np
import pandas as pd

from ..datasets.v2023 import task_class_from_row
from ..metrics.capacity import fully_free_gpus, unallocated_gpu_shares
from ..metrics.results import InflationResult


class InflationRunner:
    def __init__(self, cluster, policy, capacity: float, stop_fraction: float = 1.0):
        self.cluster = cluster
        self.policy = policy
        self.capacity = float(capacity)
        self.stop_at = float(capacity) * stop_fraction

    def run(self, pods: pd.DataFrame, seed: int = 0, sampling: str = "replacement") -> InflationResult:
        if sampling not in ("replacement", "sequential"):
            raise ValueError(f"unknown sampling mode {sampling!r}")
        rng = np.random.default_rng(seed)
        res = InflationResult()
        cumulative = 0.0
        n = len(pods)
        draws = 0
        limit = 4 * n if sampling == "replacement" else n
        while cumulative < self.stop_at and draws < limit:
            if sampling == "replacement":
                row = pods.iloc[int(rng.integers(n))]
            else:
                row = pods.iloc[draws]
            demand = float(row["gpu_demand"])
            cumulative += demand
            res.cumulative_demand = cumulative
            cls = task_class_from_row(row)
            t0 = time.perf_counter()
            node, gpu_idx = self.policy.decide(self.cluster, cls)
            dt = (time.perf_counter() - t0) * 1e3
            res.decide_runtime_ms.append(dt)
            placed = node is not None
            if placed:
                self.cluster.place(node, cls, gpu_idx=gpu_idx)
                if hasattr(self.policy, "observe_placement"):
                    self.policy.observe_placement(node.id, cls, True)
                res.placed += 1
            else:
                res.unplaced += 1
            res.timeline.append(
                {
                    "i": draws,
                    "cumulative_demand": cumulative,
                    "unallocated_shares": unallocated_gpu_shares(self.cluster),
                    "fully_free": fully_free_gpus(self.cluster),
                    "placed": placed,
                }
            )
            if res.unallocated_at_capacity is None and cumulative >= self.capacity:
                res.unallocated_at_capacity = res.timeline[-1]["unallocated_shares"]
                res.fully_free_at_capacity = res.timeline[-1]["fully_free"]
            if cumulative >= self.stop_at:
                res.crossed = True
                res.unallocated_at_crossing = unallocated_gpu_shares(self.cluster)
                res.fully_free_at_crossing = fully_free_gpus(self.cluster)
                break
            draws += 1
        return res
