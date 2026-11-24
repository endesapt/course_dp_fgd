"""Layer-1 gang runner: closed-loop fill-up over a stream (no departures —
that is Layer 2). The at-capacity metric matches the inflation benchmark
semantics (§9.5): the first event whose cumulative demand reaches 100% of
cluster capacity.

Policies never mutate the cluster (plan/commit split); this runner owns the
two-phase commit and the incremental capacity accounting (CapacityTracker).
"""

import time

from ..items import as_items
from ..metrics.capacity import fully_free_gpus, unallocated_gpu_shares
from ..metrics.results import GangResult
from .capacity import CapacityTracker
from .commit import commit_gang_plan, failure_reason


class GangRunner:
    """Replays stream items in arrival order; every item is a gang of k >= 1
    workers (singletons are k=1). No departures (Layer 1)."""

    def __init__(self, cluster, policy, capacity: float):
        self.cluster = cluster
        self.policy = policy
        self.capacity = float(capacity)

    def run(self, items) -> GangResult:
        items = as_items(items)  # public boundary tolerates dicts; schema-checked
        res = GangResult()
        packed = bool(getattr(self.policy, "packed", False))
        # Whole-GPU placements decrease both counters by exactly k*d, so track
        # them incrementally (recomputing over all nodes per item dominated
        # runtime); the final values are cross-checked below.
        tracker = CapacityTracker(self.cluster)
        unalloc = tracker.unallocated
        fully_free = tracker.fully_free
        cumulative = 0.0
        for item in items:
            k = item.gang_size
            d = item.worker_gpu_demand
            cumulative += k * d
            res.cumulative_demand = cumulative
            cls = item.task_class()
            t0 = time.perf_counter()
            plan = self.policy.decide_gang(self.cluster, cls, k)
            res.decide_runtime_ms.append((time.perf_counter() - t0) * 1e3)
            if plan is None:
                placed = False
                reason = failure_reason(self.cluster, cls, k, packed=packed)
            else:
                _, _, full_consumed = tracker.inspect_plan(cls, plan)
                placed = commit_gang_plan(self.cluster, cls, plan)
                reason = None if placed else "commit_failed"
            if item.is_gang:
                res.gangs_total += 1
                res.gangs_placed += int(placed)
                st = res.shape_stats.setdefault(f"{k}x{d:g}", {"total": 0, "placed": 0})
                st["total"] += 1
                st["placed"] += int(placed)
            else:
                res.singletons_total += 1
                res.singletons_placed += int(placed)
            if placed:
                res.items_placed += 1
                res.workers_placed += k
                unalloc -= k * d
                fully_free -= k * int(d) if d >= 1.0 else full_consumed
            else:
                res.unplaced_items += 1
                res.failure_reasons[reason] = res.failure_reasons.get(reason, 0) + 1
            res.timeline.append(
                {
                    "i": item.arrival_index,
                    "cumulative_demand": cumulative,
                    "unallocated_shares": unalloc,
                    "fully_free": fully_free,
                    "is_gang": item.is_gang,
                    "placed": placed,
                }
            )
            if res.unallocated_at_capacity is None and cumulative >= self.capacity:
                res.unallocated_at_capacity = unalloc
        if abs(unalloc - unallocated_gpu_shares(self.cluster)) > 1e-9 or (
            fully_free != fully_free_gpus(self.cluster)
        ):
            raise AssertionError("incremental free-capacity counters diverged")
        res.unallocated_at_end = unalloc
        return res
