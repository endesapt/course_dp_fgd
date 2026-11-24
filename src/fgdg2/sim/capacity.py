"""CapacityTracker — incremental free-capacity + per-node emptiness accounting.

Owns the counters that were embedded in the churn loop: unallocated GPU
shares, fully-free GPU count, empty-node count, per-node empty-since clocks
and longest-streak bookkeeping, and the used-node set. `assert_consistent()`
recomputes everything from the cluster and hard-fails on divergence — the
same guard the runners had, now in one place.
"""

from ..core.rmath import TOL
from ..metrics.capacity import fully_free_gpus, unallocated_gpu_shares


class CapacityTracker:
    def __init__(self, cluster):
        self.cluster = cluster
        self.unallocated = unallocated_gpu_shares(cluster)
        self.fully_free = fully_free_gpus(cluster)
        self.empty_nodes = sum(1 for n in cluster.nodes if n.is_empty)
        # per-node empty streaks: every node starts empty at t=0
        self.empty_since = {n.id: 0.0 for n in cluster.nodes if n.is_empty}
        self.nodes_used: set[str] = set()
        self.max_empty_streak: dict[str, float] = {}
        # live online counts — under elasticity these move; billing = ∫ online
        self.online_nodes = len(cluster.nodes)
        self.online_gpus = cluster.total_gpus
        self.ever_online: set[str] = {n.id for n in cluster.nodes}

    # -- placement path -------------------------------------------------------
    def inspect_plan(self, cls, plan):
        """Pre-commit snapshot (read-only): returns (plan_node_ids, emptied_ids,
        full_consumed_by_frac). `emptied` = nodes currently empty that the plan
        touches (their streaks close at commit); `full_consumed` = fully-free
        GPUs a fractional worker would eat."""
        nodes: list[str] = []
        emptied: list[str] = []
        full_consumed = 0
        seen = set()
        seen_gpu = set()
        for s in plan:
            node = self.cluster.node_by_id(s.node_id)
            # fractional gangs may put several workers on one node / one GPU:
            # a fully-free GPU is consumed once per distinct (node, gpu) pair
            if cls.gpu_demand < 1.0 and s.gpu_idx is not None:
                key = (s.node_id, int(s.gpu_idx))
                if key not in seen_gpu:
                    seen_gpu.add(key)
                    full_consumed += int(node.gpu_free[s.gpu_idx] >= 1.0 - TOL)
            if s.node_id in seen:
                continue
            seen.add(s.node_id)
            nodes.append(s.node_id)
            if node.is_empty:
                emptied.append(s.node_id)
        return nodes, emptied, full_consumed

    def place(self, cls, receipts, plan_nodes, emptied, full_consumed, now: float):
        """Post-commit accounting. Returns GPU shares consumed."""
        used = full = 0
        for node, pl in receipts:
            if pl.kind == "whole":
                used += len(pl.gpu_indices)
                full += len(pl.gpu_indices)
            elif pl.kind == "frac":
                used += pl.gpu_share
        full += full_consumed
        self.unallocated -= used
        self.fully_free -= full
        self.empty_nodes -= len(emptied)
        for nid in plan_nodes:
            self.nodes_used.add(nid)
        for nid in emptied:
            since = self.empty_since.pop(nid, now)
            self.max_empty_streak[nid] = max(
                self.max_empty_streak.get(nid, 0.0), now - since
            )
        return used

    # -- release path ----------------------------------------------------------
    def release(self, cls, receipts, now: float) -> list:
        """Release receipts (grouped by node so emptiness is checked once per
        node, after all its placements are freed). Returns ids of nodes that
        became fully empty at `now` — the runner uses them to schedule
        eviction checks under elasticity."""
        by_node: dict[str, list] = {}
        consumed = 0.0
        newly_empty: list = []
        for node, pl in receipts:
            by_node.setdefault(node.id, []).append(pl)
            if pl.kind == "whole":
                consumed += len(pl.gpu_indices)
            elif pl.kind == "frac":
                consumed += pl.gpu_share
        for nid, pls in by_node.items():
            node = self.cluster.node_by_id(nid)
            for pl in pls:
                if pl.kind == "frac":
                    g = pl.gpu_indices[0]
                    node.release(pl, cls)
                    if node.gpu_free[g] >= 1.0 - TOL:
                        self.fully_free += 1
                else:
                    node.release(pl, cls)
                    if pl.kind == "whole":
                        self.fully_free += len(pl.gpu_indices)
            if node.is_empty:
                self.empty_nodes += 1
                self.empty_since[nid] = now
                newly_empty.append(nid)
        self.unallocated += consumed
        return newly_empty

    # -- elasticity ------------------------------------------------------------
    def evict(self, nid, now: float):
        """Remove a fully-empty node from the online pool (scale-down).
        Closes its streak and drops its free capacity from every counter."""
        node = self.cluster.node_by_id(nid)
        if not node.is_empty:
            raise ValueError(f"cannot evict non-empty node {nid}")
        since = self.empty_since.pop(nid, now)
        self.max_empty_streak[nid] = max(
            self.max_empty_streak.get(nid, 0.0), now - since)
        self.empty_nodes -= 1
        self.unallocated -= float(node.gpu_free.sum())
        self.fully_free -= node.free_slots
        self.online_nodes -= 1
        self.online_gpus -= node.num_gpus
        return self.cluster.remove_node(nid)

    def add(self, node, now: float) -> None:
        """Bring a (fresh, empty) node online (scale-up)."""
        self.cluster.add_node(node)
        self.ever_online.add(node.id)
        self.online_nodes += 1
        self.online_gpus += node.num_gpus
        self.unallocated += float(node.gpu_free.sum())
        self.fully_free += node.free_slots
        if node.is_empty:
            self.empty_nodes += 1
            self.empty_since[node.id] = now

    # -- finalize ----------------------------------------------------------------
    def close_streaks(self, t_window_end: float) -> None:
        """Close open empty streaks at the ARRIVAL-WINDOW end — a streak still
        open when arrivals stop measures emptiness during the workload; the
        drain tail would inflate it for free."""
        for nid, since in self.empty_since.items():
            self.max_empty_streak[nid] = max(
                self.max_empty_streak.get(nid, 0.0),
                max(t_window_end - since, 0.0),
            )

    @property
    def nodes_never_used(self) -> int:
        return len(self.ever_online) - len(self.nodes_used)

    def assert_consistent(self) -> None:
        if (
            abs(self.unallocated - unallocated_gpu_shares(self.cluster)) > 1e-9
            or self.fully_free != fully_free_gpus(self.cluster)
            or self.empty_nodes
            != sum(1 for n in self.cluster.nodes if n.is_empty)
            or self.online_nodes != len(self.cluster.nodes)
            or abs(self.online_gpus - self.cluster.total_gpus) > 1e-9
        ):
            raise AssertionError("incremental free-capacity counters diverged")
